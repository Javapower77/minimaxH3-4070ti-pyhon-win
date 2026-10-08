"""Local ComfyUI backend for pruned MiniMax-H3 FL2VA checkpoints."""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import requests
import websocket
from PIL import Image

from .config import AppConfig, GenerationRequest, LoRASpec, ROOT
from .diagnostics import event, stage, traced_generation, workflow_metadata, start_worker_capture
from .assets import LATENT_UPSCALER_FILE
from .frames import duration_to_frames, frames_to_duration
from .media import fit_image_without_crop, load_rgb_image
from .lora import validate_pruned_fl2va_lora
from .prompts import expand_prompt
from .resolution import resolve_output_size, size_from_image

logger = logging.getLogger(__name__)

COMFY_HOST = "127.0.0.1"
COMFY_PORT = 18188
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
COMFY_ROOT = ROOT / ".runtime" / "ComfyUI"
COMFY_MODEL = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
COMFY_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
COMFY_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
COMFY_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
DEFAULT_UPSCALE_MODEL = "RealESRGAN_x4plus.safetensors"
DEFAULT_INTERPOLATION_MODEL = "rife_v4.25_lite.safetensors"
DEFAULT_FACE_RESTORE_MODEL = "codeformer.pth"
ProgressCallback = Callable[[float, str], None]

NODE_STAGES = {
    "1": (0.08, "🔷 Loading configured pruned transformer"),
    "2": (0.11, "🔤 Loading Qwen3-VL text encoder"),
    "3": (0.13, "🎞️ Loading video VAE"),
    "4": (0.14, "🔊 Loading audio VAE"),
    "7": (0.15, "🖼️ Loading first frame"),
    "8": (0.16, "🖼️ Loading last frame"),
    "9": (0.19, "🧠 Encoding prompt and keyframes"),
    "10": (0.20, "🎲 Preparing seeded noise"),
    "11": (0.21, "⚙️ Selecting Euler sampler"),
    "12": (0.22, "📈 Building simple sigma schedule"),
    "13": (0.23, "🧭 Preparing guidance"),
    "15": (0.25, "⚡ Sampling video and audio latents"),
    "41": (0.82, "🔎 Upscaling video latents (learned 3D model)"),
    "16": (0.84, "🎞️ Decoding video frames"),
    "17": (0.88, "🔊 Decoding stereo audio"),
    "18": (0.92, "🎬 Creating synchronized video"),
    "19": (0.96, "💾 Encoding H.264 MP4"),
    "30": (0.84, "🔎 Loading detail upscaler"),
    "31": (0.87, "✨ Restoring frame detail (tiled)"),
    "32": (0.91, "📐 Resizing enhanced frames"),
    "33": (0.93, "🔬 Sharpening enhanced frames"),
    "34": (0.84, "🙂 Restoring detected faces"),
    "35": (0.94, "🎞️ Loading RIFE interpolation model"),
    "36": (0.95, "🏃 Interpolating output frames"),
    "37": (0.96, "🎯 Resampling target frame rate"),
    "50": (0.15, "🎞️ Loading restoration source reference"),
    "51": (0.16, "🎞️ Reading source video frames"),
    "52": (0.19, "🧠 Encoding restoration prompt"),
    "53": (0.20, "🎞️ Preparing restoration video latents"),
    "54": (0.24, "🧭 Encoding source video/audio references"),
}


class PrunedComfyBackend:
    """Submit pruned FL2VA jobs to an isolated loopback-only ComfyUI worker."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.root = Path(os.getenv("MINIMAX_H3_COMFY_ROOT", COMFY_ROOT)).resolve()
        self.host = os.getenv("MINIMAX_H3_COMFY_HOST", COMFY_HOST)
        self.port = int(os.getenv("MINIMAX_H3_COMFY_PORT", str(COMFY_PORT)))
        self.url = os.getenv(
            "MINIMAX_H3_COMFY_URL", f"http://{self.host}:{self.port}"
        ).rstrip("/")
        self.process: Optional[subprocess.Popen] = None
        self.output_dir = (config.output_dir / "comfyui").resolve()
        self.input_dir = (config.upload_dir / "comfyui").resolve()
        self.model = config.comfy_model
        self.text_encoder = config.comfy_text_encoder
        self.upscale_model = config.upscale_model
        self.interpolation_model = config.interpolation_model
        self.face_restore_model = config.face_restore_model

    def _required_paths(self) -> list[Path]:
        return [
            self._python_executable(),
            self.root / "main.py",
            self.root / "models/diffusion_models" / self.model,
            self.root / "models/text_encoders" / self.text_encoder,
            self.root / "models/vae" / COMFY_VIDEO_VAE,
            self.root / "models/vae" / COMFY_AUDIO_VAE,
        ]

    def _python_executable(self) -> Path:
        configured = os.getenv("MINIMAX_H3_COMFY_PYTHON")
        if configured:
            return Path(configured).resolve()
        if os.name == "nt":
            return self.root / ".venv" / "Scripts" / "python.exe"
        return self.root / ".venv" / "bin" / "python"

    def validate_installation(self) -> None:
        missing = [str(path) for path in self._required_paths() if not path.exists()]
        if missing:
            raise RuntimeError(
                "Pruned FL2VA backend is incomplete. Run "
                "python scripts/setup_pruned_backend.py first. Missing: " + ", ".join(missing)
            )

    def _is_ready(self) -> bool:
        try:
            response = requests.get(f"{self.url}/system_stats", timeout=2)
            return response.ok
        except requests.RequestException:
            return False

    def ensure_worker(
        self,
        timeout: float = 90.0,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        report = progress_callback or (lambda fraction, message: None)
        if self._is_ready():
            event(logger, "worker_reuse", endpoint=self.url)
            report(0.04, "🟣 Pruned worker ready · validating local models…")
            return
        report(0.02, "🟣 Starting isolated pruned ComfyUI worker…")
        self.validate_installation()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        log_path = ROOT / "logs" / "comfyui-worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self._python_executable()),
            str(self.root / "main.py"),
            "--listen", self.host,
            "--port", str(self.port),
            "--disable-auto-launch",
            "--disable-all-custom-nodes",
            "--whitelist-custom-nodes", "facerestore_cf", "minimax_h3_nodes",
            "--reserve-vram", str(self.config.comfy_reserve_vram_gb),
            "--output-directory", str(self.output_dir),
            "--input-directory", str(self.input_dir),
        ]
        if self.config.comfy_cache_none:
            command.append("--cache-none")
        if self.config.comfy_cpu_vae:
            command.append("--cpu-vae")
        env = os.environ.copy()
        env.update({
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DIFFUSERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        })
        process_options: dict[str, Any] = {}
        if os.name == "nt":
            process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            process_options["start_new_session"] = True
        event(logger, "worker_start", command=command, log_file=str(log_path),
              reserve_vram_gb=self.config.comfy_reserve_vram_gb,
              cache_none=self.config.comfy_cache_none, cpu_vae=self.config.comfy_cpu_vae)
        with stage(logger, "worker_spawn"):
            self.process = subprocess.Popen(
                command, cwd=self.root, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                bufsize=1, **process_options,
            )
            start_worker_capture(self.process, log_path)
        startup_started = time.perf_counter()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._is_ready():
                event(logger, "worker_ready", pid=self.process.pid,
                    duration_seconds=time.perf_counter() - startup_started)
                logger.info("Pruned ComfyUI worker ready at %s", self.url)
                report(0.04, "🟣 Pruned worker started · validating local models…")
                return
            if self.process.poll() is not None:
                raise RuntimeError(f"Pruned ComfyUI worker exited. See {log_path}")
            time.sleep(0.5)
        raise RuntimeError(f"Timed out starting pruned ComfyUI worker. See {log_path}")

    def _log_worker_stats(self, reason: str, prompt_id: str | None = None) -> None:
        """Read-only, best-effort snapshots; never dump argv, queue or HTTP bodies."""
        try:
            response = requests.get(f"{self.url}/system_stats", timeout=2)
            response.raise_for_status()
            stats = response.json()
            system = stats.get("system", {})
            fields = ("ram_total", "ram_free", "comfyui_version", "python_version", "pytorch_version")
            device_fields = ("name", "type", "index", "vram_total", "vram_free", "torch_vram_total", "torch_vram_free")
            event(logger, "worker_stats", reason=reason, prompt_id=prompt_id,
                  system={key: system.get(key) for key in fields},
                  devices=[{key: item.get(key) for key in device_fields}
                           for item in stats.get("devices", [])])
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            event(logger, "worker_stats_unavailable", reason=reason, prompt_id=prompt_id)

    def _upload_image(self, image: Image.Image, name: str) -> str:
        payload = io.BytesIO()
        image.save(payload, format="PNG")
        payload.seek(0)
        response = requests.post(
            f"{self.url}/upload/image",
            files={"image": (name, payload, "image/png")},
            data={"type": "input", "overwrite": "true"},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        return str(Path(result.get("subfolder", "")) / result["name"])

    def _sync_loras(self, paths: list[Path]) -> None:
        """Expose selected app LoRAs to ComfyUI and verify live discovery."""
        target_dir = self.root / "models/loras"
        target_dir.mkdir(parents=True, exist_ok=True)
        expected: set[str] = set()
        for value in paths:
            source = Path(value).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"LoRA file missing: {source}")
            expected.add(source.name)
            target = target_dir / source.name
            # Never unlink a source already resident in this worker (including
            # symlinks/hard links). Still check live discovery below.
            if target.resolve() == source or (target.exists() and target.samefile(source)):
                continue
            if target.exists() or target.is_symlink():
                target.unlink()
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.symlink_to(source)
            except OSError:
                # Windows requires Developer Mode or elevated privileges for
                # symlinks. A copy preserves behavior for local workstations.
                shutil.copy2(source, temporary)
            temporary.replace(target)

        response = requests.get(f"{self.url}/models/loras", timeout=30)
        response.raise_for_status()
        available = set(response.json())
        missing = sorted(expected - available)
        if missing:
            raise RuntimeError(
                "Pruned worker did not discover selected LoRA files: "
                + ", ".join(missing)
                + ". Restart the application or rerun scripts/setup_pruned_backend.py."
            )

    @staticmethod
    def build_workflow(
        *,
        prompt: str,
        width: int,
        height: int,
        frames: int,
        seed: int,
        nfe: int,
        loras: list[tuple[str, float]],
        first_image: Optional[str] = None,
        last_image: Optional[str] = None,
        filename_prefix: str = "pruned/minimax_h3",
        model: str = COMFY_MODEL,
        text_encoder: str = COMFY_CLIP,
        upscale: bool = False,
        upscale_model: str = DEFAULT_UPSCALE_MODEL,
        upscale_factor: float = 2.0,
        detailer_strength: float = 0.0,
        target_fps: float = 23.976,
        interpolation_model: str = DEFAULT_INTERPOLATION_MODEL,
        face_restore: bool = False,
        face_restore_model: str = DEFAULT_FACE_RESTORE_MODEL,
        face_fidelity: float = 0.7,
        video_shift: float = 12.0,
        audio_shift: float = 3.0,
        latent_upscale: bool = False,
        latent_upscale_factor: float = 1.5,
        latent_upscale_device: str = "cpu",
    ) -> dict[str, dict[str, Any]]:
        image_node = "16"
        if latent_upscale:
            if latent_upscale_factor not in (1.5, 2.0):
                raise ValueError("latent_upscale_factor must be 1.5 or 2.0")
            if latent_upscale_device not in ("cpu", "cuda"):
                raise ValueError("latent_upscale_device must be cpu or cuda")
        graph: dict[str, dict[str, Any]] = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": model, "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": text_encoder, "type": "minimax", "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": COMFY_VIDEO_VAE}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": COMFY_AUDIO_VAE}},
            "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
            "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["3", 0]}},
            "17": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["15", 0], "vae": ["4", 0]}},
            "18": {"class_type": "CreateVideo", "inputs": {"images": [image_node, 0], "audio": ["17", 0], "fps": float(target_fps), "bit_depth": 8, "color_space": "sRGB"}},
            "19": {"class_type": "SaveVideo", "inputs": {"video": ["18", 0], "filename_prefix": filename_prefix, "format": "mp4", "codec": "h264"}},
        }
        if latent_upscale:
            graph["41"] = {
                "class_type": "MiniMaxH3LearnedLatentUpscale",
                "inputs": {"latent": ["15", 0], "model_name": LATENT_UPSCALER_FILE,
                           "scale": float(latent_upscale_factor), "device": latent_upscale_device},
            }
            # Only the video branch changes; preserve the original generated audio.
            graph["16"] = {"class_type": "VAEDecodeTiled", "inputs": {
                "samples": ["41", 0], "vae": ["3", 0], "tile_size": 256,
                "overlap": 64, "temporal_size": 64, "temporal_overlap": 8,
            }}
        if face_restore:
            graph["34"] = {
                "class_type": "FaceRestoreModelLoader",
                "inputs": {"model_name": face_restore_model},
            }
            graph["38"] = {
                "class_type": "FaceRestoreCFWithModel",
                "inputs": {
                    "facerestore_model": ["34", 0],
                    "image": [image_node, 0],
                    "facedetection": "retinaface_mobile0.25",
                    "codeformer_fidelity": float(face_fidelity),
                },
            }
            image_node = "38"
        if upscale:
            if not 1.0 <= float(upscale_factor) <= 2.0:
                raise ValueError("upscale_factor must be between 1.0 and 2.0 on the low-VRAM path")
            graph["30"] = {
                "class_type": "UpscaleModelLoader",
                "inputs": {"model_name": upscale_model},
            }
            graph["31"] = {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {"upscale_model": ["30", 0], "image": [image_node, 0]},
            }
            graph["32"] = {
                "class_type": "ImageScaleBy",
                "inputs": {
                    "image": ["31", 0],
                    "upscale_method": "lanczos",
                    "scale_by": float(upscale_factor) / 4.0,
                },
            }
            image_node = "32"
            if float(detailer_strength) > 0:
                graph["33"] = {
                    "class_type": "ImageSharpen",
                    "inputs": {
                        "image": [image_node, 0],
                        "sharpen_radius": 1,
                        "sigma": 1.0,
                        "alpha": float(detailer_strength),
                    },
                }
                image_node = "33"
            graph["18"]["inputs"]["images"] = [image_node, 0]
        if float(target_fps) > 24.0:
            multiplier = math.ceil(float(target_fps) / 24.0)
            graph["35"] = {
                "class_type": "FrameInterpolationModelLoader",
                "inputs": {"model_name": interpolation_model},
            }
            graph["36"] = {
                "class_type": "FrameInterpolate",
                "inputs": {
                    "interp_model": ["35", 0],
                    "images": [image_node, 0],
                    "multiplier": multiplier,
                },
            }
            graph["37"] = {
                "class_type": "MiniMaxH3ResampleFramesToFPS",
                "inputs": {
                    "images": ["36", 0],
                    "source_fps": 24.0 * multiplier,
                    "target_fps": float(target_fps),
                },
            }
            image_node = "37"
        graph["18"]["inputs"]["images"] = [image_node, 0]
        model_node = "1"
        for index, (filename, strength) in enumerate(loras, start=1):
            node = str(20 + index)
            graph[node] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"model": [model_node, 0], "lora_name": filename, "strength_model": float(strength)},
            }
            model_node = node
        if video_shift != 12.0 or audio_shift != 3.0:
            graph["40"] = {
                "class_type": "MiniMaxH3SigmaShift",
                "inputs": {"model": [model_node, 0], "shift_video": float(video_shift), "shift_audio": float(audio_shift)},
            }
            model_node = "40"
        graph["12"] = {"class_type": "BasicScheduler", "inputs": {"model": [model_node, 0], "scheduler": "simple", "steps": int(nfe), "denoise": 1.0}}
        graph["13"] = {"class_type": "BasicGuider", "inputs": {"model": [model_node, 0], "conditioning": ["9", 0]}}
        graph["15"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["13", 0], "sampler": ["11", 0], "sigmas": ["12", 0], "latent_image": ["9", 1]}}
        condition_inputs: dict[str, Any] = {
            "clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
            "width": int(width), "height": int(height), "length": int(frames),
        }
        if first_image:
            graph["7"] = {"class_type": "LoadImage", "inputs": {"image": first_image}}
            condition_inputs["first_frame"] = ["7", 0]
        if last_image:
            graph["8"] = {"class_type": "LoadImage", "inputs": {"image": last_image}}
            condition_inputs["last_frame"] = ["8", 0]
        graph["9"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": condition_inputs}
        return graph

    def _wait_for_output(
        self,
        prompt_id: str,
        *,
        socket,
        started: float,
        progress_callback: Optional[ProgressCallback] = None,
        timeout: float = 3600.0,
    ) -> tuple[Path, dict[str, float]]:
        report = progress_callback or (lambda fraction, message: None)
        deadline = time.monotonic() + timeout
        timings: dict[str, float] = {}
        active_node: Optional[str] = None
        active_started = time.perf_counter()
        last_fraction = 0.08
        current_stage = "⏳ Waiting in worker queue"
        last_heartbeat = 0.0
        last_stats = time.perf_counter()
        last_sample_time: float | None = None
        last_sample_value = 0
        last_sample_node: str | None = None

        def emit(fraction: float, label: str) -> None:
            nonlocal last_fraction
            last_fraction = max(last_fraction, float(fraction))
            report(last_fraction, label)

        while time.monotonic() < deadline:
            try:
                raw = socket.recv()
                if isinstance(raw, str):
                    message = json.loads(raw)
                    event_type = message.get("type")
                    data = message.get("data", {})
                    if data.get("prompt_id") in {None, prompt_id}:
                        if event_type == "executing":
                            node = data.get("node")
                            now = time.perf_counter()
                            if active_node is not None:
                                event(logger, "node_end", prompt_id=prompt_id, node=active_node,
                                      duration_seconds=now - active_started)
                                timings[NODE_STAGES.get(active_node, (0, f"node_{active_node}"))[1]] = (
                                    timings.get(NODE_STAGES.get(active_node, (0, f"node_{active_node}"))[1], 0.0)
                                    + now - active_started
                                )
                            active_node = str(node) if node is not None else None
                            active_started = now
                            if active_node is not None:
                                last_sample_time = None
                                last_sample_value = 0
                                last_sample_node = active_node
                                event(logger, "node_start", prompt_id=prompt_id, node=active_node,
                                      elapsed_seconds=now - started)
                                fraction, current_stage = NODE_STAGES.get(
                                    active_node, (0.24, f"🔧 Processing node {active_node}")
                                )
                                emit(fraction, f"{current_stage} · {now - started:.1f}s elapsed")
                        elif event_type == "progress":
                            now = time.perf_counter()
                            progress_node = str(data.get("node") or active_node or "unknown")
                            value = int(data.get("value", 0))
                            total = max(1, int(data.get("max", 1)))
                            fraction = value / total
                            sampling = progress_node == "15"
                            current_stage = (f"⚡ Sampling step {value}/{total} · {fraction * 100:.0f}%"
                                             if sampling else f"🔧 Node {progress_node} progress {value}/{total}")
                            if progress_node != last_sample_node or value < last_sample_value:
                                last_sample_time = None
                                last_sample_value = 0
                                last_sample_node = progress_node
                            # Sampling has few expensive steps. Other nodes are rate limited.
                            if value != last_sample_value and (sampling or last_sample_time is None
                                                              or now - last_sample_time >= 5.0 or value >= total):
                                interval = now - (last_sample_time if last_sample_time is not None else active_started)
                                delta = max(1, value - last_sample_value)
                                event(logger, "sample_progress" if sampling else "node_progress",
                                      prompt_id=prompt_id, node=progress_node, value=value, total=total,
                                      elapsed_seconds=now - started, interval_seconds=interval,
                                      seconds_per_step=interval / delta,
                                      first_step_includes_loading=last_sample_time is None)
                                last_sample_time, last_sample_value = now, value
                            emit(
                                0.25 + 0.57 * fraction if sampling else last_fraction,
                                f"{current_stage} · "
                                f"elapsed {time.perf_counter() - started:.1f}s",
                            )
                        elif event_type == "execution_error":
                            # Never include worker messages, tracebacks or input dumps.
                            node_id = str(data.get("node_id", ""))
                            node_id = node_id if node_id.isdecimal() and len(node_id) <= 16 else "unknown"
                            exception_type = data.get("exception_type")
                            if not isinstance(exception_type, str) or not re.fullmatch(
                                r"[A-Za-z_][A-Za-z0-9_.]{0,127}", exception_type
                            ):
                                exception_type = "worker error"
                            event(logger, "worker_error", prompt_id=prompt_id, node=node_id,
                                  exception_type=exception_type)
                            raise RuntimeError(
                                f"Pruned ComfyUI generation failed at node {node_id} "
                                f"({exception_type}). See worker log."
                            )
            except websocket.WebSocketTimeoutException:
                now = time.perf_counter()
                if now - last_heartbeat >= 5.0:
                    emit(last_fraction, f"{current_stage} · {now - started:.1f}s elapsed")
                    event(logger, "worker_heartbeat", prompt_id=prompt_id, node=active_node,
                          elapsed_seconds=now - started, node_elapsed_seconds=now - active_started)
                    last_heartbeat = now
            now = time.perf_counter()
            if now - last_stats >= 30.0:
                self._log_worker_stats("progress", prompt_id)
                last_stats = now
            response = requests.get(f"{self.url}/history/{prompt_id}", timeout=30)
            response.raise_for_status()
            history = response.json().get(prompt_id)
            if history:
                status = history.get("status", {})
                if status.get("status_str") == "error":
                    event(logger, "worker_error", prompt_id=prompt_id, source="history")
                    raise RuntimeError("Pruned ComfyUI generation failed. See worker log.")
                for output in history.get("outputs", {}).values():
                    for key in ("videos", "files", "gifs", "images"):
                        for item in output.get(key, []):
                            filename = item.get("filename")
                            if not filename:
                                continue
                            path = self.output_dir / item.get("subfolder", "") / filename
                            if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm"}:
                                now = time.perf_counter()
                                if active_node is not None:
                                    event(logger, "node_end", prompt_id=prompt_id, node=active_node,
                                        duration_seconds=now - active_started)
                                    label = NODE_STAGES.get(active_node, (0, f"node_{active_node}"))[1]
                                    timings[label] = timings.get(label, 0.0) + now - active_started
                                event(
                                    logger, "worker_complete", prompt_id=prompt_id,
                                    elapsed_seconds=now - started, timings=timings,
                                )
                                self._log_worker_stats("completion", prompt_id)
                                return path.resolve(), timings
        raise TimeoutError(f"Timed out waiting for pruned generation {prompt_id}")

    @traced_generation
    def generate(
        self,
        request: GenerationRequest,
        spec: LoRASpec,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        from .pipeline import GenerationResult

        report = progress_callback or (lambda fraction, message: None)
        overall_started = time.perf_counter()
        timings: dict[str, float] = {}
        if request.latent_upscale:
            model_path = self.root / "models" / "latent_upscale_models" / LATENT_UPSCALER_FILE
            node_path = self.root / "custom_nodes" / "minimax_h3_nodes" / "upstream_latent_3d.py"
            if not model_path.is_file() or not node_path.is_file():
                raise FileNotFoundError(
                    "Optional learned latent upscaler is missing. Download model type latent_upscaler "
                    "with scripts/download_models.py, then restart the worker."
                )
        with stage(logger, "worker_startup"):
            self.ensure_worker(progress_callback=report)
        if request.latent_upscale:
            response = requests.get(f"{self.url}/object_info/MiniMaxH3LearnedLatentUpscale", timeout=30)
            response.raise_for_status()
            if "MiniMaxH3LearnedLatentUpscale" not in response.json():
                raise RuntimeError("Restart the ComfyUI worker to load the learned latent upscaler node.")
        report(0.05, "🖼️ Preparing and uploading reference frames…")
        prepare_started = time.perf_counter()
        first = load_rgb_image(request.first_image) if request.first_image else None
        last = load_rgb_image(request.last_image) if request.last_image else None
        anchor = first or last
        megapixels = request.megapixels or spec.megapixels
        if anchor is not None:
            width, height = size_from_image(anchor.width, anchor.height, megapixels)
        else:
            aspect = request.aspect_ratio if request.aspect_ratio != "auto" else self.config.aspect_ratio
            width, height = resolve_output_size(megapixels, "16:9" if aspect == "auto" else aspect)
        first_name = last_name = None
        token = uuid.uuid4().hex
        if first is not None:
            first_name = self._upload_image(fit_image_without_crop(first, width, height), f"{token}-first.png")
        if last is not None:
            last_name = self._upload_image(fit_image_without_crop(last, width, height), f"{token}-last.png")
        frames = duration_to_frames(request.duration_seconds, fps=self.config.fps)
        nfe = int(request.nfe or spec.nfe)
        if request.upscale:
            upscaler = self.root / "models" / "upscale_models" / self.upscale_model
            if not upscaler.is_file():
                raise FileNotFoundError(
                    f"Upscale model missing: {upscaler}. Rerun scripts/setup_windows.ps1 "
                    "or scripts/setup_pruned_backend.py --skip-install while online."
                )
        if request.target_fps > 24.0:
            interpolation = self.root / "models" / "frame_interpolation" / self.interpolation_model
            if not interpolation.is_file():
                raise FileNotFoundError(f"Frame interpolation model missing: {interpolation}")
        if request.face_restore:
            face_model = self.root / "models" / "facerestore_models" / self.face_restore_model
            if not face_model.is_file():
                raise FileNotFoundError(f"Face restoration model missing: {face_model}")
        selected_path = spec.resolved_path(self.config.lora_dir)
        if spec.lora_format == "dmad_diffusers":
            from .dmad import prepare_dmad_lora

            if selected_path is None:
                raise ValueError("DMAD catalog entry requires a checkpoint path")
            report(0.06, "🧬 Preparing original DMAD adapter for the native H3 worker…")
            selected_path = prepare_dmad_lora(selected_path, self.config.lora_dir / ".converted")
        selected_loras = [(selected_path.name if selected_path else spec.filename, float(request.lora_scale))]
        selected_paths = [selected_path]
        for path, scale in request.extra_loras:
            validate_pruned_fl2va_lora(Path(path))
            selected_loras.append((Path(path).name, float(scale)))
            selected_paths.append(Path(path))
        report(0.07, f"🧬 Synchronizing {len(selected_loras)} pruned LoRA adapter(s)…")
        self._sync_loras([Path(path) for path in selected_paths if path is not None])
        timings["prepare"] = time.perf_counter() - prepare_started
        event(logger, "stage_end", stage="prepare", duration_seconds=timings["prepare"])
        graph = self.build_workflow(
            prompt=expand_prompt(request.prompt, structured=request.structured_prompt),
            width=width, height=height, frames=frames, seed=int(request.seed), nfe=nfe,
            loras=selected_loras, first_image=first_name, last_image=last_name,
            filename_prefix=f"pruned/{time.strftime('%Y%m%d-%H%M%S')}_{request.mode}_seed{request.seed}",
            model=self.model,
            text_encoder=self.text_encoder,
            upscale=request.upscale,
            upscale_model=self.upscale_model,
            upscale_factor=request.upscale_factor,
            detailer_strength=request.detailer_strength,
            target_fps=request.target_fps,
            interpolation_model=self.interpolation_model,
            face_restore=request.face_restore,
            face_restore_model=self.face_restore_model,
            face_fidelity=request.face_fidelity,
            video_shift=spec.video_shift if request.video_shift is None else request.video_shift,
            audio_shift=spec.audio_shift if request.audio_shift is None else request.audio_shift,
            latent_upscale=request.latent_upscale,
            latent_upscale_factor=request.latent_upscale_factor,
            latent_upscale_device=request.latent_upscale_device,
        )
        started = time.perf_counter()
        event(logger, "workflow_metadata", **workflow_metadata(graph, self.root / "models"))
        self._log_worker_stats("submission")
        report(0.08, "📤 Submitting pruned workflow to local worker…")
        socket = websocket.create_connection(
            self.url.replace("http://", "ws://").replace("https://", "wss://")
            + f"/ws?clientId={token}",
            timeout=2,
        )
        try:
            with stage(logger, "submit"):
                response = requests.post(f"{self.url}/prompt", json={"prompt": graph, "client_id": token}, timeout=60)
                if not response.ok:
                    raise RuntimeError(f"Pruned workflow rejected (HTTP {response.status_code}). See worker log.")
            event(logger, "worker_submit", prompt_id=response.json()["prompt_id"])
            output, worker_timings = self._wait_for_output(
                response.json()["prompt_id"],
                socket=socket,
                started=started,
                progress_callback=report,
            )
            timings.update(worker_timings)
        finally:
            socket.close()
            try:
                requests.post(
                    f"{self.url}/free",
                    json={"unload_models": True, "free_memory": True},
                    timeout=30,
                )
            except requests.RequestException:
                logger.warning("Could not request ComfyUI model unload", exc_info=True)
        elapsed = time.perf_counter() - started
        latent_width = round(width * request.latent_upscale_factor / 32) * 32 if request.latent_upscale else width
        latent_height = round(height * request.latent_upscale_factor / 32) * 32 if request.latent_upscale else height
        output_width = round(latent_width * request.upscale_factor) if request.upscale else latent_width
        output_height = round(latent_height * request.upscale_factor) if request.upscale else latent_height
        output_frames = (
            round((frames - 1) * request.target_fps / self.config.fps) + 1
            if request.target_fps > self.config.fps
            else frames
        )
        suffix = "_upscaled" if request.upscale else ""
        if request.latent_upscale:
            suffix = "_latent" + suffix
        final = self.config.output_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{request.mode}_seed{request.seed}_{output_width}x{output_height}_pruned{suffix}.mp4"
        final.parent.mkdir(parents=True, exist_ok=True)
        if output != final.resolve():
            with stage(logger, "output_copy"):
                shutil.copy2(output, final)
        timings["worker"] = elapsed
        timings["total"] = time.perf_counter() - overall_started
        report(1.0, f"✅ Complete · total {timings['total']:.1f}s · saved {final.name}")
        return GenerationResult(
            path=final, mode=request.mode, width=output_width, height=output_height, frames=output_frames,
            duration=(frames - 1) / float(self.config.fps), seed=int(request.seed),
            nfe=nfe, fps=float(request.target_fps), elapsed_seconds=timings["total"], lora=spec.name,
            prompt=expand_prompt(request.prompt, structured=request.structured_prompt),
            timings=timings,
        )
