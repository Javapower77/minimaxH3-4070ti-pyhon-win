"""Opt-in, source-aligned Ref2VA Restoration enhancement (never part of FL2VA)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
import hashlib
import json
import math
import re
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import uuid

import requests
from safetensors import safe_open
import websocket

from .assets import (
    RESTORE_FILE, RESTORE_SHA256,
    REF2VA_MODEL_FILE, REF2VA_MODEL_SHA256,
)
from .comfy_backend import PrunedComfyBackend, ProgressCallback
from .config import AppConfig

RESTORATION_CAPTION = "Restore this video to high visual quality by recovering fine detail and reducing pixelation, block artifacts, and compression noise while preserving the original scene, natural colors, and motion."
_VERIFIED: dict[tuple[str, int, int, str], bool] = {}


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frames: int
    fps: Fraction
    audio_tracks: int
    duration: Fraction | None = field(default=None, compare=False)
    start_time: Fraction = field(default=Fraction(0), compare=False)
    cfr: bool = field(default=True, compare=False)
    time_base: Fraction = field(default=Fraction(1, 1000000), compare=False)

    @property
    def duration_seconds(self) -> Fraction:
        return self.duration if self.duration is not None else self.frames / self.fps

    def validate(self) -> None:
        if self.width < 32 or self.height < 32 or self.width % 32 or self.height % 32:
            raise ValueError("Restoration requires source width and height to be multiples of 32; no automatic resize/crop.")
        if not 5 <= self.frames <= 3600 or self.frames % 17 != 5:
            raise ValueError("Restoration requires 17n+5 source frames (5–3600); no automatic trimming.")
        if self.fps != 24:
            raise ValueError("Restoration requires exactly 24 fps CFR. 23.976/interpolated FL2VA outputs are incompatible; FL2VA defaults are unchanged.")


@dataclass(frozen=True)
class RestorationRequest:
    source: Path
    strength: float = 1.0
    steps: int = 30
    seed: int = 42
    normalize_input: bool = False
    output_fps: str = "native"

    def validate(self) -> None:
        if not isinstance(self.normalize_input, bool) or self.output_fps not in ("native", "source"):
            raise ValueError("Restoration normalize_input must be boolean and output_fps must be native or source.")
        if not math.isfinite(self.strength) or not 0 < self.strength <= 2:
            raise ValueError("Restoration strength must be greater than 0 and at most 2.")
        if isinstance(self.steps, bool) or int(self.steps) != self.steps or not 4 <= self.steps <= 50:
            raise ValueError("Restoration steps must be an integer from 4 to 50 (local experimental default 30; not creator guidance).")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or not 0 <= self.seed < 2**64:
            raise ValueError("Restoration seed must be an unsigned 64-bit integer.")


@dataclass(frozen=True)
class RestorationResult:
    path: Path
    original: Path
    info: VideoInfo
    elapsed_seconds: float
    normalization_summary: str = "Strict source alignment; no temporal resampling."


class RestorationFailure(RuntimeError):
    """Expose a durable original even when the enhancement fails."""

    def __init__(self, original: Path, error: Exception):
        self.original = original
        super().__init__(f"Restoration failed: {error}. Original retained: {original}")


def _run(command: list[str], *, timeout: int = 300) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f"{Path(command[0]).name} failed: {completed.stderr[-2000:]}")
    return completed.stdout


def probe_video(path: Path, *, strict: bool = True) -> VideoInfo:
    """Inspect decoded input metadata separately from the native H3 contract."""
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        raise RuntimeError("Restoration requires local ffmpeg and ffprobe on PATH.")
    data = json.loads(_run([
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path),
    ]))
    videos = [s for s in data["streams"] if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")]
    if len(videos) != 1:
        raise ValueError("Restoration requires exactly one video stream.")
    stream = videos[0]
    info = VideoInfo(int(stream["width"]), int(stream["height"]), int(stream["nb_read_frames"]),
                     Fraction(stream["avg_frame_rate"]), sum(s["codec_type"] == "audio" for s in data["streams"]))
    if strict:
        info.validate()
    if stream.get("tags", {}).get("rotate", "0") != "0" or any(s.get("rotation", 0) != 0 for s in stream.get("side_data_list", [])):
        raise ValueError("Restoration does not accept rotated video; bake orientation explicitly first.")
    # Average rate alone cannot establish constant cadence. Check every decoded timestamp.
    frames = json.loads(_run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time,duration_time,pkt_duration_time", "-of", "json", str(path),
    ]))["frames"]
    times = [Fraction(frame["best_effort_timestamp_time"]) for frame in frames]
    if not times or len(times) != info.frames or info.fps <= 0 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Restoration requires constant 24 fps timestamps in strict mode; missing or non-monotonic input timestamps.")
    time_base = Fraction(stream.get("time_base", "1/1000000"))
    # Containers such as MKV quantize 24 fps to alternating 41/42 ms intervals.
    # Check the whole cadence against its grid with one mux tick of tolerance.
    tolerance = time_base + Fraction(1, 1000000)
    cfr = all(abs(t - times[0] - i / info.fps) <= tolerance for i, t in enumerate(times))
    last_duration = frames[-1].get("duration_time", frames[-1].get("pkt_duration_time"))
    duration = (times[-1] - times[0] + Fraction(last_duration)) if last_duration and Fraction(last_duration) > 0 else None
    if duration is None:
        declared = stream.get("duration")
        duration = Fraction(declared) if declared and declared != "N/A" else times[-1] - times[0] + 1 / info.fps
    if duration <= 0 or duration < times[-1] - times[0]:
        raise ValueError("Restoration input has no reliable positive video duration.")
    if cfr and abs(duration - info.frames / info.fps) <= tolerance:
        duration = info.frames / info.fps
    if not cfr:
        info = replace(info, fps=info.frames / duration)
    info = replace(info, duration=duration, start_time=times[0], cfr=cfr, time_base=time_base)
    if strict and not cfr:
        raise ValueError("Restoration requires constant 24 fps timestamps, not variable frame rate.")
    if strict and abs(times[0]) > tolerance:
        raise ValueError("Restoration requires a zero-start video timeline to preserve audio sync.")
    return info


def normalization_plan(info: VideoInfo, request: RestorationRequest) -> tuple[VideoInfo, VideoInfo, str]:
    """Pad the complete 24 fps guide, then remove only the synthetic tail."""
    request.validate()
    if info.start_time < 0:
        raise ValueError("Restoration normalization does not support negative-start video timelines; source/audio unchanged.")
    duration = info.duration_seconds
    native_frames = math.ceil(duration * 24)
    padded = 5 + 17 * max(0, math.ceil(Fraction(native_frames - 5, 17)))
    model = VideoInfo(info.width, info.height, padded, Fraction(24), 0)
    model.validate()  # Keep geometry and maximum native frame count limits.
    fps = (info.fps if request.output_fps == "source" else Fraction(24)
           if request.output_fps in (None, "native") else Fraction(str(request.output_fps)))
    final = replace(info, frames=math.ceil(duration * fps), fps=fps, cfr=True,
                    duration=math.ceil(duration * fps) / fps)
    summary = (
        f"Normalize input: {info.frames} frames at {float(info.fps):.6g} fps "
        f"({'CFR' if info.cfr else 'VFR; source mode uses average rate'}) / {float(duration):.6f}s → "
        f"24 fps CFR, {native_frames} resampled frames; pad {padded - native_frames} cloned tail frames "
        f"to {padded} (17n+5), intermediate duration adjustment +{float(model.duration_seconds - duration):.6f}s. "
        f"Trim enhanced tail to original duration; output {final.frames} frames at {float(fps):.6g} fps "
        f"/ {float(final.duration_seconds):.6f}s (frame-grid adjustment +{float(final.duration_seconds - duration):.6f}s). "
        "Playback speed preserved. Resampling drops/duplicates frames: temporal detail may be lost; "
        "matching source rate does not restore lost motion or VFR timestamps. Original audio timeline unchanged."
    )
    return model, final, summary


def normalize_video(original: Path, destination: Path, model: VideoInfo) -> VideoInfo:
    """Lossless, silent temporary guide; never rewrite the original."""
    _run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(original),
          "-map", "0:v:0", "-an", "-vf",
          "setpts=PTS-STARTPTS,fps=24:round=up,tpad=stop_mode=clone:stop=-1",
          "-frames:v", str(model.frames), "-c:v", "ffv1", str(destination)])
    actual = probe_video(destination)
    if (actual.width, actual.height, actual.frames, actual.fps) != (model.width, model.height, model.frames, model.fps):
        raise ValueError("Normalized Restoration guide does not match the planned native timeline.")
    return actual


def restore_video_timeline(enhanced: Path, destination: Path, source: VideoInfo,
                           model: VideoInfo, final: VideoInfo) -> None:
    actual = probe_video(enhanced)
    if (actual.width, actual.height, actual.frames, actual.fps) != (model.width, model.height, model.frames, model.fps):
        raise ValueError("Enhanced output does not match normalized model geometry/timeline.")
    _run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(enhanced),
          "-map", "0:v:0", "-an", "-vf",
          f"setpts=PTS-STARTPTS,fps={final.fps}:round=up,trim=end_frame={final.frames},"
          f"settb=1/1000000,setpts=PTS-STARTPTS+({float(source.start_time):.9f})/TB",
          "-frames:v", str(final.frames), "-c:v", "ffv1", "-enc_time_base", "1/1000000",
          "-fps_mode", "passthrough", "-copyts", "-avoid_negative_ts", "disabled", str(destination)])


def read_header(path: Path) -> dict:
    """Bounded local header read; no model tensors loaded into RAM/GPU."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Invalid SafeTensors file: {path.name}")
        size = struct.unpack("<Q", prefix)[0]
        if not 2 <= size <= 16 * 1024 * 1024:
            raise ValueError(f"Invalid SafeTensors header size: {path.name}")
        return json.loads(stream.read(size))


def verify_asset(path: Path, checksum: str) -> None:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, checksum)
    if key not in _VERIFIED:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != checksum:
            raise ValueError(f"Restoration asset checksum mismatch: {path.name}; install the pinned download, not a renamed FL2VA/full model.")
        _VERIFIED[key] = True


def validate_adapter(adapter: Path, model_header: dict) -> None:
    """Validate Comfy-native patches without rewriting weights or alpha/rank.

    Build flattened aliases *from* the exact checkpoint vocabulary, as Comfy's
    model_lora_keys_unet does. Reversing underscores would corrupt out_proj,
    qkv_proj, token_refiner and adaln_proj names and can hide alias collisions.
    """
    header = read_header(adapter)
    consumed = {"__metadata__"}
    pairs = 0
    aliases: dict[str, set[str]] = {}
    for target in model_header:
        if target.endswith(".weight"):
            alias = "lora_unet_" + target[:-len(".weight")].replace(".", "_")
            aliases.setdefault(alias, set()).add(target)

    def resolve(prefix: str) -> str:
        if prefix.startswith("diffusion_model."):
            module = prefix.removeprefix("diffusion_model.")
            return module if module.endswith(".bias") else module + ".weight"
        targets = aliases.get(prefix, set())
        if len(targets) > 1:
            raise ValueError(f"Ambiguous Restoration flattened alias collision: {prefix}: {sorted(targets)}")
        return next(iter(targets), "")

    patched: set[str] = set()
    for key in header:
        if key != "__metadata__" and not key.startswith(("diffusion_model.", "lora_unet_")):
            raise ValueError(f"Unsupported Restoration adapter key: {key}")
    with safe_open(str(adapter), framework="pt", device="cpu") as tensors:
        for key, entry in header.items():
            if key in consumed:
                continue
            suffix = next((s for s in (".lora_A.weight", ".lora_down.weight") if key.endswith(s)), None)
            if suffix:
                prefix = key[:-len(suffix)]
                other = prefix + (".lora_B.weight" if "lora_A" in suffix else ".lora_up.weight")
                target = resolve(prefix)
                if target not in model_header or other not in header:
                    raise ValueError(f"Unknown or unpaired Restoration target: {key}")
                module = target[:-len(".weight")] if target.endswith(".weight") else target
                if target in patched:
                    raise ValueError(f"Duplicate Restoration target collision: {target}")
                patched.add(target)
                down, up = entry["shape"], header[other]["shape"]
                expected = model_header[target]["shape"]
                reshape = prefix + ".reshape_weight"
                if reshape in header:
                    if module not in ("final_layer.video_out", "final_layer.audio_out", "final_layer.video_out.bias", "final_layer.audio_out.bias"):
                        raise ValueError(f"Unsupported reshaped Restoration target: {module}")
                    expected = tensors.get_tensor(reshape).tolist()
                    consumed.add(reshape)
                    base = model_header[target]["shape"]
                    if len(expected) != len(base) or expected[0] < base[0] or expected[1:] != base[1:]:
                        raise ValueError(f"Invalid PDD head reshape: {module}")
                # Bias head patches are represented as [rows, rank] @ [rank, 1].
                expected_matrix = expected if len(expected) == 2 else [expected[0], 1]
                if len(down) != 2 or len(up) != 2 or down[0] <= 0 or up[1] != down[0] or [up[0], down[1]] != expected_matrix:
                    raise ValueError(f"Incompatible Restoration target {module}: {up} @ {down}, expected {expected}. Full AdaLN cannot apply to the pruned curve basis; unsupported conversion is blocked.")
                consumed.update((key, other))
                alpha = prefix + ".alpha"
                if alpha in header:
                    if header[alpha]["shape"] not in ([], [1]):
                        raise ValueError(f"Invalid adapter alpha: {alpha}")
                    consumed.add(alpha)
                pairs += 1
            elif key.endswith(".diff_b"):
                weight = resolve(key[:-len(".diff_b")])
                target = weight[:-len(".weight")] + ".bias" if weight.endswith(".weight") else ""
                if target not in model_header or entry["shape"] != model_header[target]["shape"]:
                    raise ValueError(f"Incompatible adapter bias: {key}")
                if target in patched:
                    raise ValueError(f"Duplicate Restoration target collision: {target}")
                patched.add(target)
                consumed.add(key)
        unknown = set(header) - consumed
        if unknown or not pairs:
            raise ValueError(f"Unsupported/unmatched Restoration adapter patches: {sorted(unknown)}")


def mux_original_audio(enhanced: Path, original: Path, destination: Path, info: VideoInfo,
                       *, normalized: bool = False) -> None:
    """Atomic Matroska stream-copy; no AAC conversion, trimming, or audio VAE decode."""
    actual = probe_video(enhanced, strict=False) if normalized else probe_video(enhanced)
    if (actual.width, actual.height, actual.frames, actual.fps) != (info.width, info.height, info.frames, info.fps):
        raise ValueError("Enhanced output does not match source geometry/timeline; original retained.")
    if normalized and (not actual.cfr or abs(actual.start_time - info.start_time) > actual.time_base + Fraction(1, 1000000)):
        raise ValueError("Enhanced output changed source start time or constant cadence.")
    temporary = destination.with_name(destination.stem + ".partial.mkv")
    try:
        _run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-copyts", "-i", str(enhanced), "-i", str(original),
              "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1", "-c", "copy",
              "-avoid_negative_ts", "disabled", str(temporary)])
        if normalized:
            validate_normalized_mux(temporary, info)
        if info.audio_tracks:
            def audio_hash(path):
                text = _run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a",
                             "-c", "copy", "-f", "streamhash", "-hash", "sha256", "-"])
                # Never compare timebase/format headers: only ordered audio payload digest rows.
                rows = re.findall(r"^\s*\d+,a,SHA256=([^\s]+)\s*$", text, re.MULTILINE)
                if not rows:
                    rows = re.findall(r"^\s*\d+,\s*\d+,\s*([0-9a-fA-F]{64})\s*$", text, re.MULTILINE)
                if len(rows) != info.audio_tracks:
                    raise RuntimeError("Original audio packet hashes unavailable for every track.")
                return rows
            if audio_hash(original) != audio_hash(temporary):
                raise RuntimeError("Original audio packet hashes changed during mux; refusing enhanced output.")
            if normalized:
                verify_audio_timeline(original, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def validate_normalized_mux(path: Path, info: VideoInfo) -> None:
    muxed = probe_video(path, strict=False)
    expected = (info.width, info.height, info.frames, info.fps, info.audio_tracks)
    actual = (muxed.width, muxed.height, muxed.frames, muxed.fps, muxed.audio_tracks)
    if actual != expected or not muxed.cfr:
        raise ValueError("Mux changed normalized video geometry/timeline or audio track count.")
    if abs(muxed.start_time - info.start_time) > muxed.time_base + Fraction(1, 1000000):
        raise ValueError("Mux changed normalized video start time.")


def verify_audio_timeline(original: Path, muxed: Path) -> None:
    """Payload checks alone cannot detect an audio sync shift during remux."""
    def packets(path):
        data = json.loads(_run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_packets",
                               "-show_entries", "packet=stream_index,pts_time", "-of", "json", str(path)]))
        tracks: dict[int, list[Fraction]] = {}
        for packet in data["packets"]:
            tracks.setdefault(packet["stream_index"], []).append(Fraction(packet["pts_time"]))
        return list(tracks.values())
    before, after = packets(original), packets(muxed)
    if len(before) != len(after) or any(len(a) != len(b) or any(
            abs(x - y) > Fraction(1001, 1000000) for x, y in zip(a, b)) for a, b in zip(before, after)):
        raise RuntimeError("Original audio timeline changed during mux (beyond MKV millisecond resolution).")


def expanded_inputs(schema: dict) -> tuple[set[str], set[str]]:
    """Expand native V3 autogrow inputs as ComfyUI does (group.name)."""
    required, supported = set(), set()
    for section in ("required", "optional"):
        for name, value in schema.get(section, {}).items():
            if value and value[0] == "COMFY_AUTOGROW_V3":
                template = value[1]["template"]
                names = template.get("names")
                if names is None:
                    names = [f"{template['prefix']}{i}" for i in range(template["max"])]
                for i, child in enumerate(names):
                    full = f"{name}.{child}"
                    supported.add(full)
                    if i < template.get("min", 0) and template["input"].get("required"):
                        required.add(full)
            else:
                supported.add(name)
                if section == "required":
                    required.add(name)
    return required, supported


def reference_input(nodes: dict | None, group: str, prefix: str, kind: str) -> str:
    """Select the installed reference input, or fail closed on an unknown API."""
    expected = f"{group}.{prefix}0"
    if nodes is None:  # Offline graph construction; runtime always supplies live schema.
        return expected
    node = nodes.get("MiniMaxH3ReferenceToVideo")
    if node is None:
        raise RuntimeError("Restoration native node unavailable: MiniMaxH3ReferenceToVideo")
    schema = node.get("input", {})
    entries = schema.get("required", {}) | schema.get("optional", {})
    if group in entries and entries[group] and entries[group][0] == "COMFY_AUTOGROW_V3":
        template = entries[group][1].get("template", {})
        child_entries = template.get("input", {}).get("required", {}) | template.get("input", {}).get("optional", {})
        if (template.get("prefix") != prefix or template.get("max", 0) < 1
                or len(child_entries) != 1 or next(iter(child_entries.values()))[0] != kind):
            raise RuntimeError(f"Installed Restoration reference API is incompatible: {group}")
        return expected
    # Explicit flat ports are supported only when their declared type matches.
    for key in (expected, prefix + "0"):
        if key in entries and entries[key] and entries[key][0] == kind:
            return key
    raise RuntimeError(f"Installed Restoration reference API is incompatible: {group}")


class RestorationBackend(PrunedComfyBackend):
    operation = "restoration"

    def __init__(self, config: AppConfig):
        super().__init__(replace(config, comfy_cache_none=True,
                                 comfy_reserve_vram_gb=max(1.0, config.comfy_reserve_vram_gb)))
        self.model = REF2VA_MODEL_FILE

    @staticmethod
    def build_restoration_workflow(*, source: str, info: VideoInfo, request: RestorationRequest,
                           text_encoder: str, filename_prefix: str, nodes: dict | None = None,
                           audio_source: str | None = None) -> dict:
        info.validate()
        request.validate()
        graph = PrunedComfyBackend.build_workflow(
            prompt=RESTORATION_CAPTION, width=info.width, height=info.height, frames=info.frames,
            seed=request.seed, nfe=request.steps, model=REF2VA_MODEL_FILE,
            text_encoder=text_encoder, loras=[(RESTORE_FILE, request.strength)],
            target_fps=24.0, filename_prefix=filename_prefix,
        )
        graph.pop("9")
        graph.pop("17")  # generated audio is discarded, never decoded or muxed
        graph["18"]["inputs"].pop("audio")
        graph["50"] = {"class_type": "LoadVideo", "inputs": {"file": source}}
        graph["51"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["50", 0]}}
        inputs = {"clip": ["2", 0], "vae": ["3", 0], "prompt": RESTORATION_CAPTION,
                  "width": info.width, "height": info.height, "length": info.frames,
                  "ref_image_size": "match"}
        inputs[reference_input(nodes, "ref_videos", "ref_video_", "IMAGE")] = ["51", 0]
        if info.audio_tracks or audio_source:
            inputs["audio_vae"] = ["4", 0]
            if audio_source:
                graph["55"] = {"class_type": "LoadVideo", "inputs": {"file": audio_source}}
                graph["56"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["55", 0]}}
            inputs[reference_input(nodes, "ref_video_audios", "ref_video_audio_", "AUDIO")] = ["56" if audio_source else "51", 1]
        graph["54"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": inputs}
        graph["13"]["inputs"]["conditioning"] = ["54", 0]
        graph["15"]["inputs"]["latent_image"] = ["54", 1]
        return graph

    def preflight(self) -> None:
        missing = [str(p) for p in self._required_paths() if not p.is_file()]
        adapters = [(self.root / "models/loras" / RESTORE_FILE, RESTORE_SHA256)]
        missing += [str(p) for p, _ in adapters if not p.is_file()]
        if missing:
            raise FileNotFoundError("Optional Restoration assets missing. Download type restore (adapter), restore_base (21 GB base), and reuse pruned encoder/VAEs. Compatibility is unverified until every adapter key passes local validation. Missing: " + ", ".join(missing))
        base = self.root / "models/diffusion_models" / self.model
        verify_asset(base, REF2VA_MODEL_SHA256)
        header = read_header(base)
        if header.get("adaln_t_table", {}).get("shape") != [1025, 8]:
            raise ValueError("Restoration requires the pinned pruned Ref2VA curve model.")
        for path, checksum in adapters:
            verify_asset(path, checksum)
            validate_adapter(path, header)
        # The selected turbo has expanded PDD heads; older loaders silently miss these.
        requirements = {
            "comfy/weight_adapter/lora.py": "reshape_weight",
            "comfy/ldm/minimax/model.py": "_pdd_head",
            "comfy/cli_args.py": "enables_dynamic_vram",
        }
        for relative, marker in requirements.items():
            path = self.root / relative
            if not path.is_file() or marker not in path.read_text(encoding="utf-8"):
                raise RuntimeError(f"Installed ComfyUI lacks verified Restoration/PDD/DynamicVRAM support: {relative}. Run the pinned backend setup.")

    def validate_worker_profile(self) -> None:
        """Reject busy or differently configured workers before Restoration submission."""
        response = requests.get(f"{self.url}/system_stats", timeout=30)
        response.raise_for_status()
        argv = response.json().get("system", {}).get("argv", [])
        def value(flag):
            try:
                return argv[argv.index(flag) + 1]
            except (ValueError, IndexError):
                raise RuntimeError(f"Restoration worker launch profile is missing {flag}; restart with the app's pinned worker.")
        if "--cache-none" not in argv or any(flag in argv for flag in ("--disable-dynamic-vram", "--highvram", "--gpu-only")):
            raise RuntimeError("Restoration requires cache-none and DynamicVRAM; restart the worker with the app's low-memory profile.")
        if float(value("--reserve-vram")) < 1.0:
            raise RuntimeError("Restoration worker requires at least 1 GB VRAM reserve.")
        for flag, expected in (("--input-directory", self.input_dir), ("--output-directory", self.output_dir)):
            if Path(value(flag)).resolve() != expected.resolve():
                raise RuntimeError(f"Restoration worker has an incompatible {flag}; restart the worker.")
        response = requests.get(f"{self.url}/queue", timeout=30)
        response.raise_for_status()
        queue = response.json()
        if queue.get("queue_running") or queue.get("queue_pending"):
            raise RuntimeError("Restoration requires an idle worker; wait for queued generation jobs to finish.")

    def _prepare_reference(self, token: str) -> None:
        """Operation hook: stage optional image references before worker startup."""

    def _retained_source(self, source: Path, originals: Path, token: str) -> Path:
        original = originals / (token + source.suffix.lower())
        shutil.copy2(source, original)
        return original

    def _build_source_workflow(self, **kwargs) -> dict:
        return self.build_restoration_workflow(**kwargs)

    def enhance(self, request: RestorationRequest, progress_callback: ProgressCallback | None = None) -> RestorationResult:
        import logging
        from .diagnostics import event, stage, workflow_metadata

        log = logging.getLogger(__name__)
        request.validate()
        source = Path(request.source).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Restoration source video missing: {source}")
        token = uuid.uuid4().hex
        originals = self.config.upload_dir / f"{self.operation}_originals"
        originals.mkdir(parents=True, exist_ok=True)
        original = self._retained_source(source, originals, token)
        report = progress_callback or (lambda fraction, message: None)
        started = time.perf_counter()
        socket = None
        worker_started = False
        workspace = None
        worker_source = None
        worker_audio_source = None
        try:
            report(0.01, "Checking Restoration source…")
            source_info = probe_video(original, strict=False) if request.normalize_input else probe_video(original)
            info = source_info
            summary = "Strict source alignment; no temporal resampling."
            if request.normalize_input:
                info, final_info, summary = normalization_plan(source_info, request)
                report(0.02, "Planning native 24 fps normalization…")
            elif request.output_fps == "source":
                summary = "Strict source alignment; source rate is already native 24 fps."
            report(0.03, "Checking local Restoration assets…")
            with stage(log, "reference_preflight", operation=self.operation):
                self.preflight()
            # Persist input BEFORE startup so native LoadVideo sees it in its input-root combo.
            self.input_dir.mkdir(parents=True, exist_ok=True)
            worker_source = self.input_dir / (token + (".mkv" if request.normalize_input else source.suffix.lower()))
            if request.normalize_input:
                report(0.04, "Normalizing guide to 24 fps and padding frames…")
                with stage(log, "reference_normalize", operation=self.operation):
                    info = normalize_video(original, worker_source, info)
                workspace = tempfile.TemporaryDirectory(prefix="restoration_")
            else:
                report(0.04, "Preparing source guide…")
                shutil.copy2(original, worker_source)
            self._prepare_reference(token)
            if request.normalize_input and source_info.audio_tracks:
                worker_audio_source = self.input_dir / (token + "_audio" + source.suffix.lower())
                shutil.copy2(original, worker_audio_source)
            with stage(log, "worker_startup", operation=self.operation):
                self.ensure_worker(progress_callback=report)
            worker_started = True
            self.validate_worker_profile()
            # ComfyUI processes this unload request asynchronously. The queue is
            # checked idle, but this HTTP response is not an unload-completion barrier.
            response = requests.post(f"{self.url}/free", json={"unload_models": True, "free_memory": True}, timeout=60)
            response.raise_for_status()
            response = requests.get(f"{self.url}/object_info", timeout=30)
            response.raise_for_status()
            nodes = response.json()
            graph = self._build_source_workflow(source=worker_source.name, info=info, request=request,
                                           text_encoder=self.text_encoder, filename_prefix=f"{self.operation}/{token}",
                                           nodes=nodes, audio_source=worker_audio_source.name if worker_audio_source else None)
            for node in graph.values():
                name = node["class_type"]
                if name not in nodes:
                    raise RuntimeError(f"Restoration native node unavailable: {name}; restart/update the pinned worker. No external node packs are used.")
                schema = nodes[name].get("input", {})
                required, supported = expanded_inputs(schema)
                if not set(node["inputs"]) <= supported or not required <= set(node["inputs"]):
                    raise RuntimeError(f"Installed Restoration node API is incompatible: {name}")
            socket = websocket.create_connection(self.url.replace("http://", "ws://").replace("https://", "wss://") + f"/ws?clientId={token}", timeout=2)
            event(log, "workflow_metadata", operation=self.operation,
                  **workflow_metadata(graph, self.root / "models"))
            self._log_worker_stats("submission")
            with stage(log, "submit", operation=self.operation):
                response = requests.post(f"{self.url}/prompt", json={"prompt": graph, "client_id": token}, timeout=60)
                if not response.ok:
                    raise RuntimeError("Restoration workflow rejected. See worker log.")
            event(log, "worker_submit", operation=self.operation, prompt_id=response.json()["prompt_id"])
            output, _ = self._wait_for_output(response.json()["prompt_id"], socket=socket, started=started, progress_callback=report)
            report(0.98, "Validating enhanced video and stream-copying original audio…")
            destination = self.config.output_dir / f"{token}_{self.operation}.mkv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if request.normalize_input:
                report(0.98, "Restoring source timeline and trimming padded tail…")
                restored = Path(workspace.name) / "restored.mkv"
                with stage(log, "reference_restore_timeline", operation=self.operation):
                    restore_video_timeline(output, restored, source_info, info, final_info)
                report(0.99, "Preserving original audio and verifying output…")
                with stage(log, "reference_mux_verify", operation=self.operation):
                    mux_original_audio(restored, original, destination, final_info, normalized=True)
                info = probe_video(destination, strict=False)
            else:
                with stage(log, "reference_mux_verify", operation=self.operation):
                    mux_original_audio(output, original, destination, info)
            report(1.0, "Restoration complete; original audio preserved without re-encoding.")
            return RestorationResult(destination, original, info, time.perf_counter() - started, summary)
        except Exception as exc:
            raise RestorationFailure(original, exc) from exc
        finally:
            if workspace is not None:
                workspace.cleanup()
            if request.normalize_input and worker_source is not None:
                worker_source.unlink(missing_ok=True)
            if worker_audio_source is not None:
                worker_audio_source.unlink(missing_ok=True)
            if socket is not None:
                socket.close()
            if worker_started:
                try:
                    requests.post(f"{self.url}/free", json={"unload_models": True, "free_memory": True}, timeout=30).raise_for_status()
                except requests.RequestException:
                    import logging
                    logging.getLogger(__name__).warning("Could not unload Restoration worker models", exc_info=True)