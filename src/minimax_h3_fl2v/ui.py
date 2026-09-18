"""Gradio studio for MiniMax-H3 FL2VA + LoRA."""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from .offline import enforce_offline_runtime

enforce_offline_runtime()

import gradio as gr

from .config import AppConfig, GenerationRequest, load_config
from .pipeline import MiniMaxH3Engine, get_engine
from .prompts import describe_mode
from .resolution import PRESET_LABELS, SUPPORTED_ASPECT_RATIOS

logger = logging.getLogger(__name__)

LISTEN_HOST = "0.0.0.0"


class _ChunkedUploadPreviewMiddleware:
    """Remove unstable lengths from Gradio's uploaded-file preview responses.

    On Windows, the browser can request a preview while Gradio is still replacing
    the uploaded temporary image. Starlette may stat the old size and stream the
    new file, which h11 rejects as larger than the declared Content-Length.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = str(scope.get("path", ""))
        is_file_preview = scope.get("type") == "http" and (
            path.startswith("/file=")
            or path.startswith("/file/")
            or path.startswith("/gradio_api/file=")
            or path.startswith("/gradio_api/file/")
        )
        if not is_file_preview:
            await self.app(scope, receive, send)
            return

        async def send_without_length(message):
            if message["type"] == "http.response.start":
                message = dict(message)
                message["headers"] = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != b"content-length"
                ]
            await send(message)

        await self.app(scope, receive, send_without_length)


def _nic_ipv4() -> Optional[str]:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("1.1.1.1", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except OSError:
        return None


def _imds_get(url: str, timeout: float = 2.0) -> Optional[str]:
    req = urllib.request.Request(url, headers={"Metadata": "true"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore").strip()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _azure_public_ipv4() -> Optional[str]:
    import json

    nic = _imds_get(
        "http://169.254.169.254/metadata/instance/network/interface/0/"
        "ipv4/ipAddress/0/publicIpAddress?api-version=2021-02-01&format=text"
    )
    if nic:
        return nic
    payload = _imds_get("http://169.254.169.254/metadata/loadbalancer?api-version=2020-10-01")
    if not payload:
        return None
    try:
        data = json.loads(payload)
        addrs = data.get("loadbalancer", {}).get("publicIpAddresses") or []
        for item in addrs:
            ip = str(item.get("frontendIpAddress") or "").strip()
            if ip:
                return ip
    except json.JSONDecodeError:
        return None
    return None


def _print_listen_urls(port: int) -> None:
    print(f"* Bound on {LISTEN_HOST}:{port} (all interfaces, no Gradio share tunnel)")
    nic = _nic_ipv4()
    if nic:
        print(f"* Private NIC:  http://{nic}:{port}")
    public = _azure_public_ipv4()
    if public:
        print(f"* Public URL:   http://{public}:{port}")
        print("  Open that URL from the client IP allowed in your Azure NSG.")
    else:
        print(f"* Public URL:   http://<vm-public-ip>:{port}")

CSS = """
:root { --body-bg: #0b0d12; }
.gradio-container { font-family: "IBM Plex Sans", "Segoe UI", sans-serif; }
#title-block h1 { letter-spacing: 0.04em; font-weight: 650; }
.hint { color: #9aa3b2; font-size: 0.92rem; }
.progress-bar-wrap {
    padding: 12px 16px !important;
    border: 1px solid rgba(148, 163, 184, 0.38) !important;
    border-radius: 12px !important;
    background: rgba(2, 6, 23, 0.94) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.38) !important;
}
.progress-bar-wrap,
.progress-bar-wrap *,
.progress-text,
.progress-text *,
.eta-bar,
.eta-bar *,
.progress-level,
.progress-level * {
    color: #ffffff !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #ffffff !important;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.95) !important;
}
.progress-text {
    font-size: 14px !important;
    line-height: 1.45 !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
.eta-bar {
    font-size: 13px !important;
    font-weight: 750 !important;
}
.progress-bar { height: 16px !important; border-radius: 999px !important; overflow: hidden; }
.progress-level {
    background: rgba(255, 255, 255, 0.18) !important;
    border-radius: 999px !important;
}
.progress-level-inner,
.progress-bar > div {
    background: linear-gradient(90deg, #7c3aed, #06b6d4, #22c55e) !important;
    box-shadow: 0 0 18px rgba(6, 182, 212, 0.55);
}
#run-summary textarea { font-family: "IBM Plex Mono", "Cascadia Mono", monospace; }
"""


def _file_path(value) -> Optional[Path]:
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        return _file_path(value[0] if value else None)
    name = getattr(value, "name", None)
    if name:
        return Path(str(name))
    return Path(str(value))


def _lora_choices(config: AppConfig) -> list[tuple[str, str]]:
    choices = []
    for spec in config.catalog:
        mark = " ★" if spec.recommended else ""
        choices.append((f"{spec.name}{mark}", spec.id))
    return choices


def _stored_lora_choices(config: AppConfig) -> list[tuple[str, str | None]]:
    """List persistent local LoRAs, newest first, with an explicit None choice."""
    files = sorted(
        config.lora_dir.glob("*.safetensors"),
        key=lambda path: (path.stat().st_mtime, path.name.lower()),
        reverse=True,
    )
    return [("None (Turbo/catalog only)", None), *[(path.name, str(path.resolve())) for path in files]]


def _save_uploaded_lora(config: AppConfig, upload) -> tuple[gr.Dropdown, None]:
    source = _file_path(upload)
    if source is None:
        return gr.Dropdown(choices=_stored_lora_choices(config), value=None), None
    if source.suffix.lower() != ".safetensors":
        raise gr.Error("Only .safetensors LoRA files are accepted.")
    destination = (config.lora_dir / source.name).resolve()
    config.lora_dir.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination:
        shutil.copy2(source, destination)
    return (
        gr.Dropdown(choices=_stored_lora_choices(config), value=str(destination)),
        None,
    )


MAX_EXTRA_LORAS = 5
MAX_SEED = 2**63 - 1


def _resolve_seed(mode: str, value) -> int:
    if str(mode).lower() == "random each generation":
        return secrets.randbelow(MAX_SEED + 1)
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise gr.Error("Fixed seed must be an integer.") from exc
    if not 0 <= seed <= MAX_SEED:
        raise gr.Error(f"Seed must be between 0 and {MAX_SEED}.")
    return seed


def _selected_extra_loras(paths: list, scales: list) -> list[tuple[Path, float]]:
    selected: list[tuple[Path, float]] = []
    seen: set[Path] = set()
    for value, scale in zip(paths, scales):
        path = _file_path(value)
        if path is None:
            continue
        path = path.expanduser().resolve()
        if path in seen:
            raise gr.Error(f"LoRA selected more than once: {path.name}")
        strength = float(scale)
        if not 0.0 <= strength <= 2.0:
            raise gr.Error(f"LoRA strength must be between 0 and 2: {path.name}")
        seen.add(path)
        selected.append((path, strength))
    if len(selected) > MAX_EXTRA_LORAS:
        raise gr.Error(f"At most {MAX_EXTRA_LORAS} extra LoRAs may be selected.")
    return selected


def _lora_slot_updates(count) -> tuple:
    """Show exactly the requested number of additional LoRA rows."""
    try:
        selected_count = max(0, min(MAX_EXTRA_LORAS, int(count)))
    except (TypeError, ValueError):
        selected_count = 0
    return tuple(gr.update(visible=index < selected_count) for index in range(MAX_EXTRA_LORAS))


def _lora_guidance(config: AppConfig, lora_id: str, count=0, *scales) -> str:
    """Return concise schedule, strength, and compatibility guidance."""
    spec = config.lora_by_id(lora_id)
    try:
        selected_count = int(count)
    except (TypeError, ValueError):
        selected_count = 0
    active_scales = [float(value) for value in scales[:selected_count] if value is not None]
    warnings: list[str] = []
    if any(value > 1.2 for value in active_scales):
        warnings.append("⚠️ Extra strength above 1.2 can overpower identity, motion, or anatomy.")
    if selected_count >= 4:
        warnings.append("⚠️ Four or five stacked LoRAs substantially increase load time and conflict risk.")
    backend = (
        "**Pruned backend:** only 8-wide pruned-compatible extra LoRAs may be stacked."
        if spec.backend == "comfy_pruned"
        else "**Diffusers backend:** use FL2VA PEFT SafeTensors; Ref2VA/pruned adapters are incompatible."
    )
    schedule = (
        f"**Recommended:** NFE **{spec.nfe}**, Turbo strength **{spec.lora_scale:g}**, "
        f"video/audio shift **{spec.video_shift:g}/{spec.audio_shift:g}**."
    )
    strength = (
        "For extra LoRAs, begin at **0.2–0.5**, test one at a time with a fixed seed, "
        "then increase gradually."
    )
    warning_text = "\n\n" + "  \n".join(warnings) if warnings else ""
    return f"{schedule}  \n{backend}  \n{strength}{warning_text}"


def _recommended_reset_values(config: AppConfig) -> tuple:
    """Values returned by the UI's reset-to-recommended-defaults action."""
    spec = config.lora_by_id(config.default_lora_id)
    return (
        "", None, None, config.default_lora_id, 0,
        *([None] * MAX_EXTRA_LORAS),
        None, _preset_for_megapixels(spec.megapixels), "match reference (no crop)",
        5.0, spec.nfe, "Fixed", config.seed, spec.lora_scale,
        *([0.8] * MAX_EXTRA_LORAS),
        True, False, 2.0, 0.25, "23.976 fps", False, 0.7,
        spec.notes,
        _lora_guidance(config, spec.id, 0),
        "Fixed seed will be reused.",
    )


def _preset_for_megapixels(megapixels: float) -> str:
    """Choose the closest 16:9 UI preset for a catalog recommendation."""
    candidates = [(label, value) for label, (value, ratio) in PRESET_LABELS.items() if ratio == "16:9"]
    return min(candidates, key=lambda item: abs(item[1] - float(megapixels)))[0]


def build_app(config: Optional[AppConfig] = None) -> gr.Blocks:
    config = config or load_config()
    engine = get_engine(config)

    def load_model(lora_id: str):
        try:
            spec = config.lora_by_id(lora_id)
            if spec.backend == "comfy_pruned":
                from .comfy_backend import PrunedComfyBackend

                backend = PrunedComfyBackend(config)
                backend.ensure_worker()
                return (
                    f"Ready · pruned ComfyUI backend · {config.device} · "
                    f"model={config.comfy_model}"
                )
            engine.load()
            return f"Ready · {config.pretrained_path} · {config.device} · CPU offload={config.cpu_offload}"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Model load failed")
            return f"Load failed: {exc}"

    def generate(
        prompt,
        first_image,
        last_image,
        lora_id,
        extra_lora_count,
        stored_lora_1,
        stored_lora_2,
        stored_lora_3,
        stored_lora_4,
        stored_lora_5,
        lora_upload,
        duration,
        preset,
        aspect_ratio,
        nfe,
        seed_mode,
        seed,
        lora_scale,
        extra_scale_1,
        extra_scale_2,
        extra_scale_3,
        extra_scale_4,
        extra_scale_5,
        structured,
        upscale,
        upscale_factor,
        detailer_strength,
        output_fps,
        face_restore,
        face_fidelity,
        progress=gr.Progress(track_tqdm=False),
    ):
        if not prompt or not str(prompt).strip():
            raise gr.Error("Prompt is required.")
        selected_spec = config.lora_by_id(lora_id)
        if not engine.ready and selected_spec.backend != "comfy_pruned":
            progress(0.05, desc="Loading MiniMax-H3…")
            engine.load()

        uploaded = _file_path(lora_upload)
        if uploaded is not None:
            dest = config.lora_dir / uploaded.name
            if uploaded.resolve() != dest.resolve():
                shutil.copy2(uploaded, dest)

        selected_count = max(0, min(MAX_EXTRA_LORAS, int(extra_lora_count)))
        extra_loras = _selected_extra_loras(
            [stored_lora_1, stored_lora_2, stored_lora_3, stored_lora_4, stored_lora_5][
                :selected_count
            ],
            [extra_scale_1, extra_scale_2, extra_scale_3, extra_scale_4, extra_scale_5][
                :selected_count
            ],
        )
        actual_seed = _resolve_seed(seed_mode, seed)

        has_reference = bool(first_image or last_image)
        if preset and preset in PRESET_LABELS:
            megapixels, preset_aspect = PRESET_LABELS[preset]
            if aspect_ratio == "from preset" and not has_reference:
                aspect_ratio = preset_aspect
        else:
            megapixels = config.megapixels
        if has_reference:
            # Reference geometry always wins. The engine letterboxes the tiny
            # 32-grid rounding difference and never crops source pixels.
            aspect_ratio = "auto"
        elif aspect_ratio in {"from preset", "match reference (no crop)"}:
            aspect_ratio = "16:9"

        request = GenerationRequest(
            prompt=prompt,
            first_image=_file_path(first_image),
            last_image=_file_path(last_image),
            duration_seconds=float(duration),
            megapixels=float(megapixels),
            aspect_ratio=str(aspect_ratio),
            nfe=int(nfe),
            seed=actual_seed,
            lora_id=lora_id,
            extra_loras=extra_loras,
            lora_scale=float(lora_scale),
            structured_prompt=bool(structured),
            upscale=bool(upscale),
            upscale_factor=float(upscale_factor),
            detailer_strength=float(detailer_strength) if upscale else 0.0,
            target_fps=float(str(output_fps).split()[0]),
            face_restore=bool(face_restore),
            face_fidelity=float(face_fidelity),
        )
        progress(0.01, desc=f"🚀 Starting {describe_mode(first_image, last_image)} generation…")

        def report(fraction: float, message: str) -> None:
            progress(max(0.0, min(1.0, float(fraction))), desc=message)

        result = engine.generate(request, progress_callback=report)
        generated_seconds = max(result.duration, 0.001)
        real_time_factor = result.elapsed_seconds / generated_seconds
        timing_text = ""
        if result.timings:
            useful = [
                f"{name.replace('_', ' ')}={seconds:.1f}s"
                for name, seconds in result.timings.items()
                if name != "total" and seconds >= 0.05
            ]
            if useful:
                timing_text = "\nStages: " + " · ".join(useful)
        summary = (
            f"{result.mode.upper()}  {result.width}×{result.height}  "
            f"{result.frames} frames ({result.duration:.2f}s) @ {result.fps:g} fps  "
            f"NFE={result.nfe}  seed={result.seed}  "
            f"LoRA={result.lora}\n"
            f"Generation: {result.elapsed_seconds:.1f}s  ·  "
            f"{real_time_factor:.2f}× realtime  ·  "
            f"{result.frames / max(result.elapsed_seconds, 0.001):.2f} frames/s"
            f"{timing_text}\n"
            f"{result.path}"
        )
        seed_note = (
            f"Random seed chosen: {result.seed}. It has been copied into the Seed field; "
            "switch Seed mode to Fixed to reproduce this video."
            if str(seed_mode).lower() == "random each generation"
            else f"Fixed seed used: {result.seed}."
        )
        return str(result.path), summary, result.seed, seed_note

    theme = gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="slate",
        neutral_hue="slate",
    ).set(
        body_background_fill="#0b0d12",
        block_background_fill="#141821",
        border_color_primary="#2a3140",
    )

    with gr.Blocks(title=config.title) as demo:
        gr.Markdown(
            f"# {config.title}\n"
            "Local **MiniMax-H3 FL2VA** — first/last-frame to **video + stereo audio**. "
            "Offline, unfiltered. LoRAs are Diffusers PEFT **SafeTensors**. "
            "Default profile: **RTX 4070 Ti 12 GB** with the pruned FP8 backend."
        )
        status = gr.Textbox(
            label="Engine",
            value="Click Load model to validate the selected backend, or generate directly.",
            interactive=False,
        )
        with gr.Row():
            load_btn = gr.Button("Load model", variant="secondary")

        with gr.Row():
            with gr.Column(scale=5):
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=6,
                    placeholder=(
                        "A woman in a rust-red coat walks through neon rain toward a subway entrance. "
                        "Handheld camera, shallow depth of field, sodium and cyan highlights. "
                        "Rain hiss, distant traffic, heels on wet concrete, no score."
                    ),
                )
                with gr.Row():
                    first_image = gr.Image(label="First frame (optional)", type="filepath")
                    last_image = gr.Image(label="Last frame (optional, FL2VA)", type="filepath")
                with gr.Row():
                    refresh_loras = gr.Button("Refresh", scale=1)
                    extra_lora_count = gr.Dropdown(
                        choices=list(range(MAX_EXTRA_LORAS + 1)),
                        value=0,
                        label="Additional LoRAs",
                        info="Select how many independent LoRA slots to enable (0–5).",
                        scale=2,
                    )
                stored_loras = []
                extra_scales = []
                lora_rows = []
                for index in range(1, MAX_EXTRA_LORAS + 1):
                    with gr.Row(visible=False) as lora_row:
                        stored = gr.Dropdown(
                            choices=_stored_lora_choices(config),
                            value=None,
                            label=f"Extra LoRA {index}",
                            allow_custom_value=False,
                            scale=5,
                        )
                        strength = gr.Slider(
                            0.0,
                            2.0,
                            value=0.8,
                            step=0.05,
                            label=f"Strength {index}",
                            scale=2,
                        )
                    stored_loras.append(stored)
                    extra_scales.append(strength)
                    lora_rows.append(lora_row)
                lora_upload = gr.File(
                    label="Upload new LoRA once (saved into models/loras)",
                    file_types=[".safetensors"],
                    type="filepath",
                )
            with gr.Column(scale=4):
                lora_id = gr.Dropdown(
                    choices=_lora_choices(config),
                    value=config.default_lora_id,
                    label="Turbo / catalog LoRA",
                )
                lora_notes = gr.Markdown(value=config.lora_by_id(config.default_lora_id).notes)
                lora_help = gr.Markdown(
                    value=_lora_guidance(config, config.default_lora_id, 0),
                    elem_classes=["hint"],
                )
                load_btn.click(load_model, inputs=lora_id, outputs=status)
                preset = gr.Dropdown(
                    choices=list(PRESET_LABELS.keys()),
                    value=_preset_for_megapixels(
                        config.lora_by_id(config.default_lora_id).megapixels
                    ),
                    label="Canvas preset",
                )
                aspect_ratio = gr.Dropdown(
                    choices=["match reference (no crop)", "from preset", *SUPPORTED_ASPECT_RATIOS],
                    value="match reference (no crop)",
                    label="Aspect ratio",
                )
                duration = gr.Slider(5.0, 15.0, value=5.0, step=0.5, label="Duration (seconds, snapped to 17n+5 frames)")
                with gr.Accordion("Advanced settings", open=False):
                    structured = gr.Checkbox(
                        value=True,
                        label="Wrap prompt as H3-Context-IR (vision + soundscape + music)",
                    )
                    upscale = gr.Checkbox(
                        value=False,
                        label="Upscale/detail after generation (Real-ESRGAN, tiled)",
                        info="Adds processing time. Designed for the 12 GB low-VRAM backend.",
                    )
                    upscale_factor = gr.Radio(
                        choices=[1.5, 2.0],
                        value=2.0,
                        label="Output upscale factor",
                    )
                    detailer_strength = gr.Slider(
                        0.0,
                        1.0,
                        value=0.25,
                        step=0.05,
                        label="Detail/sharpen strength",
                        info="0 disables sharpening; 0.2–0.35 is conservative for video.",
                    )
                    output_fps = gr.Radio(
                        choices=["23.976 fps", "29.97 fps", "60 fps", "120 fps"],
                        value="23.976 fps",
                        label="Output frame rate",
                        info="29.97/60/120 use RIFE; higher rates require more time and RAM.",
                    )
                    face_restore = gr.Checkbox(
                        value=False,
                        label="Restore faces with CodeFormer",
                        info="Uses lightweight MobileNet face detection; may flicker on small/profile faces.",
                    )
                    face_fidelity = gr.Slider(
                        0.0,
                        1.0,
                        value=0.7,
                        step=0.05,
                        label="Face fidelity",
                        info="Higher preserves identity; lower performs stronger restoration.",
                    )
                    nfe = gr.Slider(
                        4,
                        50,
                        value=config.nfe,
                        step=1,
                        label="NFE (transformer steps)",
                        info="Use the catalog recommendation. More steps are not always better for Turbo LoRAs.",
                    )
                    lora_scale = gr.Slider(
                        0.0,
                        1.5,
                        value=config.lora_by_id(config.default_lora_id).lora_scale,
                        step=0.05,
                        label="Turbo / catalog LoRA strength",
                        info="Uses the selected catalog recommendation. Reduce it when stacking strong extra LoRAs.",
                    )
                    seed_mode = gr.Radio(
                        choices=["Fixed", "Random each generation"],
                        value="Fixed",
                        label="Seed mode",
                    )
                    seed = gr.Number(value=config.seed, precision=0, label="Seed")
                    seed_note = gr.Textbox(
                        value="Fixed seed will be reused.",
                        label="Seed used",
                        interactive=False,
                    )
                with gr.Row():
                    reset_btn = gr.Button("Reset recommended defaults", variant="secondary")
                run_btn = gr.Button("Generate video + audio", variant="primary")

        video = gr.Video(label="Output MP4 (H.264 + AAC)", autoplay=True)
        summary = gr.Textbox(
            label="Run summary and performance",
            lines=5,
            elem_id="run-summary",
        )

        lora_id.change(
            lambda lora, count, *scales: (
                config.lora_by_id(lora).nfe,
                config.lora_by_id(lora).lora_scale,
                config.lora_by_id(lora).notes,
                _lora_guidance(config, lora, count, *scales),
            ),
            inputs=[lora_id, extra_lora_count, *extra_scales],
            outputs=[nfe, lora_scale, lora_notes, lora_help],
        )
        extra_lora_count.change(
            _lora_slot_updates,
            inputs=extra_lora_count,
            outputs=lora_rows,
        ).then(
            lambda lora, count, *scales: _lora_guidance(config, lora, count, *scales),
            inputs=[lora_id, extra_lora_count, *extra_scales],
            outputs=lora_help,
        )
        for strength in extra_scales:
            strength.change(
                lambda lora, count, *scales: _lora_guidance(config, lora, count, *scales),
                inputs=[lora_id, extra_lora_count, *extra_scales],
                outputs=lora_help,
            )
        refresh_loras.click(
            lambda: tuple(
                gr.Dropdown(choices=_stored_lora_choices(config))
                for _ in range(MAX_EXTRA_LORAS)
            ),
            outputs=stored_loras,
        )
        lora_upload.upload(
            lambda upload, count: (
                *_save_uploaded_lora(config, upload),
                max(1, int(count or 0)),
                gr.update(visible=True),
            ),
            inputs=[lora_upload, extra_lora_count],
            outputs=[stored_loras[0], lora_upload, extra_lora_count, lora_rows[0]],
        )
        seed_mode.change(
            lambda mode: (
                gr.Number(interactive=str(mode).lower() == "fixed"),
                "Enter a repeatable seed."
                if str(mode).lower() == "fixed"
                else "A random seed will be generated and shown after each run.",
            ),
            inputs=seed_mode,
            outputs=[seed, seed_note],
        )

        reset_outputs = [
            prompt,
            first_image,
            last_image,
            lora_id,
            extra_lora_count,
            *stored_loras,
            lora_upload,
            preset,
            aspect_ratio,
            duration,
            nfe,
            seed_mode,
            seed,
            lora_scale,
            *extra_scales,
            structured,
            upscale,
            upscale_factor,
            detailer_strength,
            output_fps,
            face_restore,
            face_fidelity,
            lora_notes,
            lora_help,
            seed_note,
        ]
        reset_btn.click(
            lambda: _recommended_reset_values(config),
            outputs=reset_outputs,
        ).then(
            lambda: _lora_slot_updates(0),
            outputs=lora_rows,
        )

        run_btn.click(
            generate,
            inputs=[
                prompt,
                first_image,
                last_image,
                lora_id,
                extra_lora_count,
                *stored_loras,
                lora_upload,
                duration,
                preset,
                aspect_ratio,
                nfe,
                seed_mode,
                seed,
                lora_scale,
                *extra_scales,
                structured,
                upscale,
                upscale_factor,
                detailer_strength,
                output_fps,
                face_restore,
                face_fidelity,
            ],
            outputs=[video, summary, seed, seed_note],
            show_progress="full",
            show_progress_on=status,
        )

        gr.Markdown(
            "### Notes\n"
            "- **T2VA**: prompt only · **I2VA**: first frame · **FL2VA**: first + last frame.\n"
            "- A reference image controls the output ratio; only 32-pixel-grid rounding is applied. "
            "Frames are contained with padding, never cropped.\n"
            "- Turbo LoRAs are trained at **4–8 NFE**. Base model wants ~**50 NFE**.\n"
            "- Stack up to **five** local LoRAs; each has an independent strength. Uploaded files persist in `models/loras`.\n"
            "- Random seed mode shows the chosen seed and copies it into the Seed field for reproduction.\n"
            "- No Hub, no share tunnel, no safety checker. See `docs/AZURE_H100.md`."
        )

    # Gradio 6 moved these from Blocks(...) to launch(...).
    demo._minimax_theme = theme
    return demo


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    enforce_offline_runtime()
    config = load_config()
    demo = build_app(config)

    def _warmup():
        try:
            default_spec = config.lora_by_id(config.default_lora_id)
            if default_spec.backend != "comfy_pruned":
                get_engine(config).load()
            else:
                logger.info("Skipping full Diffusers warmup for pruned default backend")
        except Exception:
            logger.exception("Background model load failed")

    threading.Thread(target=_warmup, name="h3-warmup", daemon=True).start()
    os.environ["GRADIO_SERVER_NAME"] = LISTEN_HOST
    port = int(config.server_port)
    _print_listen_urls(port)
    from starlette.middleware import Middleware

    demo.queue(max_size=config.max_queue).launch(
        server_name=LISTEN_HOST,
        server_port=port,
        theme=getattr(demo, "_minimax_theme", None),
        css=CSS,
        share=False,
        ssr_mode=False,
        mcp_server=False,
        enable_monitoring=False,
        inbrowser=False,
        quiet=False,
        show_error=True,
        max_file_size="100mb",
        app_kwargs={"middleware": [Middleware(_ChunkedUploadPreviewMiddleware)]},
        allowed_paths=[str(config.output_dir), str(config.lora_dir), str(config.upload_dir)],
    )


if __name__ == "__main__":
    main()
