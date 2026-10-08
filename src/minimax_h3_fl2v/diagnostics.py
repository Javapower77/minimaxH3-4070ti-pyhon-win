"""Small, stdlib-only diagnostic helpers; never serialize requests or workflows raw.

Application handlers sanitize *after* rendering (including exception tracebacks).
Other handlers' records are not modified. Worker rotation belongs to its reader
thread, not to a subprocess holding an unmanaged Windows file handle.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
import math
import os
import re
import sys
import threading
import time
import urllib.parse
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 5
MAX_WORKER_LINE = 64 * 1024
_REDACTED = "[REDACTED]"
_lock = threading.RLock()
_active_prompts: dict[str, int] = {}
_hooks_installed = False
_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "diagnostics_request_id", default=None
)
_secrets: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "diagnostics_secrets", default=()
)
_ENV_NAMES = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "CIVITAI_API_TOKEN", "CIVITAI_TOKEN"}
_CREDENTIAL = r"(?:access[_-]?token|refresh[_-]?token|hf[_-]?token|token|password|passwd|api[_-]?key|apikey|client[_-]?secret)"
_AUTH = re.compile(r"(?i)(\b(?:authorization[\s\"']*[:=][\s\"']*)?(?:bearer|basic)\s+)[^\s\"',;<>]+")
_ASSIGNMENT = re.compile(
    rf"(?i)(\b{_CREDENTIAL}[\"']?\s*[:=]\s*)(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\s,;&\}}\]\"']+)"
)
_USERINFO = re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://)[^\s/@]+@")
_QUERY = re.compile(rf"(?i)([?&]{_CREDENTIAL}=)[^&#\s\"']*")


def _secret_variants(secret: str) -> set[str]:
    return {
        secret, json.dumps(secret, ensure_ascii=True)[1:-1],
        json.dumps(secret, ensure_ascii=False)[1:-1], repr(secret)[1:-1],
        urllib.parse.quote(secret, safe=""), urllib.parse.quote_plus(secret),
    }


def _sanitize(text: str) -> str:
    # Only credential variables: never scoop up paths, user names, or other env values.
    secrets = list(_secrets.get())
    secrets.extend(
        value for name, value in os.environ.items()
        if value and (name.upper() in _ENV_NAMES
                      or re.search(r"(?:^|_)API_?KEY$", name, re.IGNORECASE))
    )
    variants = {variant for secret in secrets if secret
                for variant in _secret_variants(secret) if variant}
    for variant in sorted(variants, key=len, reverse=True):
        text = text.replace(variant, _REDACTED)
    text = _AUTH.sub(lambda m: m[1] + _REDACTED, text)
    text = _USERINFO.sub(lambda m: m[1] + _REDACTED + "@", text)
    text = _QUERY.sub(lambda m: m[1] + _REDACTED, text)
    def redact_assignment(match):
        quote = match[2][0] if match[2][0] in ("'", '"') else ""
        return match[1] + quote + _REDACTED + quote

    return _ASSIGNMENT.sub(redact_assignment, text)


class SanitizedFormatter(logging.Formatter):
    """Render a private copy so exc_text/message caching cannot taint other handlers."""

    def format(self, record: logging.LogRecord) -> str:
        local = logging.makeLogRecord(record.__dict__.copy())
        local.exc_text = None
        return _sanitize(super().format(local))


def _handler(path: Path) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    handler.setFormatter(SanitizedFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    return handler


def _install_hooks() -> None:
    global _hooks_installed
    if _hooks_installed:
        return
    original_sys = sys.excepthook
    original_thread = threading.excepthook
    logger = logging.getLogger(__name__)

    def sys_hook(kind, value, tb):
        if issubclass(kind, KeyboardInterrupt):
            original_sys(kind, value, tb)
            return
        logger.critical("Uncaught exception", exc_info=(kind, value, tb))

    def thread_hook(args):
        if issubclass(args.exc_type, (KeyboardInterrupt, SystemExit)):
            original_thread(args)
            return
        logger.critical(
            "Uncaught thread exception",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = sys_hook
    threading.excepthook = thread_hook
    _hooks_installed = True


def initialize_logging(root: Path = ROOT) -> Path:
    """Create application.log once per resolved path and install exception hooks once."""
    with _lock:
        path = Path(root) / "logs" / "application.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Resolve only after creating parents: concurrent Windows resolution of
        # nonexistent paths can disagree with the canonical on-disk spelling.
        path = path.resolve()
        logger = logging.getLogger()
        if not any(getattr(h, "_diagnostics_path", None) == path for h in logger.handlers):
            handler = _handler(path)
            handler._diagnostics_path = path
            logger.addHandler(handler)
        if not any(isinstance(h, logging.StreamHandler)
                   and not isinstance(h, logging.FileHandler) for h in logger.handlers):
            console = logging.StreamHandler()
            console.setFormatter(SanitizedFormatter("%(levelname)s %(name)s %(message)s"))
            console._diagnostics_console = True
            logger.addHandler(console)
        if logger.level > logging.INFO:
            logger.setLevel(logging.INFO)
        for name in ("httpx", "httpcore", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)
        _install_hooks()
    return path


def _safe(value: Any) -> Any:
    """Accept JSON primitives/containers, never arbitrary repr() of request objects."""
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if type(value) is str:
        return _sanitize(value)
    if isinstance(value, Mapping):
        return {_sanitize(str(k)): _safe(v) for k, v in value.items()
                if type(k) in (str, int)}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return None


def _emit(logger, name, metadata, level=logging.INFO, exc_info=False):
    payload = _safe(metadata)
    payload["event"] = _sanitize(name)
    if _request_id.get() is not None:
        payload["request_id"] = _request_id.get()
    logger.log(level, json.dumps(payload, ensure_ascii=True, allow_nan=False), exc_info=exc_info)


def event(logger: logging.Logger, event: str, **metadata: Any) -> None:
    """Log caller-selected safe metadata, with the current generation correlation ID."""
    _emit(logger, event, metadata)


def file_metadata(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        size = None
    return {"file": _sanitize(path.name), "bytes": size}


# Class-specific allowlists deliberately do not share generic input-name rules.
_LOADERS = {
    "UNETLoader": {"unet_name": ("diffusion_models", "unet")},
    "CheckpointLoaderSimple": {"ckpt_name": ("checkpoints",)},
    "CLIPLoader": {"clip_name": ("text_encoders", "clip")},
    "DualCLIPLoader": {"clip_name1": ("text_encoders", "clip"),
                       "clip_name2": ("text_encoders", "clip")},
    "VAELoader": {"vae_name": ("vae",)},
    "LoraLoader": {"lora_name": ("loras",)},
    "LoraLoaderModelOnly": {"lora_name": ("loras",)},
    "UpscaleModelLoader": {"model_name": ("upscale_models",)},
    "FaceRestoreModelLoader": {"model_name": ("facerestore_models",)},
    "FrameInterpolationModelLoader": {"model_name": ("", "frame_interpolation")},
    "MiniMaxH3LearnedLatentUpscale": {"model_name": ("latent_upscale_models",)},
}
_INPUTS = {
    "UNETLoader": {"unet_name", "model_name"},
    "CheckpointLoaderSimple": {"ckpt_name", "model_name"},
    "CLIPLoader": {"clip_name", "model_name", "device"},
    "DualCLIPLoader": {"clip_name1", "clip_name2", "device"},
    "VAELoader": {"vae_name", "model_name"},
    "LoraLoader": {"lora_name", "model_name", "strength_model", "strength_clip"},
    "LoraLoaderModelOnly": {"lora_name", "model_name", "strength_model"},
    "UpscaleModelLoader": {"model_name"},
    "FaceRestoreModelLoader": {"model_name"},
    "FrameInterpolationModelLoader": {"model_name"},
    "MiniMaxH3LearnedLatentUpscale": {"model_name", "scale", "device"},
    "MiniMaxH3ImageToVideo": {"width", "height", "length"},
    "MiniMaxH3ReferenceToVideo": {"width", "height", "length"},
    "EmptyLatentImage": {"width", "height"},
    "EmptyHunyuanLatentVideo": {"width", "height", "length"},
    "RandomNoise": {"noise_seed", "seed"},
    "KSamplerSelect": {"sampler_name"},
    "BasicScheduler": {"scheduler", "steps", "denoise"},
    "KSampler": {"seed", "steps", "sampler_name", "scheduler", "denoise"},
    "MiniMaxH3SigmaShift": {"shift_video", "shift_audio"},
    "CreateVideo": {"fps"},
    "ImageScaleBy": {"scale_by", "upscale_method"},
    "ImageScale": {"width", "height", "upscale_method"},
    "MiniMaxH3ResampleFramesToFPS": {"source_fps", "target_fps"},
}


def _model_file(root: Path, filename: str, folders: tuple[str, ...]) -> dict[str, Any]:
    # Preserve safe relative loader subdirectories for lookup, not in the log.
    root = root.resolve()
    for folder in folders:
        candidate = (root / folder / filename).resolve()
        if candidate.is_relative_to((root / folder).resolve()) and candidate.is_file():
            return file_metadata(candidate)
    return {"file": _sanitize(filename.replace("\\", "/").rsplit("/", 1)[-1]), "bytes": None}


def workflow_metadata(graph: Mapping, model_root: Path) -> dict[str, Any]:
    """Return {'nodes': {id: {class_type, inputs, files?}}}, never node contents."""
    nodes = {}
    for node_id, node in graph.items():
        if not isinstance(node, Mapping):
            continue
        kind = node.get("class_type", "")
        if type(kind) is not str or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", kind):
            kind = "unknown"
        raw = node.get("inputs", {})
        raw = raw if isinstance(raw, Mapping) else {}
        inputs = {}
        files = {}
        loaders = _LOADERS.get(kind, {})
        for key in _INPUTS.get(kind, ()):
            value = raw.get(key)
            if key not in raw or type(value) not in (str, int, float, bool):
                continue
            if isinstance(value, str):
                if key in loaders or key == "model_name":
                    value = value.replace("\\", "/").rsplit("/", 1)[-1]
                value = value[:1024]
            inputs[key] = _safe(value)
            if key in loaders and type(raw[key]) is str:
                files[key] = _model_file(Path(model_root), raw[key], loaders[key])
        entry = {"class_type": kind, "inputs": inputs}
        if files:
            entry["files"] = files
        nodes[_sanitize(str(node_id))] = entry
    return {"nodes": nodes}


@contextmanager
def stage(logger: logging.Logger, name: str, **metadata: Any):
    """Time the entire block, including waits; traceback only, never frame locals."""
    started = time.perf_counter()
    _emit(logger, "stage_start", {**metadata, "stage": name})
    try:
        yield
    except BaseException:
        _emit(logger, "stage_error", {
            **metadata, "stage": name, "duration_seconds": time.perf_counter() - started,
        }, logging.ERROR, True)
        raise
    else:
        _emit(logger, "stage_end", {
            **metadata, "stage": name, "duration_seconds": time.perf_counter() - started,
        })


_REQUEST_FIELDS = (
    "mode", "nfe", "seed", "megapixels", "duration_seconds", "lora_id",
    "upscale", "latent_upscale", "face_restore", "target_fps",
)


def _prompt_secrets(value: Any, seen: set[int] | None = None) -> list[str]:
    """Inspect stored fields, not properties/repr; also cover nested *_prompt fields."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, Mapping):
        fields = value
    elif isinstance(value, (list, tuple)):
        return [s for item in value for s in _prompt_secrets(item, seen)]
    else:
        fields = getattr(value, "__dict__", {})
        slots = getattr(type(value), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        fields = {**fields, **{k: getattr(value, k, None) for k in slots}}
    found = []
    for key, item in fields.items():
        if isinstance(key, str) and "prompt" in key.lower() and isinstance(item, str):
            found.extend([item, *[line for line in item.splitlines() if line]])
        elif isinstance(item, (Mapping, list, tuple)) or hasattr(item, "__dict__"):
            found.extend(_prompt_secrets(item, seen))
    return found


def traced_generation(function=None, *, logger: logging.Logger | None = None):
    """Use @traced_generation, @traced_generation(logger=...), or (logger)(fn).

    The wrapper starts before the function's queue lock/load. Nested stages and
    events inherit its UUID. Only allowlisted scalar request fields and scalar
    result timings are recorded; prompt redaction is scoped to this invocation.
    """
    if isinstance(function, logging.Logger):
        return functools.partial(traced_generation, logger=function)
    if function is None:
        return functools.partial(traced_generation, logger=logger)
    log = logger if logger is not None else logging.getLogger(function.__module__)
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        request = bound.arguments.get("request")
        if request is None:
            request = next((v for k, v in bound.arguments.items()
                            if k not in ("self", "cls") and
                            (isinstance(v, Mapping) and "prompt" in v
                             or hasattr(v, "prompt"))), None)
        arguments = {k: v for k, v in bound.arguments.items() if k not in ("self", "cls")}
        secrets = _prompt_secrets(arguments)
        secret_token = _secrets.set((*_secrets.get(), *secrets))
        with _lock:
            for secret in secrets:
                _active_prompts[secret] = _active_prompts.get(secret, 0) + 1
        id_token = _request_id.set(_request_id.get() or str(uuid.uuid4()))
        started = time.perf_counter()
        try:
            metadata = {"request_type": type(request).__name__}
            for field in _REQUEST_FIELDS:
                value = (request.get(field) if isinstance(request, Mapping)
                         else getattr(request, field, None))
                if value is None or type(value) in (str, int, float, bool):
                    metadata[field] = _safe(value)
            _emit(log, "generation_start", metadata)
            result = function(*args, **kwargs)
            timings = result.get("timings") if isinstance(result, Mapping) else getattr(result, "timings", None)
            ending = {"duration_seconds": time.perf_counter() - started}
            if isinstance(timings, Mapping):
                ending["timings"] = {k: v for k, v in timings.items()
                                     if type(k) is str and type(v) in (int, float)}
            _emit(log, "generation_end", ending)
            return result
        except BaseException:
            _emit(log, "generation_error", {
                "duration_seconds": time.perf_counter() - started,
            }, logging.ERROR, True)
            raise
        finally:
            _request_id.reset(id_token)
            _secrets.reset(secret_token)
            with _lock:
                for secret in secrets:
                    remaining = _active_prompts[secret] - 1
                    if remaining:
                        _active_prompts[secret] = remaining
                    else:
                        del _active_prompts[secret]

    return wrapped


def start_worker_capture(process, path: Path) -> threading.Thread:
    """Drain a text-mode PIPE into an independent rotating log; return its daemon.

    Launch the process with stdout=PIPE, stderr=STDOUT, text=True, encoding='utf-8',
    errors='replace', bufsize=1. Oversized lines are discarded, not split into
    potentially unredactable credential fragments. Both CR and LF delimit lines.
    """
    if process.stdout is None:
        raise ValueError("Worker capture requires stdout=PIPE in text mode")
    path = Path(path)
    launch_prompts = _secrets.get()

    def capture():
        handler = None
        worker = logging.Logger(f"{__name__}.worker.{uuid.uuid4()}", logging.INFO)
        worker.propagate = False
        pipe = process.stdout
        def emit_line(line):
            # A reused worker's reader has no generation ContextVar. Consult
            # active requests on every line, including multiline/escaped echoes.
            with _lock:
                prompts = (*launch_prompts, *_active_prompts)
            payload = re.search(r"(?i)\b(?:\w*prompt\w*|text)\b", line)
            # Preserve the worker's numeric total-duration summary, not payloads.
            if re.fullmatch(r"(?:\[INFO\]\s*)?Prompt executed in (?:[0-9:.]+ seconds|[0-9:]+)", line.strip()):
                payload = None
            known = any(variant in line for prompt in prompts if prompt
                        for variant in _secret_variants(prompt) if variant)
            if payload or known:
                worker.info("[worker prompt/text payload line omitted]")
            else:
                worker.info("%s", _sanitize(line))

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = _handler(path)
            worker.addHandler(handler)
            pending = ""
            oversized = False
            while True:
                # Buffered TextIO reads larger requested sizes until filled: that
                # delays sparse startup/progress lines. Read buffered characters
                # instead, emitting CR/LF lines immediately without GPU polling.
                chunk = pipe.read(1)
                if not chunk:
                    if pending and not oversized:
                        emit_line(pending)
                    break
                for part in re.split(r"([\r\n])", chunk):
                    if part in ("\r", "\n"):
                        if pending and not oversized:
                            emit_line(pending)
                        pending = ""
                        oversized = False
                    elif not oversized:
                        if len(pending) + len(part) > MAX_WORKER_LINE:
                            pending = ""
                            oversized = True
                            worker.warning("[oversized worker line omitted]")
                        else:
                            pending += part
        except Exception:
            if handler is not None:
                worker.error("Worker capture failed", exc_info=True)
        finally:
            try:
                pipe.close()
            finally:
                if handler is not None:
                    worker.removeHandler(handler)
                    handler.close()

    thread = threading.Thread(target=capture, name="minimax-worker-log", daemon=True)
    thread.start()
    return thread