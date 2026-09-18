"""Force an air-gapped runtime: no Hub, no telemetry, no safety checkers.

The download script is the only module allowed to talk to the network.
Import and call ``enforce_offline_runtime()`` before transformers / diffusers
/ Gradio so those libraries never open a socket.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "DIFFUSERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "GRADIO_ANALYTICS_ENABLED": "False",
    "GRADIO_SHARE": "0",
    "GRADIO_MCP_SERVER": "False",
}

# This is effective when set before torch imports (app.py/generate.py do so).
CUDA_ALLOC_CONF = "expandable_segments:True"

HUB_ATTENTION_BACKENDS = {
    "_flash_3_hub",
    "_flash_3_varlen_hub",
    "flash_hub",
    "_flash_varlen_hub",
}

REQUIRED_SNAPSHOT_MARKERS = (
    "modular_model_index.json",
    "model_index.json",
)
REQUIRED_SNAPSHOT_DIRS = (
    "transformer",
    "text_encoder",
    "tokenizer",
    "processor",
    "vae",
    "audio_vae",
    "scheduler",
    "audio_scheduler",
)

SAFETY_ATTRS = (
    "safety_checker",
    "safety_checker_model",
    "nsfw_checker",
    "watermarker",
    "watermark",
    "content_filter",
)


def enforce_offline_runtime() -> None:
    """Pin process-wide flags so Hub / Gradio / telemetry stay dark."""
    for key, value in OFFLINE_ENV.items():
        os.environ[key] = value
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", CUDA_ALLOC_CONF)
    os.environ.pop("GRADIO_SHARE_SERVER_ADDRESS", None)


def allow_online_for_download() -> None:
    """Clear offline pins so ``scripts/download_models.py`` can fetch weights."""
    for key in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "DIFFUSERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
    ):
        os.environ.pop(key, None)


def local_attention_backend(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    backend = str(name).strip()
    if backend.lower() in HUB_ATTENTION_BACKENDS or backend.lower().endswith("_hub"):
        logger.warning(
            "Ignoring Hub attention backend %s (offline runtime). Using local Flash/SDPA.",
            backend,
        )
        return "_flash_3"
    return backend


def snapshot_ready(local_dir: Path) -> bool:
    root = Path(local_dir)
    return root.is_dir() and any((root / marker).is_file() for marker in REQUIRED_SNAPSHOT_MARKERS)


def require_local_snapshot(local_dir: Path) -> Path:
    root = Path(local_dir).expanduser().resolve()
    if not snapshot_ready(root):
        raise FileNotFoundError(
            f"Local MiniMax-H3 snapshot not found at {root}. "
            "This app is offline-only and will not pull from Hugging Face. "
            "While you have network access, run: python scripts/download_models.py --base"
        )
    missing_dirs = [name for name in REQUIRED_SNAPSHOT_DIRS if not (root / name).exists()]
    if missing_dirs:
        raise FileNotFoundError(
            f"Incomplete local snapshot at {root}: missing {', '.join(missing_dirs)}. "
            "Re-run: python scripts/download_models.py --base"
        )
    return root


def disable_safety_guards(pipe: Any) -> None:
    """Strip every content-filter / watermark hook if a pipeline exposes one."""
    for attr in SAFETY_ATTRS:
        if hasattr(pipe, attr):
            try:
                setattr(pipe, attr, None)
            except Exception:  # noqa: BLE001
                logger.debug("Could not clear %s", attr)
    if hasattr(pipe, "requires_safety_checker"):
        try:
            pipe.requires_safety_checker = False
        except Exception:  # noqa: BLE001
            pass
    config = getattr(pipe, "config", None)
    if isinstance(config, dict) and "requires_safety_checker" in config:
        config["requires_safety_checker"] = False


def strip_hub_kwargs(kwargs: dict) -> dict:
    """Drop credentials / Hub options that would trigger a network call."""
    for key in ("token", "revision", "proxies", "mirror", "endpoint"):
        kwargs.pop(key, None)
    kwargs["local_files_only"] = True
    return kwargs


def _is_local_weight_path(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return any(_is_local_weight_path(item) for item in value)
    path = Path(str(value)).expanduser()
    return path.exists()


def pin_component_specs_to_local(pipe: Any, local_dir: str | Path) -> int:
    """Rewrite Hub repo IDs in ModularPipeline component specs to a local snapshot.

    MiniMax-H3's ``modular_model_index.json`` keeps
    ``pretrained_model_name_or_path: MiniMaxAI/MiniMax-H3`` even after a local
    download. ``load_components()`` then hits Hugging Face and, under
    ``HF_HUB_OFFLINE``, prints a failure for every component while leaving
    them ``None``.
    """
    root = str(Path(local_dir).expanduser().resolve())
    specs = getattr(pipe, "_component_specs", None) or {}
    rewritten = 0
    for spec in specs.values():
        if getattr(spec, "default_creation_method", "from_pretrained") != "from_pretrained":
            continue
        current = getattr(spec, "pretrained_model_name_or_path", None)
        if _is_local_weight_path(current):
            continue
        spec.pretrained_model_name_or_path = root
        repo = getattr(spec, "repo", None)
        if repo and not _is_local_weight_path(repo):
            spec.repo = root
        rewritten += 1
    if rewritten:
        logger.info("Pinned %s component specs to local snapshot %s", rewritten, root)
    if hasattr(pipe, "_pretrained_model_name_or_path"):
        pipe._pretrained_model_name_or_path = root
    return rewritten


def assert_no_hub_backends(backends: Iterable[str]) -> None:
    bad = [name for name in backends if str(name).lower() in HUB_ATTENTION_BACKENDS]
    if bad:
        raise RuntimeError(f"Hub attention backends are not allowed offline: {bad}")
