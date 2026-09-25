#!/usr/bin/env python3
"""Install the isolated ComfyUI worker and pruned MiniMax-H3 FL2VA weights."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from minimax_h3_fl2v.assets import DEFAULT_COMFY_ROOT  # noqa: E402
from minimax_h3_fl2v.download import (  # noqa: E402
    download_postprocess_models,
    download_pruned_models,
)

COMFY = DEFAULT_COMFY_ROOT
COMFY_REVISION = "1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a"
FACE_RESTORE_REPO = "https://github.com/mav-rik/facerestore_cf.git"
FACE_RESTORE_REVISION = "ff4d7a5c102441d8f058dd6135797ffb57b6c6ad"


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def venv_python(venv: Path) -> Path:
    """Return the interpreter path for a virtual environment on this OS."""
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def python311_executable() -> str:
    """Find Python 3.11 without assuming the Unix ``python3.11`` command."""
    if sys.version_info[:2] == (3, 11):
        return sys.executable
    if os.name == "nt" and shutil.which("py"):
        result = subprocess.run(
            ["py", "-3.11", "-c", "import sys; print(sys.executable)"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    executable = shutil.which("python3.11")
    if executable:
        return executable
    raise RuntimeError(
        "Python 3.11 is required. Activate a Python 3.11 environment or install it first."
    )


def link_or_copy(source: Path, destination: Path) -> None:
    """Prefer a symlink, but copy on Windows when link privileges are absent."""
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true", help="Only download/check model files")
    args = parser.parse_args()

    if not args.skip_install:
        COMFY.parent.mkdir(parents=True, exist_ok=True)
        if not (COMFY / ".git").is_dir():
            run("git", "clone", "--depth", "1", "https://github.com/Comfy-Org/ComfyUI.git", str(COMFY))
        run("git", "-C", str(COMFY), "fetch", "--depth", "1", "origin", COMFY_REVISION)
        run("git", "-C", str(COMFY), "checkout", "--detach", COMFY_REVISION)
        configured_python = os.getenv("MINIMAX_H3_COMFY_PYTHON")
        venv = COMFY / ".venv"
        if configured_python:
            python = configured_python
        else:
            python_path = venv_python(venv)
            if not python_path.is_file():
                run(python311_executable(), "-m", "venv", str(venv))
            python = str(python_path)
        run(python, "-m", "pip", "install", "--upgrade", "pip")
        run(
            python, "-m", "pip", "install", "torch==2.11.0", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu128",
        )
        run(python, "-m", "pip", "install", "-r", str(COMFY / "requirements.txt"))

        face_restore = COMFY / "custom_nodes" / "facerestore_cf"
        if not (face_restore / ".git").is_dir():
            run("git", "clone", FACE_RESTORE_REPO, str(face_restore))
        run("git", "-C", str(face_restore), "fetch", "--depth", "1", "origin", FACE_RESTORE_REVISION)
        run("git", "-C", str(face_restore), "checkout", "--detach", FACE_RESTORE_REVISION)
        run(python, "-m", "pip", "install", "-r", str(face_restore / "requirements.txt"))

    app_nodes = ROOT / "comfy_nodes" / "minimax_h3_nodes"
    installed_nodes = COMFY / "custom_nodes" / "minimax_h3_nodes"
    if installed_nodes.exists() or installed_nodes.is_symlink():
        if installed_nodes.is_dir() and not installed_nodes.is_symlink():
            shutil.rmtree(installed_nodes)
        else:
            installed_nodes.unlink()
    shutil.copytree(app_nodes, installed_nodes)

    from minimax_h3_fl2v.offline import allow_online_for_download

    allow_online_for_download()
    print("Downloading pruned FL2VA transformer, text encoder, and VAEs...", flush=True)
    download_pruned_models(COMFY)
    print("Downloading post-process models (Real-ESRGAN, RIFE, CodeFormer)...", flush=True)
    download_postprocess_models(COMFY)

    lora_target = COMFY / "models/loras"
    lora_target.mkdir(parents=True, exist_ok=True)
    lora_source = Path(os.getenv("MINIMAX_H3_LORA_DIR", ROOT / "models" / "loras"))
    for source in lora_source.glob("*.safetensors"):
        destination = lora_target / source.name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        link_or_copy(source, destination)

    print("Pruned FL2VA backend is ready.")


if __name__ == "__main__":
    main()
