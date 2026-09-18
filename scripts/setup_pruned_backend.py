#!/usr/bin/env python3
"""Install the isolated ComfyUI worker and pruned MiniMax-H3 FL2VA weights."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.getenv("MINIMAX_H3_COMFY_ROOT", ROOT / ".runtime" / "ComfyUI")).resolve()
COMFY_REVISION = "1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a"
REPO = "Comfy-Org/MiniMax-H3"
UPSCALER_REPO = "Comfy-Org/Real-ESRGAN_repackaged"
UPSCALER_FILE = "RealESRGAN_x4plus.safetensors"
RIFE_REPO = "Comfy-Org/frame_interpolation"
RIFE_FILE = "frame_interpolation/rife_v4.25_lite.safetensors"
RIFE_SHA256 = "e5e5fe0286d30708f4c36aa23639a38d3d7cd0c724922c66e6b04130ae12c6e4"
FACE_RESTORE_REPO = "https://github.com/mav-rik/facerestore_cf.git"
FACE_RESTORE_REVISION = "ff4d7a5c102441d8f058dd6135797ffb57b6c6ad"
FACE_ASSETS = {
    "facerestore_models/codeformer.pth": (
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        376637898,
    ),
    "facedetection/detection_mobilenet0.25_Final.pth": (
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth",
        1789735,
    ),
    "facedetection/parsing_parsenet.pth": (
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
        85331193,
    ),
}
FILES = (
    "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae/minimax_h3_video_vae_fp16.safetensors",
    "vae/minimax_h3_audio_vae_fp32.safetensors",
)
PRUNED_SHA256 = "12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993"
UPSCALER_SHA256 = "37f9a931c215f040aa6d50f711f2cb115f713c46df1d0d6469a8bd7bfe9a60bb"


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_url(url: str, destination: Path, expected_size: int) -> None:
    if destination.is_file() and destination.stat().st_size == expected_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(url) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream, 16 * 1024 * 1024)
    if temporary.stat().st_size != expected_size:
        actual = temporary.stat().st_size
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download size mismatch for {destination.name}: expected {expected_size}, got {actual}"
        )
    temporary.replace(destination)


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

    sys.path.insert(0, str(ROOT / "src"))
    from minimax_h3_fl2v.offline import allow_online_for_download

    allow_online_for_download()
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    from huggingface_hub import hf_hub_download

    for filename in FILES:
        print(f"Downloading {REPO}/{filename}", flush=True)
        hf_hub_download(repo_id=REPO, filename=filename, local_dir=str(COMFY / "models"))

    upscale_dir = COMFY / "models" / "upscale_models"
    print(f"Downloading {UPSCALER_REPO}/{UPSCALER_FILE}", flush=True)
    hf_hub_download(
        repo_id=UPSCALER_REPO,
        filename=UPSCALER_FILE,
        local_dir=str(upscale_dir),
    )
    rife = Path(
        hf_hub_download(
            repo_id=RIFE_REPO,
            filename=RIFE_FILE,
            local_dir=str(COMFY / "models"),
        )
    )
    if sha256(rife) != RIFE_SHA256:
        raise RuntimeError(f"RIFE SHA-256 mismatch for {rife}")

    for relative_path, (url, expected_size) in FACE_ASSETS.items():
        download_url(url, COMFY / "models" / relative_path, expected_size)

    model = COMFY / "models/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
    actual = sha256(model)
    if actual != PRUNED_SHA256:
        raise RuntimeError(f"Pruned base SHA-256 mismatch: expected {PRUNED_SHA256}, got {actual}")
    upscaler = upscale_dir / UPSCALER_FILE
    upscale_actual = sha256(upscaler)
    if upscale_actual != UPSCALER_SHA256:
        raise RuntimeError(
            f"Upscaler SHA-256 mismatch: expected {UPSCALER_SHA256}, got {upscale_actual}"
        )

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
