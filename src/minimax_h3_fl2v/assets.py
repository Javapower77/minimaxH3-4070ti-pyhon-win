"""Local paths and checksums for MiniMax-H3 FL2VA runtime assets."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMFY_ROOT = Path(os.getenv("MINIMAX_H3_COMFY_ROOT", ROOT / ".runtime" / "ComfyUI")).resolve()

PRUNED_REPO = "Comfy-Org/MiniMax-H3"
PRUNED_FILES = (
    "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae/minimax_h3_video_vae_fp16.safetensors",
    "vae/minimax_h3_audio_vae_fp32.safetensors",
)
PRUNED_SHA256 = {
    "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors": (
        "12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993"
    ),
}

UPSCALER_REPO = "Comfy-Org/Real-ESRGAN_repackaged"
UPSCALER_FILE = "RealESRGAN_x4plus.safetensors"
UPSCALER_SHA256 = "37f9a931c215f040aa6d50f711f2cb115f713c46df1d0d6469a8bd7bfe9a60bb"

RIFE_REPO = "Comfy-Org/frame_interpolation"
RIFE_FILE = "frame_interpolation/rife_v4.25_lite.safetensors"
RIFE_SHA256 = "e5e5fe0286d30708f4c36aa23639a38d3d7cd0c724922c66e6b04130ae12c6e4"

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
