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

# Optional restoration; never part of the FL2VA default asset set.
RESTORE_FILE = "Restore_Enhance_Improve_rank16_v1_H3-lora.safetensors"
RESTORE_URL = "https://civitai.red/api/download/models/3389308?fileId=3278511"
RESTORE_SHA256 = "04068fccc4f7c14c3f42d0c1dc1a408eb335fa9c31149d068174e702eab47296"
RESTORE_SIZE = 149165944  # API sizeKB 145669.8671875 * 1024, exactly.

# Optional character swap; shares restore_base, but never downloads it implicitly.
CHARACTER_SWAP_LORA_REPO = "akatz-ai/MiniMax-H3-Character-Swap-LoRA"
CHARACTER_SWAP_LORA_REVISION = "62407e0cc8089c363abd9ce4b0b27662abb237af"
CHARACTER_SWAP_LORA_FILE = "h3_character_swap_pro4500_1000.safetensors"
CHARACTER_SWAP_LORA_URL = (
    f"https://huggingface.co/{CHARACTER_SWAP_LORA_REPO}/resolve/"
    f"{CHARACTER_SWAP_LORA_REVISION}/{CHARACTER_SWAP_LORA_FILE}"
)
# Verified against the pinned Hugging Face API LFS metadata.
CHARACTER_SWAP_LORA_SIZE = 155110320
CHARACTER_SWAP_LORA_SHA256 = "4b2a3f420ae804c0aa3422761ff84dbd1bf52eef6900ffab6d2e66df63cb4e79"

REF2VA_MODEL_REPO = "Comfy-Org/MiniMax-H3"
REF2VA_MODEL_REVISION = "e5eb578a89295337b8ff433a035929ce0279e0b6"
REF2VA_MODEL_FILE = "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
REF2VA_MODEL_SHA256 = "f86f2f79ebd2d76eb8eeb46091e83982e6ff51d255747e7b16e92834b392b8e9"
REF2VA_MODEL_SIZE = 20958205608

REF2VA_TURBO_REPO = "Kijai/MiniMax-H3-experimental"
REF2VA_TURBO_REVISION = "d8023be02fefbb3633b0cd335c3879f91177299d"
REF2VA_TURBO_FILE = "MiniMax-H3-Ref2VA-Acc-8Step_pruned_comfy.safetensors"
REF2VA_TURBO_URL = (
    f"https://huggingface.co/{REF2VA_TURBO_REPO}/resolve/"
    f"{REF2VA_TURBO_REVISION}/{REF2VA_TURBO_FILE}"
)
REF2VA_TURBO_SHA256 = "6f18e1c2eccb14b37322607730f26b16bf1169b56cd098ea006cffaec43d1e39"
REF2VA_TURBO_SIZE = 1725921392

UPSCALER_REPO = "Comfy-Org/Real-ESRGAN_repackaged"
UPSCALER_FILE = "RealESRGAN_x4plus.safetensors"
UPSCALER_SHA256 = "37f9a931c215f040aa6d50f711f2cb115f713c46df1d0d6469a8bd7bfe9a60bb"

RIFE_REPO = "Comfy-Org/frame_interpolation"
RIFE_FILE = "frame_interpolation/rife_v4.25_lite.safetensors"
RIFE_SHA256 = "e5e5fe0286d30708f4c36aa23639a38d3d7cd0c724922c66e6b04130ae12c6e4"

LATENT_UPSCALER_FILE = "h3_upscaler_lms_v0.1.safetensors"
LATENT_UPSCALER_URL = "https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/resolve/e25a489717c7067ed1813bbd884e64fa68b76d58/latent_upscaler/h3_upscaler_lms_v0.1.safetensors"
LATENT_UPSCALER_SHA256 = "40d4228227146245ef2b65480b647c50222465ca125de5880f192e3259e40d59"
LATENT_UPSCALER_SIZE = 690593048
LATENT_NODE_REVISION = "40316cf008b2fd8663263270669eb4da23f89d2c"
LATENT_NODE_BASE_URL = f"https://raw.githubusercontent.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/{LATENT_NODE_REVISION}/"
LATENT_NODE_FILES = {
    "upstream_latent_3d.py": ("nodes/minimax_h3_latent_upscaler_3d.py", "744063b43e0f3eec23e2485cb7c65503069946ca9690906ecb548d7515cb89e2"),
    "UPSTREAM_LICENSE": ("LICENSE", "86805fe49f63c8957e4427f7e7f26fba687c6cea05fa9249bf36f9d3316a0734"),
}

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
