#!/usr/bin/env bash
set -Eeuo pipefail

APP_ROOT=/opt/app
COMFY_ROOT="${MINIMAX_H3_COMFY_ROOT:-/opt/comfyui}"
MODEL_ROOT="${COMFY_ROOT}/models"
LORA_ROOT="${MINIMAX_H3_LORA_DIR:-/data/models/loras}"
MARKER="${MODEL_ROOT}/.minimax-h3-ready"

mkdir -p "${MODEL_ROOT}" "${LORA_ROOT}" /data/hf-home "${APP_ROOT}/outputs" "${APP_ROOT}/uploads"

check_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo >&2 "NVIDIA GPU is unavailable. Install the host NVIDIA driver and NVIDIA Container Toolkit."
    exit 1
  fi
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
}

check_models() {
  local missing=0
  local files=(
    "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    "vae/minimax_h3_video_vae_fp16.safetensors"
    "vae/minimax_h3_audio_vae_fp32.safetensors"
    "upscale_models/RealESRGAN_x4plus.safetensors"
    "frame_interpolation/rife_v4.25_lite.safetensors"
    "facerestore_models/codeformer.pth"
    "facedetection/detection_mobilenet0.25_Final.pth"
    "facedetection/parsing_parsenet.pth"
  )
  for file in "${files[@]}"; do
    if [[ ! -s "${MODEL_ROOT}/${file}" ]]; then
      echo >&2 "Missing ${MODEL_ROOT}/${file}"
      missing=1
    fi
  done
  if [[ ! -s "${LORA_ROOT}/minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors" ]]; then
    echo >&2 "Missing TaoMate LoRA in ${LORA_ROOT}"
    missing=1
  fi
  return "${missing}"
}

setup_models() {
  export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 DIFFUSERS_OFFLINE=0
  python "${APP_ROOT}/scripts/setup_pruned_backend.py" --skip-install
  python -m minimax_h3_fl2v.download --loras --lora-id taomate_fl2va_3step_ema
  touch "${MARKER}"
  echo "Model setup complete. The studio can now run offline."
}

case "${1:-serve}" in
  setup)
    setup_models
    ;;
  check)
    check_gpu
    check_models
    ;;
  serve)
    check_gpu
    if ! check_models; then
      cat >&2 <<'EOF'
Required model artifacts are absent. Run once while online:
  docker compose --profile setup run --rm setup
Then start the offline studio:
  docker compose up -d studio
EOF
      exit 2
    fi
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
    exec python "${APP_ROOT}/app.py"
    ;;
  *)
    exec "$@"
    ;;
esac
