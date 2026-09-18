# syntax=docker/dockerfile:1.7
# Ubuntu 24.04 with CUDA user-space libraries. The NVIDIA kernel driver is
# supplied by the host through NVIDIA Container Toolkit.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG COMFY_REVISION=1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf-home \
    MINIMAX_H3_COMFY_ROOT=/opt/comfyui \
    MINIMAX_H3_COMFY_PYTHON=/opt/venv/bin/python \
    MINIMAX_H3_LORA_DIR=/data/models/loras \
    MINIMAX_H3_LOCAL_DIR=/data/models/MiniMax-H3 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        gnupg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsndfile1 \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3.11 -m venv /opt/venv

WORKDIR /opt/app

COPY requirements.txt pyproject.toml README.md ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
       torch==2.11.0 torchvision torchaudio \
       --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install -r requirements.txt

RUN git clone --filter=blob:none https://github.com/Comfy-Org/ComfyUI.git /opt/comfyui \
    && git -C /opt/comfyui fetch --depth 1 origin "${COMFY_REVISION}" \
    && git -C /opt/comfyui checkout --detach "${COMFY_REVISION}" \
    && python -m pip install -r /opt/comfyui/requirements.txt \
    && rm -rf /opt/comfyui/.git

COPY . .
RUN git clone https://github.com/mav-rik/facerestore_cf.git /opt/comfyui/custom_nodes/facerestore_cf \
    && git -C /opt/comfyui/custom_nodes/facerestore_cf checkout --detach ff4d7a5c102441d8f058dd6135797ffb57b6c6ad \
    && python -m pip install -r /opt/comfyui/custom_nodes/facerestore_cf/requirements.txt \
    && rm -rf /opt/comfyui/custom_nodes/facerestore_cf/.git \
    && cp -r /opt/app/comfy_nodes/minimax_h3_nodes /opt/comfyui/custom_nodes/minimax_h3_nodes
RUN python -m pip install --no-deps . \
    && chmod +x docker/entrypoint.sh docker/healthcheck.py \
    && mkdir -p /data/models/loras /data/models/MiniMax-H3 /data/outputs \
       /data/uploads /data/hf-home /opt/comfyui/models

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "/opt/app/docker/healthcheck.py"]

ENTRYPOINT ["/opt/app/docker/entrypoint.sh"]
CMD ["serve"]
