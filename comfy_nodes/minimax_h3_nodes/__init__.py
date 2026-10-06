"""Small offline post-processing nodes owned by the MiniMax-H3 application."""

from __future__ import annotations

import torch


class ResampleFramesToFPS:
    """Select evenly spaced frames while preserving the clip's end-to-end duration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}),
                "target_fps": ("FLOAT", {"default": 29.97, "min": 1.0, "max": 240.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resample"
    CATEGORY = "MiniMax-H3/video"

    def resample(self, images: torch.Tensor, source_fps: float, target_fps: float):
        frame_count = int(images.shape[0])
        if frame_count < 2 or target_fps <= source_fps:
            return (images,)
        output_count = max(2, round((frame_count - 1) * target_fps / source_fps) + 1)
        indices = torch.linspace(0, frame_count - 1, output_count, device=images.device)
        indices = indices.round().to(torch.long).clamp_(0, frame_count - 1)
        return (images.index_select(0, indices),)


class LearnedLatentUpscale:
    """Adapt H3's joint AV latent to the pinned upstream video-only upscaler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT",),
            "model_name": ("STRING", {"default": "h3_upscaler_lms_v0.1.safetensors"}),
            "scale": ("FLOAT", {"default": 1.5, "min": 1.5, "max": 2.0}),
            "device": (["cpu", "cuda"], {"default": "cpu"}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "MiniMax-H3/video"

    def upscale(self, latent, model_name, scale, device):
        from . import upstream_latent_3d as upstream

        samples = latent["samples"]
        video = samples.unbind()[0] if getattr(samples, "is_nested", False) else samples
        if video.ndim != 5 or video.shape[1] != 24:
            raise ValueError("Expected a five-dimensional, 24-channel H3 video latent")
        if scale not in (1.5, 2.0) or device not in ("cpu", "cuda"):
            raise ValueError("Supported latent upscale settings are 1.5/2.0 and cpu/cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA latent upscaling selected but CUDA is unavailable")
        precision = "fp32" if device == "cpu" else "fp16"
        dtype = torch.float32 if device == "cpu" else torch.float16
        dev = torch.device(device)
        height = round(video.shape[-2] * 16 * scale / 32) * 2
        width = round(video.shape[-1] * 16 * scale / 32) * 2
        model = upstream.load_model(model_name, dev, precision)
        mean, std = upstream._make_norm_tensors(dev, dtype)
        try:
            with torch.inference_mode():
                normalized = (video.to(device=dev, dtype=dtype) - mean) / std
                output = model(normalized, scale=float(scale),
                               target_size=(video.shape[2], height, width), enable_chunking=True)
                output = (output * std + mean).to(device="cpu", dtype=video.dtype)
        finally:
            # Pinned architecture is reused, but its node API varies with ComfyUI
            # versions. Own the AV bridge/inference lifecycle rather than relying
            # on its newer NodeOutput API or retaining the large model in VRAM.
            model.to("cpu")
            upstream.MODEL_CACHE.clear()
            if device == "cuda":
                torch.cuda.empty_cache()
        return ({"samples": output},)


NODE_CLASS_MAPPINGS = {"MiniMaxH3ResampleFramesToFPS": ResampleFramesToFPS,
                       "MiniMaxH3LearnedLatentUpscale": LearnedLatentUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3ResampleFramesToFPS": "Resample Frames to FPS"}
