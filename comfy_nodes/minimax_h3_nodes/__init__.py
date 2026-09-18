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


NODE_CLASS_MAPPINGS = {"MiniMaxH3ResampleFramesToFPS": ResampleFramesToFPS}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3ResampleFramesToFPS": "Resample Frames to FPS"}
