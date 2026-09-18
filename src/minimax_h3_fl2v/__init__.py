"""Local MiniMax-H3 FL2VA studio (video + audio, LoRA, Gradio)."""

from .config import AppConfig, GenerationRequest, LoRASpec
from .frames import align_num_frames, duration_to_frames
from .pipeline import MiniMaxH3Engine

__all__ = [
    "AppConfig",
    "GenerationRequest",
    "LoRASpec",
    "MiniMaxH3Engine",
    "align_num_frames",
    "duration_to_frames",
]
__version__ = "1.0.0"
