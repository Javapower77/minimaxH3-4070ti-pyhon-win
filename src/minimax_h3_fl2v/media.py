"""Image loaders and MP4 mux (video frames + stereo audio)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


def load_rgb_image(path: Path) -> Image.Image:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def fit_image_without_crop(
    image: Image.Image,
    width: int,
    height: int,
    *,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Contain an image on a canvas without cropping or changing its ratio.

    MiniMax-H3's stock FL2VA step stretches the first keyframe and cover-crops
    the last keyframe. Passing an already canvas-sized image bypasses both
    branches. Any unavoidable difference caused by the 32-pixel model grid is
    represented as centered letterbox/pillarbox pixels instead of lost content.
    """
    if width < 1 or height < 1:
        raise ValueError("target dimensions must be positive")
    source = image.convert("RGB")
    scale = min(width / source.width, height / source.height)
    resized_width = min(width, max(1, round(source.width * scale)))
    resized_height = min(height, max(1, round(source.height * scale)))
    resized = source.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), fill)
    canvas.paste(resized, ((width - resized_width) // 2, (height - resized_height) // 2))
    return canvas


def save_result_video(
    result: dict,
    output_path: Path,
    fps: int = 24,
) -> Path:
    """Encode the first generated video and its audio into one MP4 file."""
    from diffusers.utils.export_utils import encode_video

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio = None
    audio_sample_rate: Optional[int] = None
    if result.get("audio") is not None:
        audio = result["audio"][0]
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio)
        audio = audio.detach()
        audio_sample_rate = int(result["sampling_rate"])

    frames = result["videos"][0]
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu()
    elif isinstance(frames, np.ndarray):
        pass

    encode_video(
        frames,
        fps=fps,
        output_path=str(output_path),
        audio=audio,
        audio_sample_rate=audio_sample_rate,
    )
    return output_path
