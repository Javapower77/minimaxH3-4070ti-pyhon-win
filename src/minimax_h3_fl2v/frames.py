"""MiniMax-H3 video VAE frame-count rules.

The visual VAE accepts frame counts of the form ``17 * n + 5`` at 24 FPS.
The shortest valid clip is 124 frames (~5.17 s). Validated range is 5–15 s.
"""

from __future__ import annotations

FPS = 24
VAE_TEMPORAL_STRIDE = 17
VAE_TEMPORAL_OFFSET = 5
MIN_FRAMES = 124  # 17 * 7 + 5
MAX_FRAMES = 362  # ~15.08 s
MIN_DURATION = 5.0
MAX_DURATION = 15.0


def align_num_frames(num_frames: int) -> int:
    """Snap a raw frame count onto the MiniMax-H3 VAE grid."""
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    aligned = num_frames + (VAE_TEMPORAL_OFFSET - (num_frames % VAE_TEMPORAL_STRIDE)) % VAE_TEMPORAL_STRIDE
    return max(MIN_FRAMES, min(MAX_FRAMES, aligned))


def duration_to_frames(duration_seconds: float, fps: int = FPS) -> int:
    """Convert a duration in seconds to a valid MiniMax-H3 frame count."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    raw = max(VAE_TEMPORAL_OFFSET, round(float(duration_seconds) * fps))
    return align_num_frames(raw)


def frames_to_duration(num_frames: int, fps: int = FPS) -> float:
    return num_frames / float(fps)
