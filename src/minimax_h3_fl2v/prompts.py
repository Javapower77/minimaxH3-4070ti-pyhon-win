"""Prompt helpers for MiniMax-H3 FL2VA.

H3-Context-IR prompts work best as a compact multimodal block:

* ``integrated_multimodal_description`` — camera, subject, motion, lighting
* ``overall_soundscape`` — diegetic audio
* ``non_diegetic_music`` — score / absence of score

The expander only wraps user text when those headings are missing. It does not
call another LLM (keeps local inference as the only token consumer).
"""

from __future__ import annotations

STRUCTURED_MARKERS = (
    "integrated_multimodal_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)

DEFAULT_SOUNDSCAPE = (
    "Natural diegetic audio matching the scene: room tone, footsteps, clothing rustle, "
    "and environment-appropriate ambience. Speech is clear when a speaker is visible."
)
DEFAULT_MUSIC = "No non-diegetic music unless the prompt explicitly requests a score."


def expand_prompt(prompt: str, *, structured: bool = True) -> str:
    text = (prompt or "").strip()
    if not text:
        raise ValueError("prompt is empty")
    if not structured:
        return text
    lowered = text.lower()
    if any(marker in lowered for marker in STRUCTURED_MARKERS):
        return text
    return (
        f"integrated_multimodal_description: {text}\n\n"
        f"overall_soundscape: {DEFAULT_SOUNDSCAPE}\n\n"
        f"non_diegetic_music: {DEFAULT_MUSIC}"
    )


def describe_mode(first_image, last_image) -> str:
    if first_image is not None and last_image is not None:
        return "FL2VA (first + last frame)"
    if first_image is not None:
        return "I2VA (first frame)"
    if last_image is not None:
        return "I2VA (last frame)"
    return "T2VA (text only)"
