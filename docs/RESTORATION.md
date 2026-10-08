# Restore / Enhance / Improve — Nugus rank16 (experimental)

An independent, opt-in Ref2VA second pass over an existing video, not an FL2VA
generation adapter, learned latent upscale, or automatic stage. FL2VA defaults
remain unchanged. No turbo adapter is enabled by default.

## Asset and prerequisites

Download types `restore` (restoration adapter only) and `restore_base` (separate
pruned Ref2VA FP8 transformer, 20,958,205,608 bytes) are opt-in, not default/all/12gb
assets. Reuse the installed encoder and video/audio VAEs. Local `ffmpeg`,
`ffprobe`, and the pinned native ComfyUI backend are required; no external
reference-node packs or inference-time downloads are needed.

- Creator: Nugus; **Restore / Enhance / Improve**, BF16, rank 16.
- [Model 2989135](https://civitai.red/models/2989135),
  [version 3389308 metadata](https://civitai.red/api/v1/model-versions/3389308),
  file **3278511**.
- Filename: `Restore_Enhance_Improve_rank16_v1_H3-lora.safetensors`.
- Size: **149,165,944 bytes**.
- SHA-256: `04068FCCC4F7C14C3F42D0C1DC1A408EB335FA9C31149D068174E702EAB47296`.

Local read-only preflight verified the downloaded adapter and pinned Ref2VA base
SHA-256 hashes. All **600 tensors** match: **200 rank-16 pairs**, with scalar
alpha **16** each, covering 50 each of `attn.out_proj`, `attn.qkv_proj`, `mlp.fc1`
and `mlp.fc2` (48 main blocks and two token-refiner blocks). There are **no AdaLN
patches** in this adapter. Its flattened Musubi `lora_unet_` names are supported
directly by installed ComfyUI; no conversion or alpha/rank adjustment is needed.

Runtime verifies hashes and **every adapter key and target shape** before
submission. Validation constructs aliases from the exact native checkpoint
vocabulary, preserves underscores within module names, and rejects ambiguous or
duplicate targets. Existing `diffusion_model.` patches remain supported. Full
AdaLN incompatible with the pruned curve basis is rejected explicitly, not
converted or silently loaded with unsupported keys dropped. Header compatibility
does not prove restoration quality or runtime memory feasibility.

## Input and controls

Choose a generated result or uploaded video. Native
`MiniMaxH3ReferenceToVideo` receives a **same-aspect-ratio video reference** and
source audio when present, not `MiniMaxH3AddGuide` guidance. No separate guide
image is required. The final audio always comes from the original, not generated
audio. Strength **1.0** is the creator setting; **30 steps** is an **unverified
local default**, not a creator recommendation. Strength, steps, seed, input
normalization and output-rate controls remain independent.

Strict mode is the default: exactly **24 fps CFR**, width/height multiples of
**32**, and **17n+5 frames** within 5..3600 (largest matching count: 3592).
The unchanged FL2VA 23.976 fps default is not strict-compatible. No automatic
resize, crop or orientation correction is performed.

## Optional normalization

Enable **Normalize input** only when temporal loss is acceptable. Arbitrary
FPS/VFR is resampled from decoded timestamps to native 24 fps **without changing
playback speed**, using a temporary lossless FFV1 video reference. The final
frame is duplicated to reach the next 17n+5 count; the source is never cut to
fit. Geometry and maximum-frame restrictions still apply. Rotated video, invalid
timestamps and negative-start video are rejected before GPU startup. Positive
video offsets are restored; negative AAC audio priming is checked independently.

After rendering, only the synthetic padded tail is trimmed. Original duration
is rounded **up to a whole output frame**, extending it by less than one frame.
**Native 24 fps** is the default. **Match source rate** emits CFR at the source
rate, or decoded frame-count/duration average for VFR; it does not restore VFR
timestamps. Strict mode uses native 24 fps for either choice.

Frames can be dropped or duplicated, losing motion detail. Matching source FPS
cannot recover discarded motion; no optical-flow interpolation is performed.
Progress and the final summary report input cadence, duration, native resampling,
padding and output timing. Temporary FFV1 files need disk space and are cleaned
on success or failure.

## Original audio and failure safety

Output is downloadable **MKV**: all original audio tracks are **stream-copied**,
never re-encoded, replaced by generated audio, or cut using `-shortest`. Audio
may extend beyond video. This preserves encoded audio streams, not a
byte-identical container. Silent inputs remain silent.

The source video start time is restored before muxing. Ordered audio payload
digest rows are compared, not streamhash headers/timebases. Normalized packet
timestamps are checked within MKV millisecond resolution, including negative
AAC priming. Unsafe timestamp preservation fails instead of silently shifting
audio. The original and its durable retained copy are never edited; retain them
until the final MKV is checked. Failure is not a successful preserving render.

## Runtime and memory limits

Use an idle pinned worker with DynamicVRAM, cache-none and at least 1 GB VRAM
reserve. Runtime checks reused worker arguments, directories and queue. Worker
unloading via `/free` is asynchronous, not a completion barrier.

**No GPU restoration render or 12 GB feasibility has been validated.** The
separate ~21 GB transformer and longer references increase memory requirements;
input validation bounds are not memory guarantees. Start with short, small
inputs only after the downloaded adapter passes the all-key shape gate.
