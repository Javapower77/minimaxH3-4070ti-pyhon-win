# Optional learned H3 latent upscale

The experimental FP16 [h3_upscaler_lms_v0.1](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/docs/latent-upscaler.md)
was fine-tuned for 2,000 steps on a sharpness dataset. It is a standalone neural
upscaler, **not the restoration LoRA or a second diffusion pass**.

## Setup and selection

Download model type `latent_upscaler` with `scripts/download_models.py`, then
restart the local ComfyUI worker/application. Alternatively, add
`--latent-upscaler` when running `scripts/setup_pruned_backend.py`.
Optional assets are not included in the default `12gb` download.

The downloader installs:

- Worker `models/latent_upscale_models/h3_upscaler_lms_v0.1.safetensors`
  (690,593,048 bytes, about 659 MiB).
- SHA-256 `40d4228227146245ef2b65480b647c50222465ca125de5880f192e3259e40d59`.
- Pinned [LBH-123-AI 3D inference code](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)
  at revision `40316cf008b2fd8663263270669eb4da23f89d2c` and its MIT license.
  Existing ComfyUI dependencies (`torch`, `einops`, `safetensors`) suffice.

In Advanced options select **Learned latent upscale (experimental)**, choose
1.5× or 2×, and CPU or CUDA. CLI flags: `--latent-upscale`,
`--latent-upscale-factor`, `--latent-upscale-device`.

## Processing order

FL2VA generation → learned video-latent upscale → tiled video VAE decode
→ optional CodeFormer → optional pixel Real-ESRGAN/sharpen → optional RIFE → MP4.

Generated audio is unchanged. Frame count and timing are preserved; dimensions
are aligned to 32 pixels and reported accordingly. Pixel and latent upscale
options are independent and their factors compound. Only `comfy_pruned` catalog
entries support this stage; Diffusers requests fail explicitly.

## Memory and quality

Default: off, CPU/1.5×. CPU inference uses FP32 compute with the FP16 model
export and can be slow and RAM-intensive. CUDA uses FP16, temporal chunking
and model offload afterward. 2× spatial scaling quadruples latent area; tiled
VAE decode helps but does not guarantee freedom from out-of-memory errors.
No high-resolution diffusion refinement is added. Malformed text or missing
detail cannot reliably be reconstructed. Real-model quality and peak memory
on 12 GB remain unverified.

Validation: the downloaded FP16 file passed SHA-256 verification and strict
architecture loading (345,280,216 parameters). A tiny real-model CPU inference
through the joint-AV bridge preserved video time/channels and returned finite
2× output. This is not a full-video quality, CUDA, or peak-memory benchmark.

## Separate restoration enhancement

The opt-in **Restore / Enhance / Improve — Nugus rank16** section uses a
separate Ref2VA second pass, not the FL2VA adapter stack or this latent upscaler.
Native `MiniMaxH3ReferenceToVideo` takes a same-aspect-ratio video reference and
source audio when present, not a guide. All original audio is copied to the
final MKV, never replaced by generated audio. No turbo is enabled by default;
strength 1.0 is the creator setting and 30 steps is an unverified local default.
Optional downloads, normalization, the runtime all-key shape gate, unknown
adapter header compatibility and unvalidated GPU feasibility are documented in
[RESTORATION.md](RESTORATION.md). Normal FL2VA generation is unchanged.
