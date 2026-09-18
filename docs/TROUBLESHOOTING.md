# Troubleshooting

## `CUDA was requested but no CUDA device is available`

- `nvidia-smi` must show an H100.
- You are inside `.venv` and torch was installed with the **cu128** wheel,
  not the CPU wheel.
- Windows 11: re-run `.\scripts\setup_windows.ps1` in PowerShell.
- Ubuntu: re-run `bash scripts/setup_venv.sh`.

### Windows reports that `python3.11` cannot be found

Use the PowerShell setup script instead of invoking a Unix-style Python command:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\scripts\setup_windows.ps1
```

It resolves Python 3.11 through `py -3.11`, uses
`.venv\Scripts\python.exe`, and creates the ComfyUI environment with the correct
Windows directory layout. Verify installed interpreters with `py -0p`.

## Out of memory / `CUDA error: out of memory`

Single H100 80 GB needs CPU offload for full bf16 FL2VA.

- Confirm `cpu_offload: true` in `configs/default.yaml`.
- Use 8-step or 4-step turbo LoRA, not 50 NFE base, for first tests.
- Drop megapixels from `1.0` to `0.5`.
- Keep duration at 5 s.
- The default reserve is now `20GB`; raise it further if using a long or
  unusually shaped canvas.
- Do not load Ref2VA (`transformer_ref`).

The engine now offloads every managed component before and after each request,
clears CUDA caches, and catches OOM so another request can be attempted without
restarting. `app.py` also sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before importing Torch.

## `Local MiniMax-H3 snapshot not found`

Inference never falls back to Hugging Face. Download once while online:

```bash
python scripts/download_models.py --all
```

Confirm `models/MiniMax-H3/modular_model_index.json` (or `model_index.json`)
and the `transformer/`, `text_encoder/`, `vae/` folders exist.

## `Failed to load required components` / Hub `LocalEntryNotFoundError`

`modular_model_index.json` still names `MiniMaxAI/MiniMax-H3`. The engine
rewrites those specs to `models/MiniMax-H3` before `load_components()`. If
you still see Hub errors:

- Restart `python app.py` after pulling this pin-to-local fix.
- Confirm `models/MiniMax-H3/scheduler/scheduler_config.json` exists.
- Diffusers must be installed **from git main** (`requirements.txt`).
- `python -c "import diffusers; print(diffusers.__version__)"` should be a
  recent `0.36.dev` / main build that includes `MiniMaxH3Blocks`.
- Re-run `python scripts/download_models.py --base`.

## `LoRA checkpoint is not a Diffusers PEFT LoRA`

You pointed at a ComfyUI file. Use:

```
minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors
```

not `*_comfyui_*.safetensors`.

## `LoRA file missing`

```bash
python scripts/download_models.py --loras --lora-id fl2va_turbo_8step_768p
```

If Hub returns 404, the filename may have moved. Check
https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/main and edit
`configs/loras.yaml`.

## Flash Attention 3 failed

The engine logs a warning and falls back to SDPA. Generation still works,
slightly slower. To force fallback:

```
MINIMAX_H3_ATTENTION_BACKEND=
```

## Slow first generation

Pipeline load moves Qwen3-VL-32B and the transformer through CPU offload.
Wait for the Engine box to say `Ready`. Later runs reuse the process.

## Second generation fails with `requires_grad=True on inference tensor`

This was caused by Diffusers reactivating the same PEFT adapter in training
mode after the first request had run under `torch.inference_mode`. The engine
now activates each `BaseTunerLayer` with `inference_mode=True`, skips redundant
activation when the adapter/scale are unchanged, and generates under
`torch.no_grad` so later LoRA changes remain legal.

Restart `python app.py` once after updating: tensors held by an already-running
old process retain their inference-only state.

## Gradio not reachable

- App binds `0.0.0.0:7860`. Use an SSH tunnel instead of opening the NSG
  to the world.
- `sudo ss -lptn 'sport = :7860'`
- Check `journalctl -u minimax-h3-studio -e` if using systemd.

## Wrong aspect / cropped faces

Reference images override canvas presets and retain their actual ratio. The video
VAE still requires multiples of 32, so a few padding pixels may be visible. The
app contains both keyframes without cropping. If the first and last images have
different ratios, the first frame defines the canvas and the second is
letterboxed/pillarboxed to fit it completely.

## Audio missing / video silent

`encode_video` muxes `result["audio"]` at `result["sampling_rate"]`
(48 kHz). If ffmpeg is missing, install it (`scripts/azure_h100_setup.sh`
already does). Play the MP4 in VLC, not a player that drops the audio track.

## Python version

Only **3.11** is supported. 3.12+ may work later; this repo refuses other
versions in `setup_venv.sh` and `requires-python`.
