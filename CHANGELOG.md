# Changelog

All notable changes to this project are documented here. The project currently
uses date-based development entries because no tagged release series exists yet.

## [Unreleased]

### Unreleased additions

- Dedicated documentation for the default TaoMate FL2VA 3-step EMA avg-rank-19
  LoRA, including provenance, checksum, backend compatibility, and recommended
  RTX 4070 Ti settings.
- Silveroxides DARE-TIES pruned v1 LoRA in the catalog for the 12 GB ComfyUI
  backend, with SHA-256 `9AAB6353CE76F0A1C6A6FBBEAA1A4C60DECED365E328BC14DDB0B7B9F0849B48`.
- Pruned AdaLN validation now also checks direct `.diff` patches, not only
  LoRA A/down matrices.
- This changelog.

## [2026-09-17]

### 2026-09-17 additions

- Selectable output frame rates: 23.976, 29.97, 60, and 120 fps.
- Native ComfyUI RIFE 4.25 Lite frame interpolation with duration-preserving
  frame resampling.
- Optional CodeFormer face restoration with the lightweight MobileNet RetinaFace
  detector and configurable fidelity.
- Optional tiled Real-ESRGAN x4plus enhancement at 1.5× or 2× output size.
- Configurable detail/sharpen pass after upscaling.
- Automatic download and validation of RIFE, CodeFormer, face-detection,
  ParseNet, and Real-ESRGAN assets.
- App-owned ComfyUI frame-rate resampling node.

### 2026-09-17 changes

- Progress messages now show one stable stage label and one elapsed-time value.
- Generation progress renders in one Engine status area instead of being repeated
  over every output component.
- Gradio upload preview responses use chunked transfer on file routes to avoid
  intermittent Windows `Content-Length` races.
- Embedded ComfyUI defaults to private port 18188 to avoid common VS Code and
  standalone ComfyUI port conflicts on 8188.

### 2026-09-17 fixes

- Repeated `still working` fragments in long-running progress messages.
- Intermittent `h11.LocalProtocolError` while previewing uploaded images on
  Windows.
- Pruned-backend model loading incorrectly requiring the optional full Diffusers
  snapshot.
- Windows ComfyUI virtual-environment paths and subprocess startup behavior.

## [2026-09-15]

### 2026-09-15 additions

- Windows 11 setup and launcher scripts for Python 3.11.
- Dockerfile and Docker Compose environment based on Ubuntu 24.04 and CUDA 12.8.
- GPU, model, and health preflight checks for container startup.
- Persistent Compose volumes for models, LoRAs, outputs, uploads, and Hub cache.
- RTX 4070 Ti low-memory profile using the pruned FP8 FL2VA transformer and
  NVFP4 Qwen3-VL text encoder.
- TaoMate FL2VA 3-step EMA compact avg-rank-19 BF16 LoRA as the default catalog
  selection.
- Streamed Civitai download with SHA-256 validation.
- Cross-platform fallback from symbolic links to file copies on Windows.

### 2026-09-15 changes

- Default generation settings changed to 864×480, 4 NFE, Euler/simple, and
  TaoMate strength 0.75 for 12 GB VRAM.
- CLI and UI route pruned catalog entries directly to the isolated ComfyUI
  worker without loading the full Diffusers model.

### 2026-09-15 fixes

- H100-specific `--reserve-vram 20` behavior on 12 GB GPUs.
- Windows `python3.11` command assumptions.
- Hard-coded Linux `.venv/bin/python` paths.
- Worker startup collisions caused by fixed port 8188.

## [2026-09-14]

### 2026-09-14 additions

- Initial MiniMax-H3 FL2VA local studio.
- Gradio UI and CLI generation paths.
- Diffusers backend with CPU offload for H100-class hardware.
- LoRA catalog, validation, swapping, and stacking.
- First/last-frame geometry preservation without cropping.
- Offline runtime enforcement and local-only model loading.
- Pruned ComfyUI backend and Dasiwa Turbo Multistep catalog entries.
