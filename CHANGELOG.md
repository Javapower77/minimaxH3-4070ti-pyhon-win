# Changelog

## Restoration replacement

- Replaced the LMS application section with Nugus Restore / Enhance / Improve
  rank16 (Civitai version 3389308, file 3278511), using native Ref2VA video and
  optional soundtrack references, not guide anchoring. Downloads are opt-in
  `restore` / `restore_base`; no turbo is applied by default. Normalization and
  original-audio preservation remain independent of FL2VA generation.
- Metadata and byte count verified; anonymous header access returned 401.
  Pruned compatibility requires local checksum and all-key shape validation;
  no GPU render is claimed. Earlier LMS entries below are historical.

All notable changes to this project are documented here. The project currently
uses date-based development entries because no tagged release series exists yet.

## [Unreleased]

### Unreleased additions

- Experimental normal-generation `dmad_full_dareties_v4_step600` catalog choice
  and `--type dmad_dareties` alias, pinned Civitai 2837571/3391964 file 3281488.
  Local 3,913,285,640-byte FP32 file matches SHA-256
  `ca34641d06c4e714d27a8540fd20721ef5daf54e9d58fa8be169d33b2d2699a9`;
  all 208 native pairs match the verified pruned FL2VA base (ranks128/384,
  AdaLN dropped, no direct diffs). Root-relative catalog `local_path` is now
  parsed and honored by downloads; resident worker files/hard links are never
  unlinked by synchronization, while live discovery remains checked.
  App uses 8 steps, strength1.0, shifts12/3, Euler/simple, 0.4 MP; creator
  recommends 6–8 steps, er_sde/lcm/simple. No GPU run or model download;
  TaoMate/original DMAD defaults and character swap/restoration unchanged.
- Independent native image+video **Character swap** section with editable target
  prompt, 20-step requested baseline (not prescribed by the author card), strength
  1.0, `res_multistep`/`simple`, turbo off by default and no restoration LoRA. Preserves the
  source canvas without resizing controls or a forced 0.4 MP limit; warns about
  memory exhaustion and unproven face-only, temporal and lip-sync fidelity.
- Character-only normalization/output-FPS controls removed: source must be exact
  24 fps CFR, 32-aligned, `17n+5` frames and zero-start, with no normalization or
  resampling. Persistent image/video originals and verified-remux safeguards for
  all original audio tracks remain. Restoration features and docs are unchanged.
- Default-off Ref2VA turbo checkbox forwards `use_turbo: bool = False`; effective
  steps are 8 when enabled, otherwise requested steps. Toggle selects 8 disabled
  steps or resets to 20 editable steps. Turbo uses Euler/simple, shifts 12/3 and
  fixed strength 1.0 before the character LoRA, whose strength stays independently
  editable at default 1.0.
- Optional worker-LoRA-only `--type character_swap`, excluded from aggregate
  download groups; reuses the ~21 GB `restore_base` and encoder/VAEs. Pins
  `akatz-ai/MiniMax-H3-Character-Swap-LoRA` revision
  `62407e0cc8089c363abd9ce4b0b27662abb237af`, file
  `h3_character_swap_pro4500_1000.safetensors` (155,110,320 bytes), SHA-256
  `4b2a3f420ae804c0aa3422761ff84dbd1bf52eef6900ffab6d2e66df63cb4e79`.
  Actual local 416 tensor keys/shapes pass against pruned Ref2VA FP8; pinned base
  and character hashes match. GPU rendering is untested. No downloads needed
  for this documentation work.
- Optional `--type ref2va_turbo` pins Kijai/MiniMax-H3-experimental revision
  `d8023be02fefbb3633b0cd335c3879f91177299d`, file
  `MiniMax-H3-Ref2VA-Acc-8Step_pruned_comfy.safetensors` (1,725,921,392 bytes),
  SHA-256 `6f18e1c2eccb14b37322607730f26b16bf1169b56cd098ea006cffaec43d1e39`.
  Actual header: 578 tensors, AdaLN across 50 blocks, curve-compatible input 8 /
  output 96,768, not full-width. Actual local checksum, every target, and ordered
  turbo-first stack pass preflight with matching 32× PDD weight/bias banks.
  Malformed reshapes, incompatible later patches, and full-width AdaLN fail closed;
  no GPU render or 12 GB fit is validated.
- `docs/CHARACTER_SWAP.md` documents the author's pruned INT8 training base,
  native-worker boundaries versus fused/GGUF/custom-pack workflows, and upstream
  Community License territorial limits. `workflows/CharacterSwap.json` and
  existing changes remain untouched; existing `.gitignore` already excludes
  downloaded models and uploads.
- LMS-only optional input normalization (default off): timestamp-based arbitrary
  FPS/VFR to 24 FPS, cloned tail padding to the next `17n+5`, enhanced-tail trim,
  native/source output rate and reported duration/temporal-loss adjustments.
  Original source/audio timelines preserved; payload-only audio digest checks
  tolerate differing streamhash timebase headers and CFR validation tolerates
  container timestamp quantization. Offline mocks and tiny ffmpeg tests added.
- Separate LMS / Sharpness section for generated or uploaded 24 fps video,
  using an aligned full-source guide, optional pinned pruned Ref2VA FP8 base,
  LMS r64 and Ref2VA eight-step turbo. Independent strength/steps/seed controls,
  retained original source and lossless original-audio remux to MKV. Added
  optional `lms`/`lms_base` downloads and worker-profile/idle-queue checks.
  FL2VA defaults remain unchanged; 12 GB real-render feasibility is unverified.
- Optional Alissonerdx learned H3 latent upscale stage on the currently generated
  video before tiled VAE decode; independent 1.5×/2× and CPU/CUDA controls,
  preserved audio, aligned output metadata, CLI flags and optional
  `--type latent_upscaler` checkpoint/pinned MIT node downloads. Default off.
  LMS Ref2VA sharpening is excluded per FL2VA-only scope; no second diffusion
  refinement is implied. GPU quality and 12 GB peak-memory validation pending.
- Dasiwa PDMD + DMAD 4-step r256 Fro1 as a separate experimental native pruned
  catalog choice (`dasiwa_pdmd_dmad_4step_r256`, Civitai 3385247 file 3274094)
  with `--type pdmd_dmad`, SHA-256 verification, 4 NFE, shifts 12/3 and 0.4 MP.
  Civitai downloads accept a process-environment `CIVITAI_API_TOKEN` via HTTPS
  Bearer authentication without placing credentials in URLs or the catalog.
- Separate Dasiwa DMAD + Hyperflow 4-step r256 experimental catalog option
  (`dasiwa_dmad_hyperflow_4step_r256`, Civitai 3383490 file 3272202) and
  `--type dmad_hyperflow` download with SHA-256 verification. Uses the native
  pruned backend, Euler/simple, 4 NFE, shifts 12/3 and 0.4 MP; leaves original
  DMAD and TaoMate defaults unchanged. Documents Civitai authentication and
  unverified local tensor/render compatibility.
- Original DMAD 4-step `lora_critic` as a separate experimental FL2VA catalog
  option and `--type dmad` download, preserving TaoMate as default. Includes
  lossless cached Diffusers-to-native conversion, independent QKV rank fusion,
  FFN gate reordering, and coherent video/audio shifts 12/2. Uses Euler/simple,
  not the paper's re-noise sampler; does not substitute Civitai's `full_critic`.
- Typed model downloader `scripts/download_models.py --type …` covering pruned
  ComfyUI weights, post-process assets, catalog LoRAs (TaoMate, Silveroxides,
  Dasiwa v1 ranks, Dasiwa turbo-multistep-v2, LightX2V), Diffusers FL2VA, and
  the `12gb` / `h100` profiles.
- Catalogued Dasiwa FL2VA turbo-multistep-v2 Hyperflow+EMA600 pruned r128
  (`dasiwa_multistep_v2_r128_pruned`, Civitai 3357658 file 3245274, SHA-256
  `C27839218F19CF444F0D2EAFCA38C1B64A48FC30CE02AEEF94CA5804CEA6ECB7`). Download
  with `--type dasiwa_v2`. Default remaining 8 NFE / 1.0 MP on the isolated
  pruned ComfyUI backend. FL2VA only.
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
