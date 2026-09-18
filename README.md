# MiniMax-H3 FL2VA local studio

Local **first / last frame → video + stereo audio** inference for
**[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)**. The default
profile targets an **RTX 4070 Ti 12 GB + 64 GB RAM**; the original H100
Diffusers path remains available.

This is **not a chat LLM**. MiniMax-H3 is an open video+audio generator
(T2VA / I2VA / FL2VA). Text is a **prompt**, not a conversation. The
Qwen3-VL-32B encoder is used only to condition the DiT.

| Item | Value |
| --- | --- |
| Base model | `MiniMaxAI/MiniMax-H3` (Apache-2.0) |
| Default workflow | `fl2va` (loads `transformer/`, **not** `transformer_ref/`) |
| Default turbo LoRA | TaoMate FL2VA 3-step EMA, compact avg-rank-19 BF16 |
| UI | Gradio 6 (local, no share, no analytics) |
| Weights | SafeTensors + PEFT LoRA |
| Python | **3.11 venv** on Windows 11 or Ubuntu 24.04 |
| Default GPU | RTX 4070 Ti 12 GB, CUDA 12.8, ComfyUI DynamicVRAM |

Official sources:

- Model card: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Diffusers API: [MiniMax-H3 pipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3)
- Turbo LoRAs: [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
- Community LoRA: [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
- Default TaoMate release: [Civitai model version 3322352](https://civitai.red/models/2837571?modelVersionId=3322352)
- Release history: [CHANGELOG.md](CHANGELOG.md)

## Default TaoMate model profile

The 12 GB profile does not run the full Diffusers checkpoint directly. It combines
the pruned/quantized MiniMax-H3 FL2VA backend with the compact TaoMate Turbo LoRA:

| Component | Artifact | Purpose |
| --- | --- | --- |
| FL2VA transformer | `minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | ~21 GB FP8-scaled pruned video/audio transformer |
| Text encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | ~15.7 GB NVFP4 Qwen3-VL-32B conditioning encoder |
| Turbo LoRA | `minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors` | Compact ~173 MB TaoMate FL2VA acceleration adapter |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | Video latent decode |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | Stereo audio decode |

TaoMate catalog settings:

| Setting | Default | Notes |
| --- | --- | --- |
| Catalog ID | `taomate_fl2va_3step_ema` | Default entry in `configs/default.yaml` |
| Backend | `comfy_pruned` | Uses the isolated loopback ComfyUI worker |
| NFE | `4` | Upstream guidance is 4–6 steps |
| Sampler / scheduler | Euler / simple | No CFG |
| LoRA strength | `0.75` | Recommended starting strength |
| Canvas | `0.4` MP / 864×480 at 16:9 | Chosen for 12 GB VRAM |
| Video/audio shift | `12.0` / `3.0` | Catalog defaults |
| SHA-256 | `DE9663D974A884B477556748239C6F28239F7CA1825BE270F98F023FF5DAB6A7` | Verified after download |

Despite the upstream “3-step EMA” name, this application defaults to **4 NFE**
because the published recommendation is 4–6 steps and the extra step is more
stable for the low-memory workflow. The selected 173 MB avg-rank-19 BF16 file is
different from the 2.31 GB full-FP32 and 1.21 GB rank-128 variants on the same
Civitai release.

The TaoMate adapter is intended for the **pruned FL2VA architecture**. Do not load
it through the full Diffusers transformer or stack it with Ref2VA adapters. The
application validates this boundary and routes the catalog entry automatically.

Install only the default TaoMate LoRA:

```powershell
.\.venv\Scripts\python.exe -m minimax_h3_fl2v.download `
  --loras --lora-id taomate_fl2va_3step_ema
```

See [docs/LORA.md](docs/LORA.md#taomate-fl2va-3-step-ema-default-12-gb-profile)
for compatibility, file provenance, and tuning details.

## What FL2VA does

| Mode | Inputs | Output |
| --- | --- | --- |
| **T2VA** | text prompt | ~5–15 s video + audio |
| **I2VA** | prompt + first frame | video starts from that image |
| **FL2VA** | prompt + first + last frame | video interpolates between the two stills |

Constraints from the official model card:

- 24 FPS, duration **5–15 s**
- frame count `17n + 5` (shortest clip = **124 frames ≈ 5.17 s**)
- native **768p**, 16:9 / 9:16
- native audio **48 kHz stereo**
- **no CFG**
- text encoder is **Qwen3-VL-32B**

## Why CPU offload on one H100

A single 80 GB H100 **cannot** hold full bf16 MiniMax-H3 + Qwen3-VL-32B.
offload with a **12 GB** reserve. This project uses that recipe.
The official single-GPU recipe is Diffusers `ComponentsManager` auto CPU
offload. This project uses a more conservative **20 GB** reserve and explicitly
evicts managed components between requests to leave room for VAE decode buffers.

Optional faster path (not the default): load
`lightx2v/MiniMax-H3-int8c` (INT8 transformer, ~33 GB) and skip CPU
offload. See [docs/AZURE_H100.md](docs/AZURE_H100.md).

## Quick start (Azure Linux + H100)

```bash
cd /mnt/disk2TB/minimaxH3
bash scripts/azure_h100_setup.sh
bash scripts/setup_venv.sh
source .venv/bin/activate
cp .env.example .env          # HF_TOKEN is only for this download step
python scripts/download_models.py --all   # last online step
python scripts/smoke_test.py
python app.py                 # Gradio stays local → http://<vm-ip>:7860
```

CLI:

```bash
python generate.py \
  --prompt "$(cat examples/fl2va_prompt.txt)" \
  --first-image /path/to/first.png \
  --last-image /path/to/last.png \
  --lora-id fl2va_turbo_8step_768p \
  --duration 5 \
  --aspect-ratio 16:9 \
  --nfe 8 \
  --seed 42
```

MP4 files land in `outputs/`.

## Windows 11 + RTX 4070 Ti setup (default)

Requirements:

- Windows 11 with an up-to-date NVIDIA Studio or Game Ready driver.
- Python 3.11 registered with the Windows Python launcher (`py -3.11`).
- Git for Windows and FFmpeg available on `PATH`.
- At least 64 GB RAM, substantial free NVMe space, and a system-managed page file
  (or a manually configured page file of at least 16 GB).

The setup script creates or repairs `.venv`, installs the CUDA 12.8 PyTorch
packages, installs the pinned ComfyUI worker, downloads the FP8/NVFP4 model
files, and downloads the TaoMate LoRA:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\scripts\setup_windows.ps1
```

The model download is approximately 43 GB plus caches. After setup, start the
studio with:

```powershell
.\scripts\run_windows.ps1
```

Open `http://localhost:7860`. To choose another port:

```powershell
.\scripts\run_windows.ps1 -Port 7861
```

The private ComfyUI worker defaults to `127.0.0.1:18188` to avoid VS Code and
standalone ComfyUI processes that commonly occupy port 8188. Override both ports
when needed:

```powershell
.\scripts\run_windows.ps1 -Port 7868 -ComfyPort 19188
```

If the environment already contains all Python packages and only the model
downloads are required, run:

```powershell
.\.venv\Scripts\python.exe scripts\setup_pruned_backend.py
.\.venv\Scripts\python.exe -m minimax_h3_fl2v.download --loras --lora-id taomate_fl2va_3step_ema
```

Do not run the setup with an activated Python 3.14 environment. The script
explicitly locates Python 3.11 and both application and ComfyUI environments use
Windows `Scripts\python.exe` paths.

The default TaoMate catalog entry uses the pruned ComfyUI model files under
`.runtime\ComfyUI\models`; it does **not** require `models\MiniMax-H3`. The
large full Diffusers snapshot is needed only after selecting a non-pruned
catalog entry. Download that optional backend with:

```powershell
.\scripts\setup_windows.ps1 -SkipComfyInstall -IncludeFullDiffusersBase
```

The **Load model** button checks the backend selected in the LoRA dropdown. With
TaoMate selected it starts/validates the local ComfyUI worker instead of trying
to load `models\MiniMax-H3`.

## Ubuntu + RTX 4070 Ti 12 GB setup

Use Ubuntu with an NVIDIA driver compatible with CUDA 12.8, at least 64 GB of
RAM, substantial free NVMe space, and a swap file. The FP8 transformer (~21 GB),
NVFP4 text encoder (~15.7 GB), VAEs, and runtime buffers can approach the
available host memory during model transitions.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/setup_pruned_backend.py
python -m minimax_h3_fl2v.download --loras --lora-id taomate_fl2va_3step_ema
python app.py
```

The default render is 864×480, 124 frames (~5.17 seconds), 4 NFE,
Euler/simple, and LoRA strength 0.75. Start there. 608×352 (`0.2` MP) is the
fallback for OOMs; 960×544 (`0.5` MP) may work but has substantially larger
attention and VAE peaks. Generation will be much slower than on an H100 because
model blocks are streamed between RAM and 12 GB VRAM.

The Civitai catalog entry downloads the 173 MB avg-rank-19 file and verifies
SHA-256 `DE9663D974A884B477556748239C6F28239F7CA1825BE270F98F023FF5DAB6A7`.
Do not use the 2.31 GB FP32 variant on this hardware unless specifically needed.

### Optional upscale and detail restoration

Enable **Upscale/detail after generation** under Advanced settings to process
decoded frames with `RealESRGAN_x4plus.safetensors`. ComfyUI applies the model
in tiles and automatically reduces tile size after a CUDA OOM, making it suitable
for the 4070 Ti. The native 4× result is resized to the selected **1.5×** or
**2×** output factor, then an optional conservative sharpening pass is applied.
Audio is carried into the enhanced MP4 unchanged.

Recommended 12 GB settings:

| Input | Output | Detail strength |
| --- | --- | --- |
| 608×352 | 1216×704 (2×) | 0.20–0.35 |
| 864×480 | 1296×720 (1.5×) | 0.20–0.30 |
| 864×480 | 1728×960 (2×) | 0.15–0.25; slower |

The Windows setup script downloads and verifies the 66.9 MB BSD-licensed
Real-ESRGAN model. Existing installations can add only the missing model with:

```powershell
.\.venv\Scripts\python.exe scripts\setup_pruned_backend.py --skip-install
```

CLI example:

```powershell
.\.venv\Scripts\python.exe generate.py --prompt "A cinematic landscape" --upscale --upscale-factor 1.5 --detailer-strength 0.25
```

### Frame-rate interpolation and face restoration

The Advanced settings also provide:

- **23.976 fps**: cinema-rate output without interpolation.
- **29.97 fps**: RIFE 2× generation followed by duration-preserving resampling.
- **60 fps**: RIFE 3× generation followed by resampling.
- **120 fps**: RIFE 5× interpolation. This is the slowest and most memory-heavy option.
- **CodeFormer face restoration** using the lightweight MobileNet RetinaFace detector.

RIFE changes temporal smoothness, not clip duration. The lightweight
`rife_v4.25_lite.safetensors` model uses ComfyUI's native low-VRAM loading.
CodeFormer processes detected faces at 512×512 and composites them back into each
frame. Start with fidelity `0.7`; higher values preserve identity, while lower
values perform stronger reconstruction. Face restoration can flicker when faces
are very small, occluded, or shown in profile because it operates per frame.

For a 4070 Ti, enable features incrementally: face restoration first, then 1.5×
upscaling, then 29.97/60 fps. Combining 2× upscale, face restoration, and 120 fps
on a five-second clip produces hundreds of high-resolution frames and can be very
slow despite tiled/offloaded processing.

CLI example:

```powershell
.\.venv\Scripts\python.exe generate.py `
  --prompt "A close cinematic portrait" `
  --face-restore --face-fidelity 0.7 `
  --fps 60
```

## Docker Compose on Ubuntu 24.04 + NVIDIA

The image uses Ubuntu 24.04 and CUDA 12.8 user-space libraries. **The NVIDIA
kernel driver is not installed in the container.** Install the current proprietary
NVIDIA driver for the RTX 4070 Ti and NVIDIA Container Toolkit on the Ubuntu
24.04 host. Docker passes the host GPU into the container.

Host prerequisites:

```bash
# Install a recommended NVIDIA driver, reboot, and verify it first.
sudo ubuntu-drivers install
sudo reboot
nvidia-smi

# Install Docker Engine from Docker's official Ubuntu repository, then install
# NVIDIA Container Toolkit from NVIDIA's official repository.
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# This must print the RTX 4070 Ti from inside a CUDA container.
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

For exact repository installation commands, use the current official Docker
Engine and NVIDIA Container Toolkit documentation; repository signing keys and
package URLs can change.

Prepare and start the application:

```bash
cp .env.compose.example .env
docker compose build

# One-time online download into persistent Docker volumes (~43 GB plus caches).
docker compose --profile setup run --rm setup

# Runtime is offline and requests one NVIDIA GPU.
docker compose up -d studio
docker compose ps
docker compose logs -f studio
```

Open `http://localhost:7860`. Set `MINIMAX_H3_PORT` in `.env` to publish a
different host port. The named volumes preserve ComfyUI model weights, the
TaoMate LoRA, generated videos, uploads, and Hugging Face cache across image
rebuilds. To stop without deleting models:

```bash
docker compose down
```

Do not use `docker compose down -v` unless all downloaded weights and generated
outputs may be deleted. The default container limits are 60 GB RAM and 76 GB
RAM+swap. Configure at least 16 GB of fast host swap on NVMe, close other GPU
applications, and reduce `MINIMAX_H3_COMFY_RESERVE_VRAM_GB` only if ComfyUI
cannot use enough of the 12 GB card.

Container artifacts:

- `Dockerfile`: Ubuntu 24.04, CUDA 12.8 runtime, Python 3.11, PyTorch, pinned ComfyUI.
- `compose.yaml`: setup profile, offline studio service, GPU reservation, volumes.
- `docker/entrypoint.sh`: GPU/model preflight, one-time setup, offline startup.
- `docker/healthcheck.py`: local Gradio health probe.
- `.env.compose.example`: ports and memory tuning.
- `.dockerignore`: excludes local models, environments, outputs, and credentials.

## Layout

```text
minimaxH3/
├── app.py / generate.py
├── configs/default.yaml          # RTX 4070 Ti low-memory runtime
├── configs/loras.yaml            # SafeTensors LoRA catalog
├── src/minimax_h3_fl2v/          # pipeline, LoRA, Gradio, CLI
├── scripts/setup_venv.sh         # Python 3.11 + torch cu128
├── scripts/setup_windows.ps1     # Windows 11 environment + model setup
├── scripts/run_windows.ps1       # Windows launcher
├── scripts/setup_pruned_backend.py # ComfyUI + quantized models + enhancements
├── CHANGELOG.md                   # Date-based release and feature history
├── docs/                         # Azure, LoRA, prompting, troubleshooting
└── models/                       # created on download (gitignored)
```

## Recommended H100 settings

| Setting | Value |
| --- | --- |
| LoRA | FL2VA Turbo 8-step v1.0 768p |
| NFE | 8 (scheduler grid = 9; the extra point is terminal sigma) |
| `lora_alpha` | 128 |
| `lora_scale` | 1.0 |
| video shift | 6 |
| audio shift | 3 |
| canvas | 1.0 MP 16:9 → **1376×768** |
| duration | 5 s (124 frames) |
| attention | `_flash_3`, then local SDPA (no Hub kernels) |
| offload | on, `memory_reserve_margin=20GB`; evict between requests |

4-step 768p is faster and slightly softer. Base model (no LoRA) wants **~50 NFE**.

## LoRA rules

- Only **Diffusers PEFT** files (`*.lora_A.default.weight` / `*.lora_B.default.weight`).
- **Do not** load ComfyUI-named files from the same Hub repo.
- Targets: `to_q`, `to_k`, `to_v`, `to_out.0`, `ff.net.0.proj`, `ff.net.2`.
- Gradio keeps adapters **unfused** so you can swap LoRAs without reloading 60 GB.
- Optional second `.safetensors` (style) stacks on the turbo adapter.
- UI uploads are copied once to `models/loras/` and remain available in the
  five **Extra LoRA** dropdowns. Use **Refresh** after copying a file there
  manually. Up to five local LoRAs can be stacked with independent strengths,
  in addition to the selected catalog/Turbo adapter. Duplicate files are rejected.
- **Random each generation** creates a fresh 63-bit seed for every run, displays
  it after generation, and copies it into the Seed field. Switch the mode to
  **Fixed** to reproduce a result with that seed.
- The Dasiwa Turbo Multistep catalog entries automatically use an isolated local
  ComfyUI worker with the matching pruned FL2VA base. Run
  `python scripts/setup_pruned_backend.py` once while online to install or repair
  that backend; normal catalog entries continue using Diffusers.
- Both backends report live model/LoRA loading, prompt and keyframe encoding,
  sampling step and percentage, elapsed time, VAE decoding, and MP4 encoding.
  The final summary records end-to-end generation time, realtime factor,
  frames/second, and a per-stage timing breakdown for parameter comparisons.
  outside the UI.

Details: [docs/LORA.md](docs/LORA.md).

## Reference geometry / no crop

When a first or last reference frame is present, its actual width:height ratio
always controls the output canvas. Named presets no longer override it. H3
requires a 32-pixel spatial grid, so width/height are rounded to that grid while
minimizing ratio error. Both keyframes are then **contained** on that canvas:
the full source remains visible, and any small mismatch is letterboxed or
pillarboxed instead of stretched or cropped.

## Prompting

H3-Context-IR likes a compact multimodal block. The UI checkbox wraps a
plain sentence into that format **without calling another model**.

See [docs/PROMPTING.md](docs/PROMPTING.md) and `examples/fl2va_prompt.txt`.

## Offline + unfiltered

Runtime (`app.py` / `generate.py`) is **air-gapped**:

- loads only `models/MiniMax-H3` and `models/loras`
- `local_files_only=True`, `HF_HUB_OFFLINE=1`
- no Gradio share tunnel, no Gradio analytics, no Google Fonts CDN
- no FlashAttention Hub kernel (`_flash_3_hub` is rejected)
- no safety checker, NSFW filter, watermark, or prompt blacklist

The **only** network path is `python scripts/download_models.py` while you
still have Hub access. After that, inference does not contact Hugging Face.

## Security / networking

Gradio binds **`0.0.0.0:7860`** (all NICs) with `share=False` — no Gradio
tunnel. From a browser on the IP allowed in NSG:

```text
http://<vm-public-ip>:7860
```

On this VM that is typically `http://xx.xx.xx.xx:7860`.

NSG must allow TCP **7860** from your client IP only (not `0.0.0.0/0`).
SSH tunnel still works if you prefer not to open the NSG:

```bash
ssh -L 7860:127.0.0.1:7860 azureuser@<vm-public-ip>
```

## License

This studio code is Apache-2.0. MiniMax-H3 weights are Apache-2.0.
Turbo LoRAs keep their upstream licenses (LightX2V / community).
