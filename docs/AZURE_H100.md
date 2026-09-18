# Azure Linux H100 runbook

Target: **one NVIDIA H100 80 GB** VM, Ubuntu 22.04 or 24.04, this repo on
`/mnt/disk2TB`.

## 1. Pick a SKU

| SKU | GPUs | GPU RAM | Notes |
| --- | --- | --- | --- |
| `Standard_NC40ads_H100_v5` | 1× H100 | 80 GB | Best match for this project |
| `Standard_NC80ads_H100_v5` | 2× H100 | 2×80 GB | Not required; the app uses GPU 0 |
| `Standard_ND96isr_H100_v5` | 8× H100 | 8×80 GB | Overkill; use FSDP2 only if you load both transformer partitions |

NC_H100_v5 VMs are billed while allocated. Stop/deallocate when idle.

Region: pick a region that actually lists H100 quota
(`eastus2`, `southcentralus`, `westeurope`, `swedencentral`, …). Check:

```bash
az vm list-skus --location eastus2 --size Standard_NC40ads_H100_v5 --output table
az vm list-usage --location eastus2 --query "[?contains(localName, 'H100')]" -o table
```

## 2. Disk

The FL2VA Diffusers snapshot is large (transformer ~62 GB + Qwen3-VL-32B
text encoder ~65 GB + VAEs). Put **everything** on the 2 TB data disk:

| Path | Role |
| --- | --- |
| `/mnt/disk2TB/minimaxH3` | this repo |
| `/mnt/disk2TB/minimaxH3/.venv` | Python 3.11 venv |
| `/mnt/disk2TB/minimaxH3/.hf_cache` | Hugging Face cache |
| `/mnt/disk2TB/minimaxH3/models` | local snapshot + LoRAs |
| `/mnt/disk2TB/minimaxH3/outputs` | MP4 results |

OS disk (30–128 GB) is not enough.

```bash
export HF_HOME=/mnt/disk2TB/minimaxH3/.hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/disk2TB/minimaxH3/.hf_cache/hub
```

Those are already in `.env.example`.

## 3. NVIDIA driver

Use the Azure N-series GPU driver (not a random `apt` CUDA toolkit as
the only install):

```bash
# Ubuntu example — follow current Microsoft docs if the package name moved
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
nvidia-smi   # Driver 550+ , CUDA 12.4+
```

Microsoft doc: [N-series GPU driver setup](https://learn.microsoft.com/azure/virtual-machines/linux/n-series-driver-setup).

You do **not** need a full local CUDA toolkit. The PyTorch **cu128**
wheel bundles the user-mode CUDA libraries.

## 4. Host packages + venv

```bash
cd /mnt/disk2TB/minimaxH3
bash scripts/azure_h100_setup.sh
bash scripts/setup_venv.sh
source .venv/bin/activate
```

`setup_venv.sh` enforces **Python 3.11**, installs:

```
torch (cu128) → requirements.txt → pip install -e .
```

Confirm:

```text
torch 2.x+cu128   cuda True   device NVIDIA H100   bf16 True   capability 9.0
```

## 5. Download weights

```bash
cp .env.example .env
# optional: HF_TOKEN=hf_...
python scripts/download_models.py --all
```

The downloader **skips** `transformer_ref/` (~62 GB, Ref2VA only). This
studio is FL2VA. After this step, `app.py` / `generate.py` stay offline.

Expected local tree:

```
models/
├── MiniMax-H3/
│   ├── modular_model_index.json
│   ├── model_index.json
│   ├── transformer/
│   ├── text_encoder/
│   ├── tokenizer/ processor/
│   ├── vae/ audio_vae/
│   └── scheduler/ audio_scheduler/
└── loras/
    ├── minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors
    └── …
```

## 6. Memory recipe (single 80 GB)

Default `configs/default.yaml`:

- `workflow: fl2va` — do not load `transformer_ref`
- `dtype: bfloat16`
- `cpu_offload: true`
- `memory_reserve_margin: 20GB`
- Flash Attention 3 via local `_flash_3` (then SDPA). Hub kernels are disabled.

What lives on the H100 at peak:

1. Active transformer blocks (paged in by the offload manager)
2. Working latents for 124×1376×768
3. 20 GB target headroom for VAE decode and allocator spikes

If you OOM:

1. Drop to 544p (`0.5` MP, 960×544) or 4-step 768p LoRA
2. Keep duration at 5 s
3. Raise `memory_reserve_margin` above `20GB`
4. Or switch to `lightx2v/MiniMax-H3-int8c` and set `cpu_offload: false`

## 7. Launch

```bash
source .venv/bin/activate
python app.py
```

The process binds **`0.0.0.0:7860`**. Open the public URL from the client
IP allowed in NSG (startup prints it):

```text
http://<vm-public-ip>:7860
```

SSH tunnel (optional, if you do not open 7860):

```bash
ssh -L 7860:127.0.0.1:7860 azureuser@<public-ip>
# open http://127.0.0.1:7860
```

NSG (required for direct public-IP access):

```bash
az network nsg rule create \
  --resource-group <rg> \
  --nsg-name <nsg> \
  --name AllowGradioFromMe \
  --priority 1100 \
  --destination-port-ranges 7860 \
  --source-address-prefixes <YOUR_IP>/32 \
  --access Allow --protocol Tcp
```

Optional systemd unit:

```bash
sudo cp scripts/minimax-h3-studio.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now minimax-h3-studio
sudo journalctl -u minimax-h3-studio -f
```

Edit `User=` in the unit if your login is not `azureuser`.

## 8. First-run timing

| Step | Typical on H100 + data disk |
| --- | --- |
| Hub download (FL2VA only) | 20–60 min depending on region |
| Pipeline load (CPU offload) | 3–8 min |
| 8-NFE 768p 5 s clip | ~1–3 min after warmup |
| 50-NFE base model | several minutes |

The Gradio process warms the pipeline in a background thread. The first
Generate click waits if load is still running.

## 9. Cost hygiene

- Deallocate the VM when idle (`az vm deallocate`).
- Do not leave Gradio up overnight unless you need it.
- Keep outputs on the data disk; they are large MP4s.
- One generation at a time (the engine holds a GPU lock).
