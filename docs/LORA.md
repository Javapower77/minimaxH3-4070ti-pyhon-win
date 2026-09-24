# LoRA (SafeTensors / PEFT)

MiniMax-H3 turbo LoRAs are **PEFT adapters** stored as **SafeTensors**.
This studio loads them with Diffusers' official
`MiniMaxH3ModularPipeline.load_lora_weights` (same conversion path as
the Hugging Face MiniMax-H3 docs).

## File format

Accepted keys:

```text
<module>.lora_A.default.weight
<module>.lora_B.default.weight
```

Also rewritten automatically:

| Incoming key | Normalized to |
| --- | --- |
| `base_model.model.*.lora_A.weight` | `*.lora_A.default.weight` |
| `transformer.*.lora_A.weight` | `*.lora_A.default.weight` |
| `diffusion_model.*.lora_A.weight` | `*.lora_A.default.weight` |

Rejected:

- ComfyUI-named files (`*_comfyui_*.safetensors`, `lora_unet_*`, `lora_up` / `lora_down`)
- non-`.safetensors` pickles
- unpaired A/B matrices

Official MiniMax-H3 LoRAs are **mixed-rank** (64 on attention/FFN, 16 on AdaLN). Diffusers' `load_lora_weights` injects those adapters and honors a `__metadata__` `alpha` when present. Alpha-less files load at `alpha == rank`.

Typical target modules:

```text
to_q  to_k  to_v  to_out.0
ff.net.0.proj  ff.net.2
AdaLN / modulation projections (rank 16)
```

## Effective scale

```text
effective_scale = lora_scale * lora_alpha / rank
```

LightX2V 8-step 768p files often record `alpha` in SafeTensors metadata
(Diffusers honors that). The 544p mixed-AR files typically load at
`alpha == rank`. Catalog `lora_alpha` values in `configs/loras.yaml` are
kept as documentation for the UI; runtime scaling is `lora_scale` on the
adapter plus whatever alpha the official loader applied.

`lora_scale` is the UI strength slider (default `1.0`).

## Catalog

| id | File | NFE | alpha | canvas |
| --- | --- | --- | --- | --- |
| `fl2va_turbo_8step_768p` | `minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors` | 8 | 128 | 1.0 MP |
| `fl2va_turbo_4step_768p` | `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors` | 4 | 128 | 1.0 MP |
| `fl2va_turbo_8step` | `minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors` | 8 | 8 | 0.5 MP |
| `fl2va_turbo_4step_v01` | `minimax_h3_fl2v_turbo_4step_v0.1.safetensors` | 4 | 8 | 0.5 MP |
| `larryvrh_turbo_v4` | `minimax_h3_turbo_v4_step600_ema.safetensors` | 6 | 8 | 1.0 MP |
| `taomate_fl2va_3step_ema` | `minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors` | 4 | 19 | 0.4 MP |
| `silveroxides_dareties_pruned_v1` | `minimax_h3_fl2v_turbo_silver_dareties_comfy_pruned_v1.safetensors` | 8 | mixed/~64 | 0.5 MP |
| `dasiwa_multistep_r48_pruned` | `minimax_h3_fl2va_bf16_turbo_multistep_fro099_r48_pruned.safetensors` | 8 | dynamic/r48 | 1.0 MP |
| `dasiwa_multistep_r96_pruned` | `minimax_h3_fl2va_bf16_turbo_multistep_fro099_r96_pruned.safetensors` | 8 | dynamic/r96 | 1.0 MP |
| `dasiwa_multistep_r144_pruned` | `minimax_h3_fl2va_bf16_turbo_multistep_fro099_r144_pruned.safetensors` | 8 | dynamic/r144 | 1.0 MP |
| `dasiwa_multistep_r512_pruned` | `minimax_h3_fl2va_bf16_turbo_multistep_fro099_r512_pruned.safetensors` | 8 | dynamic/r512 | 1.0 MP |
| `none` | — | 50 | — | 1.0 MP |

Hub: [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo),
[larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora),
and [silveroxides/MiniMax-H3_tests](https://huggingface.co/silveroxides/MiniMax-H3_tests).

### TaoMate FL2VA 3-step EMA: default 12 GB profile

The default low-memory entry is the TaoMate FL2VA 3-step EMA release from
[Civitai model version 3322352](https://civitai.red/models/2837571?modelVersionId=3322352).
This project downloads file ID `3208405`:

`minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors`

| Property | Value |
| --- | --- |
| File size | 177,439 KiB (about 173 MB) |
| Precision / rank | BF16, averaged rank 19 |
| SHA-256 | `DE9663D974A884B477556748239C6F28239F7CA1825BE270F98F023FF5DAB6A7` |
| Workflow | MiniMax-H3 FL2VA, pruned architecture |
| Recommended NFE | 4–6; application default 4 |
| Sampler / scheduler | Euler / simple |
| Strength | 0.75 |
| Default canvas | 0.4 MP; 864×480 for 16:9 |
| Backend | Isolated local ComfyUI worker with DynamicVRAM |

The model name says “3-step,” but the release page currently recommends 4–6
steps. The studio therefore uses 4 transformer evaluations. Increasing to 5 or 6
can improve difficult motion but costs proportionally more time; reducing to 3 is
not the supported default.

The LoRA is only one part of the low-memory profile. It runs with the
`minimax_h3_fl2va_pruned_fp8_scaled.safetensors` transformer and
`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` text encoder. DynamicVRAM streams
those components through the 4070 Ti while keeping staged weights in system RAM.

Download and verify the adapter:

```powershell
.\.venv\Scripts\python.exe -m minimax_h3_fl2v.download `
  --loras --lora-id taomate_fl2va_3step_ema
```

Install or repair the complete pruned backend and post-processing assets:

```powershell
.\.venv\Scripts\python.exe scripts\setup_pruned_backend.py
```

Compatibility rules:

- Do not apply this adapter to the full Diffusers FL2VA transformer.
- Do not combine it with Ref2VA adapters.
- Extra LoRAs used with it must target the same pruned, 8-wide AdaLN architecture.
- Keep strength near 0.75 before stacking style adapters.
- Start at 0.2–0.4 MP on a 12 GB card; upscale decoded frames afterward when needed.

The larger rank-128 BF16 and full FP32 files are not the default because they add
RAM, disk, and loading pressure without making the base transformer fit in VRAM.

### Silveroxides DARE-TIES pruned v1

Optional 12 GB catalog entry from
[silveroxides/MiniMax-H3_tests](https://huggingface.co/silveroxides/MiniMax-H3_tests).
This project catalogs only the pruned file:

`minimax_h3_fl2v_turbo_silver_dareties_comfy_pruned_v1.safetensors`

| Property | Value |
| --- | --- |
| File size | 765,582,016 bytes (about 766 MB) |
| Precision / rank | BF16 DARE-TIES merge, mixed ranks clustered around 64–102 |
| SHA-256 | `9AAB6353CE76F0A1C6A6FBBEAA1A4C60DECED365E328BC14DDB0B7B9F0849B48` |
| Workflow | MiniMax-H3 FL2VA, pruned architecture |
| AdaLN | Direct `diff` / `diff_b` patches at width 8 (`96768×8`) |
| Recommended NFE | 6–8; application default 8 |
| Sampler / scheduler | Euler / simple |
| Strength | 0.9 |
| Default canvas | 0.5 MP; 960×544 for 16:9 |
| Backend | Isolated local ComfyUI worker with DynamicVRAM |

Community reports on this exact filename are that 0.9 strength stays crisp at
960×544. Related DARE-TIES pruned files are commonly run at 6–8 steps with
Euler/simple. The sibling `minimax_h3_fl2v_turbo_silver_dareties_comfy_full_v1.safetensors`
(829 MB) is **not** catalogued; it targets unpruned AdaLN and will not load on
the 12 GB pruned backend.

Download and verify the adapter:

```powershell
.\.venv\Scripts\python.exe -m minimax_h3_fl2v.download `
  --loras --lora-id silveroxides_dareties_pruned_v1
```

Compatibility rules:

- Do not apply this adapter to the full Diffusers FL2VA transformer.
- Do not combine it with Ref2VA adapters.
- Extra LoRAs used with it must target the same pruned, 8-wide AdaLN architecture.
- Keep strength near 0.9 before stacking style adapters.
- Start at 0.5 MP on a 12 GB card; upscale decoded frames afterward when needed.

### Civitai Turbo Multistep v1 architecture notice

The four Dasiwa entries come from Civitai model version `3315815`. The creator
recommends Euler/simple at 4–8 NFE, with 8 NFE preferred. These files declare
`output_mode=pruned`, `adaln_target_width=8`, and contain direct `diff`/`diff_b`
AdaLN patches. Selecting one automatically routes generation to the isolated
local ComfyUI backend and its matching
`minimax_h3_fl2va_pruned_bf16.safetensors` base. Standard catalog entries keep
using the original Diffusers backend.

Install or repair this backend while online with:

```bash
source .venv/bin/activate
python scripts/setup_pruned_backend.py
```

The worker is installed under `.runtime/ComfyUI`, listens only on
`127.0.0.1:18188` by default, starts automatically when a pruned entry is selected, and
supports text, first-frame, last-frame, and first+last-frame FL2VA generation.
It uses Euler/simple and unloads its models after each request to return VRAM.
Only LoRAs compatible with the pruned 8-wide AdaLN architecture may be stacked
with these entries.

Before every pruned request, the selected catalog and extra LoRA files are
atomically linked from `models/loras/` into the worker's model directory and
checked against its live `/models/loras` list. Newly uploaded or copied LoRAs
therefore become available without reinstalling or restarting the worker.

Installed backend components are the 21.0 GB FP8-scaled pruned FL2VA transformer,
Qwen3-VL-32B NVFP4 text encoder, FP16 video VAE, and FP32 audio VAE from
`Comfy-Org/MiniMax-H3`. ComfyUI is pinned to the revision tested by the setup
script. A real five-frame smoke render with the rank-48 adapter is used to
validate model loading, mixed LoRA patches, sampling, both VAEs, and MP4 muxing.

Optional post-processing uses `Comfy-Org/Real-ESRGAN_repackaged` via ComfyUI's
built-in `UpscaleModelLoader` and tiled `ImageUpscaleWithModel` nodes. This is a
pixel-space restoration pass, not another diffusion/detailing pass, so it avoids
loading a second generative model on the 12 GB GPU. A mild built-in sharpening
node can be applied after resizing the native 4× output to 1.5× or 2×.

Frame-rate enhancement uses ComfyUI's built-in `FrameInterpolate` node with the
compact RIFE 4.25 Lite model. For non-integer broadcast rates, the app first
interpolates at an integer multiple and then selects evenly spaced frames to
preserve duration at 29.97, 60, or 120 fps. Optional CodeFormer restoration uses
the pinned `facerestore_cf` node with `retinaface_mobile0.25` to limit VRAM use.

Locally verified Civitai SHA-256 checksums for ranks 48, 96, 144, and 512 match
the publisher. The rank-512 file is `5,136,520,680` bytes and has SHA-256
`70F1A59B130162CB15E5D9DB8ACFA227A4C460405ED31882B215A570E1BCBCD2`.

## Scheduler grid

MiniMax-H3's scheduler treats `num_inference_steps` as **sigma grid
points including terminal zero**. For **N** transformer evaluations the
code sends **N + 1**. The UI NFE slider is the human-facing N.

## Swapping and stacking

Gradio **does not fuse** LoRAs. Unfused adapters can be deleted and
replaced without reloading the 62 GB transformer.

You can stack:

1. Catalog turbo LoRA (`adapter_name=turbo`)
2. Up to five persistent local SafeTensors (`adapter_name=extra_1` through
   `extra_5`), each with an independent strength from `0.0` to `2.0`.

The same file cannot occupy multiple slots or duplicate the active catalog
adapter. Additional adapters increase host/GPU memory use and initial load time;
begin with one or two and add more only when the visual combination warrants it.
2. Extra uploaded `.safetensors` (`adapter_name=style`)

Do **not** stack a Ref2VA LoRA onto the FL2VA transformer. Ref2VA uses
`transformer_ref/` and is out of scope for this studio.

## Adding your own LoRA

1. Export a PEFT adapter as SafeTensors with the keys above.
2. Copy it to `models/loras/`.
3. Add an entry to `configs/loras.yaml`.
4. Restart Gradio.

Or upload it in the UI; the file persists in `models/loras/` and becomes
available in every **Extra LoRA** selector without a catalog edit.

## Fuse (CLI only)

Fusing bakes A/B into the base weights and unloads the adapter. Faster
for a long batch of the **same** LoRA; you cannot swap afterwards
without reloading the pipeline. The Gradio path never fuses.
