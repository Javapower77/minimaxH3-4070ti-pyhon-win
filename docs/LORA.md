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
| `dasiwa_multistep_v2_r128_pruned` | `minimax_h3_hyperflow_EMA600_pruned_r128_fro0995_turbo_lora.safetensors` | 8 | r128 | 1.0 MP |
| `dmad_4step_lora_critic` | `dmad_minimax_h3_4step_lora_critic.safetensors` | 4 | r128 | 0.4 MP |
| `dasiwa_dmad_hyperflow_4step_r256` | `minimax_h3_DMAD_Hyperflow_4step_r256_add_fro0995_turbo_lora.safetensors` | 4 | r256 | 0.4 MP |
| `dasiwa_pdmd_dmad_4step_r256` | `minimax_h3_PDMD_DMAD_4step_r256_add_fro1_turbo_lora.safetensors` | 4 | r256 | 0.4 MP |
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
.\.venv\Scripts\python.exe scripts\download_models.py --type taomate
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
.\.venv\Scripts\python.exe scripts\download_models.py --type silveroxides
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

Download all four ranks:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py --type dasiwa
```

Download a single rank with its catalog id:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py --type dasiwa_multistep_r48_pruned
```

### Dasiwa Turbo Multistep v2 Hyperflow + EMA600

Optional pruned catalog entry from Civitai model version
[3357658](https://civitai.red/models/2929860/minimax-h3-turbo-or-ref2va-or-fl2va-or-hybrid?modelVersionId=3357658)
(`turbo-multistep-v2`). This project downloads file ID `3245274`:

`minimax_h3_hyperflow_EMA600_pruned_r128_fro0995_turbo_lora.safetensors`

| Property | Value |
| --- | --- |
| File size | 885,254 KiB (about 864 MB) |
| Precision / rank | BF16, pruned rank 128, Fro 0.995 |
| SHA-256 | `C27839218F19CF444F0D2EAFCA38C1B64A48FC30CE02AEEF94CA5804CEA6ECB7` |
| Workflow | MiniMax-H3 FL2VA, pruned architecture |
| Recommended NFE | 4–8; application default 8 |
| Sampler / scheduler | Euler / simple |
| Strength | 1.0 |
| Default canvas | 1.0 MP |
| Backend | Isolated local ComfyUI worker with DynamicVRAM |

The release is a Hyperflow 1.0 + factorized EMA600 blend with a recalculated
pruned extraction. The listing also mentions Ref2VA and hybrid checkpoints;
this studio remains FL2VA-only and routes the adapter to the pruned ComfyUI
backend. Do not apply it to the full Diffusers transformer.

Download and verify the adapter:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py --type dasiwa_v2
```

### Original DMAD 4-step lora_critic (experimental)

The selectable `dmad_4step_lora_critic` option downloads the original
[DMAD paper checkpoint](https://github.com/Yzmblog/DMAD) from
[ZhengmingYu/DMAD](https://huggingface.co/ZhengmingYu/DMAD), pinned to revision
`2cb62f3d699c0afabb4459216db78d76fc801e35`:

- Filename: `dmad_minimax_h3_4step_lora_critic.safetensors` (1,383,674,304 bytes).
- SHA-256: `61D0865D6BF426E328CAEAB6610A28A677C68E9F2F04C607657C6451E38B5980`.
- Rank/alpha 128, strength 1.0, EMA iteration 800; attention and FFN adapters
   for all 50 transformer blocks and two token-refiner blocks.
- Application defaults: 4 NFE, video/audio shifts 12/2, 0.4 MP for the 12 GB
   profile. TaoMate remains the application's default selection.

**Checkpoint distinction:** Civitai version `3382801` publishes `full_critic`
compatibility files, including a compressed avg-rank-39 version. Those are
different trained weights and are not substituted for `lora_critic` here.

Download with model type `dmad` or catalog ID `dmad_4step_lora_critic`.
Select **DMAD 4-step lora_critic r128 (experimental FL2VA transfer)** in the UI.
On the first generation, the app validates the original release's tensor
dimensions and converts its Diffusers keys into a cached ComfyUI-native file.
The original remains unchanged. Independent Q/K/V factors are concatenated
and block-diagonally fused (rank 384), and gated FFN output halves are reordered;
no recompression, base-model merging, or AdaLN patches are performed. Allow
about 2.0 GiB extra disk space plus temporary host RAM for conversion. The cache
is content-addressed and reused on subsequent generations. Do not select the
unconverted original as an extra LoRA; use the catalog option.

**Compatibility limitations:** DMAD was trained on the T2VA transformer at
1344×768 and 124 frames. The adapted attention/FFN dimensions match the installed
pruned FL2VA backbone, but matching dimensions do not establish generation
quality or first/last-frame compatibility. Start with text-only generation;
frame conditioning and reduced-resolution quality remain unverified. The app
uses Euler/simple, the upstream-supported Euler alternative, **not** the
paper's stochastic re-noise step rule. Outputs will differ from paper examples.
The upstream consumer-GPU path targets 24 GB, not this app's 12 GB profile.
Original weights retain the MiniMax H3 Community License; upstream code is
Apache-2.0. This app remains FL2VA-only and does not add Ref2VA workflows.

### Dasiwa DMAD + Hyperflow 4-step r256 (experimental)

[Civitai version 3383490](https://civitai.red/models/2929860/minimax-h3-turbo-or-ref2va-or-fl2va-or-hybrid?modelVersionId=3383490)
is `turbo-4step-DH-v1`, a Darksidewalker blend of DMAD (using a Silveroxides
extraction) and Hyperflow, rank 256 / Fro 0.995. This is **not** a replacement
for the original DMAD paper checkpoint or Dasiwa's Hyperflow+EMA600 v2.

- Catalog ID: `dasiwa_dmad_hyperflow_4step_r256`.
- File: `minimax_h3_DMAD_Hyperflow_4step_r256_add_fro0995_turbo_lora.safetensors`.
- Civitai file ID: `3272202`; BF16, 1,366,710.734375 KiB (~1.30 GiB).
- SHA-256: `3D6B77D9A3775DA27CC0057D41DC0DF339E2707D6E941A072C8FEB291CB29920`.
- Creator settings: 4 steps (possibly more), Euler/simple or Euler/beta,
  video/audio shifts 12/3. Application defaults: 4 NFE, strength 1.0,
  Euler/simple, reduced 0.4 MP for the 12 GB profile.

Download using model type `dmad_hyperflow` or its catalog ID, then select
**Dasiwa DMAD + Hyperflow 4-step r256 (experimental)** in the studio. The
option routes to the existing native pruned ComfyUI workflow, bypassing the
original DMAD Diffusers converter. This studio only supports FL2VA, regardless
of the publisher's Ref2VA/hybrid examples. The rank-256 adapter is larger than
TaoMate; start at 0.4 MP before increasing resolution or stacking adapters.

Authenticated download is currently required by Civitai (HTTP 401 during
header inspection). If the downloader returns 401, sign in to Civitai and
download file `3272202` manually into `models/loras/` with the exact filename
above; rerun the download type to validate the existing file's checksum.
Tensor layout, loading against the local pruned backbone, and rendered
quality remain unverified locally. The publisher's example metadata references
a different DMAD filename, so it is not treated as proof for this exact blend.
No additional sampler or Ref2VA workflow is installed.

### Dasiwa PDMD + DMAD 4-step r256 (experimental)

[Civitai 3385247](https://civitai.red/models/2929860/minimax-h3-turbo-or-ref2va-or-fl2va-or-hybrid?modelVersionId=3385247)
(`turbo-4step-PD-v1`) blends PDMD and DMAD at rank 256 / Fro1. The creator
describes it as an attempt to combine PDMD audio quality with DMAD motion and
visual fidelity; these are publisher claims, not local benchmark results.

- Catalog ID: `dasiwa_pdmd_dmad_4step_r256`; download type: `pdmd_dmad`.
- File: `minimax_h3_PDMD_DMAD_4step_r256_add_fro1_turbo_lora.safetensors`.
- Civitai file ID `3274094`, BF16, 2,422,472.03125 KiB (~2.31 GiB).
- SHA-256: `B729385D6B338E7D07462AAFCCDAAB76FA7DD9817A53E300F3264D842643BC6C`.
- Creator: 4 steps (possibly more), Euler/simple or Euler/beta, shifts 12/3.
   Application: 4 NFE, strength 1.0, Euler/simple, 0.4 MP for the 12 GB profile.

This is a separate native pruned ComfyUI choice, not an update to the original
DMAD converter or the Hyperflow blend. The studio remains FL2VA-only; publisher
Ref2VA/hybrid examples do not establish quality on this local FL2VA base.
Start at 0.4 MP without extra adapters: this file is larger than the previous
DMAD-Hyperflow blend. Real render quality remains unverified.

For authenticated downloads, provide `CIVITAI_API_TOKEN` through the process
environment. The downloader uses an HTTPS Civitai-only Authorization header
and requests removes it on cross-host redirects. Never put tokens in catalog
URLs, source files, or shell history. A manually downloaded file in
`models/loras/` can also be verified by rerunning the download type.

### Installing the shared pruned backend

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
