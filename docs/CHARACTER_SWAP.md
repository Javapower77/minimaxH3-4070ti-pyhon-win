# Character swap — independent Ref2VA pass (experimental)

Replace a selected person in an existing video using a character reference image
and an editable target-identifying prompt. This is an **independent native
image + video reference pass**, not FL2VA generation, restoration, a face-swap
guarantee, or an automatic postprocessing stage. Existing generation and
restoration settings remain separate.

## Inputs and baseline

Open **Character swap — independent Ref2VA pass**, upload a reference image and
source video, identify the target in the prompt, then choose **Swap character**.
The native `MiniMaxH3ReferenceToVideo` node receives the image as `<Picture 1>`
and the source video as `<Video 1>`. No additional trigger word was trained.

The default prompt targets the person in the purple shirt. Edit that description
to match the actual subject; describe the replacement identity, outfit and art
style, and ask to preserve the camera, background, lighting, objects and other
people. Preservation instructions express intent, not a strict editing mask.

| Setting | Application baseline |
| --- | --- |
| Steps | **20**, editable from 4–50 when turbo is off; **8**, disabled when on |
| Character LoRA strength | **1.0**, independently editable |
| Seed | **42**, independently editable |
| Sampler / scheduler | Normal: **`res_multistep` / `simple`**; turbo: **Euler / `simple`**, shifts **12/3** |
| Ref2VA turbo checkbox | **OFF** by default; optional pinned eight-step adapter |
| Turbo strength / order | Fixed **1.0**; turbo first, then character LoRA at its independently editable strength |
| Restoration LoRA | **Not loaded** |
| Input / output rate | Strict source **24 fps CFR**; native **24 fps**, no resampling |

Strength **1.0** is the author's starting setting. **20 steps is the requested
application baseline, not a proven author/model-card recommendation**: the
current pinned card does not prescribe 20 inference steps. The character adapter
is not a Turbo distillation LoRA; the optional acceleration adapter is separate.
Enabling **Ref2VA turbo** sets and locks the steps control to **8**; disabling it
restores **20** and makes steps editable. Character strength remains editable at
default **1.0** and never changes the fixed turbo strength. `CharacterSwapRequest`
forwards `use_turbo: bool = False`; `effective_steps` is 8 in turbo mode, otherwise
the requested steps. GPU quality and these runtime settings have not been
validated locally.

## Preserve the source canvas

Output uses the source video's width and height. There are **no resizing,
cropping, aspect-ratio or megapixel controls** in this section; the FL2VA
**0.4 MP profile is not forced onto character swap**. Both dimensions must be
multiples of **32**. Unsupported geometry is rejected, not silently resized.
Rotated video must have its orientation baked explicitly before submission.

Use a readable single-frame JPEG, PNG, WebP, BMP or TIFF reference image. A
temporary RGB PNG bakes its EXIF orientation without stretching or cropping it
to the output canvas; the retained original image is not rewritten.
The native reference node may internally downscale conditioning frames/images
while preserving aspect ratio; source-sized output is not untouched source detail.

**Memory warning:** preserving a large canvas or using a long clip can exhaust
VRAM and host memory. The separate ~21 GB transformer, reference encoding and
runtime buffers are additional costs. DynamicVRAM and valid input dimensions
are not a 12 GB feasibility guarantee. Start with short, small source clips;
there is no automatic low-memory downscale.

## Strict timing — no normalization

Character swap requires **exactly 24 fps CFR**, **`17n+5` frames**, and the 32-grid
canvas. The shared validator accepts 5–3600 frames, with 3592 the largest
matching count; this is an input bound, not a useful-duration or memory promise.
Input needs a **zero-start video timeline**. The unchanged 23.976 fps FL2VA
default is not strict-compatible.

There is **no character-swap normalize checkbox or output-frame-rate selector**,
and the request exposes neither `normalize_input` nor `output_fps`. Unsupported
rates, VFR, frame counts, dimensions and nonzero-start video are rejected, not
normalized, resized, padded or resampled. The worker guide is a byte-for-byte
source copy, not a normalized transcode. Restoration's separate optional
normalization and output-rate controls remain unchanged; they are not applied
implicitly to character swap.

## Original image, video and audio

Persistent copies of the image and video are retained under
`uploads/character_swap_originals/` and exposed as downloadable originals,
including after backend failure. Keep them until the final result is checked;
failure is not a successful preserving render. Original uploads are never edited.

The final result is **MKV**. **All source audio tracks are stream-copied** from
the original, not generated, re-encoded, or used as conditioning in this pass.
Silent sources remain silent. Audio is not cut with `-shortest` and may extend
beyond the video. Ordered audio payload digests are checked and `-copyts` with
timestamp shifting disabled preserves the source timestamps during remux.
Unlike restoration's normalized-output path, strict character swap does not
perform an additional audio packet-timestamp comparison after remux. This is
encoded-stream preservation, not a byte-identical container or proven A/V fidelity.

Copying source audio **does not repair lip-sync drift**. Exact face-only edits,
identity fidelity, unchanged facial expressions, temporal alignment, camera-cut
timing and background preservation are **not guaranteed**. The author reports
short continuous shots as more promising than long windows; that is qualitative
upstream evidence, not a local benchmark or established maximum duration.

## Pinned adapter and optional download

Author: **Akatz Labs / akatz-ai**. Upstream:
[akatz-ai/MiniMax-H3-Character-Swap-LoRA](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA).

| Artifact field | Pinned value |
| --- | --- |
| Revision | `62407e0cc8089c363abd9ce4b0b27662abb237af` |
| File | `h3_character_swap_pro4500_1000.safetensors` |
| Size | **155,110,320 bytes** |
| SHA-256 | `4b2a3f420ae804c0aa3422761ff84dbd1bf52eef6900ffab6d2e66df63cb4e79` |
| Worker destination | `.runtime/ComfyUI/models/loras/h3_character_swap_pro4500_1000.safetensors` |

For an installation missing this optional adapter only:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py --type character_swap
```

This type fetches **only the pinned worker LoRA**. It does not download the
restoration adapter, full Diffusers snapshot, transformer, encoder or VAEs, and
is excluded from default, `12gb`, `backend`, `loras`, `h100` and `all` groups.
Reuse the existing **`restore_base`** pruned Ref2VA FP8-scaled transformer
(`minimax_h3_ref2va_pruned_fp8_scaled.safetensors`, **20,958,205,608 bytes**,
approximately 21 GB), installed text encoder and video/audio VAEs. Only if that
base is missing is the separate optional `--type restore_base` needed. Sharing
assets does not apply the restoration LoRA.

For optional eight-step acceleration, download **`--type ref2va_turbo`** separately.
This opt-in type fetches the turbo worker LoRA only, not the character adapter,
base, restoration adapter or encoder/VAEs; leave the checkbox off if it is absent.

| Turbo artifact field | Pinned value |
| --- | --- |
| Repository | `Kijai/MiniMax-H3-experimental` |
| Revision | `d8023be02fefbb3633b0cd335c3879f91177299d` |
| File | `MiniMax-H3-Ref2VA-Acc-8Step_pruned_comfy.safetensors` |
| Size | **1,725,921,392 bytes** |
| SHA-256 | `6f18e1c2eccb14b37322607730f26b16bf1169b56cd098ea006cffaec43d1e39` |
| Worker destination | `.runtime/ComfyUI/models/loras/MiniMax-H3-Ref2VA-Acc-8Step_pruned_comfy.safetensors` |

Existing files are checked for the pinned byte count and SHA-256 rather than
blindly replaced. Downloaded models, `.runtime/`, outputs and uploads are already
ignored by the existing `.gitignore`; no ignore-file change is required.
**No downloads are needed for the current local documentation/validation work.**

## Compatibility and validation status

The actual local adapter header contains **416 tensors**. All adapter keys and
target shapes pass the shared validator against the actual pruned **Ref2VA FP8**
base: **208 paired targets, all 416 tensor keys checked**. Both local base and
adapter **SHA-256 hashes match the pinned artifacts**. This is structural
compatibility evidence, not proof of correct rendering, quality, or memory feasibility.
Runtime preflight verifies the pinned base/adapter hashes, checks every adapter
key and target shape, and requires the pinned native DynamicVRAM worker before
GPU submission. **No GPU character-swap render has been tested locally.**

The actual pinned turbo header contains **578 tensors**, with **AdaLN across
50 blocks**, curve-compatible **input width 8** and **output width 96,768**.
It is **not a full-width AdaLN adapter**. The actual local turbo size and SHA-256
match the pinned artifact. Every target passes against the actual native base,
including all four PDD head weight/bias reshapes: video **[3072, 5376] / [3072]**,
audio **[1024, 5376] / [1024]**, matching **32×** output banks. Ordered preflight
validates turbo first, then character against the expanded header; the current
character adapter does not patch these heads. Integer reshape metadata, matched
bank counts, every key and shape, and native PDD loader/model support are required.
Incompatible later head patches or full-width AdaLN blocks are rejected before
worker startup; no unsupported conversion or skipped patches are used.
Both real local turbo-on and turbo-off preflights passed on CPU. This does not
establish successful GPU rendering, quality, or a 12 GB fit for either mode.

The [pinned model card](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA/blob/62407e0cc8089c363abd9ce4b0b27662abb237af/README.md)
records training on **`minimax_h3_ref2va_pruned_int8_convrot.safetensors`**, with
a frozen Ostris Ref2VA training assistant. Those training assets are not merged
into this LoRA. Author hybrid/Turbo evaluation configurations are separate
experiments, not universal compatibility claims for this application's FP8 base.

The supplied user workflow **`workflows/CharacterSwap.json` remains untouched**.
Its runtime-fused/hybrid, GGUF or custom-pack dependencies are not blindly
imported or installed. This integration constructs a native image+video graph
for the pinned worker instead; a workflow file alone does not establish that its
models, loaders or third-party nodes are interchangeable.

## License and territorial limits

The adapter uses the **MiniMax H3 Community License Agreement**, **not
Apache-2.0**. Its upstream terms include acceptable-use, distribution, commercial
and territorial provisions. The standard territorial grant excludes the
**US, EU, UK and Republic of Korea** and describes separate authorization.
Read the complete [upstream LICENSE](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA/blob/62407e0cc8089c363abd9ce4b0b27662abb237af/LICENSE)
and attribution/NOTICE before use or redistribution. This project's license and
the separately licensed dataset do not expand the adapter's grant.
