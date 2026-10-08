"""Strict native Ref2VA character replacement with optional Ref2VA acceleration."""

from dataclasses import dataclass
from pathlib import Path
import shutil
import uuid

from PIL import Image, ImageOps

from .assets import (CHARACTER_SWAP_LORA_FILE, CHARACTER_SWAP_LORA_SHA256,
                     REF2VA_MODEL_FILE, REF2VA_MODEL_SHA256,
                     REF2VA_TURBO_FILE, REF2VA_TURBO_SHA256, REF2VA_TURBO_SIZE)
from .comfy_backend import PrunedComfyBackend, ProgressCallback
from . import restoration as shared

CHARACTER_SWAP_PROMPT = (
    "Replace only the person in the purple shirt in <Video 1> with the character "
    "in <Picture 1>. Keep the replacement character's identity, outfit, and art style "
    "from <Picture 1>. Preserve the source video's camera, background, lighting, "
    "objects, and all other people. Match the target person's position, scale, pose, "
    "and movement. Do not show the reference sheet or its background."
)


@dataclass(frozen=True)
class CharacterSwapRequest:
    source_video: Path
    reference_image: Path
    prompt: str = CHARACTER_SWAP_PROMPT
    steps: int = 20
    strength: float = 1.0
    seed: int = 42
    use_turbo: bool = False

    @property
    def effective_steps(self) -> int:
        return 8 if self.use_turbo else self.steps

    @property
    def source(self) -> Path:
        return self.source_video

    def validate(self) -> None:
        shared.RestorationRequest(self.source, self.strength, self.steps, self.seed,
                                  False, "native").validate()
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("Character swap requires an editable target-identifying prompt.")
        if not isinstance(self.use_turbo, bool):
            raise ValueError("Character swap use_turbo must be a boolean.")


def validate_adapter_stack(adapters: list[Path], model_header: dict) -> None:
    """Fail closed on every patch against the shape produced by preceding adapters.

    Native PDD requires equal bank counts in both output heads and their biases.
    Do not pad later LoRAs or bypass full-width AdaLN mismatches: the pinned
    pruned adapter uses [96768, 8] projections, not a full timestep embedder.
    Only tiny reshape metadata is loaded; weights stay on disk.
    """
    header = {key: dict(value) for key, value in model_header.items()}
    heads = ("final_layer.video_out", "final_layer.audio_out")
    for adapter in adapters:
        shared.validate_adapter(adapter, header)
        patches = shared.read_header(adapter)
        reshapes = {}
        with shared.safe_open(str(adapter), framework="pt", device="cpu") as tensors:
            for key, entry in patches.items():
                if not key.endswith(".reshape_weight"):
                    continue
                module = key.removeprefix("diffusion_model.").removesuffix(".reshape_weight")
                target = module if module.endswith(".bias") else module + ".weight"
                base = header[target]["shape"]
                shape = tensors.get_tensor(key).tolist()
                if (entry["dtype"] not in ("I32", "I64") or entry["shape"] != [len(base)]
                        or not isinstance(shape, list)
                        or any(type(dim) is not int or dim <= 0 for dim in shape)
                        or shape[0] % base[0] or shape[1:] != base[1:]):
                    raise ValueError(f"Invalid integer PDD head bank reshape: {key}")
                reshapes[target] = shape
        if reshapes:
            expected = {head + suffix for head in heads for suffix in (".weight", ".bias")}
            if set(reshapes) != expected:
                raise ValueError("PDD requires matching video/audio head weight and bias reshapes.")
            banks = {reshapes[target][0] // header[target]["shape"][0] for target in expected}
            if len(banks) != 1:
                raise ValueError("PDD video/audio weight and bias bank counts must match.")
            for target, shape in reshapes.items():
                header[target]["shape"] = shape


@dataclass(frozen=True)
class CharacterSwapResult:
    output_path: Path
    original_path: Path
    original_image_path: Path
    message: str


class CharacterSwapFailure(RuntimeError):
    def __init__(self, original_path: Path, original_image_path: Path, error: Exception):
        self.original_path = original_path
        self.original_image_path = original_image_path
        super().__init__(f"Character swap failed: {error}. Original video: {original_path}; image: {original_image_path}")


class CharacterSwapBackend(shared.RestorationBackend):
    operation = "character_swap"

    def enhance(self, request, progress_callback=None):
        """Do not expose restoration's normalization API on character backends."""
        raise ValueError("Character swap requires swap(CharacterSwapRequest); restoration enhancement is not supported here.")

    @staticmethod
    def build_character_swap_workflow(*, source, reference_image, info, request,
                                      text_encoder, filename_prefix, nodes=None,
                                      audio_source=None) -> dict:
        info.validate()
        request.validate()
        loras = [(REF2VA_TURBO_FILE, 1.0)] if request.use_turbo else []
        loras.append((CHARACTER_SWAP_LORA_FILE, request.strength))
        graph = PrunedComfyBackend.build_workflow(
            prompt=request.prompt, width=info.width, height=info.height, frames=info.frames,
            seed=request.seed, nfe=request.effective_steps, model=REF2VA_MODEL_FILE,
            text_encoder=text_encoder, loras=loras,
            target_fps=24.0, filename_prefix=filename_prefix)
        graph.pop("9")
        graph.pop("17")
        graph["18"]["inputs"].pop("audio")
        graph["11"]["inputs"]["sampler_name"] = "euler" if request.use_turbo else "res_multistep"
        graph["50"] = {"class_type": "LoadVideo", "inputs": {"file": source}}
        graph["51"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["50", 0]}}
        graph["53"] = {"class_type": "LoadImage", "inputs": {"image": reference_image}}
        inputs = {"clip": ["2", 0], "vae": ["3", 0], "prompt": request.prompt,
                  "width": info.width, "height": info.height, "length": info.frames,
                  "ref_image_size": "match"}
        inputs[shared.reference_input(nodes, "ref_images", "ref_image_", "IMAGE")] = ["53", 0]
        inputs[shared.reference_input(nodes, "ref_videos", "ref_video_", "IMAGE")] = ["51", 0]
        # Source audio is remuxed separately, not generated or used as conditioning.
        graph["54"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": inputs}
        graph["13"]["inputs"]["conditioning"] = ["54", 0]
        graph["15"]["inputs"]["latent_image"] = ["54", 1]
        return graph

    def _build_source_workflow(self, **kwargs) -> dict:
        kwargs["request"] = self._swap_request
        return self.build_character_swap_workflow(reference_image=self._worker_image.name, **kwargs)

    def preflight(self, request: CharacterSwapRequest | None = None) -> None:
        request = request or getattr(self, "_swap_request", None)
        use_turbo = request is not None and request.use_turbo
        adapter = self.root / "models/loras" / CHARACTER_SWAP_LORA_FILE
        turbo = self.root / "models/loras" / REF2VA_TURBO_FILE
        adapters = [turbo, adapter] if use_turbo else [adapter]
        missing = [str(p) for p in [*self._required_paths(), *adapters] if not p.is_file()]
        if missing:
            raise FileNotFoundError("Optional character swap assets missing; download character_swap (and ref2va_turbo only when enabled), reuse restore_base and encoder/VAEs: " + ", ".join(missing))
        base = self.root / "models/diffusion_models" / self.model
        shared.verify_asset(base, REF2VA_MODEL_SHA256)
        shared.verify_asset(adapter, CHARACTER_SWAP_LORA_SHA256)
        header = shared.read_header(base)
        if header.get("adaln_t_table", {}).get("shape") != [1025, 8]:
            raise ValueError("Character swap requires the pinned pruned Ref2VA curve base.")
        if use_turbo:
            if turbo.stat().st_size != REF2VA_TURBO_SIZE:
                raise ValueError("Ref2VA turbo asset size mismatch; install the pinned optional download.")
            shared.verify_asset(turbo, REF2VA_TURBO_SHA256)
        try:
            validate_adapter_stack(adapters, header)
        except ValueError as exc:
            raise ValueError(f"Character swap LoRA incompatible with actual Ref2VA base {base.name}: {exc}") from exc
        dynamic = self.root / "comfy/cli_args.py"
        if not dynamic.is_file() or "enables_dynamic_vram" not in dynamic.read_text(encoding="utf-8"):
            raise RuntimeError("Character swap requires the pinned native DynamicVRAM worker; run backend setup.")
        if use_turbo:
            for relative, marker in (("comfy/weight_adapter/lora.py", "reshape_weight"),
                                     ("comfy/ldm/minimax/model.py", "_pdd_head")):
                path = self.root / relative
                if not path.is_file() or marker not in path.read_text(encoding="utf-8"):
                    raise RuntimeError(f"Character swap turbo requires native PDD support: {relative}; run backend setup.")

    def _prepare_reference(self, token: str) -> None:
        self._worker_image = self.input_dir / f"{token}_character.png"
        with Image.open(self._original_image) as image:
            # Bake EXIF orientation, never stretch/crop to the output canvas.
            ImageOps.exif_transpose(image).convert("RGB").save(self._worker_image)

    def _retained_source(self, source: Path, originals: Path, token: str) -> Path:
        return self._original_video

    def swap(self, request: CharacterSwapRequest, progress: ProgressCallback | None = None) -> CharacterSwapResult:
        request.validate()
        source_image = Path(request.reference_image).expanduser().resolve()
        source_video = Path(request.source_video).expanduser().resolve()
        if not source_image.is_file():
            raise FileNotFoundError(f"Character reference image missing: {source_image}")
        if not source_video.is_file():
            raise FileNotFoundError(f"Character source video missing: {source_video}")
        originals = self.config.upload_dir / "character_swap_originals"
        originals.mkdir(parents=True, exist_ok=True)
        self._original_image = originals / (uuid.uuid4().hex + source_image.suffix.lower())
        shutil.copy2(source_image, self._original_image)
        self._original_video = originals / (uuid.uuid4().hex + source_video.suffix.lower())
        shutil.copy2(source_video, self._original_video)
        self._worker_image = None
        try:
            with Image.open(self._original_image) as image:
                if image.format not in ("JPEG", "PNG", "WEBP", "BMP", "TIFF") or getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Use a readable single-frame JPEG, PNG, WebP, BMP, or TIFF reference image.")
                image.load()
            def report(fraction, message):
                if progress:
                    progress(fraction, message.replace("Restoration", "Character swap").replace("restoration", "character swap"))
            self._swap_request = request
            # Private lifecycle bridge: character swap cannot request normalization
            # or output-rate conversion. Restoration's public features are unchanged.
            lifecycle_request = shared.RestorationRequest(
                source_video, request.strength, request.effective_steps, request.seed,
                normalize_input=False, output_fps="native")
            result = super().enhance(lifecycle_request, report)
            return CharacterSwapResult(result.path, result.original, self._original_image,
                f"Character swap · {result.info.width}×{result.info.height} · {result.info.frames} frames · "
                f"{float(result.info.fps):.6g} fps · steps={request.effective_steps} · strength={request.strength:g} · turbo={request.use_turbo}\n"
                f"{result.normalization_summary}\nOriginal audio stream-copy verified. "
                "Experimental character replacement, not guaranteed face-only editing or lip sync. "
                "Source canvas preserved: long/high-resolution clips may exceed VRAM; GPU rendering unvalidated.")
        except shared.RestorationFailure as exc:
            raise CharacterSwapFailure(exc.original, self._original_image, exc.__cause__ or exc) from exc
        except Exception as exc:
            raise CharacterSwapFailure(self._original_video, self._original_image, exc) from exc
        finally:
            if self._worker_image is not None:
                self._worker_image.unlink(missing_ok=True)