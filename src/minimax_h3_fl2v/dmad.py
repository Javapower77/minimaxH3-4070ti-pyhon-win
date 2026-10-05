"""Lossless DMAD Diffusers-to-native H3 LoRA conversion for ComfyUI.

This converts adapter factors, not the base model or the trained workflow.
T2VA-to-FL2VA transfer and Euler sampling remain experimental.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

CONVERSION_VERSION = "1"


def convert_dmad_state_dict(state: dict[str, torch.Tensor], alpha: float) -> dict[str, torch.Tensor]:
    """Preserve each delta exactly, including independent split Q/K/V factors."""
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    suffixes = {
        ".lora.down.weight": "A", ".lora.up.weight": "B",
        ".lora_A.weight": "A", ".lora_B.weight": "B",
    }
    for key, value in state.items():
        for suffix, part in suffixes.items():
            if key.endswith(suffix):
                module = key[:-len(suffix)].removeprefix("transformer.")
                pair = pairs.setdefault(module, {})
                if part in pair:
                    raise ValueError(f"Duplicate DMAD factor: {key}")
                pair[part] = value
                break
        else:
            raise ValueError(f"Unsupported DMAD tensor: {key}")
    if not pairs or any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise ValueError("DMAD requires complete A/B pairs")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("DMAD alpha must be finite and positive")
    for module, pair in pairs.items():
        a, b = pair["A"], pair["B"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] <= 0 or a.shape[0] != b.shape[1]:
            raise ValueError(f"Invalid DMAD factor shapes: {module}")

    prefixes = {module.rsplit(".attn.", 1)[0] for module in pairs if ".attn." in module}
    result: dict[str, torch.Tensor] = {}

    def emit(target: str, a: torch.Tensor, b: torch.Tensor, scaling: float) -> None:
        result[target + ".lora_down.weight"] = a.contiguous()
        result[target + ".lora_up.weight"] = (b * scaling).contiguous()
        # Comfy scales alpha / rank: the original scaling is already in B.
        result[target + ".alpha"] = torch.tensor(float(a.shape[0]))

    for prefix in sorted(prefixes):
        if prefix.startswith("transformer_blocks.") and prefix[len("transformer_blocks."):].isdigit():
            target = prefix.replace("transformer_blocks.", "blocks.", 1)
        elif prefix.startswith("token_refiner.refiner_blocks.") and prefix[len("token_refiner.refiner_blocks."):].isdigit():
            target = prefix.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.", 1)
        else:
            raise ValueError(f"Unsupported DMAD block: {prefix}")
        qkv = []
        for projection in ("to_q", "to_k", "to_v"):
            module = prefix + ".attn." + projection
            if module not in pairs:
                raise ValueError(f"Missing DMAD projection: {module}")
            qkv.append(pairs.pop(module))
        if len({tuple(p["A"].shape) for p in qkv}) != 1 or len({tuple(p["B"].shape) for p in qkv}) != 1:
            raise ValueError(f"Inconsistent DMAD QKV shapes: {prefix}")
        # Native H3 uses [q_all; k_all; v_all], NOT per-head interleaving.
        # Rank 384 is necessary to preserve three independent rank-128 deltas.
        emit(target + ".attn.qkv_proj", torch.cat([p["A"] for p in qkv]),
             torch.block_diag(*(p["B"] for p in qkv)), alpha / qkv[0]["A"].shape[0])
        for source, destination in (("attn.to_out.0", "attn.out_proj"),
                                    ("ff.net.0.proj", "mlp.fc1"), ("ff.net.2", "mlp.fc2")):
            module = prefix + "." + source
            if module not in pairs:
                raise ValueError(f"Missing DMAD projection: {module}")
            pair = pairs.pop(module)
            b = pair["B"]
            if source == "ff.net.0.proj":
                if b.shape[0] % 2:
                    raise ValueError(f"Odd DMAD gated FFN width: {module}")
                # Diffusers [gate; value] -> native [value; gate].
                b = torch.cat(b.chunk(2)[::-1])
            emit(target + "." + destination, pair["A"], b, alpha / pair["A"].shape[0])
    if pairs:
        raise ValueError(f"Unmapped DMAD modules: {sorted(pairs)[:3]}")
    return result


def prepare_dmad_lora(source: Path, cache_dir: Path) -> Path:
    """Cache a derived native adapter, preserving the downloaded original."""
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"DMAD LoRA missing: {source}. Download model type dmad.")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    fingerprint = digest.hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"dmad_lora_critic_native_v{CONVERSION_VERSION}_{fingerprint[:16]}.safetensors"
    if destination.is_file():
        with safe_open(str(destination), framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
            if metadata.get("source_sha256") == fingerprint and metadata.get("conversion_version") == CONVERSION_VERSION:
                return destination
    with safe_open(str(source), framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        alpha = float(metadata.get("lora_alpha", "128"))
        # Pin the supported release's dimensions and modules before conversion.
        if len(checkpoint.keys()) != 624:
            raise ValueError("Expected the original DMAD lora_critic release (624 tensors)")
        expected = {}
        for prefix, count in (("transformer_blocks", 50), ("token_refiner.refiner_blocks", 2)):
            for index in range(count):
                for projection, inputs, outputs in (("attn.to_q", 5376, 7168), ("attn.to_k", 5376, 7168),
                                                   ("attn.to_v", 5376, 7168), ("attn.to_out.0", 7168, 5376),
                                                   ("ff.net.0.proj", 5376, 28672), ("ff.net.2", 14336, 5376)):
                    module = f"{prefix}.{index}.{projection}"
                    expected[module] = {"A": (128, inputs), "B": (outputs, 128)}
        for key in checkpoint.keys():
            normalized = key.replace(".lora.down.weight", ".lora_A.weight").replace(".lora.up.weight", ".lora_B.weight")
            module, separator, part = normalized.rpartition(".lora_")
            shape = tuple(checkpoint.get_slice(key).get_shape())
            if not separator or module not in expected or expected[module].get(part.removesuffix(".weight")) != shape:
                raise ValueError(f"Unexpected original DMAD tensor/shape: {key} {shape}")
    state = convert_dmad_state_dict(load_file(str(source), device="cpu"), alpha)
    temporary = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        save_file(state, str(temporary), metadata={
            **metadata, "source_sha256": fingerprint, "conversion_version": CONVERSION_VERSION,
            "key_format": "ComfyUI native MiniMax-H3; lossless split-QKV fusion and FFN reorder",
            "compatibility": "Experimental T2VA adapter transfer to pruned FL2VA; no AdaLN patches",
        })
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination