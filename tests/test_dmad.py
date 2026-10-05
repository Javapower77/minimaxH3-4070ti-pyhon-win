import pytest
import torch
from safetensors.torch import save_file

from minimax_h3_fl2v.config import load_config
from minimax_h3_fl2v.dmad import convert_dmad_state_dict, prepare_dmad_lora
from minimax_h3_fl2v.lora import validate_pruned_fl2va_lora


def _state(prefix="transformer_blocks.0"):
    generator = torch.Generator().manual_seed(42)
    state = {}
    for module, inputs, outputs in (("attn.to_q", 3, 4), ("attn.to_k", 3, 4),
                                    ("attn.to_v", 3, 4), ("attn.to_out.0", 4, 3),
                                    ("ff.net.0.proj", 3, 10), ("ff.net.2", 5, 3)):
        down, up = ("lora.down", "lora.up") if module.startswith("attn") else ("lora_A", "lora_B")
        state[f"{prefix}.{module}.{down}.weight"] = torch.randn(2, inputs, generator=generator)
        state[f"{prefix}.{module}.{up}.weight"] = torch.randn(outputs, 2, generator=generator)
    return state


@pytest.mark.parametrize("prefix,target", [("transformer_blocks.0", "blocks.0"),
                                          ("token_refiner.refiner_blocks.1", "token_refiner.blocks.1")])
def test_dmad_conversion_preserves_independent_qkv_and_ffn_deltas(prefix, target):
    state = _state(prefix)
    converted = convert_dmad_state_dict(state, alpha=3)
    qkv = target + ".attn.qkv_proj"
    delta = converted[qkv + ".lora_up.weight"] @ converted[qkv + ".lora_down.weight"]
    expected = torch.cat([state[f"{prefix}.attn.{p}.lora.up.weight"] @
                          state[f"{prefix}.attn.{p}.lora.down.weight"] * 1.5
                          for p in ("to_q", "to_k", "to_v")])
    torch.testing.assert_close(delta, expected)
    assert converted[qkv + ".lora_down.weight"].shape == (6, 3)
    assert converted[qkv + ".alpha"].item() == 6
    for source, destination in (("attn.to_out.0", "attn.out_proj"),
                                ("ff.net.0.proj", "mlp.fc1"), ("ff.net.2", "mlp.fc2")):
        a, b = ("lora.down", "lora.up") if source.startswith("attn") else ("lora_A", "lora_B")
        expected = state[f"{prefix}.{source}.{b}.weight"] @ state[f"{prefix}.{source}.{a}.weight"] * 1.5
        if destination == "mlp.fc1":
            expected = torch.cat(expected.chunk(2)[::-1])
        actual = converted[f"{target}.{destination}.lora_up.weight"] @ converted[f"{target}.{destination}.lora_down.weight"]
        torch.testing.assert_close(actual, expected)
    assert len(converted) == 12
    assert not any("adaln" in key for key in converted)


def test_dmad_conversion_rejects_unknown_modules_and_incomplete_pairs():
    state = _state()
    state["unmapped.lora_A.weight"] = torch.ones(2, 3)
    state["unmapped.lora_B.weight"] = torch.ones(3, 2)
    with pytest.raises(ValueError, match="Unmapped DMAD"):
        convert_dmad_state_dict(state, 2)
    del state["unmapped.lora_B.weight"]
    with pytest.raises(ValueError, match="complete A/B"):
        convert_dmad_state_dict(state, 2)


def test_dmad_prepare_rejects_nonrelease_without_overwriting_source(tmp_path):
    source = tmp_path / "dmad.safetensors"
    save_file(_state(), str(source))
    original = source.read_bytes()
    with pytest.raises(ValueError, match="624 tensors"):
        prepare_dmad_lora(source, tmp_path / ".converted")
    assert source.read_bytes() == original
    assert not list((tmp_path / ".converted").glob("*.safetensors"))


@pytest.mark.parametrize("prefix", ["", "transformer.", "diffusion_model."])
def test_dmad_raw_extra_is_rejected_but_native_conversion_is_accepted(tmp_path, prefix):
    source = tmp_path / "original.safetensors"
    save_file({prefix + key: value for key, value in _state().items()}, str(source))
    with pytest.raises(ValueError, match="converts it automatically"):
        validate_pruned_fl2va_lora(source)
    native = tmp_path / "native.safetensors"
    save_file(convert_dmad_state_dict(_state(), 2), str(native))
    validate_pruned_fl2va_lora(native)


def test_dmad_catalog_preserves_original_checkpoint_and_defaults():
    config = load_config()
    spec = config.lora_by_id("dmad_4step_lora_critic")
    assert spec.filename == "dmad_minimax_h3_4step_lora_critic.safetensors"
    assert "ZhengmingYu/DMAD/resolve/2cb62f3" in spec.download_url
    assert spec.sha256 == "61D0865D6BF426E328CAEAB6610A28A677C68E9F2F04C607657C6451E38B5980"
    assert spec.backend == "comfy_pruned"
    assert spec.lora_format == "dmad_diffusers"
    assert (spec.nfe, spec.lora_alpha, spec.lora_scale) == (4, 128, 1.0)
    assert (spec.video_shift, spec.audio_shift, spec.megapixels) == (12, 2, 0.4)
    assert "Experimental" in spec.notes
    assert config.default_lora_id == "taomate_fl2va_3step_ema"


def test_dmad_hyperflow_catalog_is_separate_native_blend():
    config = load_config()
    spec = config.lora_by_id("dasiwa_dmad_hyperflow_4step_r256")
    original = config.lora_by_id("dmad_4step_lora_critic")
    assert spec.filename == "minimax_h3_DMAD_Hyperflow_4step_r256_add_fro0995_turbo_lora.safetensors"
    assert spec.download_url == "https://civitai.red/api/download/models/3383490?fileId=3272202"
    assert spec.sha256 == "3D6B77D9A3775DA27CC0057D41DC0DF339E2707D6E941A072C8FEB291CB29920"
    assert (spec.nfe, spec.lora_alpha, spec.lora_scale) == (4, 256, 1.0)
    assert (spec.video_shift, spec.audio_shift, spec.megapixels) == (12, 3, 0.4)
    assert spec.backend == "comfy_pruned"
    assert spec.lora_format == "native"
    assert original.lora_format == "dmad_diffusers"
    assert original.audio_shift == 2
    assert "3383490" in spec.notes and "FL2VA only" in spec.notes
    assert config.default_lora_id == "taomate_fl2va_3step_ema"