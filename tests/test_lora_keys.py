import torch
from safetensors.torch import save_file

from minimax_h3_fl2v.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    _normalize_keys,
    _validate_state_dict,
    validate_pruned_fl2va_lora,
    validate_lora_workflow,
)


def test_normalize_peft_prefixes():
    raw = {
        "base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(8, 16),
        "base_model.model.transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(16, 8),
    }
    out = _normalize_keys(raw)
    assert any(key.endswith(LORA_A_SUFFIX) for key in out)
    assert any(key.endswith(LORA_B_SUFFIX) for key in out)


def test_validate_official_layout():
    state = {
        "transformer_blocks.0.attn.to_q" + LORA_A_SUFFIX: torch.zeros(8, 32),
        "transformer_blocks.0.attn.to_q" + LORA_B_SUFFIX: torch.zeros(32, 8),
    }
    assert _validate_state_dict(state, path=__import__("pathlib").Path("demo.safetensors")) == 8


def test_validate_mixed_rank_official():
    state = {
        "transformer_blocks.0.attn.to_q" + LORA_A_SUFFIX: torch.zeros(64, 32),
        "transformer_blocks.0.attn.to_q" + LORA_B_SUFFIX: torch.zeros(32, 64),
        "transformer_blocks.0.adaLN_modulation.1" + LORA_A_SUFFIX: torch.zeros(16, 8),
        "transformer_blocks.0.adaLN_modulation.1" + LORA_B_SUFFIX: torch.zeros(8, 16),
    }
    assert _validate_state_dict(state, path=__import__("pathlib").Path("mixed.safetensors")) == 64


def test_reject_comfyui_keys():
    state = {
        "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": torch.zeros(8, 32),
        "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight": torch.zeros(32, 8),
    }
    try:
        _validate_state_dict(state, path=__import__("pathlib").Path("x_comfyui.safetensors"))
    except ValueError as exc:
        assert "ComfyUI" in str(exc)
    else:
        raise AssertionError("expected ComfyUI LoRA rejection")


def test_reject_ref2va_lora_in_fl2va_workflow(tmp_path):
    path = tmp_path / "sharpness.safetensors"
    save_file(
        {"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.zeros(4, 8)},
        path,
        metadata={"ss_base_model_version": "minimax_h3_ref2va"},
    )
    try:
        validate_lora_workflow(path, "fl2va")
    except ValueError as exc:
        assert "Ref2VA" in str(exc)
        assert "FL2VA" in str(exc)
        assert "cannot be stacked" in str(exc)
    else:
        raise AssertionError("expected Ref2VA LoRA rejection")


def test_allow_generic_lora_in_fl2va_workflow(tmp_path):
    path = tmp_path / "style.safetensors"
    save_file(
        {"transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(4, 8)},
        path,
        metadata={"ss_base_model_version": "minimax_h3"},
    )
    validate_lora_workflow(path, "fl2va")


def test_reject_pruned_lora_on_full_diffusers_transformer(tmp_path):
    path = tmp_path / "multistep-pruned.safetensors"
    save_file(
        {"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.zeros(4, 8)},
        path,
        metadata={
            "output_mode": "pruned",
            "adaln_target_width": "8",
            "adaln_source_width": "2688",
        },
    )
    try:
        validate_lora_workflow(path, "fl2va")
    except ValueError as exc:
        assert "pruned" in str(exc)
        assert "8-wide AdaLN" in str(exc)
        assert "2688-wide AdaLN" in str(exc)
    else:
        raise AssertionError("expected pruned architecture rejection")


def test_reject_incomplete_safetensors_with_actionable_error(tmp_path):
    path = tmp_path / "partial.safetensors"
    path.write_bytes(b"partial download")
    try:
        validate_lora_workflow(path, "fl2va")
    except ValueError as exc:
        assert "incomplete or corrupt" in str(exc)
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("expected incomplete SafeTensors rejection")


def test_pruned_validator_accepts_eight_wide_adaln(tmp_path):
    path = tmp_path / "pruned-style.safetensors"
    save_file(
        {"diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight": torch.zeros(4, 8)},
        path,
        metadata={"ss_base_model_version": "minimax_h3_fl2va"},
    )
    validate_pruned_fl2va_lora(path)


def test_pruned_validator_rejects_full_adaln(tmp_path):
    path = tmp_path / "full-style.safetensors"
    save_file(
        {"diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight": torch.zeros(4, 2688)},
        path,
    )
    try:
        validate_pruned_fl2va_lora(path)
    except ValueError as exc:
        assert "input width 2688" in str(exc)
    else:
        raise AssertionError("expected full AdaLN rejection")
