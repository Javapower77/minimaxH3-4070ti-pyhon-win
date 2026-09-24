from minimax_h3_fl2v.config import load_config


def test_default_low_vram_config(monkeypatch):
    monkeypatch.delenv("GRADIO_SERVER_PORT", raising=False)
    cfg = load_config()
    assert cfg.workflow == "fl2va"
    assert cfg.cpu_offload is True
    assert cfg.memory_reserve_margin == "20GB"
    assert cfg.dtype == "bfloat16"
    assert cfg.default_lora_id == "taomate_fl2va_3step_ema"
    assert cfg.comfy_model == "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
    assert cfg.comfy_text_encoder == "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    assert cfg.comfy_reserve_vram_gb == 1.0
    assert cfg.comfy_cache_none is True
    assert cfg.upscale_model == "RealESRGAN_x4plus.safetensors"
    assert cfg.interpolation_model == "rife_v4.25_lite.safetensors"
    assert cfg.face_restore_model == "codeformer.pth"
    assert cfg.server_name == "0.0.0.0"
    assert cfg.server_port == 7860
    assert cfg.share is False
    spec = cfg.lora_by_id(cfg.default_lora_id)
    assert spec.nfe == 4
    assert spec.lora_alpha == 19
    assert spec.lora_scale == 0.75
    assert spec.backend == "comfy_pruned"
    assert spec.sha256 == "DE9663D974A884B477556748239C6F28239F7CA1825BE270F98F023FF5DAB6A7"
    assert spec.filename.endswith(".safetensors")


def test_silveroxides_dareties_pruned_is_catalogued():
    cfg = load_config()
    spec = cfg.lora_by_id("silveroxides_dareties_pruned_v1")
    assert spec.filename == "minimax_h3_fl2v_turbo_silver_dareties_comfy_pruned_v1.safetensors"
    assert spec.repo == "silveroxides/MiniMax-H3_tests"
    assert spec.nfe == 8
    assert spec.lora_scale == 0.9
    assert spec.megapixels == 0.5
    assert spec.backend == "comfy_pruned"
    assert spec.sha256 == "9AAB6353CE76F0A1C6A6FBBEAA1A4C60DECED365E328BC14DDB0B7B9F0849B48"
    assert "full_v1" in spec.notes


def test_civitai_multistep_ranks_are_catalogued():
    cfg = load_config()
    expected = {
        "dasiwa_multistep_r48_pruned": 48,
        "dasiwa_multistep_r96_pruned": 96,
        "dasiwa_multistep_r144_pruned": 144,
        "dasiwa_multistep_r512_pruned": 512,
    }
    for lora_id, rank in expected.items():
        spec = cfg.lora_by_id(lora_id)
        assert f"r{rank}_pruned" in spec.filename
        assert spec.nfe == 8
        assert spec.video_shift == 12.0
        assert spec.backend == "comfy_pruned"
        assert "pruned ComfyUI backend" in spec.notes
