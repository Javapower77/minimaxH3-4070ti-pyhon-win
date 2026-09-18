from pathlib import Path

from minimax_h3_fl2v.config import AppConfig, LoRASpec
from minimax_h3_fl2v.ui import (
    _ChunkedUploadPreviewMiddleware,
    MAX_SEED,
    _lora_guidance,
    _lora_slot_updates,
    _recommended_reset_values,
    _resolve_seed,
    _save_uploaded_lora,
    _selected_extra_loras,
    _stored_lora_choices,
)


def test_upload_preview_middleware_removes_stale_content_length():
    messages = []

    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", b"4"), (b"content-type", b"image/png")],
            }
        )
        await send({"type": "http.response.body", "body": b"more-than-four"})

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        messages.append(message)

    import asyncio

    asyncio.run(
        _ChunkedUploadPreviewMiddleware(app)(
            {"type": "http", "path": "/gradio_api/file=C:/temp/upload.png"}, receive, send
        )
    )
    assert (b"content-length", b"4") not in messages[0]["headers"]
    assert (b"content-type", b"image/png") in messages[0]["headers"]


def test_stored_lora_choices_include_local_safetensors(tmp_path):
    cfg = AppConfig(lora_dir=tmp_path)
    (tmp_path / "style-a.safetensors").write_bytes(b"a")
    (tmp_path / "ignore.txt").write_text("x")
    choices = _stored_lora_choices(cfg)
    assert choices[0] == ("None (Turbo/catalog only)", None)
    assert ("style-a.safetensors", str((tmp_path / "style-a.safetensors").resolve())) in choices
    assert all("ignore.txt" not in label for label, _ in choices)


def test_uploaded_lora_is_persisted_and_selected(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    upload = incoming / "custom.safetensors"
    upload.write_bytes(b"weights")
    lora_dir = tmp_path / "loras"
    cfg = AppConfig(lora_dir=lora_dir)
    update, cleared = _save_uploaded_lora(cfg, str(upload))
    assert (lora_dir / "custom.safetensors").read_bytes() == b"weights"
    assert update.value == str((lora_dir / "custom.safetensors").resolve())
    assert cleared is None


def test_select_five_loras_with_independent_strengths(tmp_path):
    paths = []
    for index in range(5):
        path = tmp_path / f"lora-{index}.safetensors"
        path.write_bytes(b"x")
        paths.append(str(path))
    selected = _selected_extra_loras(paths, [0.2, 0.4, 0.6, 0.8, 1.0])
    assert [scale for _, scale in selected] == [0.2, 0.4, 0.6, 0.8, 1.0]


def test_seed_fixed_and_random_modes():
    assert _resolve_seed("Fixed", 123456) == 123456
    random_seed = _resolve_seed("Random each generation", 123456)
    assert 0 <= random_seed <= MAX_SEED


def test_lora_slot_updates_show_exact_count():
    updates = _lora_slot_updates(3)
    assert [update["visible"] for update in updates] == [True, True, True, False, False]


def test_lora_guidance_warns_for_strong_large_stack():
    cfg = AppConfig(
        catalog=[
            LoRASpec(
                id="turbo",
                name="Turbo",
                nfe=8,
                lora_scale=1.0,
                video_shift=6,
                audio_shift=3,
            )
        ]
    )
    guidance = _lora_guidance(cfg, "turbo", 4, 0.4, 1.3, 0.5, 0.6)
    assert "NFE **8**" in guidance
    assert "above 1.2" in guidance
    assert "Four or five" in guidance


def test_reset_values_restore_zero_extra_loras():
    cfg = AppConfig(
        default_lora_id="turbo",
        seed=42,
        catalog=[
            LoRASpec(
                id="turbo", name="Turbo", nfe=8, lora_scale=1.0, notes="recommended"
            )
        ],
    )
    values = _recommended_reset_values(cfg)
    assert values[3] == "turbo"
    assert values[4] == 0
    assert values[5:10] == (None, None, None, None, None)
    assert values[15] == "Fixed"