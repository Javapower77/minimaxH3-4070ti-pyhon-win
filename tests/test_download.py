import hashlib
from pathlib import Path

import pytest

from minimax_h3_fl2v.config import load_config
from minimax_h3_fl2v.download import (
    LORA_GROUPS,
    download_loras,
    download_url_file,
    resolve_types,
)


def test_default_type_is_12gb():
    assert resolve_types(model_type=None, base=False, loras=False, all_flag=False) == [
        "pruned",
        "postprocess",
        "taomate",
    ]


def test_type_all_is_pruned_postprocess_and_catalog_loras():
    assert resolve_types(model_type="all", base=False, loras=False, all_flag=False) == [
        "pruned",
        "postprocess",
        "loras",
    ]


def test_legacy_all_flag_is_h100():
    assert resolve_types(model_type=None, base=False, loras=False, all_flag=True) == [
        "base",
        "loras",
    ]


def test_type_h100_downloads_diffusers_base_and_loras():
    assert resolve_types(model_type="h100", base=False, loras=False, all_flag=False) == [
        "base",
        "loras",
    ]


def test_type_backend_is_pruned_plus_postprocess():
    assert resolve_types(model_type="backend", base=False, loras=False, all_flag=False) == [
        "pruned",
        "postprocess",
    ]


def test_aliases_and_catalog_ids():
    catalog_ids = (
        "taomate_fl2va_3step_ema",
        "silveroxides_dareties_pruned_v1",
        "dasiwa_multistep_v2_r128_pruned",
    )
    assert resolve_types(
        model_type="silveroxides_dareties_pruned_v1",
        base=False,
        loras=False,
        all_flag=False,
        catalog_ids=catalog_ids,
    ) == ["silveroxides"]
    assert resolve_types(
        model_type="taomate_fl2va_3step_ema",
        base=False,
        loras=False,
        all_flag=False,
        catalog_ids=catalog_ids,
    ) == ["taomate"]
    assert resolve_types(
        model_type="dasiwa_multistep_r48_pruned",
        base=False,
        loras=False,
        all_flag=False,
        catalog_ids=("dasiwa_multistep_r48_pruned",),
    ) == ["dasiwa_multistep_r48_pruned"]
    assert resolve_types(
        model_type="dasiwa_multistep_v2_r128_pruned",
        base=False,
        loras=False,
        all_flag=False,
        catalog_ids=catalog_ids,
    ) == ["dasiwa_v2"]
    assert resolve_types(
        model_type="dasiwa_v2",
        base=False,
        loras=False,
        all_flag=False,
        catalog_ids=catalog_ids,
    ) == ["dasiwa_v2"]


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown model type: not_a_model"):
        resolve_types(model_type="not_a_model", base=False, loras=False, all_flag=False)


def test_dmad_type_and_original_checkpoint_download(monkeypatch, tmp_path):
    cfg = load_config()
    cfg.lora_dir = tmp_path
    assert LORA_GROUPS["dmad"] == ("dmad_4step_lora_critic",)
    assert resolve_types(model_type="dmad", base=False, loras=False, all_flag=False) == ["dmad"]
    assert resolve_types(model_type="dmad_4step_lora_critic", base=False, loras=False,
                         all_flag=False, catalog_ids=("dmad_4step_lora_critic",)) == ["dmad_4step_lora_critic"]
    captured = []

    def fake_download(url, destination, *, expected_sha256=None):
        captured.append((url, expected_sha256))
        destination.write_bytes(b"original")
        return destination

    monkeypatch.setattr("minimax_h3_fl2v.download.download_url_file", fake_download)
    spec = cfg.lora_by_id("dmad_4step_lora_critic")
    assert download_loras(cfg, ids=[spec.id]) == [tmp_path / spec.filename]
    assert captured == [(spec.download_url, spec.sha256)]


def test_dasiwa_group_contains_all_ranks():
    assert LORA_GROUPS["dasiwa"] == (
        "dasiwa_multistep_r48_pruned",
        "dasiwa_multistep_r96_pruned",
        "dasiwa_multistep_r144_pruned",
        "dasiwa_multistep_r512_pruned",
    )
    assert LORA_GROUPS["dasiwa_v2"] == ("dasiwa_multistep_v2_r128_pruned",)


@pytest.mark.parametrize("model_type,lora_id", [
    ("dmad_hyperflow", "dasiwa_dmad_hyperflow_4step_r256"),
    ("pdmd_dmad", "dasiwa_pdmd_dmad_4step_r256"),
])
def test_dmad_blend_type_downloads_only_selected_blend(monkeypatch, tmp_path, model_type, lora_id):
    cfg = load_config()
    cfg.lora_dir = tmp_path
    spec = cfg.lora_by_id(lora_id)
    assert LORA_GROUPS[model_type] == (spec.id,)
    assert LORA_GROUPS["dmad"] == ("dmad_4step_lora_critic",)
    assert resolve_types(model_type=model_type, base=False, loras=False, all_flag=False) == [model_type]
    assert resolve_types(model_type=spec.id, base=False, loras=False, all_flag=False,
                         catalog_ids=(spec.id,)) == [spec.id]
    captured = []

    def fake_download(url, destination, *, expected_sha256=None):
        captured.append((url, expected_sha256))
        destination.write_bytes(b"blend")
        return destination

    monkeypatch.setattr("minimax_h3_fl2v.download.download_url_file", fake_download)
    assert download_loras(cfg, ids=list(LORA_GROUPS[model_type])) == [tmp_path / spec.filename]
    assert captured == [(spec.download_url, spec.sha256)]


@pytest.mark.parametrize("url,authenticated", [
    ("https://civitai.red/api/download/models/3385247?fileId=3274094", True),
    ("https://civitai.com/api/download/models/3385247?fileId=3274094", True),
    ("https://huggingface.co/example/model", False),
    ("https://civitai.red.example.invalid/file", False),
    ("http://civitai.red/file", False),
])
def test_download_token_only_sent_to_https_civitai(monkeypatch, tmp_path, caplog, url, authenticated):
    token = "synthetic-test-credential"
    monkeypatch.setenv("CIVITAI_API_TOKEN", token)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("minimax_h3_fl2v.download.get_token", lambda: None)
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"model"

    def get(request_url, **kwargs):
        captured.update(kwargs)
        assert token not in request_url
        return Response()

    monkeypatch.setattr("minimax_h3_fl2v.download.requests.get", get)
    with caplog.at_level("INFO"):
        download_url_file(url, tmp_path / "model.safetensors")
    assert captured.get("headers", {}) == ({"Authorization": f"Bearer {token}"} if authenticated else {})
    assert token not in caplog.text


def test_download_url_file_skips_existing_checksum(tmp_path: Path):
    destination = tmp_path / "payload.bin"
    payload = b"ok"
    destination.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    result = download_url_file(
        "https://example.invalid/payload.bin",
        destination,
        expected_sha256=expected,
    )
    assert result == destination
    assert destination.read_bytes() == payload


def test_download_loras_uses_dasiwa_civitai_url(monkeypatch, tmp_path: Path):
    cfg = load_config()
    cfg.lora_dir = tmp_path
    captured: list[str] = []

    def fake_download(url, destination, *, expected_sha256=None, expected_size=None):
        captured.append(url)
        destination.write_bytes(b"lora")
        return destination

    monkeypatch.setattr("minimax_h3_fl2v.download.download_url_file", fake_download)
    paths = download_loras(cfg, ids=["dasiwa_multistep_r48_pruned"])
    spec = cfg.lora_by_id("dasiwa_multistep_r48_pruned")
    assert captured == [spec.download_url]
    assert paths == [tmp_path / spec.filename]
    assert spec.download_url.endswith("fileId=3201294")


def test_download_loras_uses_dasiwa_v2_civitai_url(monkeypatch, tmp_path: Path):
    cfg = load_config()
    cfg.lora_dir = tmp_path
    captured: list[str] = []

    def fake_download(url, destination, *, expected_sha256=None, expected_size=None):
        captured.append(url)
        destination.write_bytes(b"lora")
        return destination

    monkeypatch.setattr("minimax_h3_fl2v.download.download_url_file", fake_download)
    paths = download_loras(cfg, ids=["dasiwa_multistep_v2_r128_pruned"])
    spec = cfg.lora_by_id("dasiwa_multistep_v2_r128_pruned")
    assert captured == [spec.download_url]
    assert paths == [tmp_path / spec.filename]
    assert spec.download_url.endswith("fileId=3245274")
