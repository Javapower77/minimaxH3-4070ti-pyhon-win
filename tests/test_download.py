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
    catalog_ids = ("taomate_fl2va_3step_ema", "silveroxides_dareties_pruned_v1")
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


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown model type: not_a_model"):
        resolve_types(model_type="not_a_model", base=False, loras=False, all_flag=False)


def test_dasiwa_group_contains_all_ranks():
    assert LORA_GROUPS["dasiwa"] == (
        "dasiwa_multistep_r48_pruned",
        "dasiwa_multistep_r96_pruned",
        "dasiwa_multistep_r144_pruned",
        "dasiwa_multistep_r512_pruned",
    )


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
