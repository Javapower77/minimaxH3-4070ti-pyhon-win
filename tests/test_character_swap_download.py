"""Isolated character-swap downloads: no real weights, network, or GPU."""

import hashlib
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from minimax_h3_fl2v import assets, download


@pytest.fixture
def config(monkeypatch):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "CIVITAI_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(download, "get_token", lambda: None)
    monkeypatch.setattr(download.sys.stdin, "isatty", lambda: True)
    return SimpleNamespace(hf_token=None, catalog=[])


@pytest.fixture
def payload(monkeypatch):
    content = b"isolated character-swap adapter"
    monkeypatch.setattr(download, "CHARACTER_SWAP_LORA_SIZE", len(content))
    monkeypatch.setattr(download, "CHARACTER_SWAP_LORA_SHA256", hashlib.sha256(content).hexdigest())
    return content


def test_pinned_metadata():
    assert assets.CHARACTER_SWAP_LORA_REPO == "akatz-ai/MiniMax-H3-Character-Swap-LoRA"
    assert assets.CHARACTER_SWAP_LORA_REVISION == "62407e0cc8089c363abd9ce4b0b27662abb237af"
    assert assets.CHARACTER_SWAP_LORA_FILE == "h3_character_swap_pro4500_1000.safetensors"
    assert assets.CHARACTER_SWAP_LORA_SIZE == 155110320
    assert assets.CHARACTER_SWAP_LORA_SHA256 == (
        "4b2a3f420ae804c0aa3422761ff84dbd1bf52eef6900ffab6d2e66df63cb4e79"
    )
    assert assets.CHARACTER_SWAP_LORA_URL == (
        f"https://huggingface.co/{assets.CHARACTER_SWAP_LORA_REPO}/resolve/"
        f"{assets.CHARACTER_SWAP_LORA_REVISION}/{assets.CHARACTER_SWAP_LORA_FILE}"
    )


def test_optional_type_and_shared_base(config):
    assert "character_swap" in download.known_model_types(config)
    assert download.resolve_types(
        model_type="character_swap", base=False, loras=False, all_flag=False,
    ) == ["character_swap"]
    assert download._download_providers(["character_swap", "restore_base"], config) == {"huggingface"}


@pytest.mark.parametrize("kind,all_flag", [
    (None, False), (None, True), ("all", False), ("h100", False),
    ("12gb", False), ("backend", False), ("loras", False),
])
def test_excluded_from_default_sets(kind, all_flag):
    selected = download.resolve_types(
        model_type=kind, base=False, loras=False, all_flag=all_flag,
    )
    assert "character_swap" not in selected


def test_routing_downloads_only_adapter(config, monkeypatch, tmp_path):
    fetch = Mock(return_value=tmp_path / "adapter")
    monkeypatch.setattr(download, "download_character_swap", fetch)
    monkeypatch.setattr(download, "allow_online_for_download", lambda: None)
    for name in (
        "download_restore", "download_restore_base", "download_base_model", "download_loras",
        "download_pruned_models", "download_postprocess_models", "download_latent_upscaler",
        "download_ref2va_turbo",
        "hf_hub_download", "snapshot_download",
    ):
        monkeypatch.setattr(download, name, Mock(side_effect=AssertionError("unexpected asset")))
    download.run_downloads(["character_swap"], config=config, comfy_root=tmp_path)
    fetch.assert_called_once_with(tmp_path)


@pytest.mark.parametrize("custom_root", [True, False])
def test_worker_destination_and_integrity_arguments(monkeypatch, tmp_path, custom_root):
    monkeypatch.setattr(download, "DEFAULT_COMFY_ROOT", tmp_path)
    destination = tmp_path / "models" / "loras" / assets.CHARACTER_SWAP_LORA_FILE
    fetch = Mock(return_value=destination)
    monkeypatch.setattr(download, "download_url_file", fetch)
    assert download.download_character_swap(tmp_path if custom_root else None) == [destination]
    fetch.assert_called_once_with(
        assets.CHARACTER_SWAP_LORA_URL, destination,
        expected_sha256=assets.CHARACTER_SWAP_LORA_SHA256,
        expected_size=assets.CHARACTER_SWAP_LORA_SIZE,
    )


@pytest.mark.parametrize("state", ["valid", "wrong_size", "wrong_hash"])
def test_existing_assets_retained(monkeypatch, tmp_path, payload, state):
    destination = tmp_path / "models" / "loras" / assets.CHARACTER_SWAP_LORA_FILE
    destination.parent.mkdir(parents=True)
    content = payload if state == "valid" else (b"short" if state == "wrong_size" else b"x" * len(payload))
    destination.write_bytes(content)
    monkeypatch.setattr(download, "download_url_file", Mock(side_effect=AssertionError("replacement")))
    if state == "valid":
        assert download.download_character_swap(tmp_path) == [destination]
    else:
        with pytest.raises(RuntimeError, match="incorrect size|SHA-256 mismatch"):
            download.download_character_swap(tmp_path)
    assert destination.read_bytes() == content


@pytest.mark.parametrize("state", ["valid", "wrong_size", "wrong_hash"])
def test_streamed_download_verifies_before_install(config, monkeypatch, tmp_path, payload, state):
    content = payload if state == "valid" else (b"short" if state == "wrong_size" else b"x" * len(payload))
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.iter_content.return_value = [content]
    monkeypatch.setattr(download.requests, "get", Mock(return_value=response))
    destination = tmp_path / "models" / "loras" / assets.CHARACTER_SWAP_LORA_FILE
    if state == "valid":
        assert download.download_character_swap(tmp_path) == [destination]
        assert destination.read_bytes() == payload
    else:
        with pytest.raises(RuntimeError, match="size mismatch|SHA-256 mismatch"):
            download.download_character_swap(tmp_path)
        assert not destination.exists()
    assert not destination.with_suffix(".safetensors.part").exists()


@pytest.mark.parametrize("source", ["prompt", "config", "env", "legacy_env", "cache", "anonymous"])
@pytest.mark.parametrize("failure", [False, True])
def test_credentials_are_hf_only_masked_and_temporary(
    config, monkeypatch, tmp_path, payload, caplog, capsys, source, failure,
):
    secret = "synthetic-character-swap-secret"
    if source == "config":
        config.hf_token = secret
    elif source in {"env", "legacy_env"}:
        monkeypatch.setenv("HF_TOKEN" if source == "env" else "HUGGING_FACE_HUB_TOKEN", secret)
    elif source == "cache":
        monkeypatch.setattr(download, "get_token", lambda: secret)
    monkeypatch.setenv("CIVITAI_API_TOKEN", "unrelated-civitai-secret")
    original_hf = config.hf_token
    original_env = {key: os.getenv(key) for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "CIVITAI_API_TOKEN")}
    prompt = Mock(return_value=secret if source == "prompt" else "")
    monkeypatch.setattr(download.getpass, "getpass", prompt)
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.iter_content.return_value = [payload]
    if failure:
        response.raise_for_status.side_effect = RuntimeError("synthetic download failure")
    get = Mock(return_value=response)
    monkeypatch.setattr(download.requests, "get", get)

    def run():
        with download._download_credentials(["character_swap"], config):
            download.download_character_swap(tmp_path)

    with caplog.at_level("INFO"):
        if failure:
            with pytest.raises(RuntimeError, match="synthetic download failure"):
                run()
        else:
            run()
    assert get.call_args.args == (assets.CHARACTER_SWAP_LORA_URL,)
    assert get.call_args.kwargs.get("headers", {}) == (
        {} if source == "anonymous" else {"Authorization": f"Bearer {secret}"}
    )
    if source in {"prompt", "anonymous"}:
        prompt.assert_called_once()
        assert "Hugging Face" in prompt.call_args.args[0]
        assert "hidden" in prompt.call_args.args[0]
    else:
        prompt.assert_not_called()
    assert config.hf_token == original_hf
    assert {key: os.getenv(key) for key in original_env} == original_env
    assert secret not in caplog.text + capsys.readouterr().out


@pytest.mark.parametrize("no_input,tty", [(True, True), (False, False)])
def test_noninteractive_character_swap_credentials(config, monkeypatch, no_input, tty):
    monkeypatch.setattr(download.sys.stdin, "isatty", lambda: tty)
    monkeypatch.setattr(download.getpass, "getpass", Mock(side_effect=AssertionError("unexpected prompt")))
    with download._download_credentials(["character_swap"], config, no_input=no_input):
        assert config.hf_token is None
        assert "HF_TOKEN" not in os.environ


def test_cli_routes_only_character_swap(config, monkeypatch, tmp_path):
    monkeypatch.setattr(download, "load_config", lambda: config)
    run = Mock()
    monkeypatch.setattr(download, "run_downloads", run)
    download.main(["--type", "character_swap", "--comfy-root", str(tmp_path), "--no-input"])
    run.assert_called_once_with(["character_swap"], lora_ids=None, config=config, comfy_root=tmp_path)