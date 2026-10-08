"""Opt-in Ref2VA turbo downloads: no real weights, network, or workers."""

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
    content = b"isolated Ref2VA turbo adapter"
    monkeypatch.setattr(download, "REF2VA_TURBO_SIZE", len(content))
    monkeypatch.setattr(download, "REF2VA_TURBO_SHA256", hashlib.sha256(content).hexdigest())
    return content


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    for name in ("hf_hub_download", "snapshot_download"):
        monkeypatch.setattr(download, name, Mock(side_effect=AssertionError("unexpected network")))
    monkeypatch.setattr(download.requests, "get", Mock(side_effect=AssertionError("unexpected network")))
    monkeypatch.setattr(download.urllib.request, "urlopen", Mock(side_effect=AssertionError("unexpected network")))


def test_pinned_metadata():
    assert assets.REF2VA_TURBO_REPO == "Kijai/MiniMax-H3-experimental"
    assert assets.REF2VA_TURBO_REVISION == "d8023be02fefbb3633b0cd335c3879f91177299d"
    assert assets.REF2VA_TURBO_FILE == "MiniMax-H3-Ref2VA-Acc-8Step_pruned_comfy.safetensors"
    assert assets.REF2VA_TURBO_SIZE == 1725921392
    assert assets.REF2VA_TURBO_SHA256 == (
        "6f18e1c2eccb14b37322607730f26b16bf1169b56cd098ea006cffaec43d1e39"
    )
    assert assets.REF2VA_TURBO_URL == (
        "https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/"
        "d8023be02fefbb3633b0cd335c3879f91177299d/"
        "MiniMax-H3-Ref2VA-Acc-8Step_pruned_comfy.safetensors"
    )


def test_optional_type_and_provider(config):
    assert "ref2va_turbo" in download.known_model_types(config)
    assert download.resolve_types(
        model_type="ref2va_turbo", base=False, loras=False, all_flag=False,
    ) == ["ref2va_turbo"]
    assert download._download_providers(["ref2va_turbo"], config) == {"huggingface"}


@pytest.mark.parametrize("kind,all_flag,expected", [
    (None, False, ["pruned", "postprocess", "taomate"]),
    (None, True, ["base", "loras"]),
    ("all", False, ["pruned", "postprocess", "loras"]),
    ("h100", False, ["base", "loras"]),
    ("12gb", False, ["pruned", "postprocess", "taomate"]),
    ("backend", False, ["pruned", "postprocess"]),
    ("loras", False, ["loras"]),
    ("restore", False, ["restore"]),
    ("restore_base", False, ["restore_base"]),
    ("character_swap", False, ["character_swap"]),
])
def test_off_and_bulk_sets_unchanged(config, monkeypatch, tmp_path, kind, all_flag, expected):
    selected = download.resolve_types(
        model_type=kind, base=False, loras=False, all_flag=all_flag,
    )
    assert selected == expected
    monkeypatch.setattr(download, "allow_online_for_download", lambda: None)
    for name in (
        "download_restore", "download_restore_base", "download_character_swap",
        "download_base_model", "download_loras", "download_pruned_models",
        "download_postprocess_models", "download_latent_upscaler",
    ):
        monkeypatch.setattr(download, name, Mock())
    turbo = Mock(side_effect=AssertionError("turbo must remain opt-in"))
    monkeypatch.setattr(download, "download_ref2va_turbo", turbo)
    download.run_downloads(selected, config=config, comfy_root=tmp_path)
    turbo.assert_not_called()
    assert not (tmp_path / "models").exists()


def test_routing_downloads_only_turbo(config, monkeypatch, tmp_path):
    fetch = Mock(return_value=[tmp_path / "adapter"])
    monkeypatch.setattr(download, "download_ref2va_turbo", fetch)
    monkeypatch.setattr(download, "allow_online_for_download", lambda: None)
    for name in (
        "download_restore", "download_restore_base", "download_character_swap",
        "download_base_model", "download_loras", "download_pruned_models",
        "download_postprocess_models", "download_latent_upscaler",
    ):
        monkeypatch.setattr(download, name, Mock(side_effect=AssertionError("unexpected asset")))
    download.run_downloads(["ref2va_turbo"], config=config, comfy_root=tmp_path)
    fetch.assert_called_once_with(tmp_path)


@pytest.mark.parametrize("custom_root", [True, False])
def test_worker_destination_and_integrity_arguments(monkeypatch, tmp_path, custom_root):
    monkeypatch.setattr(download, "DEFAULT_COMFY_ROOT", tmp_path)
    destination = tmp_path / "models" / "loras" / assets.REF2VA_TURBO_FILE
    fetch = Mock(return_value=destination)
    monkeypatch.setattr(download, "download_url_file", fetch)
    assert download.download_ref2va_turbo(tmp_path if custom_root else None) == [destination]
    fetch.assert_called_once_with(
        assets.REF2VA_TURBO_URL, destination,
        expected_sha256=assets.REF2VA_TURBO_SHA256,
        expected_size=assets.REF2VA_TURBO_SIZE,
    )


@pytest.mark.parametrize("state", ["valid", "wrong_size", "wrong_hash"])
def test_existing_assets_retained(monkeypatch, tmp_path, payload, state):
    destination = tmp_path / "models" / "loras" / assets.REF2VA_TURBO_FILE
    destination.parent.mkdir(parents=True)
    content = payload if state == "valid" else (b"short" if state == "wrong_size" else b"x" * len(payload))
    destination.write_bytes(content)
    monkeypatch.setattr(download, "download_url_file", Mock(side_effect=AssertionError("replacement")))
    if state == "valid":
        assert download.download_ref2va_turbo(tmp_path) == [destination]
    else:
        with pytest.raises(RuntimeError, match="incorrect size|SHA-256 mismatch"):
            download.download_ref2va_turbo(tmp_path)
    assert destination.read_bytes() == content


@pytest.mark.parametrize("state", ["valid", "wrong_size", "wrong_hash"])
def test_streamed_download_verifies_before_install(config, monkeypatch, tmp_path, payload, state):
    content = payload if state == "valid" else (b"short" if state == "wrong_size" else b"x" * len(payload))
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.iter_content.return_value = [content[:5], b"", content[5:]]
    monkeypatch.setattr(download.requests, "get", Mock(return_value=response))
    destination = tmp_path / "models" / "loras" / assets.REF2VA_TURBO_FILE
    if state == "valid":
        assert download.download_ref2va_turbo(tmp_path) == [destination]
        assert destination.read_bytes() == payload
    else:
        with pytest.raises(RuntimeError, match="size mismatch|SHA-256 mismatch"):
            download.download_ref2va_turbo(tmp_path)
        assert not destination.exists()
    assert not destination.with_suffix(".safetensors.part").exists()


@pytest.mark.parametrize("source", ["prompt", "config", "env", "legacy_env", "cache", "anonymous"])
@pytest.mark.parametrize("failure", [False, True])
def test_credentials_are_hf_only_masked_and_temporary(
    config, monkeypatch, tmp_path, payload, caplog, capsys, source, failure,
):
    secret = "synthetic-ref2va-turbo-secret"
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
        with download._download_credentials(["ref2va_turbo"], config):
            download.download_ref2va_turbo(tmp_path)

    with caplog.at_level("INFO"):
        if failure:
            with pytest.raises(RuntimeError, match="synthetic download failure"):
                run()
        else:
            run()
    assert get.call_args.args == (assets.REF2VA_TURBO_URL,)
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
    captured = capsys.readouterr()
    assert secret not in caplog.text + captured.out + captured.err


@pytest.mark.parametrize("no_input,tty", [(True, True), (False, False)])
def test_noninteractive_credentials(config, monkeypatch, no_input, tty):
    monkeypatch.setattr(download.sys.stdin, "isatty", lambda: tty)
    monkeypatch.setattr(download.getpass, "getpass", Mock(side_effect=AssertionError("unexpected prompt")))
    with download._download_credentials(["ref2va_turbo"], config, no_input=no_input):
        assert config.hf_token is None
        assert "HF_TOKEN" not in os.environ


def test_cli_routes_only_ref2va_turbo(config, monkeypatch, tmp_path):
    monkeypatch.setattr(download, "load_config", lambda: config)
    run = Mock()
    monkeypatch.setattr(download, "run_downloads", run)
    monkeypatch.setattr(download.getpass, "getpass", Mock(side_effect=AssertionError("unexpected prompt")))
    download.main(["--type", "ref2va_turbo", "--comfy-root", str(tmp_path), "--no-input"])
    run.assert_called_once_with(["ref2va_turbo"], lora_ids=None, config=config, comfy_root=tmp_path)


def test_cli_help_exposes_optional_type(capsys):
    with pytest.raises(SystemExit) as exc:
        download.main(["--help"])
    assert exc.value.code == 0
    assert "ref2va_turbo" in capsys.readouterr().out