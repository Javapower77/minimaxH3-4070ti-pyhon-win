import os

import pytest

from minimax_h3_fl2v.config import load_config
from minimax_h3_fl2v.download import _download_credentials, _download_providers


@pytest.fixture
def credentials(monkeypatch):
    cfg = load_config()
    cfg.hf_token = None
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "CIVITAI_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("minimax_h3_fl2v.download.get_token", lambda: None)
    monkeypatch.setattr("minimax_h3_fl2v.download.sys.stdin.isatty", lambda: True)
    return cfg


def test_provider_selection(credentials):
    assert _download_providers(["pdmd_dmad"], credentials) == {"civitai"}
    assert _download_providers(["dmad"], credentials) == {"huggingface"}
    assert _download_providers(["pruned", "taomate"], credentials) == {"huggingface", "civitai"}
    assert _download_providers(["loras"], credentials, ["dasiwa_pdmd_dmad_4step_r256"]) == {"civitai"}


@pytest.mark.parametrize("failure", [False, True])
def test_prompts_once_per_provider_and_cleans_up(credentials, monkeypatch, failure):
    prompts = []

    def prompt(label):
        prompts.append(label)
        return "synthetic-secret"

    monkeypatch.setattr("minimax_h3_fl2v.download.getpass.getpass", prompt)
    try:
        with _download_credentials(["pruned", "taomate", "pdmd_dmad"], credentials):
            assert os.getenv("HF_TOKEN") == "synthetic-secret"
            assert os.getenv("CIVITAI_API_TOKEN") == "synthetic-secret"
            assert credentials.hf_token == "synthetic-secret"
            if failure:
                raise RuntimeError("download failed")
    except RuntimeError:
        assert failure
    assert len(prompts) == 2
    assert credentials.hf_token is None
    assert "HF_TOKEN" not in os.environ and "CIVITAI_API_TOKEN" not in os.environ


def test_existing_credentials_skip_prompt(credentials, monkeypatch):
    monkeypatch.setenv("CIVITAI_API_TOKEN", "existing-test-secret")
    monkeypatch.setattr("minimax_h3_fl2v.download.get_token", lambda: "cached-test-secret")
    monkeypatch.setattr("minimax_h3_fl2v.download.getpass.getpass", lambda *args: pytest.fail("unexpected prompt"))
    with _download_credentials(["pruned", "taomate"], credentials):
        assert credentials.hf_token == "cached-test-secret"
    assert credentials.hf_token is None
    assert os.getenv("CIVITAI_API_TOKEN") == "existing-test-secret"


@pytest.mark.parametrize("no_input,tty", [(True, True), (False, False)])
def test_noninteractive_never_prompts(credentials, monkeypatch, no_input, tty):
    monkeypatch.setattr("minimax_h3_fl2v.download.sys.stdin.isatty", lambda: tty)
    monkeypatch.setattr("minimax_h3_fl2v.download.getpass.getpass", lambda *args: pytest.fail("unexpected prompt"))
    with _download_credentials(["pruned", "taomate"], credentials, no_input=no_input):
        assert credentials.hf_token is None


def test_empty_token_allows_anonymous_download(credentials, monkeypatch):
    monkeypatch.setattr("minimax_h3_fl2v.download.getpass.getpass", lambda *args: "")
    with _download_credentials(["pdmd_dmad"], credentials):
        assert "CIVITAI_API_TOKEN" not in os.environ