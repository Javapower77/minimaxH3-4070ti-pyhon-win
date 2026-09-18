import os
from pathlib import Path

from minimax_h3_fl2v.config import load_config
from minimax_h3_fl2v.offline import (
    HUB_ATTENTION_BACKENDS,
    disable_safety_guards,
    enforce_offline_runtime,
    local_attention_backend,
    pin_component_specs_to_local,
    require_local_snapshot,
    snapshot_ready,
    strip_hub_kwargs,
)
from minimax_h3_fl2v.pipeline import ATTENTION_BACKEND_FALLBACKS
from minimax_h3_fl2v.prompts import expand_prompt


class _FakePipe:
    safety_checker = object()
    watermarker = object()
    requires_safety_checker = True


def test_enforce_offline_runtime_pins_env():
    os.environ["GRADIO_SHARE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "0"
    enforce_offline_runtime()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["DIFFUSERS_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"
    assert os.environ["GRADIO_SHARE"] == "0"


def test_config_never_shares_and_never_uses_hub_attention():
    old_share = os.environ.get("GRADIO_SHARE")
    old_attn = os.environ.get("MINIMAX_H3_ATTENTION_BACKEND")
    os.environ["GRADIO_SHARE"] = "1"
    os.environ["MINIMAX_H3_ATTENTION_BACKEND"] = "_flash_3_hub"
    try:
        cfg = load_config()
        assert cfg.share is False
        assert cfg.attention_backend == "_flash_3"
        assert "_flash_3_hub" not in ATTENTION_BACKEND_FALLBACKS
        assert not any(name in HUB_ATTENTION_BACKENDS for name in ATTENTION_BACKEND_FALLBACKS)
    finally:
        if old_share is None:
            os.environ.pop("GRADIO_SHARE", None)
        else:
            os.environ["GRADIO_SHARE"] = old_share
        if old_attn is None:
            os.environ.pop("MINIMAX_H3_ATTENTION_BACKEND", None)
        else:
            os.environ["MINIMAX_H3_ATTENTION_BACKEND"] = old_attn


def test_missing_snapshot_does_not_fall_back_to_hub(tmp_path):
    empty = tmp_path / "MiniMax-H3"
    empty.mkdir()
    assert snapshot_ready(empty) is False
    try:
        require_local_snapshot(empty)
    except FileNotFoundError as exc:
        assert "offline-only" in str(exc)
        assert "huggingface.co" not in str(exc).lower()
    else:
        raise AssertionError("expected missing local snapshot to fail closed")


def test_pretrained_path_requires_local_files():
    cfg = load_config()
    if snapshot_ready(cfg.local_dir):
        assert Path(cfg.pretrained_path) == cfg.local_dir.resolve()
        return
    try:
        _ = cfg.pretrained_path
    except FileNotFoundError:
        return
    raise AssertionError("pretrained_path must not return a Hub repo id")


class _Spec:
    def __init__(self, path, *, method="from_pretrained", repo=None):
        self.pretrained_model_name_or_path = path
        self.default_creation_method = method
        self.repo = repo


class _Pipe:
    def __init__(self, specs):
        self._component_specs = specs
        self._pretrained_model_name_or_path = "MiniMaxAI/MiniMax-H3"


def test_pin_component_specs_rewrites_hub_ids(tmp_path):
    local = tmp_path / "MiniMax-H3"
    local.mkdir()
    pipe = _Pipe(
        {
            "scheduler": _Spec("MiniMaxAI/MiniMax-H3", repo="MiniMaxAI/MiniMax-H3"),
            "tokenizer": _Spec(None),
            "image_processor": _Spec(None, method="from_config"),
            "vae": _Spec(str(local)),
        }
    )
    rewritten = pin_component_specs_to_local(pipe, local)
    assert rewritten == 2
    assert pipe._component_specs["scheduler"].pretrained_model_name_or_path == str(local.resolve())
    assert pipe._component_specs["scheduler"].repo == str(local.resolve())
    assert pipe._component_specs["tokenizer"].pretrained_model_name_or_path == str(local.resolve())
    assert pipe._component_specs["image_processor"].pretrained_model_name_or_path is None
    assert pipe._component_specs["vae"].pretrained_model_name_or_path == str(local)
    assert pipe._pretrained_model_name_or_path == str(local.resolve())


def test_pin_then_load_scheduler_from_local_snapshot():
    cfg = load_config()
    if not snapshot_ready(cfg.local_dir):
        return
    from diffusers import ModularPipeline

    enforce_offline_runtime()
    root = str(require_local_snapshot(cfg.local_dir))
    pipe = ModularPipeline.from_pretrained(
        root, **strip_hub_kwargs({"workflow": cfg.workflow, "local_files_only": True})
    )
    pin_component_specs_to_local(pipe, root)
    pipe.load_components(
        names=["scheduler", "audio_scheduler", "tokenizer"],
        local_files_only=True,
        pretrained_model_name_or_path=root,
    )
    assert pipe.scheduler is not None
    assert pipe.audio_scheduler is not None
    assert pipe.tokenizer is not None
    assert "MiniMaxAI/" not in str(pipe._component_specs["scheduler"].pretrained_model_name_or_path)


def test_strip_hub_kwargs_drops_tokens():
    out = strip_hub_kwargs({"token": "hf_secret", "revision": "main", "workflow": "fl2va"})
    assert out["local_files_only"] is True
    assert "token" not in out
    assert "revision" not in out
    assert out["workflow"] == "fl2va"


def test_hub_attention_backends_rewritten():
    assert local_attention_backend("_flash_3_hub") == "_flash_3"
    assert local_attention_backend("_flash_3") == "_flash_3"


def test_safety_guards_stripped():
    pipe = _FakePipe()
    disable_safety_guards(pipe)
    assert pipe.safety_checker is None
    assert pipe.watermarker is None
    assert pipe.requires_safety_checker is False


def test_prompts_are_not_censored():
    text = "A graphic fight with blood, weapons, and nudity, no cuts."
    out = expand_prompt(text)
    assert "graphic fight" in out.lower()
    assert "blood" in out.lower()
    assert "nudity" in out.lower()
    assert "cannot" not in out.lower()
    assert "safety" not in out.lower()
