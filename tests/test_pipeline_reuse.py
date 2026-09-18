from pathlib import Path

import torch
from safetensors.torch import save_file

from minimax_h3_fl2v.config import AppConfig, LoRASpec
from minimax_h3_fl2v.pipeline import MiniMaxH3Engine


class _Transformer:
    def named_parameters(self):
        return []

    def requires_grad_(self, value):
        return self


class _Pipe:
    def __init__(self):
        self.load_calls = []
        self.unload_calls = 0

    def unload_lora_weights(self):
        self.unload_calls += 1

    def load_lora_weights(self, path, **kwargs):
        self.load_calls.append((path, kwargs))


def _write_test_lora(path):
    save_file(
        {"transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(4, 8)},
        path,
        metadata={"ss_base_model_version": "minimax_h3"},
    )


def test_second_generation_does_not_reactivate_unchanged_adapter(tmp_path):
    lora = tmp_path / "turbo.safetensors"
    _write_test_lora(lora)
    spec = LoRASpec(id="turbo", name="Turbo", filename=lora.name)
    engine = MiniMaxH3Engine(AppConfig(lora_dir=tmp_path))
    engine.pipe = _Pipe()
    engine.transformer = _Transformer()
    activations = []
    engine._set_adapters_for_inference = lambda names, weights: (
        activations.append((tuple(names), tuple(weights))),
        setattr(engine, "_active_adapter_names", tuple(names)),
        setattr(engine, "_active_adapter_weights", tuple(weights)),
    )

    engine.apply_lora(spec, scale=1.0)
    engine.apply_lora(spec, scale=1.0)

    assert len(engine.pipe.load_calls) == 1
    assert activations == [(('turbo',), (1.0,))]


def test_scale_change_reactivates_without_reloading_weights(tmp_path):
    lora = tmp_path / "turbo.safetensors"
    _write_test_lora(lora)
    spec = LoRASpec(id="turbo", name="Turbo", filename=lora.name)
    engine = MiniMaxH3Engine(AppConfig(lora_dir=tmp_path))
    engine.pipe = _Pipe()
    engine.transformer = _Transformer()
    activations = []

    def activate(names, weights):
        activations.append((tuple(names), tuple(weights)))
        engine._active_adapter_names = tuple(names)
        engine._active_adapter_weights = tuple(weights)

    engine._set_adapters_for_inference = activate
    engine.apply_lora(spec, scale=1.0)
    engine.apply_lora(spec, scale=0.75)

    assert len(engine.pipe.load_calls) == 1
    assert activations == [(('turbo',), (1.0,)), (('turbo',), (0.75,))]


class _Hook:
    def __init__(self):
        self.model_id = "transformer"
        self.calls = 0

    def offload(self):
        self.calls += 1


class _Manager:
    def __init__(self, hooks):
        self.model_hooks = hooks


def test_release_cuda_memory_offloads_all_managed_components(monkeypatch):
    hooks = [_Hook(), _Hook()]
    engine = MiniMaxH3Engine(AppConfig())
    engine.components_manager = _Manager(hooks)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    engine._release_cuda_memory()
    assert [hook.calls for hook in hooks] == [1, 1]


def test_five_extra_loras_are_loaded_and_activated(tmp_path):
    files = []
    for index in range(5):
        path = tmp_path / f"style-{index}.safetensors"
        _write_test_lora(path)
        files.append(path)
    engine = MiniMaxH3Engine(AppConfig(lora_dir=tmp_path))
    engine.pipe = _Pipe()
    engine.transformer = _Transformer()
    activations = []

    def activate(names, weights):
        activations.append((tuple(names), tuple(weights)))
        engine._active_adapter_names = tuple(names)
        engine._active_adapter_weights = tuple(weights)

    engine._set_adapters_for_inference = activate
    base = LoRASpec(id="none", name="Base")
    engine.apply_lora(base, extra_loras=[(path, 0.5 + i * 0.1) for i, path in enumerate(files)])

    assert [kwargs["adapter_name"] for _, kwargs in engine.pipe.load_calls] == [
        "extra_1", "extra_2", "extra_3", "extra_4", "extra_5"
    ]
    assert activations[0][0] == ("extra_1", "extra_2", "extra_3", "extra_4", "extra_5")


def test_more_than_five_extra_loras_is_rejected(tmp_path):
    files = []
    for index in range(6):
        path = tmp_path / f"style-{index}.safetensors"
        path.write_bytes(b"weights")
        files.append(path)
    engine = MiniMaxH3Engine(AppConfig(lora_dir=tmp_path))
    engine.pipe = _Pipe()
    engine.transformer = _Transformer()
    try:
        engine.apply_lora(LoRASpec(id="none", name="Base"), extra_loras=[(path, 1.0) for path in files])
    except ValueError as exc:
        assert "five" in str(exc).lower()
    else:
        raise AssertionError("expected six-LoRA stack to be rejected")


def test_incompatible_lora_is_rejected_before_current_adapters_are_unloaded(tmp_path):
    path = tmp_path / "ref-style.safetensors"
    save_file(
        {"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.zeros(4, 8)},
        path,
        metadata={"ss_base_model_version": "minimax_h3_ref2va"},
    )
    engine = MiniMaxH3Engine(AppConfig(lora_dir=tmp_path, workflow="fl2va"))
    engine.pipe = _Pipe()
    engine.transformer = _Transformer()

    try:
        engine.apply_lora(LoRASpec(id="none", name="Base"), extra_loras=[(path, 1.0)])
    except ValueError as exc:
        assert "Ref2VA" in str(exc)
    else:
        raise AssertionError("expected incompatible workflow rejection")
    assert engine.pipe.unload_calls == 0
    assert engine.pipe.load_calls == []


def test_unpruned_modular_denoise_block_emits_detailed_steps():
    class Denoise:
        def progress_bar(self, iterable=None, total=None):
            return "original"

    class Blocks:
        sub_blocks = {"text_encoder": object(), "denoise.denoise": Denoise()}

    class Pipe:
        blocks = Blocks()

    engine = MiniMaxH3Engine(AppConfig())
    engine.pipe = Pipe()
    events = []
    block, original = engine._install_denoise_progress(
        lambda fraction, message: events.append((fraction, message)),
        0.0,
    )
    with block.progress_bar(total=4) as progress:
        for _ in range(4):
            progress.update()
    engine._restore_denoise_progress(block, original)

    assert [f"step {index}/4" in events[index - 1][1] for index in range(1, 5)] == [
        True,
        True,
        True,
        True,
    ]
    assert "decoding video/audio" in events[-1][1]
    assert block.progress_bar() == "original"