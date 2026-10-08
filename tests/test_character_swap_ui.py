"""UI/engine contract tests for the independently implemented character-swap backend."""

from dataclasses import dataclass
from pathlib import Path
import inspect
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from minimax_h3_fl2v import ui
from minimax_h3_fl2v.comfy_backend import PrunedComfyBackend
from minimax_h3_fl2v.config import AppConfig, GenerationRequest, LoRASpec
from minimax_h3_fl2v.pipeline import MiniMaxH3Engine


@pytest.fixture
def contract(monkeypatch):
    """Use only the public contract; do not require the upcoming worker module."""
    module = ModuleType("minimax_h3_fl2v.character_swap")
    module.CHARACTER_SWAP_PROMPT = "Replace the character using the reference image."

    @dataclass
    class CharacterSwapRequest:
        source_video: Path
        reference_image: Path
        prompt: str = module.CHARACTER_SWAP_PROMPT
        steps: int = 20
        strength: float = 1.0
        seed: int = 42
        use_turbo: bool = False

        @property
        def effective_steps(self):
            return 8 if self.use_turbo else self.steps

    class CharacterSwapFailure(RuntimeError):
        def __init__(self, original_path, original_image_path):
            super().__init__("worker failed; originals retained")
            self.original_path = original_path
            self.original_image_path = original_image_path

    module.CharacterSwapRequest = CharacterSwapRequest
    module.CharacterSwapFailure = CharacterSwapFailure
    module.CharacterSwapBackend = Mock()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.fixture
def config(tmp_path):
    return AppConfig(
        output_dir=tmp_path / "outputs", upload_dir=tmp_path / "uploads",
        lora_dir=tmp_path / "loras", default_lora_id="turbo",
        catalog=[LoRASpec(id="turbo", name="Turbo", backend="comfy_pruned")],
    )


@pytest.fixture
def app(config, contract, monkeypatch, tmp_path):
    result = SimpleNamespace(
        original_path=tmp_path / "persistent_original.mp4",
        original_image_path=tmp_path / "persistent_reference.png",
        output_path=tmp_path / "swapped.mkv", message="Source geometry preserved.",
    )
    engine = SimpleNamespace(
        ready=True, swap_character=Mock(return_value=result),
        generate=Mock(), enhance_restoration=Mock(), load=Mock(),
    )
    monkeypatch.setattr(ui, "get_engine", lambda config: engine)
    demo = ui.build_app(config)
    swap = next(fn for fn in demo.fns.values() if fn.fn.__name__ == "swap_character")
    yield SimpleNamespace(demo=demo, swap=swap, engine=engine, result=result)
    demo.close()


def invoke(app, **overrides):
    values = dict(reference_image="reference.png", source_video="source.mp4",
                  prompt="Editable prompt", steps=20, strength=1.0, seed=42,
                  use_turbo=False, progress=Mock())
    values.update(overrides)
    return app.swap.fn(**values)


def test_independent_defaults_and_no_generation_or_restoration_controls(app, contract):
    assert [component.value for component in app.swap.inputs] == [
        None, None, contract.CHARACTER_SWAP_PROMPT, 20, 1.0, 42, False,
    ]
    assert isinstance(app.swap.inputs[0], ui.gr.File)
    assert app.swap.inputs[0].type == "filepath"
    assert isinstance(app.swap.inputs[1], ui.gr.File)
    assert app.swap.inputs[1].type == "filepath"
    assert app.swap.inputs[2].interactive is True
    assert app.swap.inputs[3].interactive is True
    assert app.swap.inputs[4].interactive is True
    assert isinstance(app.swap.inputs[6], ui.gr.Checkbox)
    signature = inspect.signature(app.swap.fn)
    assert signature.parameters["use_turbo"].default is False
    assert "normalize_input" not in signature.parameters
    assert "output_fps" not in signature.parameters
    assert all(isinstance(component, ui.gr.File) and component.interactive is False
               for component in app.swap.outputs[:3])
    for name in ("generate", "enhance_restoration"):
        other = next(fn for fn in app.demo.fns.values() if fn.fn.__name__ == name)
        assert not set(app.swap.inputs) & set(other.inputs)
        assert not set(app.swap.outputs) & set(other.outputs)
    assert not any("canvas" in component.label.lower() or "lora" in component.label.lower()
                   for component in app.swap.inputs)
    assert not any("normalize" in component.label.lower() or "frame rate" in component.label.lower()
                   for component in app.swap.inputs)
    app.engine.swap_character.assert_not_called()


def test_exactly_one_dedicated_progress_target(app):
    event = next(event for event in app.demo.config["dependencies"]
                 if event["id"] == app.swap._id)
    status = app.swap.outputs[-1]
    assert isinstance(status, ui.gr.Textbox)
    assert status.elem_id == "character-swap-progress"
    assert event["show_progress"] == "full"
    assert event["show_progress_on"] == [status._id]
    assert status not in app.swap.inputs
    assert all(output._id not in event["show_progress_on"] for output in app.swap.outputs[:-1])


@pytest.mark.parametrize("use_turbo", [False, True])
def test_request_progress_and_persistent_downloads(app, contract, use_turbo):
    progress = Mock()
    outputs = invoke(app, prompt="Keep source composition", steps=27, strength=0.75,
                     seed=123, use_turbo=use_turbo, progress=progress)
    request = app.engine.swap_character.call_args.args[0]
    assert request == contract.CharacterSwapRequest(
        Path("source.mp4"), Path("reference.png"), prompt="Keep source composition",
        steps=27, strength=0.75, seed=123, use_turbo=use_turbo,
    )
    assert isinstance(request.use_turbo, bool)
    assert request.effective_steps == (8 if use_turbo else 27)
    assert outputs[:3] == (str(app.result.output_path), str(app.result.original_path),
                           str(app.result.original_image_path))
    assert outputs[3] == app.result.message
    app.engine.swap_character.call_args.kwargs["progress"](0.5, "Sampling step 10/20")
    progress.assert_called_once_with(0.5, desc="Sampling step 10/20")
    app.engine.generate.assert_not_called()
    app.engine.enhance_restoration.assert_not_called()
    app.engine.load.assert_not_called()


def test_turbo_toggle_updates_only_steps_and_preserves_editable_character_strength(app):
    steps, strength, turbo = app.swap.inputs[3], app.swap.inputs[4], app.swap.inputs[6]
    toggle = next(fn for fn in app.demo.fns.values() if fn.inputs == [turbo])
    assert toggle.outputs == [steps]
    for enabled in (True, False, True, False):
        assert toggle.fn(enabled) == ui.gr.update(
            value=8 if enabled else 20, interactive=not enabled,
        )
    assert strength.value == 1.0
    assert strength.interactive is True
    app.engine.swap_character.assert_not_called()


def test_restoration_normalization_controls_remain_unchanged(app):
    restoration = next(fn for fn in app.demo.fns.values()
                       if fn.fn.__name__ == "enhance_restoration")
    assert restoration.inputs[5].value is False
    assert isinstance(restoration.inputs[5], ui.gr.Checkbox)
    assert restoration.inputs[6].value == "native"
    assert isinstance(restoration.inputs[6], ui.gr.Radio)
    assert [value for _, value in restoration.inputs[6].choices] == ["native", "source"]
    signature = inspect.signature(restoration.fn)
    assert "normalize_input" in signature.parameters
    assert "output_fps" in signature.parameters


@pytest.mark.parametrize("missing", ["reference_image", "source_video"])
def test_missing_source_retains_previous_result(app, missing):
    outputs = invoke(app, **{missing: None})
    assert outputs[:3] == (ui.gr.update(),) * 3
    assert "Select a reference image and a source video" in outputs[3]
    app.engine.swap_character.assert_not_called()


def test_backend_failure_retains_result_and_returns_both_originals(app, contract):
    failure = contract.CharacterSwapFailure(Path("durable.mp4"), Path("durable.png"))
    app.engine.swap_character.side_effect = failure
    outputs = invoke(app)
    assert outputs[:3] == (ui.gr.update(), "durable.mp4", "durable.png")
    assert "worker failed" in outputs[3]
    assert outputs[4] == "Character swap failed; originals retained."


@pytest.mark.parametrize("error", [ValueError("invalid settings"), RuntimeError("worker offline")])
def test_request_rejection_retains_all_existing_downloads(app, error):
    app.engine.swap_character.side_effect = error
    outputs = invoke(app)
    assert outputs[:3] == (ui.gr.update(),) * 3
    assert str(error) in outputs[3]
    assert "source.mp4" in outputs[3] and "reference.png" in outputs[3]


def test_missing_backend_does_not_break_existing_studio(app, monkeypatch):
    monkeypatch.setitem(sys.modules, "minimax_h3_fl2v.character_swap", None)
    outputs = invoke(app)
    assert outputs[:3] == (ui.gr.update(),) * 3
    assert "backend unavailable" in outputs[3]
    demo = ui.build_app()
    try:
        swap = next(fn for fn in demo.fns.values() if fn.fn.__name__ == "swap_character")
        assert swap.inputs[2].value == ""
    finally:
        demo.close()


def test_engine_shared_lock_serializes_swap_with_generation(config, contract, monkeypatch):
    engine = MiniMaxH3Engine(config)
    events, failures = [], []
    entered, proceed = threading.Event(), threading.Event()

    def release():
        assert engine._lock.locked()
        events.append("release")

    def swap(request, progress=None):
        assert engine._lock.locked()
        assert engine.status == "swapping character with Ref2VA backend"
        events.append("swap")
        entered.set()
        assert proceed.wait(5)
        return "swapped"

    def generate(self, request, spec, progress_callback=None):
        assert engine._lock.locked()
        events.append("generation")
        return "generated"

    monkeypatch.setattr(engine, "_release_cuda_memory", release)
    contract.CharacterSwapBackend.return_value.swap.side_effect = swap
    monkeypatch.setattr(PrunedComfyBackend, "generate", generate)

    def run(action):
        try:
            action()
        except BaseException as exc:
            failures.append(exc)

    request = contract.CharacterSwapRequest(Path("source.mp4"), Path("reference.png"))
    first = threading.Thread(target=run, args=(lambda: engine.swap_character(request),))
    second = threading.Thread(target=run, args=(lambda: engine.generate(
        GenerationRequest(prompt="Landscape", lora_id="turbo")),))
    first.start()
    try:
        assert entered.wait(5)
        second.start()
        assert events == ["release", "swap"]
    finally:
        proceed.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert not failures
    assert events == ["release", "swap", "release", "generation"]
    assert engine.status == "ready"
    contract.CharacterSwapBackend.assert_called_once_with(config)


@pytest.mark.parametrize("fails", [False, True])
def test_engine_progress_failure_status_and_lock_release(config, contract, monkeypatch, fails):
    engine = MiniMaxH3Engine(config)
    request = contract.CharacterSwapRequest(Path("source.mp4"), Path("reference.png"))
    progress, release = Mock(), Mock()
    monkeypatch.setattr(engine, "_release_cuda_memory", release)
    worker = contract.CharacterSwapBackend.return_value.swap
    if fails:
        failure = contract.CharacterSwapFailure(Path("durable.mp4"), Path("durable.png"))
        worker.side_effect = failure
        with pytest.raises(contract.CharacterSwapFailure) as caught:
            engine.swap_character(request, progress=progress)
        assert caught.value is failure
        assert engine.status == "Character swap failed; originals retained"
    else:
        worker.return_value = "result"
        assert engine.swap_character(request, progress=progress) == "result"
        assert engine.status == "ready"
    worker.assert_called_once_with(request, progress=progress)
    release.assert_called_once_with()
    assert not engine._lock.locked()
    assert engine.ready is False