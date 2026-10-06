from pathlib import Path
from types import SimpleNamespace

import pytest

from minimax_h3_fl2v import cli, ui
from minimax_h3_fl2v.config import AppConfig, LoRASpec
from minimax_h3_fl2v.config import GenerationRequest, load_config
from minimax_h3_fl2v.comfy_backend import PrunedComfyBackend
from minimax_h3_fl2v.pipeline import MiniMaxH3Engine
from minimax_h3_fl2v.download import download_latent_upscaler, resolve_types, _download_providers
from minimax_h3_fl2v.assets import LATENT_UPSCALER_FILE, LATENT_UPSCALER_SHA256, LATENT_UPSCALER_SIZE


def _config(tmp_path):
    return AppConfig(
        default_lora_id="turbo",
        lora_dir=tmp_path,
        catalog=[LoRASpec(id="turbo", name="Turbo", backend="comfy_pruned")],
    )


def _engine(requests):
    def generate(request, **kwargs):
        requests.append(request)
        return SimpleNamespace(
            path=Path("output.mp4"), width=640, height=360, frames=125,
            duration=5.0, fps=23.976, nfe=8, elapsed_seconds=1.0,
            lora="turbo", mode="fl2va", seed=42, timings={},
        )

    return SimpleNamespace(ready=True, generate=generate)


def test_cli_latent_defaults():
    args = cli.build_parser().parse_args(["--prompt", "A quiet landscape"])
    assert (args.latent_upscale, args.latent_upscale_factor, args.latent_upscale_device) == (
        False, 1.5, "cpu",
    )


@pytest.mark.parametrize("pixel", [False, True])
def test_latent_workflow_preserves_audio_and_processing_order(pixel):
    graph = PrunedComfyBackend.build_workflow(
        prompt="test", width=864, height=480, frames=124, seed=42, nfe=4,
        loras=[("turbo.safetensors", 0.75)], latent_upscale=True,
        latent_upscale_factor=1.5, latent_upscale_device="cpu", upscale=pixel,
        face_restore=True, target_fps=60,
    )
    assert graph["41"]["inputs"]["latent"] == ["15", 0]
    assert graph["41"]["inputs"]["model_name"] == LATENT_UPSCALER_FILE
    assert graph["16"]["class_type"] == "VAEDecodeTiled"
    assert graph["16"]["inputs"]["samples"] == ["41", 0]
    assert graph["17"]["inputs"]["samples"] == ["15", 0]
    assert graph["38"]["inputs"]["image"] == ["16", 0]
    if pixel:
        assert graph["31"]["inputs"]["image"] == ["38", 0]


def test_disabled_latent_workflow_is_unchanged():
    graph = PrunedComfyBackend.build_workflow(prompt="test", width=864, height=480,
        frames=124, seed=42, nfe=4, loras=[])
    assert "41" not in graph
    assert graph["16"]["class_type"] == "VAEDecode"
    assert graph["16"]["inputs"]["samples"] == ["15", 0]


def test_diffusers_latent_upscale_is_rejected_before_loading():
    engine = MiniMaxH3Engine(AppConfig(catalog=[LoRASpec(id="none", name="Base")]))
    with pytest.raises(ValueError, match="not supported by the Diffusers"):
        engine.generate(GenerationRequest(prompt="test", lora_id="none", latent_upscale=True))


def test_missing_optional_latent_assets_fail_before_worker(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    backend = PrunedComfyBackend(cfg)
    backend.root = tmp_path
    monkeypatch.setattr(backend, "ensure_worker", lambda **kw: pytest.fail("worker started before asset validation"))
    with pytest.raises(FileNotFoundError, match="model type latent_upscaler"):
        backend.generate(GenerationRequest(prompt="test", latent_upscale=True), cfg.catalog[0])


def test_optional_download_uses_fp16_and_pinned_node_hashes(tmp_path, monkeypatch):
    calls = []

    def download(url, destination, *, expected_sha256=None, expected_size=None):
        calls.append((url, destination, expected_sha256, expected_size))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"test")
        return destination

    monkeypatch.setattr("minimax_h3_fl2v.download.download_url_file", download)
    paths = download_latent_upscaler(tmp_path)
    assert len(paths) == 3
    assert calls[0][1] == tmp_path / "models/latent_upscale_models" / LATENT_UPSCALER_FILE
    assert calls[0][2:] == (LATENT_UPSCALER_SHA256, LATENT_UPSCALER_SIZE)
    assert all("fp32" not in call[0] for call in calls)
    assert all(call[2] for call in calls)
    assert (tmp_path / "custom_nodes/minimax_h3_nodes/__init__.py").is_file()
    assert resolve_types(model_type="latent_upscaler", base=False, loras=False, all_flag=False) == ["latent_upscaler"]
    assert _download_providers(["latent_upscaler"], load_config()) == {"huggingface"}


def test_bridge_uses_pinned_network_not_version_dependent_node_api(monkeypatch):
    import importlib.util
    import sys
    import torch
    from minimax_h3_fl2v.config import ROOT

    path = ROOT / "comfy_nodes/minimax_h3_nodes/__init__.py"
    spec = importlib.util.spec_from_file_location("test_h3_bridge", path, submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    events = []

    class Model:
        def __call__(self, latent, *, scale, target_size, enable_chunking):
            events.append((scale, target_size, enable_chunking))
            return torch.nn.functional.interpolate(latent, size=target_size, mode="trilinear")

        def to(self, device):
            events.append(device)

    upstream = SimpleNamespace(load_model=lambda *args: Model(), MODEL_CACHE={"cached": 1},
        _make_norm_tensors=lambda *args: (torch.zeros(1,24,1,1,1), torch.ones(1,24,1,1,1)))
    monkeypatch.setitem(sys.modules, spec.name + ".upstream_latent_3d", upstream)
    monkeypatch.setattr(module, "upstream_latent_3d", upstream, raising=False)

    class AV:
        is_nested = True

        def unbind(self):
            return torch.ones(1,24,3,4,6), torch.zeros(1,32,2,10)

    output = module.LearnedLatentUpscale().upscale({"samples": AV()}, LATENT_UPSCALER_FILE, 1.5, "cpu")[0]
    assert output["samples"].shape == (1,24,3,6,8)
    assert events == [(1.5, (3,6,8), True), "cpu"]
    assert not upstream.MODEL_CACHE


@pytest.mark.parametrize("factor", ["1.5", "2"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("pixel_upscale", [False, True])
def test_cli_passes_independent_latent_settings(monkeypatch, tmp_path, factor, device, pixel_upscale):
    requests = []
    monkeypatch.setattr(cli, "enforce_offline_runtime", lambda: None)
    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))
    monkeypatch.setattr(cli, "MiniMaxH3Engine", lambda config: _engine(requests))
    argv = [
        "--prompt", "A quiet landscape", "--latent-upscale",
        "--latent-upscale-factor", factor, "--latent-upscale-device", device,
    ]
    if pixel_upscale:
        argv.append("--upscale")
    cli.main(argv)
    request, = requests
    assert request.latent_upscale is True
    assert request.latent_upscale_factor == float(factor)
    assert request.latent_upscale_device == device
    assert request.upscale is pixel_upscale
    assert request.upscale_factor == 2.0
    assert request.detailer_strength == (0.25 if pixel_upscale else 0.0)


@pytest.mark.parametrize("flag,value", [
    ("--latent-upscale-factor", "1.0"),
    ("--latent-upscale-factor", "3"),
    ("--latent-upscale-device", "mps"),
])
def test_cli_rejects_unsupported_latent_choices(flag, value):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--prompt", "A quiet landscape", flag, value])


def test_ui_reset_and_generation_wiring(monkeypatch, tmp_path):
    requests = []
    config = _config(tmp_path)
    monkeypatch.setattr(ui, "get_engine", lambda config: _engine(requests))
    demo = ui.build_app(config)
    try:
        generate = next(fn for fn in demo.fns.values() if fn.fn.__name__ == "generate")
        reset = next(fn for fn in demo.fns.values() if len(fn.outputs) == 36)
        values = reset.fn()
        assert len(values) == len(reset.outputs)
        assert [component.label for component in reset.outputs[30:33]] == [
            "Learned latent upscale (experimental)", "Latent upscale factor", "Latent upscale device",
        ]
        assert values[30:33] == (False, 1.5, "cpu")
        assert generate.inputs[30:33] == reset.outputs[30:33]
        reset_by_id = {component._id: value for component, value in zip(reset.outputs, values)}
        assert tuple(component.value for component in generate.inputs[30:33]) == values[30:33]
        inputs = [reset_by_id[component._id] for component in generate.inputs]
        inputs[0] = "A quiet landscape"
        inputs[30:33] = [True, 2.0, "cuda"]
        outputs = generate.fn(*inputs, progress=lambda *args, **kwargs: None)
        request, = requests
        assert (request.latent_upscale, request.latent_upscale_factor, request.latent_upscale_device) == (
            True, 2.0, "cuda",
        )
        assert request.upscale is False
        assert request.upscale_factor == 2.0
        assert request.detailer_strength == 0.0
        assert outputs[0] == "output.mp4"
        assert len(outputs) == len(generate.outputs) == 4
    finally:
        demo.close()