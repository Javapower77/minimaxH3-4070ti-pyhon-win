import pytest
import json
import os
from pathlib import Path

from minimax_h3_fl2v.comfy_backend import PrunedComfyBackend
from minimax_h3_fl2v.config import AppConfig, GenerationRequest, LoRASpec
from minimax_h3_fl2v.pipeline import MiniMaxH3Engine, _ProgressBridge


def test_pruned_workflow_contains_first_last_and_stacked_loras():
    graph = PrunedComfyBackend.build_workflow(
        prompt="test",
        width=1344,
        height=768,
        frames=124,
        seed=42,
        nfe=8,
        loras=[("rank144.safetensors", 1.0), ("style.safetensors", 0.4)],
        first_image="first.png",
        last_image="last.png",
    )
    assert graph["1"]["inputs"]["unet_name"] == "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
    assert graph["21"]["class_type"] == "LoraLoaderModelOnly"
    assert graph["22"]["inputs"]["model"] == ["21", 0]
    assert graph["9"]["inputs"]["first_frame"] == ["7", 0]
    assert graph["9"]["inputs"]["last_frame"] == ["8", 0]
    assert graph["12"]["inputs"]["steps"] == 8
    assert graph["11"]["inputs"]["sampler_name"] == "euler"


def test_pruned_workflow_accepts_configured_models():
    graph = PrunedComfyBackend.build_workflow(
        prompt="test",
        width=864,
        height=480,
        frames=124,
        seed=1,
        nfe=4,
        loras=[("turbo.safetensors", 0.75)],
        model="custom-fp8.safetensors",
        text_encoder="custom-nvfp4.safetensors",
    )
    assert graph["1"]["inputs"]["unet_name"] == "custom-fp8.safetensors"
    assert graph["2"]["inputs"]["clip_name"] == "custom-nvfp4.safetensors"


def test_pruned_workflow_adds_tiled_upscale_and_detailer():
    graph = PrunedComfyBackend.build_workflow(
        prompt="test",
        width=864,
        height=480,
        frames=124,
        seed=1,
        nfe=4,
        loras=[("turbo.safetensors", 0.75)],
        upscale=True,
        upscale_model="RealESRGAN_x4plus.safetensors",
        upscale_factor=2.0,
        detailer_strength=0.25,
    )
    assert graph["30"]["class_type"] == "UpscaleModelLoader"
    assert graph["31"]["class_type"] == "ImageUpscaleWithModel"
    assert graph["32"]["inputs"]["scale_by"] == pytest.approx(0.5)
    assert graph["33"]["inputs"]["alpha"] == pytest.approx(0.25)
    assert graph["18"]["inputs"]["images"] == ["33", 0]


def test_pruned_workflow_can_upscale_without_sharpening():
    graph = PrunedComfyBackend.build_workflow(
        prompt="test",
        width=864,
        height=480,
        frames=124,
        seed=1,
        nfe=4,
        loras=[("turbo.safetensors", 0.75)],
        upscale=True,
        upscale_factor=1.5,
        detailer_strength=0.0,
    )
    assert "33" not in graph
    assert graph["32"]["inputs"]["scale_by"] == pytest.approx(0.375)
    assert graph["18"]["inputs"]["images"] == ["32", 0]


@pytest.mark.parametrize(
    ("target_fps", "multiplier"),
    [(29.97, 2), (60.0, 3), (120.0, 5)],
)
def test_pruned_workflow_interpolates_standard_frame_rates(target_fps, multiplier):
    graph = PrunedComfyBackend.build_workflow(
        prompt="test",
        width=608,
        height=352,
        frames=124,
        seed=1,
        nfe=4,
        loras=[("turbo.safetensors", 0.75)],
        target_fps=target_fps,
        interpolation_model="rife_v4.25_lite.safetensors",
    )
    assert graph["35"]["inputs"]["model_name"] == "rife_v4.25_lite.safetensors"
    assert graph["36"]["inputs"]["multiplier"] == multiplier
    assert graph["37"]["inputs"]["target_fps"] == pytest.approx(target_fps)
    assert graph["18"]["inputs"]["images"] == ["37", 0]
    assert graph["18"]["inputs"]["fps"] == pytest.approx(target_fps)


def test_pruned_workflow_restores_faces_before_upscale_and_rife():
    graph = PrunedComfyBackend.build_workflow(
        prompt="test",
        width=608,
        height=352,
        frames=124,
        seed=1,
        nfe=4,
        loras=[("turbo.safetensors", 0.75)],
        face_restore=True,
        face_fidelity=0.8,
        upscale=True,
        detailer_strength=0.25,
        target_fps=60.0,
    )
    assert graph["38"]["inputs"]["facedetection"] == "retinaface_mobile0.25"
    assert graph["38"]["inputs"]["codeformer_fidelity"] == pytest.approx(0.8)
    assert graph["31"]["inputs"]["image"] == ["38", 0]
    assert graph["36"]["inputs"]["images"] == ["33", 0]


def test_comfy_python_path_matches_platform(tmp_path, monkeypatch):
    monkeypatch.delenv("MINIMAX_H3_COMFY_PYTHON", raising=False)
    backend = PrunedComfyBackend(AppConfig())
    backend.root = tmp_path / "ComfyUI"
    expected = (
        backend.root / ".venv" / "Scripts" / "python.exe"
        if os.name == "nt"
        else backend.root / ".venv" / "bin" / "python"
    )
    assert backend._python_executable() == expected


def test_comfy_python_path_honors_environment(tmp_path, monkeypatch):
    custom = tmp_path / "custom-python.exe"
    monkeypatch.setenv("MINIMAX_H3_COMFY_PYTHON", str(custom))
    backend = PrunedComfyBackend(AppConfig())
    assert backend._python_executable() == custom.resolve()


def test_comfy_endpoint_uses_configurable_port(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_COMFY_HOST", "127.0.0.1")
    monkeypatch.setenv("MINIMAX_H3_COMFY_PORT", "19188")
    monkeypatch.delenv("MINIMAX_H3_COMFY_URL", raising=False)
    backend = PrunedComfyBackend(AppConfig())
    assert backend.host == "127.0.0.1"
    assert backend.port == 19188
    assert backend.url == "http://127.0.0.1:19188"


def test_pruned_catalog_routes_without_loading_diffusers(monkeypatch):
    spec = LoRASpec(
        id="pruned",
        name="Pruned",
        filename="pruned.safetensors",
        backend="comfy_pruned",
    )
    config = AppConfig(catalog=[spec])
    engine = MiniMaxH3Engine(config)
    expected = object()
    monkeypatch.setattr(engine, "_load_unlocked", lambda: (_ for _ in ()).throw(AssertionError("Diffusers loaded")))
    monkeypatch.setattr(
        "minimax_h3_fl2v.comfy_backend.PrunedComfyBackend.generate",
        lambda self, request, selected, progress_callback=None: expected,
    )
    assert engine.generate(GenerationRequest(prompt="test", lora_id="pruned")) is expected


def test_sync_loras_links_new_files_and_checks_live_discovery(tmp_path, monkeypatch):
    comfy = tmp_path / "ComfyUI"
    source = tmp_path / "new-style.safetensors"
    source.write_bytes(b"weights")
    backend = PrunedComfyBackend(AppConfig())
    backend.root = comfy

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [source.name]

    monkeypatch.setattr("minimax_h3_fl2v.comfy_backend.requests.get", lambda *args, **kwargs: Response())
    backend._sync_loras([source])

    linked = comfy / "models/loras" / source.name
    assert linked.is_file()
    if linked.is_symlink():
        assert linked.resolve() == source.resolve()
    else:
        assert linked.read_bytes() == source.read_bytes()


def test_diffusers_progress_bridge_reports_sampling_steps(monkeypatch):
    events = []
    times = iter([10.0, 11.0, 12.0, 13.0])
    monkeypatch.setattr("minimax_h3_fl2v.pipeline.time.perf_counter", lambda: next(times))
    bridge = _ProgressBridge(2, lambda fraction, message: events.append((fraction, message)), 10.0)
    with bridge:
        bridge.update()
        bridge.update()
    assert "step 1/2" in events[0][1]
    assert "50%" in events[0][1]
    assert "step 2/2" in events[1][1]
    assert events[1][0] == pytest.approx(0.8)
    assert "Sampling complete" in events[2][1]


def test_pruned_progress_stays_monotonic_during_timeout_heartbeat(tmp_path, monkeypatch):
    backend = PrunedComfyBackend(AppConfig(output_dir=tmp_path))
    backend.output_dir = tmp_path
    output = tmp_path / "result.mp4"
    output.write_bytes(b"video")
    events = []

    class Socket:
        def __init__(self):
            self.calls = 0

        def recv(self):
            self.calls += 1
            if self.calls == 1:
                return json.dumps(
                    {"type": "executing", "data": {"prompt_id": "job", "node": "15"}}
                )
            raise __import__("websocket").WebSocketTimeoutException()

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "job": {
                    "status": {"status_str": "success"},
                    "outputs": {"19": {"videos": [{"filename": output.name}]}}
                }
            }

    monkeypatch.setattr("minimax_h3_fl2v.comfy_backend.requests.get", lambda *a, **k: Response())
    path, _ = backend._wait_for_output(
        "job",
        socket=Socket(),
        started=0.0,
        progress_callback=lambda fraction, message: events.append((fraction, message)),
        timeout=1,
    )
    assert path == output.resolve()
    assert all(right[0] >= left[0] for left, right in zip(events, events[1:]))
    assert all(message.count("elapsed") <= 1 for _, message in events)
    assert all("still working · still working" not in message for _, message in events)
