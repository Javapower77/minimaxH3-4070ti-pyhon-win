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


def test_dmad_workflow_sets_coherent_video_and_audio_shifts():
    graph = PrunedComfyBackend.build_workflow(
        prompt="test", width=864, height=480, frames=124, seed=42, nfe=4,
        loras=[("dmad-native.safetensors", 1.0)], video_shift=12.0, audio_shift=2.0,
    )
    assert graph["40"]["class_type"] == "MiniMaxH3SigmaShift"
    assert graph["40"]["inputs"] == {"model": ["21", 0], "shift_video": 12.0, "shift_audio": 2.0}
    assert graph["12"]["inputs"]["model"] == ["40", 0]
    assert graph["13"]["inputs"]["model"] == ["40", 0]
    assert graph["12"]["inputs"]["steps"] == 4


def test_dmad_generation_converts_and_syncs_native_checkpoint(tmp_path, monkeypatch):
    from minimax_h3_fl2v.config import load_config

    cfg = load_config()
    cfg.lora_dir = tmp_path
    spec = cfg.lora_by_id("dmad_4step_lora_critic")
    original = tmp_path / spec.filename
    native = tmp_path / ".converted" / "dmad-native.safetensors"
    native.parent.mkdir()
    original.write_bytes(b"original")
    native.write_bytes(b"converted")
    backend = PrunedComfyBackend(cfg)
    monkeypatch.setattr(backend, "ensure_worker", lambda **kwargs: None)

    def convert(path, cache):
        assert path == original
        assert cache == native.parent
        return native

    def sync(paths):
        assert paths == [native]

    def build(**kwargs):
        assert kwargs["loras"] == [(native.name, 1.0)]
        assert (kwargs["video_shift"], kwargs["audio_shift"], kwargs["nfe"]) == (12, 2, 4)
        raise RuntimeError("stop before submitting GPU job")

    monkeypatch.setattr("minimax_h3_fl2v.dmad.prepare_dmad_lora", convert)
    monkeypatch.setattr(backend, "_sync_loras", sync)
    monkeypatch.setattr(backend, "build_workflow", build)
    with pytest.raises(RuntimeError, match="stop before submitting"):
        backend.generate(GenerationRequest(prompt="test", lora_id=spec.id, nfe=4, megapixels=0.4), spec)


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


@pytest.mark.parametrize("lora_id", ["dasiwa_dmad_hyperflow_4step_r256", "dasiwa_pdmd_dmad_4step_r256"])
def test_dmad_blend_generation_uses_native_file_without_original_converter(tmp_path, monkeypatch, lora_id):
    from minimax_h3_fl2v.config import load_config

    cfg = load_config()
    cfg.lora_dir = tmp_path
    spec = cfg.lora_by_id(lora_id)
    source = tmp_path / spec.filename
    source.write_bytes(b"native blend")
    backend = PrunedComfyBackend(cfg)
    monkeypatch.setattr(backend, "ensure_worker", lambda **kwargs: None)
    monkeypatch.setattr("minimax_h3_fl2v.dmad.prepare_dmad_lora",
                        lambda *args: pytest.fail("Original DMAD converter must not run for the blend"))

    def sync(paths):
        assert paths == [source]

    def build(**kwargs):
        assert kwargs["loras"] == [(spec.filename, 1.0)]
        assert (kwargs["video_shift"], kwargs["audio_shift"], kwargs["nfe"]) == (12, 3, 4)
        raise RuntimeError("stop before GPU submission")

    monkeypatch.setattr(backend, "_sync_loras", sync)
    monkeypatch.setattr(backend, "build_workflow", build)
    with pytest.raises(RuntimeError, match="stop before GPU submission"):
        backend.generate(GenerationRequest(prompt="test", lora_id=spec.id, nfe=4, megapixels=0.4), spec)


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


@pytest.mark.parametrize("discovered", [True, False])
def test_sync_loras_preserves_resident_source_and_checks_discovery(tmp_path, monkeypatch, discovered):
    backend = PrunedComfyBackend(AppConfig())
    backend.root = tmp_path / "ComfyUI"
    source = backend.root / "models/loras/resident.safetensors"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"resident weights")
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return [source.name] if discovered else []

    def get(url, **kwargs):
        calls.append(url)
        return Response()

    monkeypatch.setattr("minimax_h3_fl2v.comfy_backend.requests.get", get)
    monkeypatch.setattr("minimax_h3_fl2v.comfy_backend.shutil.copy2", lambda *args: pytest.fail("Resident file copied"))
    for _ in range(2):
        if discovered:
            backend._sync_loras([source])
        else:
            with pytest.raises(RuntimeError, match="did not discover"):
                backend._sync_loras([source])
        assert source.read_bytes() == b"resident weights"
        assert not source.is_symlink()
    assert calls == [f"{backend.url}/models/loras"] * 2


def test_sync_loras_preserves_hardlinked_source(tmp_path, monkeypatch):
    backend = PrunedComfyBackend(AppConfig())
    backend.root = tmp_path / "ComfyUI"
    source = tmp_path / "hardlink.safetensors"
    source.write_bytes(b"weights")
    target = backend.root / "models/loras" / source.name
    target.parent.mkdir(parents=True)
    target.hardlink_to(source)

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return [source.name]

    monkeypatch.setattr("minimax_h3_fl2v.comfy_backend.requests.get", lambda *a, **k: Response())
    backend._sync_loras([source])
    assert target.samefile(source)


def test_dmad_dareties_generation_reuses_local_native_path(tmp_path, monkeypatch):
    from minimax_h3_fl2v.config import load_config

    cfg = load_config()
    spec = cfg.lora_by_id("dmad_full_dareties_v4_step600")
    spec.local_path = tmp_path / "ComfyUI/models/loras" / spec.filename
    spec.local_path.parent.mkdir(parents=True)
    spec.local_path.write_bytes(b"native fixture")
    backend = PrunedComfyBackend(cfg)
    monkeypatch.setattr(backend, "ensure_worker", lambda **kwargs: None)
    monkeypatch.setattr("minimax_h3_fl2v.dmad.prepare_dmad_lora", lambda *args: pytest.fail("Original converter invoked"))
    monkeypatch.setattr(backend, "_sync_loras", lambda paths: paths == [spec.local_path] or pytest.fail("Wrong source"))

    def build(**kwargs):
        assert kwargs["loras"] == [(spec.filename, 1.0)]
        assert (kwargs["nfe"], kwargs["video_shift"], kwargs["audio_shift"]) == (8, 12, 3)
        raise RuntimeError("stop before GPU submission")

    monkeypatch.setattr(backend, "build_workflow", build)
    with pytest.raises(RuntimeError, match="stop before GPU submission"):
        backend.generate(GenerationRequest(prompt="test", lora_id=spec.id, nfe=8, megapixels=0.4), spec)


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
            if self.calls == 2:
                return json.dumps(
                    {"type": "progress", "data": {"prompt_id": "job", "value": 3, "max": 8}}
                )
            raise __import__("websocket").WebSocketTimeoutException()

    class Response:
        calls = 0

        def raise_for_status(self):
            return None

        def json(self):
            Response.calls += 1
            if Response.calls < 3:
                return {}
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
    assert "Sampling step 3/8" in events[2][1]
    assert "38%" in events[2][1] and "elapsed" in events[2][1]


@pytest.fixture
def controlled_worker(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from minimax_h3_fl2v import comfy_backend as module

    backend = PrunedComfyBackend(AppConfig(output_dir=tmp_path))
    backend.output_dir = tmp_path
    output = tmp_path / "controlled.mp4"
    output.write_bytes(b"video")
    clock = SimpleNamespace(now=100.0, complete=False)
    reports, records, stats = [], [], []
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(module, "event", lambda logger, name, **fields: records.append({"event": name, **fields}))
    monkeypatch.setattr(backend, "_log_worker_stats", lambda reason, prompt_id=None: stats.append((clock.now, reason, prompt_id)))

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            if not clock.complete:
                return {}
            return {"job": {"status": {"status_str": "success"},
                            "outputs": {"19": {"videos": [{"filename": output.name}]}}}}

    def get(url, **kwargs):
        assert url == f"{backend.url}/history/job"
        return Response()

    monkeypatch.setattr(module.requests, "get", get)

    def run(messages):
        items = iter(messages)

        class Socket:
            def recv(self):
                elapsed, kind, data = next(items)
                clock.now = 100.0 + elapsed
                if kind == "complete":
                    clock.complete = True
                    raise module.websocket.WebSocketTimeoutException()
                if kind == "timeout":
                    raise module.websocket.WebSocketTimeoutException()
                return json.dumps({"type": kind, "data": {"prompt_id": "job", **data}})

        return backend._wait_for_output(
            "job", socket=Socket(), started=100.0, timeout=120,
            progress_callback=lambda fraction, label: reports.append((fraction, label)),
        )

    return SimpleNamespace(run=run, reports=reports, records=records, stats=stats, output=output)


def test_worker_heartbeat_elapsed_advances_without_fraction_regression(controlled_worker):
    worker = controlled_worker
    path, _ = worker.run([
        (0, "executing", {"node": "15"}),
        (2, "progress", {"value": 1, "max": 4}),
        (7, "timeout", {}), (9, "timeout", {}), (12, "complete", {}),
    ])
    assert path == worker.output.resolve()
    heartbeats = [r for r in worker.records if r["event"] == "worker_heartbeat"]
    assert [r["elapsed_seconds"] for r in heartbeats] == [7, 12]
    assert [r["node_elapsed_seconds"] for r in heartbeats] == [7, 12]
    assert "7.0s elapsed" in worker.reports[-2][1]
    assert "12.0s elapsed" in worker.reports[-1][1]
    assert worker.reports[-2][0] == worker.reports[-1][0]
    assert all(a[0] <= b[0] for a, b in zip(worker.reports, worker.reports[1:]))


def test_sampling_intervals_include_first_step_loading_and_reset(controlled_worker):
    worker = controlled_worker
    worker.run([
        (0, "executing", {"node": "15"}),
        (8, "progress", {"value": 1, "max": 4}),
        (11, "progress", {"value": 2, "max": 4}),
        (12, "progress", {"value": 2, "max": 4}),
        (15, "progress", {"value": 4, "max": 4}),
        (16, "executing", {"node": "16"}), (17, "complete", {}),
    ])
    steps = [r for r in worker.records if r["event"] == "sample_progress"]
    assert [r["value"] for r in steps] == [1, 2, 4]
    assert [r["interval_seconds"] for r in steps] == [8, 3, 4]
    assert [r["seconds_per_step"] for r in steps] == [8, 3, 2]
    assert [r["first_step_includes_loading"] for r in steps] == [True, False, False]
    ends = [r for r in worker.records if r["event"] == "node_end"]
    assert [(r["node"], r["duration_seconds"]) for r in ends] == [("15", 16), ("16", 1)]


def test_worker_stats_and_non_sampling_progress_are_rate_limited(controlled_worker):
    worker = controlled_worker
    worker.run([
        (0, "executing", {"node": "16"}),
        (1, "progress", {"value": 1, "max": 10}),
        (2, "progress", {"value": 2, "max": 10}),
        (6, "progress", {"value": 3, "max": 10}),
        (7, "progress", {"value": 10, "max": 10}),
        (29, "timeout", {}), (30, "timeout", {}), (31, "timeout", {}),
        (59, "timeout", {}), (60, "timeout", {}), (61, "complete", {}),
    ])
    assert worker.stats == [(130, "progress", "job"), (160, "progress", "job"),
                            (161, "completion", "job")]
    progress = [r for r in worker.records if r["event"] == "node_progress"]
    assert [r["value"] for r in progress] == [1, 3, 10]
    assert [r["interval_seconds"] for r in progress] == [1, 5, 1]


@pytest.mark.parametrize("malformed", [False, True])
def test_worker_execution_error_never_exposes_prompt_or_body(controlled_worker, malformed):
    worker = controlled_worker
    private = "private prompt content"
    data = {"node_id": private if malformed else "15",
            "exception_type": private if malformed else "RuntimeError",
            "exception_message": private, "traceback": [private],
            "current_inputs": {"prompt": private}, "body": private}
    with pytest.raises(RuntimeError) as error:
        worker.run([(1, "execution_error", data)])
    rendered = str(error.value) + json.dumps(worker.records) + repr(worker.reports)
    assert private not in rendered and "current_inputs" not in rendered and "traceback" not in rendered
    record = next(r for r in worker.records if r["event"] == "worker_error")
    assert record["node"] == ("unknown" if malformed else "15")
    assert record["exception_type"] == ("worker error" if malformed else "RuntimeError")


def test_worker_launch_uses_pipe_and_reports_log_path(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from minimax_h3_fl2v import comfy_backend as module

    backend = PrunedComfyBackend(AppConfig(output_dir=tmp_path / "outputs", upload_dir=tmp_path / "uploads"))
    backend.root = tmp_path / "ComfyUI"
    readiness = iter([False, True])
    monkeypatch.setattr(backend, "_is_ready", lambda: next(readiness))
    monkeypatch.setattr(backend, "validate_installation", lambda: None)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    calls, captures, records = [], [], []
    process = SimpleNamespace(pid=123, stdout=object())

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return process

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module, "start_worker_capture", lambda proc, path: captures.append((proc, path)))
    monkeypatch.setattr(module, "event", lambda logger, name, **fields: records.append({"event": name, **fields}))
    backend.ensure_worker()
    options = calls[0][1]
    assert options["stdout"] == module.subprocess.PIPE
    assert options["stderr"] == module.subprocess.STDOUT
    assert options["text"] and options["encoding"] == "utf-8"
    assert options["errors"] == "replace" and options["bufsize"] == 1
    assert options["env"]["HF_HUB_OFFLINE"] == "1"
    path = tmp_path / "logs/comfyui-worker.log"
    assert captures == [(process, path)]
    assert next(r for r in records if r["event"] == "worker_start")["log_file"] == str(path)
