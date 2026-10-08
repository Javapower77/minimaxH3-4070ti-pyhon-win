"""CPU-only tests load diagnostics directly, avoiding package/GPU side effects."""

import importlib.util
import io
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def diagnostics():
    path = Path(__file__).resolve().parents[1] / "src/minimax_h3_fl2v/diagnostics.py"
    spec = importlib.util.spec_from_file_location("isolated_diagnostics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def log(diagnostics):
    output = io.StringIO()
    logger = logging.Logger("diagnostic-test", logging.DEBUG)
    logger.propagate = False
    handler = logging.StreamHandler(output)
    handler.setFormatter(diagnostics.SanitizedFormatter("%(message)s"))
    logger.addHandler(handler)
    yield logger, output
    handler.close()


def test_initialize_paths_idempotence_and_thread_safety(diagnostics, tmp_path, monkeypatch):
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    noisy = {name: logging.getLogger(name).level for name in ("httpx", "httpcore", "urllib3")}
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    # Keep pytest's handlers intact; only isolate the root list while this test runs.
    monkeypatch.setattr(root, "handlers", [])
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            paths = list(pool.map(lambda _: diagnostics.initialize_logging(tmp_path), range(24)))
        path = tmp_path / "logs/application.log"
        assert all(p == path.resolve() for p in paths)
        files = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(files) == 1
        assert files[0].maxBytes == 5 * 1024 * 1024
        assert files[0].backupCount == 5
        assert files[0].encoding == "utf-8"
        assert len(root.handlers) == 2
        hooks = sys.excepthook, threading.excepthook
        diagnostics.initialize_logging(tmp_path / ".")
        assert hooks == (sys.excepthook, threading.excepthook)
        root.info("application test")
        assert "application test" in path.read_text(encoding="utf-8")
        for name in noisy:
            assert logging.getLogger(name).level == logging.WARNING
        second = diagnostics.initialize_logging(tmp_path / "other")
        assert second.is_file()
        assert len(root.handlers) == 3
    finally:
        for handler in root.handlers:
            handler.close()
        root.setLevel(previous_level)
        for name, level in noisy.items():
            logging.getLogger(name).setLevel(level)
    assert previous_handlers  # pytest logging was not closed or replaced globally.


def test_redacts_rendered_messages_tracebacks_and_only_credential_env(diagnostics, log, monkeypatch):
    logger, output = log
    monkeypatch.setenv("HF_TOKEN", "hf-secret-value")
    monkeypatch.setenv("OPENAI_API_KEY", "api-secret-value")
    monkeypatch.setenv("UNRELATED_SETTING", "keep-this-value")
    secrets = ["bearer-secret", "basic-secret", "assigned-token", "password with spaces",
               "key-secret", "url-user", "url-password", "query-secret", "hf-secret-value",
               "api-secret-value"]
    message = (
        'Authorization: Bearer bearer-secret Authorization: Basic basic-secret '
        'token=assigned-token password="password with spaces" api_key=key-secret '
        'https://url-user:url-password@example.org/path?token=query-secret&x=ok '
        'hf-secret-value api-secret-value keep-this-value'
    )
    try:
        raise RuntimeError(message)
    except RuntimeError:
        record = logger.makeRecord(logger.name, logging.ERROR, __file__, 1,
                                   "failure %s", (message,), sys.exc_info())
        original = record.__dict__.copy()
        logger.handle(record)
        assert record.__dict__ == original
        plain = logging.Formatter("%(message)s").format(record)
        assert "bearer-secret" in plain
    rendered = output.getvalue()
    assert "Traceback" in rendered and "RuntimeError" in rendered
    assert "keep-this-value" in rendered and "&x=ok" in rendered
    assert all(secret not in rendered for secret in secrets)


def test_file_metadata_missing(diagnostics, tmp_path):
    assert diagnostics.file_metadata(tmp_path / "missing") == {"file": "missing", "bytes": None}


def test_event_redaction_preserves_json(diagnostics, log):
    logger, output = log
    diagnostics.event(logger, "safe", token="private-token", password='private-"password')
    payload = json.loads(output.getvalue())
    assert payload["token"] == "[REDACTED]"
    assert payload["password"] == "[REDACTED]"


def test_decorator_direct_prompt_and_context_cleanup_on_bad_property(diagnostics, log):
    logger, output = log

    @diagnostics.traced_generation(logger=logger)
    def generate(prompt):
        raise ValueError(prompt)

    with pytest.raises(ValueError):
        generate("direct private prompt")
    assert "direct private prompt" not in output.getvalue()

    class BadRequest:
        prompt = "secret property prompt"

        @property
        def mode(self):
            raise ValueError("bad property")

    @diagnostics.traced_generation(logger=logger)
    def other(request):
        pass

    with pytest.raises(ValueError, match="bad property"):
        other(BadRequest())
    assert diagnostics._secrets.get() == ()
    assert diagnostics._request_id.get() is None


def test_workflow_allowlists_file_sizes_no_content(diagnostics, tmp_path):
    model = tmp_path / "diffusion_models/sub/model.safetensors"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"never-log-model-content")
    lora = tmp_path / "loras/lora.safetensors"
    lora.parent.mkdir()
    lora.write_bytes(b"lora")
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "sub/model.safetensors",
                    "prompt": "private prompt", "password": "secret"}},
        "2": {"class_type": "LoraLoader", "inputs": {"lora_name": lora.name,
                    "strength_model": 0.5, "strength_clip": 0.25}},
        "3": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"width": 768,
                    "height": 512, "length": 89, "prompt": "private prompt",
                    "first_frame": "private-image.png", "clip": ["1", 0]}},
        "4": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "private-output"}},
        "5": {"class_type": "LoadVideo", "inputs": {"file": "private-video.mp4"}},
        "6": {"class_type": "UnknownNode", "inputs": {"width": 999, "text": "private"}},
        "7": {"class_type": "BasicScheduler", "inputs": {"steps": 8, "scheduler": "simple"}},
        "8": {"class_type": "MiniMaxH3LearnedLatentUpscale", "inputs": {
                    "model_name": "missing.safetensors", "scale": 1.5, "device": "cpu"}},
    }
    result = diagnostics.workflow_metadata(graph, tmp_path)
    nodes = result["nodes"]
    assert nodes["1"]["inputs"] == {"unet_name": model.name}
    assert nodes["1"]["files"]["unet_name"] == {"file": model.name, "bytes": model.stat().st_size}
    assert nodes["2"]["files"]["lora_name"]["bytes"] == 4
    assert nodes["3"]["inputs"] == {"width": 768, "height": 512, "length": 89}
    assert nodes["6"]["inputs"] == {}
    assert nodes["8"]["files"]["model_name"]["bytes"] is None
    serialized = json.dumps(result)
    for forbidden in ("private", "never-log", "filename_prefix", "prompt", "first_frame", "sub/"):
        assert forbidden not in serialized
    assert graph["1"]["inputs"]["unet_name"] == "sub/model.safetensors"


def test_workflow_does_not_stat_outside_model_directory(diagnostics, tmp_path):
    outside = tmp_path / "private.safetensors"
    outside.write_bytes(b"private")
    graph = {"1": {"class_type": "VAELoader", "inputs": {"vae_name": "../private.safetensors"}}}
    data = diagnostics.workflow_metadata(graph, tmp_path)["nodes"]["1"]
    assert data["files"]["vae_name"] == {"file": outside.name, "bytes": None}


def test_stage_timing_and_error_traceback(diagnostics, log, monkeypatch):
    logger, output = log
    ticks = iter([10.0, 12.5, 20.0, 21.0])
    monkeypatch.setattr(diagnostics.time, "perf_counter", lambda: next(ticks))
    with diagnostics.stage(logger, "queue_lock", count=2):
        pass
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert records[0] == {"event": "stage_start", "stage": "queue_lock", "count": 2}
    assert records[1]["duration_seconds"] == 2.5
    with pytest.raises(ValueError):
        with diagnostics.stage(logger, "load"):
            raise ValueError("token=private-stage-secret")
    text = output.getvalue()
    assert '"event": "stage_error"' in text and '"duration_seconds": 1.0' in text
    assert "Traceback" in text and "private-stage-secret" not in text


def test_decorated_error_redacts_nested_multiline_prompt(diagnostics, log):
    logger, output = log
    prompt = 'unique first line\nunique second line "quoted"'
    request = SimpleNamespace(prompt=prompt, seed=42, mode="t2va", nfe=8,
                              first_image="private.png", nested={"negative_prompt": "avoid-me"})

    @diagnostics.traced_generation(logger=logger)
    def generate(request):
        diagnostics.event(logger, "inside")
        with diagnostics.stage(logger, "load"):
            raise RuntimeError(f"{request.prompt!r}; avoid-me; {request.prompt}")

    with pytest.raises(RuntimeError) as raised:
        generate(request)
    text = output.getvalue()
    assert "unique first" not in text and "unique second" not in text
    assert "avoid-me" not in text and "private.png" not in text
    assert "Traceback" in text and '"event": "generation_error"' in text
    ids = [json.loads(line)["request_id"] for line in text.splitlines() if line.startswith("{")]
    assert len(ids) >= 4 and len(set(ids)) == 1
    assert len(ids[0]) == 36
    assert prompt in str(raised.value)  # re-raise without mutating the exception.
    diagnostics.event(logger, "outside")
    assert "request_id" not in json.loads(output.getvalue().splitlines()[-1])
    assert diagnostics._secrets.get() == ()


def test_decorated_success_timings_cover_entire_call(diagnostics, log, monkeypatch):
    logger, output = log
    ticks = iter([5.0, 8.0])
    monkeypatch.setattr(diagnostics.time, "perf_counter", lambda: next(ticks))
    result = SimpleNamespace(timings={"queue": 1.0, "load": 2.0, "prompt": "not-a-timing"})

    @diagnostics.traced_generation(logger)
    def generate(request):
        return result

    assert generate({"prompt": "not-logged", "seed": 3, "upscale": True}) is result
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert records[0]["request_type"] == "dict"
    assert records[0]["seed"] == 3
    assert records[1]["duration_seconds"] == 3.0
    assert records[1]["timings"] == {"queue": 1.0, "load": 2.0}
    assert "not-logged" not in output.getvalue()


def test_hooks_preserve_keyboard_interrupt(diagnostics, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "excepthook", lambda *args: calls.append(args))
    monkeypatch.setattr(threading, "excepthook", lambda args: calls.append(args))
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    level = root.level
    noisy = {name: logging.getLogger(name).level for name in ("httpx", "httpcore", "urllib3")}
    try:
        diagnostics.initialize_logging(tmp_path)
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
        thread_args = SimpleNamespace(exc_type=KeyboardInterrupt)
        threading.excepthook(thread_args)
        assert len(calls) == 2 and calls[1] is thread_args
        sys.excepthook(ValueError, ValueError("api_key=hook-secret"), None)
        threading.excepthook(SimpleNamespace(exc_type=RuntimeError,
                            exc_value=RuntimeError("password=thread-secret"), exc_traceback=None))
        text = (tmp_path / "logs/application.log").read_text(encoding="utf-8")
        assert "Uncaught exception" in text and "Uncaught thread exception" in text
        assert "hook-secret" not in text and "thread-secret" not in text
    finally:
        for handler in root.handlers:
            handler.close()
        root.setLevel(level)
        for name, value in noisy.items():
            logging.getLogger(name).setLevel(value)


def test_worker_capture_rotation_cr_bounds_and_cleanup(diagnostics, tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "MAX_BYTES", 512)
    created = []
    original = diagnostics._handler

    def track(path):
        handler = original(path)
        created.append(handler)
        return handler

    monkeypatch.setattr(diagnostics, "_handler", track)
    contents = "Authorization: Bearer worker-secret\rprogress 1\rprogress 2\n"
    contents += "token=" + "oversized-secret" * 6000 + "\r"
    contents += "\n".join(f"safe-line-{i:03d} " + "x" * 100 for i in range(40))
    contents += "\nfinal token=final-secret"  # EOF without newline.
    pipe = io.StringIO(contents)
    process = SimpleNamespace(stdout=pipe)
    path = tmp_path / "worker/worker.log"
    thread = diagnostics.start_worker_capture(process, path)
    thread.join(timeout=5)
    assert thread.daemon and not thread.is_alive()
    assert pipe.closed and created[0].stream is None
    files = list(path.parent.glob("worker.log*"))
    assert 2 <= len(files) <= 6
    text = "".join(f.read_text(encoding="utf-8") for f in files)
    assert "worker-secret" not in text and "final-secret" not in text
    assert "oversized-secret" not in text
    assert "final" in text
    assert all(f.stat().st_size < 700 for f in files)
    renamed = path.with_suffix(".renamed")
    path.rename(renamed)  # no unmanaged open handle remains, including on Windows.


def test_worker_cr_and_split_chunk_secrets(diagnostics, tmp_path):
    pipe = io.StringIO("x" * 4080 + "\rAuthorization: Bearer split-secret\rnext\n")
    path = tmp_path / "worker.log"
    thread = diagnostics.start_worker_capture(SimpleNamespace(stdout=pipe), path)
    thread.join(timeout=5)
    assert not thread.is_alive() and pipe.closed
    text = path.read_text(encoding="utf-8")
    assert "split-secret" not in text and "next" in text and "[REDACTED]" in text


def test_worker_retains_safe_total_duration_but_omits_prompt_payload(diagnostics, tmp_path):
    pipe = io.StringIO('[INFO] Prompt executed in 510.68 seconds\n'
                       '[INFO] Prompt executed in 00:49:20\n'
                       'prompt: private content\n')
    path = tmp_path / "worker.log"
    thread = diagnostics.start_worker_capture(SimpleNamespace(stdout=pipe), path)
    thread.join(timeout=5)
    assert not thread.is_alive()
    text = path.read_text(encoding="utf-8")
    assert "510.68 seconds" in text and "00:49:20" in text
    assert "private content" not in text


def test_worker_cleanup_on_read_failure(diagnostics, tmp_path):
    class BrokenPipe(io.StringIO):
        def read(self, size=-1):
            raise OSError("token=read-secret")

    pipe = BrokenPipe()
    path = tmp_path / "broken.log"
    thread = diagnostics.start_worker_capture(SimpleNamespace(stdout=pipe), path)
    thread.join(timeout=5)
    assert not thread.is_alive() and pipe.closed
    assert "read-secret" not in path.read_text(encoding="utf-8")
    path.unlink()


def test_worker_requires_pipe(diagnostics, tmp_path):
    with pytest.raises(ValueError, match="stdout=PIPE"):
        diagnostics.start_worker_capture(SimpleNamespace(stdout=None), tmp_path / "worker.log")


def test_reused_worker_sparse_lines_and_active_prompt_redaction(diagnostics, tmp_path):
    import queue

    chunks = queue.Queue()
    written = threading.Event()
    path = tmp_path / "worker.log"
    original_handler = diagnostics._handler

    def handler_factory(path):
        handler = original_handler(path)
        original_emit = handler.emit

        def emit(record):
            original_emit(record)
            written.set()

        handler.emit = emit
        return handler

    diagnostics._handler = handler_factory

    class SparsePipe:
        closed = False

        def read(self, size):
            assert size == 1
            return chunks.get(timeout=5)

        def close(self):
            self.closed = True

    def send(line):
        written.clear()
        for char in line + "\n":
            chunks.put(char)
        assert written.wait(5), "Sparse line must flush before EOF/next worker output"

    pipe = SparsePipe()
    thread = diagnostics.start_worker_capture(SimpleNamespace(stdout=pipe), path)
    try:
        send("loading safe model")
        assert "loading safe model" in path.read_text(encoding="utf-8")

        @diagnostics.traced_generation
        def generate(request):
            send("RuntimeError: " + request.prompt.splitlines()[0])
            send("ValueError: " + json.dumps(request.prompt)[1:-1])
            send("{'prompt': 'unknown payload not registered'}")
            send("text=another unregistered payload")

        generate(SimpleNamespace(prompt='unique private line\nsecond private "line"'))
        assert diagnostics._active_prompts == {}
        send("sampling step 1/4")
    finally:
        chunks.put("")
        thread.join(5)
    assert not thread.is_alive() and pipe.closed
    text = path.read_text(encoding="utf-8")
    assert "private" not in text and "unregistered" not in text
    assert text.count("[worker prompt/text payload line omitted]") == 4
    assert "sampling step 1/4" in text