"""Offline contracts for character_swap and its real restoration shared hooks.

Only tiny CPU SafeTensors and local image fixtures are used. Worker, network,
media tools, and GPU entry points fail closed unless explicitly mocked.
"""

from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image
import pytest
import torch
from safetensors.torch import save_file

from minimax_h3_fl2v import assets, character_swap as swap, download, restoration as shared
from minimax_h3_fl2v.comfy_backend import PrunedComfyBackend
from minimax_h3_fl2v.config import AppConfig


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Unexpected network, GPU, subprocess, or real worker access")

    monkeypatch.setattr(shared.requests.sessions.Session, "request", blocked)
    for name in ("get", "post"):
        monkeypatch.setattr(shared.requests, name, blocked)
    monkeypatch.setattr(shared.websocket, "create_connection", blocked)
    monkeypatch.setattr(shared.subprocess, "run", blocked)
    monkeypatch.setattr(shared.subprocess, "Popen", blocked)
    monkeypatch.setattr(PrunedComfyBackend, "ensure_worker", blocked)
    monkeypatch.setattr(torch.cuda, "_lazy_init", blocked)
    monkeypatch.setattr(download, "hf_hub_download", blocked)
    monkeypatch.setattr(download, "snapshot_download", blocked)
    shared._VERIFIED.clear()
    yield
    shared._VERIFIED.clear()


@pytest.fixture
def info():
    return shared.VideoInfo(64, 32, 22, Fraction(24), 2)


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_COMFY_ROOT", str(tmp_path / "worker"))
    monkeypatch.setenv("MINIMAX_H3_COMFY_PYTHON", str(tmp_path / "worker/python.exe"))
    return AppConfig(output_dir=tmp_path / "outputs", upload_dir=tmp_path / "uploads",
                     lora_dir=tmp_path / "loras")


def request(**settings):
    return swap.CharacterSwapRequest(Path("source.mp4"), Path("character.png"), **settings)


def workflow(info, *, nodes=None, audio_source=None, **settings):
    return swap.CharacterSwapBackend.build_character_swap_workflow(
        source="source.mp4", reference_image="character.png", info=info,
        request=request(**settings), text_encoder="encoder.safetensors",
        filename_prefix="character_swap/test", nodes=nodes, audio_source=audio_source)


def reference_schema():
    def group(prefix):
        return ["COMFY_AUTOGROW_V3", {"template": {
            "prefix": prefix, "min": 0, "max": 10,
            "input": {"required": {prefix: ["IMAGE"]}},
        }}]

    return {"input": {"required": {
        "clip": ["CLIP"], "vae": ["VAE"], "prompt": ["STRING"],
        "width": ["INT"], "height": ["INT"], "length": ["INT"],
        "ref_image_size": [["match"]]}, "optional": {
        "ref_images": group("ref_image_"), "ref_videos": group("ref_video_")}}}


@pytest.mark.parametrize("audio_tracks", [0, 2])
@pytest.mark.parametrize("audio_source", [None, "original_soundtrack.mp4"])
def test_native_workflow_defaults_decode_video_only_and_resolve_all_links(info, audio_tracks, audio_source):
    graph = workflow(replace(info, audio_tracks=audio_tracks), audio_source=audio_source)
    assert graph["1"]["inputs"]["unet_name"] == assets.REF2VA_MODEL_FILE
    assert graph["50"] == {"class_type": "LoadVideo", "inputs": {"file": "source.mp4"}}
    assert graph["51"] == {"class_type": "GetVideoComponents", "inputs": {"video": ["50", 0]}}
    assert graph["53"] == {"class_type": "LoadImage", "inputs": {"image": "character.png"}}
    assert graph["54"] == {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
        "clip": ["2", 0], "vae": ["3", 0], "prompt": swap.CHARACTER_SWAP_PROMPT,
        "width": 64, "height": 32, "length": 22, "ref_image_size": "match",
        "ref_images.ref_image_0": ["53", 0], "ref_videos.ref_video_0": ["51", 0]}}
    assert graph["13"]["inputs"]["conditioning"] == ["54", 0]
    assert graph["15"]["inputs"]["latent_image"] == ["54", 1]
    assert graph["21"]["inputs"] == {"model": ["1", 0],
        "lora_name": assets.CHARACTER_SWAP_LORA_FILE, "strength_model": 1.0}
    assert graph["13"]["inputs"]["model"] == ["21", 0]
    assert graph["11"]["inputs"]["sampler_name"] == "res_multistep"
    assert graph["12"]["inputs"]["scheduler"] == "simple"
    assert graph["12"]["inputs"]["steps"] == 20
    assert graph["16"] == {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["3", 0]}}
    assert graph["18"]["inputs"]["images"] == ["16", 0]
    assert graph["18"]["inputs"]["fps"] == 24.0
    assert graph["19"]["inputs"]["video"] == ["18", 0]
    assert graph["19"]["inputs"]["filename_prefix"] == "character_swap/test"
    assert "audio" not in graph["18"]["inputs"]
    assert not {"9", "17", "22", "55", "56"} & graph.keys()
    assert not {"VAEDecodeAudio", "MiniMaxH3ImageToVideo", "EmptyMiniMaxH3LatentAV"} & {
        node["class_type"] for node in graph.values()}
    adapters = [node["inputs"]["lora_name"] for node in graph.values() if "lora_name" in node["inputs"]]
    assert adapters == [assets.CHARACTER_SWAP_LORA_FILE]
    assert assets.RESTORE_FILE not in json.dumps(graph) and "turbo" not in json.dumps(graph).lower()
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert len(value) == 2 and value[0] in graph and isinstance(value[1], int)


def test_workflow_editable_prompt_steps_strength_seed_and_no_config_mutation(info, config):
    before = replace(config)
    backend = swap.CharacterSwapBackend(config)
    prompt = "Replace only the left dancer in <Video 1> with <Picture 1>."
    graph = workflow(info, prompt=prompt, steps=37, strength=0.65, seed=123)
    assert graph["54"]["inputs"]["prompt"] == prompt
    assert graph["12"]["inputs"]["steps"] == 37
    assert graph["21"]["inputs"]["strength_model"] == 0.65
    assert graph["10"]["inputs"]["noise_seed"] == 123
    assert config == before and backend.model == assets.REF2VA_MODEL_FILE
    assert backend.operation == "character_swap"
    assert backend.config.comfy_cache_none and backend.config.comfy_reserve_vram_gb >= 1


@pytest.mark.parametrize("steps", [4, 20, 37, 50])
def test_turbo_graph_orders_fixed_acceleration_before_editable_character_adapter(info, monkeypatch, steps):
    build = Mock(wraps=PrunedComfyBackend.build_workflow)
    monkeypatch.setattr(PrunedComfyBackend, "build_workflow", build)
    graph = workflow(info, use_turbo=True, steps=steps, strength=0.65, seed=123,
                     prompt="Replace the dancer in <Video 1> with <Picture 1>.")
    assert graph["21"] == {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["1", 0], "lora_name": assets.REF2VA_TURBO_FILE, "strength_model": 1.0}}
    assert graph["22"] == {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["21", 0], "lora_name": assets.CHARACTER_SWAP_LORA_FILE, "strength_model": 0.65}}
    assert graph["12"]["inputs"] == {"model": ["22", 0], "scheduler": "simple", "steps": 8, "denoise": 1.0}
    assert graph["13"]["inputs"]["model"] == ["22", 0]
    assert graph["11"]["inputs"]["sampler_name"] == "euler"
    assert graph["10"]["inputs"]["noise_seed"] == 123
    assert graph["54"]["inputs"]["prompt"] == build.call_args.kwargs["prompt"]
    # Native 12/3 shifts need no override node; verify the actual builder defaults.
    import inspect
    signature = inspect.signature(build._mock_wraps)
    assert signature.parameters["video_shift"].default == 12.0
    assert signature.parameters["audio_shift"].default == 3.0
    assert "video_shift" not in build.call_args.kwargs and "audio_shift" not in build.call_args.kwargs
    assert "40" not in graph and "audio" not in graph["18"]["inputs"]
    assert [node["inputs"]["lora_name"] for node in graph.values()
            if "lora_name" in node["inputs"]] == [assets.REF2VA_TURBO_FILE, assets.CHARACTER_SWAP_LORA_FILE]
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert len(value) == 2 and value[0] in graph and isinstance(value[1], int)


@pytest.mark.parametrize("style", ["native", "qualified", "flat"])
def test_workflow_uses_live_typed_image_and_video_ports(info, style):
    schema = reference_schema()
    if style != "native":
        schema["input"]["optional"] = {
            (f"{group}.{prefix}0" if style == "qualified" else prefix + "0"): ["IMAGE"]
            for group, prefix in (("ref_images", "ref_image_"), ("ref_videos", "ref_video_"))}
    graph = workflow(info, nodes={"MiniMaxH3ReferenceToVideo": schema})
    inputs = graph["54"]["inputs"]
    image = "ref_image_0" if style == "flat" else "ref_images.ref_image_0"
    video = "ref_video_0" if style == "flat" else "ref_videos.ref_video_0"
    assert inputs[image] == ["53", 0] and inputs[video] == ["51", 0]
    assert not any("audio" in key for key in inputs)


@pytest.mark.parametrize("group", ["ref_images", "ref_videos"])
@pytest.mark.parametrize("fault", ["missing", "prefix", "type", "max", "extra_child"])
def test_workflow_rejects_incompatible_native_ports(info, group, fault):
    schema = reference_schema()
    optional = schema["input"]["optional"]
    template = optional[group][1]["template"]
    if fault == "missing":
        optional.pop(group)
    elif fault == "prefix":
        template["prefix"] = "unknown_"
    elif fault == "type":
        template["input"]["required"][template["prefix"]] = ["AUDIO"]
    elif fault == "max":
        template["max"] = 0
    else:
        template["input"]["required"]["extra"] = ["IMAGE"]
    with pytest.raises(RuntimeError, match=group):
        workflow(info, nodes={"MiniMaxH3ReferenceToVideo": schema})


@pytest.mark.parametrize("settings", [
    {"prompt": ""}, {"prompt": " \n\t"}, {"prompt": None}, {"prompt": 20},
    {"strength": 0}, {"strength": -1}, {"strength": 2.01},
    {"strength": float("nan")}, {"strength": float("inf")},
    {"steps": True}, {"steps": 3}, {"steps": 51}, {"steps": 8.5},
    {"seed": True}, {"seed": -1}, {"seed": 2**64}, {"seed": 1.5},
    {"use_turbo": 0}, {"use_turbo": 1}, {"use_turbo": "yes"},
    {"use_turbo": None}, {"use_turbo": []},
])
def test_request_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        request(**settings).validate()


@pytest.mark.parametrize("settings", [
    {}, {"strength": 0.05, "steps": 4, "seed": 0},
    {"strength": 2, "steps": 50, "seed": 2**64 - 1},
    {"use_turbo": False}, {"use_turbo": True},
    {"use_turbo": True, "steps": 37},
])
def test_request_defaults_and_valid_boundaries(settings):
    actual = request(**settings)
    actual.validate()
    assert actual.source == actual.source_video
    assert actual.effective_steps == (8 if actual.use_turbo else actual.steps)
    assert not hasattr(actual, "normalize_input") and not hasattr(actual, "output_fps")
    if not settings:
        assert (actual.steps, actual.strength, actual.seed, actual.use_turbo) == (20, 1.0, 42, False)
        assert actual.prompt == swap.CHARACTER_SWAP_PROMPT


@pytest.mark.parametrize("settings", [{"normalize_input": True}, {"output_fps": "source"},
                                      {"output_fps": 24}, {"output_fps": "native"}])
def test_removed_rate_and_normalization_arguments_are_not_accepted(settings):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        request(**settings)


def flattened_pair(module, *, rows=8, columns=4, rank=2):
    prefix = "lora_unet_" + module.replace(".", "_")
    return {prefix + ".alpha": torch.tensor(7.0),
            prefix + ".lora_down.weight": torch.zeros(rank, columns),
            prefix + ".lora_up.weight": torch.zeros(rows, rank)}


@pytest.fixture
def tiny_assets(config):
    backend = swap.CharacterSwapBackend(config)
    for path in backend._required_paths():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tiny local dependency")
    model = backend.root / "models/diffusion_models" / backend.model
    save_file({"adaln_t_table": torch.zeros(1025, 8),
               "blocks.0.attn.out_proj.weight": torch.zeros(8, 4),
               "token_refiner.blocks.1.attn.qkv_proj.weight": torch.zeros(24, 4)}, str(model))
    adapter = backend.root / "models/loras" / assets.CHARACTER_SWAP_LORA_FILE
    adapter.parent.mkdir(parents=True, exist_ok=True)
    tensors = flattened_pair("blocks.0.attn.out_proj")
    tensors.update(flattened_pair("token_refiner.blocks.1.attn.qkv_proj", rows=24))
    save_file(tensors, str(adapter), metadata={"test": "tiny CPU fixture"})
    dynamic = backend.root / "comfy/cli_args.py"
    dynamic.parent.mkdir(parents=True)
    dynamic.write_text("enables_dynamic_vram", encoding="utf-8")
    return SimpleNamespace(backend=backend, model=model, adapter=adapter,
                           tensors=tensors, dynamic=dynamic)


def test_preflight_real_tiny_headers_all_keys_and_no_restoration_or_pdd_assets(tiny_assets, monkeypatch):
    ctx = tiny_assets
    verify = Mock()
    monkeypatch.setattr(shared, "verify_asset", verify)
    validate = Mock(wraps=shared.validate_adapter)
    monkeypatch.setattr(shared, "validate_adapter", validate)
    ctx.backend.preflight()
    assert [call.args for call in verify.call_args_list] == [
        (ctx.model, assets.REF2VA_MODEL_SHA256), (ctx.adapter, assets.CHARACTER_SWAP_LORA_SHA256)]
    validate.assert_called_once_with(ctx.adapter, shared.read_header(ctx.model))
    assert not (ctx.backend.root / "models/loras" / assets.RESTORE_FILE).exists()
    assert not (ctx.backend.root / "comfy/weight_adapter/lora.py").exists()
    assert not (ctx.backend.root / "comfy/ldm/minimax/model.py").exists()


@pytest.mark.parametrize("fault", ["adapter", "base", "encoder", "curve", "dynamic", "checksum", "later_pair"])
def test_preflight_rejects_missing_or_incompatible_assets_without_worker(tiny_assets, monkeypatch, fault):
    ctx = tiny_assets
    verify = Mock()
    monkeypatch.setattr(shared, "verify_asset", verify)
    if fault in {"adapter", "base", "encoder"}:
        path = {"adapter": ctx.adapter, "base": ctx.model,
                "encoder": ctx.backend.root / "models/text_encoders" / ctx.backend.text_encoder}[fault]
        path.unlink()
        error, match = FileNotFoundError, "Optional character swap assets missing"
    elif fault == "curve":
        save_file({"adaln_t_table": torch.zeros(1025, 48)}, str(ctx.model))
        error, match = ValueError, "pruned Ref2VA curve"
    elif fault == "dynamic":
        ctx.dynamic.write_text("outdated", encoding="utf-8")
        error, match = RuntimeError, "DynamicVRAM"
    elif fault == "checksum":
        verify.side_effect = ValueError("checksum mismatch")
        error, match = ValueError, "checksum mismatch"
    else:
        ctx.tensors["lora_unet_token_refiner_blocks_1_attn_qkv_proj.lora_up.weight"] = torch.zeros(23, 2)
        save_file(ctx.tensors, str(ctx.adapter))
        error, match = ValueError, "Character swap LoRA incompatible.*token_refiner"
    with pytest.raises(error, match=match):
        ctx.backend.preflight()
    if fault in {"adapter", "base", "encoder"}:
        verify.assert_not_called()


@pytest.mark.parametrize("module", ["blocks.0.attn.out_proj", "blocks.47.mlp.fc1",
    "token_refiner.blocks.1.attn.qkv_proj", "blocks.0.adaln_proj.linear", "final_layer.adaln_proj.linear"])
def test_shared_validator_supports_real_flattened_vocabulary_without_rewriting(tmp_path, module):
    adapter = tmp_path / "adapter.safetensors"
    model = tmp_path / "base.safetensors"
    save_file(flattened_pair(module), str(adapter))
    save_file({module + ".weight": torch.zeros(8, 4)}, str(model))
    before = adapter.read_bytes()
    shared.validate_adapter(adapter, shared.read_header(model))
    assert adapter.read_bytes() == before
    with shared.safe_open(str(adapter), framework="pt", device="cpu") as loaded:
        prefix = "lora_unet_" + module.replace(".", "_")
        assert loaded.get_tensor(prefix + ".alpha").item() == 7
        assert loaded.get_slice(prefix + ".lora_down.weight").get_shape() == [2, 4]


@pytest.mark.parametrize("fault,match", [
    ("later_pair", "Incompatible Restoration target token_refiner"),
    ("unknown_target", "Unknown or unpaired"), ("missing_up", "Unknown or unpaired"),
    ("extra", "Unsupported/unmatched"), ("orphan_alpha", "Unsupported/unmatched"),
    ("orphan_up", "Unsupported/unmatched"), ("alpha_shape", "Invalid adapter alpha"),
    ("rank", "Incompatible Restoration target"), ("foreign", "Unsupported Restoration adapter key"),
    ("alias_collision", "flattened alias collision"), ("full_adaln", "Full AdaLN"),
])
def test_shared_validator_rejects_every_invalid_key_after_a_valid_pair(tiny_assets, fault, match):
    ctx = tiny_assets
    prefix = "lora_unet_token_refiner_blocks_1_attn_qkv_proj"
    tensors = dict(ctx.tensors)
    header = shared.read_header(ctx.model)
    if fault == "later_pair":
        tensors[prefix + ".lora_up.weight"] = torch.zeros(23, 2)
    elif fault == "unknown_target":
        tensors.update(flattened_pair("unknown.module"))
    elif fault == "missing_up":
        tensors.pop(prefix + ".lora_up.weight")
    elif fault == "extra":
        tensors[prefix + ".unknown"] = torch.zeros(1)
    elif fault == "orphan_alpha":
        tensors["lora_unet_unknown.alpha"] = torch.tensor(1.0)
    elif fault == "orphan_up":
        tensors["lora_unet_unknown.lora_up.weight"] = torch.zeros(8, 2)
    elif fault == "alpha_shape":
        tensors[prefix + ".alpha"] = torch.zeros(2)
    elif fault == "rank":
        tensors[prefix + ".lora_up.weight"] = torch.zeros(24, 3)
    elif fault == "foreign":
        tensors["foreign.weight"] = torch.zeros(1)
    elif fault == "alias_collision":
        header["token_refiner.blocks.1.attn.qkv.proj.weight"] = {"shape": [24, 4]}
    else:
        tensors.update(flattened_pair("blocks.0.adaln_proj.linear", rows=48))
        header["blocks.0.adaln_proj.linear.weight"] = {"shape": [8, 4]}
    save_file(tensors, str(ctx.adapter))
    before = ctx.adapter.read_bytes()
    with pytest.raises(ValueError, match=match):
        shared.validate_adapter(ctx.adapter, header)
    assert ctx.adapter.read_bytes() == before


@pytest.fixture
def lifecycle(config, tmp_path, monkeypatch, info):
    backend = swap.CharacterSwapBackend(config)
    video = tmp_path / "source.MP4"
    video.write_bytes(b"original video and all audio tracks")
    image = tmp_path / "reference.JPG"
    exif = Image.Exif()
    exif[274] = 6  # 90 degrees clockwise: 19x11 becomes 11x19.
    Image.new("RGB", (19, 11), "red").save(image, exif=exif)
    ctx = SimpleNamespace(backend=backend, video=video, image=image, events=[], graphs=[],
                          staged_images=[])
    monkeypatch.setattr(shared, "probe_video", Mock(return_value=info))
    monkeypatch.setattr(backend, "preflight", Mock(side_effect=lambda: ctx.events.append("preflight")))

    def start(**kwargs):
        ctx.events.append("worker")
        assert backend._worker_image.is_file()
        with Image.open(backend._worker_image) as staged:
            assert staged.size == (11, 19) != (info.width, info.height)
            assert staged.mode == "RGB" and staged.format == "PNG"
            assert staged.getexif().get(274, 1) == 1
        ctx.staged_images.append(backend._worker_image)
        guides = list(backend.input_dir.glob("*.mp4"))
        assert len(guides) == 1 and guides[0].is_file()
        assert guides[0].read_bytes() == video.read_bytes()
        assert not list(backend.input_dir.glob("*_audio.*"))

    monkeypatch.setattr(backend, "ensure_worker", Mock(side_effect=start))
    monkeypatch.setattr(backend, "validate_worker_profile", Mock(side_effect=lambda: ctx.events.append("profile")))
    graph = workflow(info)
    schema = {node["class_type"]: {"input": {"required": {key: [] for key in node["inputs"]}}}
              for node in graph.values()}
    schema["MiniMaxH3ReferenceToVideo"] = reference_schema()
    ctx.schema = schema
    monkeypatch.setattr(shared.requests, "get", Mock(return_value=Mock(json=Mock(return_value=schema))))

    def post(url, **kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        ctx.events.append(endpoint)
        if endpoint == "free":
            assert kwargs["json"] == {"unload_models": True, "free_memory": True}
        else:
            assert endpoint == "prompt" and ctx.events.index("free") < ctx.events.index("prompt")
            submitted = kwargs["json"]["prompt"]
            assert submitted["53"]["inputs"]["image"] == backend._worker_image.name
            assert (backend.input_dir / submitted["53"]["inputs"]["image"]).is_file()
            assert submitted["54"]["inputs"]["length"] == info.frames
            assert not any("audio" in key for key in submitted["54"]["inputs"])
            ctx.graphs.append(submitted)
        return Mock(ok=True, json=Mock(return_value={"prompt_id": "job"}))

    monkeypatch.setattr(shared.requests, "post", Mock(side_effect=post))
    ctx.socket = Mock()
    monkeypatch.setattr(shared.websocket, "create_connection", Mock(return_value=ctx.socket))
    ctx.output = tmp_path / "worker_result.mp4"
    ctx.output.write_bytes(b"rendered silent video")
    ctx.wait = Mock(return_value=(ctx.output, {}))
    monkeypatch.setattr(backend, "_wait_for_output", ctx.wait)

    def mux(enhanced, original, destination, actual_info):
        assert enhanced == ctx.output and actual_info == info
        assert original == backend._original_video and original != video
        assert original.read_bytes() == video.read_bytes()
        destination.write_bytes(b"rendered video with original audio")
        ctx.events.append("mux")

    monkeypatch.setattr(shared, "mux_original_audio", Mock(side_effect=mux))
    return ctx


def assert_originals(ctx, result):
    assert result.original_path != ctx.video and result.original_image_path != ctx.image
    assert result.original_path.read_bytes() == ctx.video.read_bytes()
    assert result.original_image_path.read_bytes() == ctx.image.read_bytes()
    originals = ctx.backend.config.upload_dir / "character_swap_originals"
    assert result.original_path.parent == result.original_image_path.parent == originals
    assert set(originals.iterdir()) == {result.original_path, result.original_image_path}


def test_swap_real_shared_hooks_persist_raw_originals_stage_orientation_before_worker(lifecycle):
    ctx = lifecycle
    progress = Mock()
    result = ctx.backend.swap(swap.CharacterSwapRequest(ctx.video, ctx.image), progress)
    assert_originals(ctx, result)
    assert result.output_path.is_file() and result.output_path.name.endswith("_character_swap.mkv")
    assert ctx.events == ["preflight", "worker", "profile", "free", "prompt", "mux", "free"]
    ctx.wait.assert_called_once()
    assert ctx.wait.call_args.args == ("job",) and ctx.wait.call_args.kwargs["socket"] is ctx.socket
    ctx.socket.close.assert_called_once()
    assert not ctx.staged_images[0].exists()
    assert "steps=20" in result.message and "strength=1" in result.message
    assert "Original audio stream-copy verified" in result.message
    assert "64×32" in result.message
    assert all("restoration" not in call.args[1].lower() for call in progress.call_args_list)
    assert progress.call_args.args[0] == 1.0
    # Raw EXIF-bearing JPEG is retained, not replaced by the oriented PNG.
    with Image.open(result.original_image_path) as original:
        assert original.size == (19, 11) and original.getexif()[274] == 6


@pytest.mark.parametrize("kind", ["corrupt", "gif", "animated_png"])
def test_invalid_image_retains_both_raw_originals_before_any_shared_worker(lifecycle, kind):
    ctx = lifecycle
    if kind == "corrupt":
        ctx.image.write_bytes(b"not an image")
    else:
        ctx.image = ctx.image.with_suffix(".gif" if kind == "gif" else ".png")
        first = Image.new("RGB", (13, 17), "red")
        if kind == "gif":
            first.save(ctx.image)
        else:
            first.save(ctx.image, save_all=True, append_images=[Image.new("RGB", (13, 17), "blue")], duration=100)
    with pytest.raises(swap.CharacterSwapFailure) as caught:
        ctx.backend.swap(swap.CharacterSwapRequest(ctx.video, ctx.image))
    assert_originals(ctx, caught.value)
    ctx.backend.preflight.assert_not_called()
    ctx.backend.ensure_worker.assert_not_called()
    shared.probe_video.assert_not_called()
    shared.requests.post.assert_not_called()
    assert not list(ctx.backend.config.output_dir.glob("*_character_swap.mkv"))


@pytest.mark.parametrize("stage", ["probe", "preflight", "prepare_image", "startup", "profile", "worker", "mux"])
def test_shared_failure_retains_both_originals_and_cleans_image(lifecycle, monkeypatch, stage):
    ctx = lifecycle
    error = RuntimeError("synthetic " + stage)
    if stage == "probe":
        shared.probe_video.side_effect = error
    elif stage == "prepare_image":
        prepare = ctx.backend._prepare_reference
        def broken_prepare(token):
            prepare(token)
            raise error
        monkeypatch.setattr(ctx.backend, "_prepare_reference", broken_prepare)
    else:
        target = {"preflight": ctx.backend.preflight, "startup": ctx.backend.ensure_worker,
                  "profile": ctx.backend.validate_worker_profile, "worker": ctx.wait,
                  "mux": shared.mux_original_audio}[stage]
        target.side_effect = error
    with pytest.raises(swap.CharacterSwapFailure, match="synthetic " + stage) as caught:
        ctx.backend.swap(swap.CharacterSwapRequest(ctx.video, ctx.image))
    assert_originals(ctx, caught.value)
    assert caught.value.__cause__ is not None
    assert ctx.backend._worker_image is None or not ctx.backend._worker_image.exists()
    if stage in {"probe", "preflight", "prepare_image"}:
        ctx.backend.ensure_worker.assert_not_called()
    if stage in {"worker", "mux"}:
        ctx.socket.close.assert_called_once()
        assert ctx.events[-1] == "free"


@pytest.mark.parametrize("missing", ["video", "image"])
def test_missing_input_rejected_without_persistence(lifecycle, missing):
    ctx = lifecycle
    getattr(ctx, missing).unlink()
    with pytest.raises(FileNotFoundError, match="Character.*missing"):
        ctx.backend.swap(swap.CharacterSwapRequest(ctx.video, ctx.image))
    assert not (ctx.backend.config.upload_dir / "character_swap_originals").exists()
    ctx.backend.ensure_worker.assert_not_called()


@pytest.mark.parametrize("fault", ["missing_node", "unknown_input", "new_required", "rejected_prompt"])
def test_live_schema_or_submission_failure_retains_inputs_without_waiting(lifecycle, fault):
    ctx = lifecycle
    if fault == "missing_node":
        ctx.schema.pop("MiniMaxH3ReferenceToVideo")
    elif fault == "unknown_input":
        ctx.schema["MiniMaxH3ReferenceToVideo"]["input"]["required"].pop("ref_image_size")
    elif fault == "new_required":
        ctx.schema["MiniMaxH3ReferenceToVideo"]["input"]["required"]["new_required"] = ["INT"]
    else:
        post = shared.requests.post.side_effect
        def reject(url, **kwargs):
            response = post(url, **kwargs)
            if url.endswith("/prompt"):
                response.ok = False
                response.text = "synthetic rejected prompt"
            return response
        shared.requests.post.side_effect = reject
    with pytest.raises(swap.CharacterSwapFailure, match="unavailable|incompatible|rejected") as caught:
        ctx.backend.swap(swap.CharacterSwapRequest(ctx.video, ctx.image))
    assert_originals(ctx, caught.value)
    ctx.wait.assert_not_called()
    assert not ctx.backend._worker_image.exists() and ctx.events[-1] == "free"
    if fault == "rejected_prompt":
        ctx.socket.close.assert_called_once()
    else:
        shared.websocket.create_connection.assert_not_called()


def mock_source_probe(monkeypatch, *, changes=None, times=None):
    """Keep the real probe/strict validation; replace only ffprobe execution."""
    stream = {"codec_type": "video", "width": 64, "height": 32,
              "nb_read_frames": "22", "avg_frame_rate": "24/1", "time_base": "1/1000000"}
    stream.update(changes or {})
    times = [i / 24 for i in range(22)] if times is None else times
    run = Mock(side_effect=[json.dumps({"streams": [stream, {"codec_type": "audio"},
                                                   {"codec_type": "audio"}]}),
                            json.dumps({"frames": [{"best_effort_timestamp_time": str(t)}
                                                   for t in times]})])
    monkeypatch.setattr(shared.shutil, "which", lambda name: name)
    monkeypatch.setattr(shared, "_run", run)
    return run


@pytest.mark.parametrize("fault,match", [
    ("rate", "exactly 24 fps"), ("fractional_rate", "exactly 24 fps"),
    ("vfr", "constant 24 fps"), ("nonzero", "zero-start"),
    ("width", "multiples of 32"), ("height", "multiples of 32"),
    ("frames", "17n\\+5"), ("missing_timestamp", "constant 24 fps"),
    ("duplicate_timestamp", "constant 24 fps"),
])
@pytest.mark.parametrize("use_turbo", [False, True])
def test_strict_real_probe_rejects_source_before_worker_and_retains_originals(
        lifecycle, monkeypatch, fault, match, use_turbo):
    ctx = lifecycle
    # Recover the implementation replaced by the lifecycle's success-path spy.
    monkeypatch.setattr(shared, "probe_video", REAL_PROBE_VIDEO)
    changes, times = {}, [i / 24 for i in range(22)]
    if fault in {"rate", "fractional_rate"}:
        changes["avg_frame_rate"] = "30/1" if fault == "rate" else "24000/1001"
    elif fault == "width":
        changes["width"] = 65
    elif fault == "height":
        changes["height"] = 31
    elif fault == "frames":
        changes["nb_read_frames"] = "23"
        times.append(22 / 24)
    elif fault == "vfr":
        times[10] += 0.01
    elif fault == "nonzero":
        times = [t + 0.125 for t in times]
    elif fault == "missing_timestamp":
        times.pop()
    else:
        times[10] = times[9]
    run = mock_source_probe(monkeypatch, changes=changes, times=times)
    with pytest.raises(swap.CharacterSwapFailure, match=match) as caught:
        ctx.backend.swap(swap.CharacterSwapRequest(ctx.video, ctx.image, use_turbo=use_turbo))
    assert_originals(ctx, caught.value)
    assert run.call_count == (1 if changes else 2)
    for call in run.call_args_list:
        assert call.args[0][0] == "ffprobe"
        assert call.args[0][-1] == str(caught.value.original_path)
    ctx.backend.preflight.assert_not_called()
    ctx.backend.ensure_worker.assert_not_called()
    ctx.wait.assert_not_called()
    shared.requests.post.assert_not_called()
    assert not list(ctx.backend.config.output_dir.glob("*_character_swap.mkv"))


REAL_PROBE_VIDEO = shared.probe_video


@pytest.mark.parametrize("use_turbo", [False, True])
def test_strict_source_lifecycle_never_resamples_and_reports_effective_steps(
        lifecycle, monkeypatch, use_turbo):
    ctx = lifecycle
    probe = Mock(wraps=REAL_PROBE_VIDEO)
    monkeypatch.setattr(shared, "probe_video", probe)
    run = mock_source_probe(monkeypatch)
    forbidden = Mock(side_effect=AssertionError("Character swap must not resample"))
    for name in ("normalization_plan", "normalize_video", "restore_video_timeline"):
        monkeypatch.setattr(shared, name, forbidden)
    result = ctx.backend.swap(swap.CharacterSwapRequest(
        ctx.video, ctx.image, steps=37, strength=0.65, use_turbo=use_turbo))
    assert_originals(ctx, result)
    probe.assert_called_once_with(result.original_path)
    assert run.call_count == 2
    forbidden.assert_not_called()
    assert "Strict source alignment; no temporal resampling." in result.message
    assert f"steps={8 if use_turbo else 37}" in result.message
    assert f"turbo={use_turbo}" in result.message
    assert "24 fps" in result.message and "22 frames" in result.message
    assert ctx.graphs[0]["12"]["inputs"]["steps"] == (8 if use_turbo else 37)
    assert not ctx.staged_images[0].exists()


@pytest.mark.parametrize("normalize", [False, True])
def test_character_backend_rejects_inherited_restoration_entry_point(lifecycle, normalize):
    ctx = lifecycle
    with pytest.raises(ValueError, match="requires swap"):
        ctx.backend.enhance(shared.RestorationRequest(
            ctx.video, normalize_input=normalize, output_fps="source"))
    shared.probe_video.assert_not_called()
    ctx.backend.preflight.assert_not_called()
    ctx.backend.ensure_worker.assert_not_called()
    shared.requests.post.assert_not_called()


def native_pair(module, shape, rank=2):
    prefix = "diffusion_model." + module
    return {prefix + ".lora_A.weight": torch.zeros(rank, shape[1] if len(shape) == 2 else 1),
            prefix + ".lora_B.weight": torch.zeros(shape[0], rank)}


def pdd_tensors(banks=2, dtype=torch.int64):
    base, patches = {"adaln_t_table": torch.zeros(1025, 8),
                     "blocks.0.adaln_proj.linear.weight": torch.zeros(12, 8)}, {}
    for head, rows in (("video_out", 4), ("audio_out", 2)):
        for suffix, shape in (("", [rows, 3]), (".bias", [rows])):
            module = "final_layer." + head + suffix
            base[module if suffix else module + ".weight"] = torch.zeros(*shape)
            expanded = [rows * banks, *shape[1:]]
            patches.update(native_pair(module, expanded))
            patches["diffusion_model." + module + ".reshape_weight"] = torch.tensor(expanded, dtype=dtype)
    patches.update(native_pair("blocks.0.adaln_proj.linear", [12, 8]))
    return base, patches


@pytest.fixture
def tiny_stack(tmp_path):
    base, patches = pdd_tensors()
    model, turbo, character = [tmp_path / name for name in
                               ("base.safetensors", "turbo.safetensors", "character.safetensors")]
    save_file(base, str(model))
    save_file(patches, str(turbo), metadata={"test": "tiny CPU PDD"})
    character_tensors = flattened_pair("blocks.0.adaln_proj.linear", rows=12, columns=8)
    save_file(character_tensors, str(character))
    return SimpleNamespace(model=model, turbo=turbo, character=character,
                           patches=patches, character_tensors=character_tensors)


@pytest.mark.parametrize("banks", [2, 32])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_real_pdd_stack_accepts_all_four_integer_head_reshapes_and_curve_adaln(
        tiny_stack, banks, dtype):
    ctx = tiny_stack
    base, patches = pdd_tensors(banks, dtype)
    save_file(base, str(ctx.model))
    save_file(patches, str(ctx.turbo))
    # Later adapters must match the shapes actually produced by the turbo.
    later = dict(ctx.character_tensors)
    for head, rows in (("video_out", 4), ("audio_out", 2)):
        for suffix, shape in (("", [rows * banks, 3]), (".bias", [rows * banks])):
            later.update(native_pair("final_layer." + head + suffix, shape))
    save_file(later, str(ctx.character))
    header = shared.read_header(ctx.model)
    before_header = json.dumps(header, sort_keys=True)
    before_files = [path.read_bytes() for path in (ctx.model, ctx.turbo, ctx.character)]
    swap.validate_adapter_stack([ctx.turbo, ctx.character], header)
    assert json.dumps(header, sort_keys=True) == before_header
    assert [path.read_bytes() for path in (ctx.model, ctx.turbo, ctx.character)] == before_files


@pytest.mark.parametrize("fault,match", [
    ("columns", "Invalid PDD head reshape"), ("rank", "Invalid PDD head reshape"),
    ("nonmultiple", "Invalid integer PDD"), ("float", "Invalid integer PDD"),
    ("zero", "Invalid PDD head reshape"), ("negative", "Invalid PDD head reshape"),
    ("missing_bias", "matching video/audio"), ("orphan", "Unsupported/unmatched"),
    ("unequal_banks", "bank counts must match"), ("unmatched_adaln", "Unknown or unpaired"),
    ("full_adaln", "Full AdaLN"),
])
def test_real_pdd_stack_rejects_malformed_metadata_and_adaln(tiny_stack, fault, match):
    ctx = tiny_stack
    patches = dict(ctx.patches)
    prefix = "diffusion_model.final_layer.video_out"
    key = prefix + ".reshape_weight"
    if fault == "columns":
        patches[key] = torch.tensor([8, 4])
    elif fault == "rank":
        patches[key] = torch.tensor([8])
    elif fault == "float":
        patches[key] = patches[key].float()
    elif fault in {"nonmultiple", "zero", "negative"}:
        rows = {"nonmultiple": 6, "zero": 0, "negative": -8}[fault]
        patches[key] = torch.tensor([rows, 3])
        if rows > 0:
            patches[prefix + ".lora_B.weight"] = torch.zeros(rows, 2)
    elif fault == "orphan":
        patches.pop(prefix + ".lora_A.weight")
        patches.pop(prefix + ".lora_B.weight")
    elif fault == "missing_bias":
        prefix += ".bias"
        for suffix in (".reshape_weight", ".lora_A.weight", ".lora_B.weight"):
            patches.pop(prefix + suffix)
    elif fault == "unequal_banks":
        for suffix, shape in (("", [6, 3]), (".bias", [6])):
            module = "final_layer.audio_out" + suffix
            patches.update(native_pair(module, shape))
            patches["diffusion_model." + module + ".reshape_weight"] = torch.tensor(shape)
    elif fault == "unmatched_adaln":
        patches.update(native_pair("blocks.47.adaln_proj.linear", [12, 8]))
    else:
        patches.update(native_pair("blocks.0.adaln_proj.linear", [12, 48]))
    save_file(patches, str(ctx.turbo))
    before = ctx.turbo.read_bytes()
    with pytest.raises(ValueError, match=match):
        swap.validate_adapter_stack([ctx.turbo, ctx.character], shared.read_header(ctx.model))
    assert ctx.turbo.read_bytes() == before


@pytest.mark.parametrize("module,shape", [
    ("final_layer.video_out", [4, 3]), ("final_layer.audio_out", [2, 3]),
    ("final_layer.video_out.bias", [4]), ("final_layer.audio_out.bias", [2]),
])
def test_later_adapter_cannot_patch_old_head_dimensions_after_turbo(tiny_stack, module, shape):
    ctx = tiny_stack
    later = dict(ctx.character_tensors)
    later.update(native_pair(module, shape))
    save_file(later, str(ctx.character))
    header = shared.read_header(ctx.model)
    shared.validate_adapter(ctx.character, header)  # Individually valid against the old base.
    with pytest.raises(ValueError, match="Incompatible Restoration target final_layer"):
        swap.validate_adapter_stack([ctx.turbo, ctx.character], header)


@pytest.fixture
def verified_turbo_assets(tiny_assets, monkeypatch):
    ctx = tiny_assets
    base, patches = pdd_tensors()
    base.update({"blocks.0.attn.out_proj.weight": torch.zeros(8, 4),
                 "token_refiner.blocks.1.attn.qkv_proj.weight": torch.zeros(24, 4)})
    save_file(base, str(ctx.model))
    ctx.turbo = ctx.backend.root / "models/loras" / assets.REF2VA_TURBO_FILE
    save_file(patches, str(ctx.turbo))
    ctx.turbo_tensors = patches
    for relative, marker in (("comfy/weight_adapter/lora.py", "reshape_weight"),
                             ("comfy/ldm/minimax/model.py", "_pdd_head")):
        path = ctx.backend.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker, encoding="utf-8")
    for constant, path in (("REF2VA_MODEL_SHA256", ctx.model),
                           ("CHARACTER_SWAP_LORA_SHA256", ctx.adapter),
                           ("REF2VA_TURBO_SHA256", ctx.turbo)):
        monkeypatch.setattr(swap, constant, hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(swap, "REF2VA_TURBO_SIZE", ctx.turbo.stat().st_size)
    return ctx


@pytest.mark.parametrize("use_turbo", [False, True])
def test_preflight_requires_and_verifies_turbo_only_when_enabled(verified_turbo_assets, monkeypatch, use_turbo):
    ctx = verified_turbo_assets
    if not use_turbo:
        ctx.turbo.unlink()
        (ctx.backend.root / "comfy/weight_adapter/lora.py").unlink()
        (ctx.backend.root / "comfy/ldm/minimax/model.py").unlink()
    verify = Mock(wraps=shared.verify_asset)
    monkeypatch.setattr(shared, "verify_asset", verify)
    ctx.backend.preflight(request(use_turbo=use_turbo))
    assert [call.args[0] for call in verify.call_args_list] == (
        [ctx.model, ctx.adapter, ctx.turbo] if use_turbo else [ctx.model, ctx.adapter])
    assert all(call.args[1] == hashlib.sha256(call.args[0].read_bytes()).hexdigest()
               for call in verify.call_args_list)


@pytest.mark.parametrize("fault,match", [
    ("missing", "assets missing"), ("size", "size mismatch"), ("checksum", "checksum mismatch"),
    ("later_pair", "LoRA incompatible.*token_refiner"),
    ("old_head", "LoRA incompatible.*final_layer.video_out"),
    ("pdd_loader", "native PDD support"), ("pdd_model", "native PDD support"),
])
def test_turbo_preflight_failure_keeps_originals_without_starting_worker(
        verified_turbo_assets, lifecycle, monkeypatch, fault, match):
    ctx, life = verified_turbo_assets, lifecycle
    # Both fixtures share config/root; use the real preflight on lifecycle's worker.
    monkeypatch.setattr(life.backend, "preflight", lambda: ctx.backend.preflight(request(use_turbo=True)))
    if fault == "missing":
        ctx.turbo.unlink()
    elif fault == "size":
        monkeypatch.setattr(swap, "REF2VA_TURBO_SIZE", ctx.turbo.stat().st_size + 1)
    elif fault == "checksum":
        monkeypatch.setattr(swap, "REF2VA_TURBO_SHA256", "0" * 64)
    elif fault in {"later_pair", "old_head"}:
        tensors = dict(ctx.tensors)
        if fault == "later_pair":
            tensors["lora_unet_token_refiner_blocks_1_attn_qkv_proj.lora_up.weight"] = torch.zeros(23, 2)
        else:
            tensors.update(native_pair("final_layer.video_out", [4, 3]))
        save_file(tensors, str(ctx.adapter))
        # Hash verification passes: malformed later patches, not damaged downloads.
        monkeypatch.setattr(swap, "CHARACTER_SWAP_LORA_SHA256", hashlib.sha256(ctx.adapter.read_bytes()).hexdigest())
    else:
        relative = "comfy/weight_adapter/lora.py" if fault == "pdd_loader" else "comfy/ldm/minimax/model.py"
        (ctx.backend.root / relative).write_text("outdated", encoding="utf-8")
    with pytest.raises(swap.CharacterSwapFailure, match=match) as caught:
        life.backend.swap(swap.CharacterSwapRequest(life.video, life.image, use_turbo=True))
    assert_originals(life, caught.value)
    life.backend.ensure_worker.assert_not_called()
    life.wait.assert_not_called()
    shared.requests.post.assert_not_called()