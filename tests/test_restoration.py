"""Offline Restoration contract tests: tiny local adapters, no models or GPU worker."""

from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from minimax_h3_fl2v import assets, download, restoration, ui
from minimax_h3_fl2v.comfy_backend import PrunedComfyBackend
from minimax_h3_fl2v.config import AppConfig, GenerationRequest, LoRASpec
from minimax_h3_fl2v.pipeline import MiniMaxH3Engine


@pytest.fixture(autouse=True)
def no_external_io(monkeypatch):
    """An accidentally unmocked download, ffmpeg or worker must fail locally."""
    def blocked(*args, **kwargs):
        pytest.fail("Unexpected network, subprocess or worker access")

    monkeypatch.setattr(restoration.requests, "get", blocked)
    monkeypatch.setattr(restoration.requests, "post", blocked)
    monkeypatch.setattr(restoration.websocket, "create_connection", blocked)
    monkeypatch.setattr(restoration.subprocess, "run", blocked)
    monkeypatch.setattr(restoration.subprocess, "Popen", blocked)
    monkeypatch.setattr(download, "hf_hub_download", blocked)
    monkeypatch.setattr(download, "snapshot_download", blocked)
    restoration._VERIFIED.clear()
    yield
    restoration._VERIFIED.clear()


@pytest.fixture
def info():
    return restoration.VideoInfo(64, 32, 22, Fraction(24), 2)


@pytest.fixture
def config(tmp_path):
    return AppConfig(
        output_dir=tmp_path / "outputs", upload_dir=tmp_path / "uploads",
        lora_dir=tmp_path / "loras", default_lora_id="turbo",
        catalog=[LoRASpec(id="turbo", name="Turbo", backend="comfy_pruned")],
    )


def workflow(info, *, nodes=None, audio_source=None, **settings):
    return restoration.RestorationBackend.build_restoration_workflow(
        source="source.mp4", info=info,
        request=restoration.RestorationRequest(Path("source.mp4"), **settings),
        text_encoder="encoder.safetensors", filename_prefix="restoration/test",
        nodes=nodes, audio_source=audio_source,
    )


def reference_schema():
    """Live native V3 groups, including optional original soundtrack references."""
    def autogrow(prefix, kind):
        return ["COMFY_AUTOGROW_V3", {"template": {
            "prefix": prefix, "min": 0, "max": 10,
            "input": {"required": {prefix: [kind]}},
        }}]

    return {"input": {
        "required": {"clip": ["CLIP"], "vae": ["VAE"], "prompt": ["STRING"],
            "width": ["INT"], "height": ["INT"], "length": ["INT"],
            "ref_image_size": [["match"]]},
        "optional": {"audio_vae": ["VAE"],
            "ref_videos": autogrow("ref_video_", "IMAGE"),
            "ref_video_audios": autogrow("ref_video_audio_", "AUDIO")},
    }}


def test_workflow_is_source_aligned_ref2va_not_fl2va(info):
    graph = workflow(info, strength=0.75, steps=12, seed=123)
    assert graph["1"]["inputs"]["unet_name"] == assets.REF2VA_MODEL_FILE
    assert assets.REF2VA_MODEL_FILE != AppConfig().comfy_model
    assert graph["50"] == {"class_type": "LoadVideo", "inputs": {"file": "source.mp4"}}
    assert graph["51"]["inputs"] == {"video": ["50", 0]}
    assert graph["54"] == {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
        "clip": ["2", 0], "vae": ["3", 0], "prompt": restoration.RESTORATION_CAPTION,
        "width": 64, "height": 32, "length": 22, "ref_image_size": "match",
        "ref_videos.ref_video_0": ["51", 0], "audio_vae": ["4", 0],
        "ref_video_audios.ref_video_audio_0": ["51", 1]}}
    assert graph["13"]["inputs"]["conditioning"] == ["54", 0]
    assert graph["15"]["inputs"]["latent_image"] == ["54", 1]
    assert graph["12"]["inputs"]["steps"] == 12
    assert graph["10"]["inputs"]["noise_seed"] == 123
    assert graph["21"]["inputs"] == {"model": ["1", 0],
        "lora_name": assets.RESTORE_FILE, "strength_model": 0.75}
    assert graph["13"]["inputs"]["model"] == ["21", 0]
    assert sum(node["class_type"] == graph["21"]["class_type"] for node in graph.values()) == 1
    assert not {"22", "52", "53", "55", "56"} & graph.keys()
    assert graph["18"]["inputs"]["fps"] == 24.0
    assert "audio" not in graph["18"]["inputs"]
    assert "9" not in graph and "17" not in graph
    assert not {"VAEDecodeAudio", "MiniMaxH3ImageToVideo", "LoadImage",
        "MiniMaxH3AddGuide", "EmptyMiniMaxH3LatentAV"} & {
        node["class_type"] for node in graph.values()}
    # Every graph link resolves; no dangling reference to discarded audio/conditioning.
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in graph


def test_video_only_native_graph_has_no_audio_vae_or_soundtrack_reference(info):
    graph = workflow(replace(info, audio_tracks=0))
    inputs = graph["54"]["inputs"]
    assert inputs["ref_videos.ref_video_0"] == ["51", 0]
    assert "audio_vae" not in inputs
    assert not any(key.startswith("ref_video_audio") for key in inputs)
    assert not {"55", "56", "17"} & graph.keys()
    assert "audio" not in graph["18"]["inputs"]
    assert graph["12"]["inputs"]["steps"] == 30


def test_fl2va_defaults_remain_unchanged(info, config):
    before = replace(config)
    backend = restoration.RestorationBackend(config)
    workflow(info)
    assert config == before
    assert backend.model == assets.REF2VA_MODEL_FILE
    assert GenerationRequest(prompt="Landscape").target_fps == 23.976
    graph = PrunedComfyBackend.build_workflow(
        prompt="Landscape", width=64, height=32, frames=22, seed=42, nfe=8, loras=[])
    assert graph["1"]["inputs"]["unet_name"] == config.comfy_model
    assert graph["17"]["class_type"] == "VAEDecodeAudio"
    assert graph["18"]["inputs"]["audio"] == ["17", 0]
    assert graph["18"]["inputs"]["fps"] == 23.976
    assert "54" not in graph


@pytest.mark.parametrize("reserve", [0.0, 0.5, 1.0, 2.5])
def test_restoration_forces_cache_none_and_minimum_reserve(config, reserve):
    config.comfy_cache_none = False
    config.comfy_reserve_vram_gb = reserve
    backend = restoration.RestorationBackend(config)
    assert backend.config.comfy_cache_none is True
    assert backend.config.comfy_reserve_vram_gb == max(1.0, reserve)
    assert config.comfy_cache_none is False
    assert config.comfy_reserve_vram_gb == reserve


@pytest.mark.parametrize("changes", [
    {"width": 0}, {"width": 31}, {"width": 65}, {"height": 16}, {"height": 33},
    {"frames": 4}, {"frames": 6}, {"frames": 3601},
    {"fps": Fraction(24000, 1001)}, {"fps": Fraction(60)},
])
def test_rejects_incompatible_source_without_normalization(info, changes):
    with pytest.raises(ValueError):
        replace(info, **changes).validate()


@pytest.mark.parametrize("frames", [5, 22, 124, 3592])
def test_accepts_17n_plus_5_frames(info, frames):
    replace(info, frames=frames).validate()


@pytest.mark.parametrize("settings", [
    {"strength": 0}, {"strength": -1}, {"strength": 2.01},
    {"strength": float("nan")}, {"strength": float("inf")},
    {"steps": True}, {"steps": 3}, {"steps": 51}, {"steps": 8.5},
    {"seed": True}, {"seed": -1}, {"seed": 2**64}, {"seed": 1.5},
])
def test_request_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        restoration.RestorationRequest(Path("source.mp4"), **settings).validate()


@pytest.mark.parametrize("settings", [{}, {"strength": 2, "steps": 4, "seed": 0},
    {"strength": 0.05, "steps": 50, "seed": 2**64 - 1}])
def test_request_defaults_and_valid_boundaries(settings):
    request = restoration.RestorationRequest(Path("source.mp4"), **settings)
    request.validate()
    if not settings:
        assert (request.strength, request.steps, request.seed) == (1.0, 30, 42)


def mock_probe(monkeypatch, *, stream_changes=None, times=None, extra_streams=None):
    stream = {"codec_type": "video", "width": 64, "height": 32,
        "nb_read_frames": "5", "avg_frame_rate": "24/1"}
    stream.update(stream_changes or {})
    streams = [stream, {"codec_type": "audio"}, {"codec_type": "audio"}]
    streams.extend(extra_streams or [])
    times = [i / 24 for i in range(5)] if times is None else times
    run = Mock(side_effect=[json.dumps({"streams": streams}), json.dumps({"frames": [
        {"best_effort_timestamp_time": str(t)} for t in times]})])
    monkeypatch.setattr(restoration.shutil, "which", lambda name: name)
    monkeypatch.setattr(restoration, "_run", run)
    return run


def test_probe_checks_decoded_cfr_timestamps_and_audio_count(monkeypatch):
    run = mock_probe(monkeypatch)
    assert restoration.probe_video(Path("source.mp4")) == restoration.VideoInfo(64, 32, 5, Fraction(24), 2)
    first, second = [call.args[0] for call in run.call_args_list]
    assert first[0] == second[0] == "ffprobe"
    assert "-count_frames" in first
    assert "-show_frames" in second
    assert "frame=best_effort_timestamp_time,duration_time,pkt_duration_time" in second
    assert first[-1] == second[-1] == "source.mp4"


@pytest.mark.parametrize("times,match", [
    ([0, 1/24, 2/24, 4/24, 5/24], "constant 24 fps"),
    ([0, 1/24, 2/24, 3/24], "constant 24 fps"),
    ([0, 1/24, 1/24, 3/24, 4/24], "constant 24 fps"),
    ([1 + i/24 for i in range(5)], "zero-start"),
])
def test_probe_rejects_vfr_missing_frames_and_nonzero_start(monkeypatch, times, match):
    mock_probe(monkeypatch, times=times)
    with pytest.raises(ValueError, match=match):
        restoration.probe_video(Path("source.mp4"))


@pytest.mark.parametrize("changes", [{"avg_frame_rate": "24000/1001"},
    {"tags": {"rotate": "90"}}, {"side_data_list": [{"rotation": -90}]}])
def test_probe_rejects_rate_and_rotation(monkeypatch, changes):
    run = mock_probe(monkeypatch, stream_changes=changes)
    with pytest.raises(ValueError):
        restoration.probe_video(Path("source.mp4"))
    assert run.call_count == 1


def test_probe_ignores_attached_cover_but_rejects_multiple_video_streams(monkeypatch):
    mock_probe(monkeypatch, extra_streams=[{"codec_type": "video", "disposition": {"attached_pic": 1}}])
    assert restoration.probe_video(Path("source.mp4")).frames == 5
    mock_probe(monkeypatch, extra_streams=[{"codec_type": "video"}])
    with pytest.raises(ValueError, match="exactly one video"):
        restoration.probe_video(Path("source.mp4"))


def test_probe_requires_local_binaries(monkeypatch):
    monkeypatch.setattr(restoration.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg and ffprobe"):
        restoration.probe_video(Path("source.mp4"))


def adapter_file(tmp_path, tensors):
    path = tmp_path / "adapter.safetensors"
    save_file(tensors, str(path), metadata={"test": "tiny CPU fixture"})
    return path


def pair(module="blocks.0.attn.q", *, rows=8, columns=4, rank=2, peft=True):
    prefix = "diffusion_model." + module
    down, up = ("lora_A", "lora_B") if peft else ("lora_down", "lora_up")
    return {f"{prefix}.{down}.weight": torch.zeros(rank, columns),
        f"{prefix}.{up}.weight": torch.zeros(rows, rank)}


@pytest.mark.parametrize("peft", [True, False])
def test_adapter_validates_real_safetensors_pairs_alpha_and_adaln(tmp_path, peft):
    tensors = pair(peft=peft)
    tensors.update(pair("blocks.0.adaln", rows=8, columns=2, peft=peft))
    tensors["diffusion_model.blocks.0.attn.q.alpha"] = torch.tensor(2.0)
    tensors["diffusion_model.blocks.0.attn.q.diff_b"] = torch.zeros(8)
    model = {"blocks.0.attn.q.weight": {"shape": [8, 4]},
        "blocks.0.attn.q.bias": {"shape": [8]}, "blocks.0.adaln.weight": {"shape": [8, 2]}}
    path = adapter_file(tmp_path, tensors)
    assert restoration.read_header(path)["diffusion_model.blocks.0.adaln.lora_A.weight" if peft
        else "diffusion_model.blocks.0.adaln.lora_down.weight"]["shape"] == [2, 2]
    restoration.validate_adapter(path, model)


@pytest.mark.parametrize("module,base,expanded", [
    ("final_layer.video_out", [4, 3], [8, 3]),
    ("final_layer.audio_out", [4, 3], [8, 3]),
    ("final_layer.video_out.bias", [4], [8]),
    ("final_layer.audio_out.bias", [4], [8]),
])
def test_adapter_supports_expanded_pdd_heads(tmp_path, module, base, expanded):
    tensors = pair(module, rows=expanded[0], columns=expanded[1] if len(expanded) == 2 else 1)
    tensors[f"diffusion_model.{module}.reshape_weight"] = torch.tensor(expanded)
    target = module if module.endswith(".bias") else module + ".weight"
    restoration.validate_adapter(adapter_file(tmp_path, tensors), {target: {"shape": base}})


def flattened_pair(module, *, rows=8, columns=4, rank=2):
    prefix = "lora_unet_" + module.replace(".", "_")
    # SafeTensors sorts alpha ahead of the factors in the downloaded adapter.
    return {prefix + ".alpha": torch.tensor(7.0),
        prefix + ".lora_down.weight": torch.zeros(rank, columns),
        prefix + ".lora_up.weight": torch.zeros(rows, rank)}


@pytest.mark.parametrize("module", [
    "blocks.0.attn.out_proj", "blocks.0.attn.qkv_proj",
    "blocks.47.mlp.fc1", "blocks.47.mlp.fc2",
    "token_refiner.blocks.0.attn.out_proj", "token_refiner.blocks.1.attn.qkv_proj",
    "token_refiner.blocks.1.mlp.fc1", "blocks.0.adaln_proj.linear",
    "final_layer.adaln_proj.linear",
])
def test_flattened_native_vocabulary_alpha_rank_and_no_conversion(tmp_path, module, monkeypatch):
    tensors = flattened_pair(module)
    path = adapter_file(tmp_path, tensors)
    before = path.read_bytes()
    # Ordinary validation must inspect shapes only, not load factor/alpha data.
    real_open = restoration.safe_open

    class HeaderOnly:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_tensor(self, key):
            raise AssertionError(f"Unexpected tensor load: {key}")

    monkeypatch.setattr(restoration, "safe_open", lambda *args, **kwargs: HeaderOnly())
    restoration.validate_adapter(path, {module + ".weight": {"shape": [8, 4]}})
    assert path.read_bytes() == before
    with real_open(str(path), framework="pt", device="cpu") as loaded:
        prefix = "lora_unet_" + module.replace(".", "_")
        assert loaded.get_tensor(prefix + ".alpha").item() == 7.0
        assert loaded.get_slice(prefix + ".lora_down.weight").get_shape() == [2, 4]
        assert loaded.get_slice(prefix + ".lora_up.weight").get_shape() == [8, 2]


def test_flattened_checks_all_pairs_and_all_auxiliary_keys(tmp_path):
    tensors = flattened_pair("blocks.0.attn.out_proj")
    tensors.update(flattened_pair("token_refiner.blocks.1.attn.qkv_proj", rows=24))
    model = {"blocks.0.attn.out_proj.weight": {"shape": [8, 4]},
        "token_refiner.blocks.1.attn.qkv_proj.weight": {"shape": [24, 4]}}
    restoration.validate_adapter(adapter_file(tmp_path, tensors), model)
    tensors["lora_unet_token_refiner_blocks_1_attn_qkv_proj.lora_up.weight"] = torch.zeros(23, 2)
    with pytest.raises(ValueError, match="Incompatible Restoration target token_refiner"):
        restoration.validate_adapter(adapter_file(tmp_path, tensors), model)


@pytest.mark.parametrize("case,match", [
    ("unknown", "Unknown or unpaired"), ("unpaired", "Unknown or unpaired"),
    ("naive_dotted", "Unknown or unpaired"), ("rank", "Incompatible Restoration target"),
    ("zero_rank", "Incompatible Restoration target"), ("alpha", "Invalid adapter alpha"),
    ("orphan_alpha", "Unsupported/unmatched"), ("orphan_up", "Unsupported/unmatched"),
    ("extra", "Unsupported/unmatched"), ("full_adaln", "Full AdaLN"),
    ("alias_collision", "flattened alias collision"), ("duplicate_alias", "target collision"),
    ("duplicate_format", "target collision"),
])
def test_flattened_rejects_unknown_incompatible_or_colliding_patches(tmp_path, case, match):
    module = "blocks.0.attn.out_proj"
    prefix = "lora_unet_blocks_0_attn_out_proj"
    tensors = flattened_pair(module)
    model = {module + ".weight": {"shape": [8, 4]}}
    if case == "unknown":
        tensors.update(flattened_pair("blocks.0.attn.unknown_proj"))
    elif case == "unpaired":
        tensors.pop(prefix + ".lora_up.weight")
    elif case == "naive_dotted":
        model = {"blocks.0.attn.out.proj.weight": {"shape": [8, 4]}}
        # Same flattened spelling but different exact vocabulary is ambiguous
        # only when both modules exist; a dotted key must not be rewritten.
        tensors = pair(module, peft=False)
    elif case == "rank":
        tensors[prefix + ".lora_up.weight"] = torch.zeros(8, 3)
    elif case == "zero_rank":
        tensors = flattened_pair(module, rank=0)
    elif case == "alpha":
        tensors[prefix + ".alpha"] = torch.zeros(2)
    elif case == "orphan_alpha":
        tensors["lora_unet_unknown.alpha"] = torch.tensor(2.0)
    elif case == "orphan_up":
        tensors["lora_unet_unknown.lora_up.weight"] = torch.zeros(8, 2)
    elif case == "extra":
        tensors[prefix + ".unknown"] = torch.zeros(1)
    elif case == "full_adaln":
        tensors.update(flattened_pair("blocks.0.adaln_proj.linear", rows=48, columns=4))
        model["blocks.0.adaln_proj.linear.weight"] = {"shape": [48, 8]}
    elif case == "alias_collision":
        model["blocks.0.attn.out.proj.weight"] = {"shape": [8, 4]}
    elif case == "duplicate_alias":
        tensors.update(pair(module, peft=False))
    elif case == "duplicate_format":
        tensors[prefix + ".lora_A.weight"] = torch.zeros(2, 4)
        tensors[prefix + ".lora_B.weight"] = torch.zeros(8, 2)
    with pytest.raises(ValueError, match=match):
        restoration.validate_adapter(adapter_file(tmp_path, tensors), model)


def test_flattened_vector_alpha_and_bias(tmp_path):
    module = "token_refiner.blocks.0.attn.out_proj"
    prefix = "lora_unet_" + module.replace(".", "_")
    tensors = flattened_pair(module)
    tensors[prefix + ".alpha"] = torch.tensor([3.0])
    tensors[prefix + ".diff_b"] = torch.zeros(8)
    model = {module + ".weight": {"shape": [8, 4]}, module + ".bias": {"shape": [8]}}
    restoration.validate_adapter(adapter_file(tmp_path, tensors), model)
    tensors[prefix + ".diff_b"] = torch.zeros(9)
    with pytest.raises(ValueError, match="Incompatible adapter bias"):
        restoration.validate_adapter(adapter_file(tmp_path, tensors), model)


@pytest.mark.parametrize("case,match", [
    ("unknown", "Unsupported/unmatched"), ("foreign", "Unsupported Restoration adapter key"),
    ("unpaired", "Unknown or unpaired"), ("unknown_target", "Unknown or unpaired"),
    ("full_adaln", "Full AdaLN"), ("rank", "Incompatible Restoration target"),
    ("alpha", "Invalid adapter alpha"), ("bias", "Incompatible adapter bias"),
    ("reshape", "Unsupported reshaped"), ("empty", "Unsupported/unmatched"),
])
def test_adapter_rejects_every_unmatched_or_incompatible_patch(tmp_path, case, match):
    tensors = pair()
    model = {"blocks.0.attn.q.weight": {"shape": [8, 4]}}
    if case == "unknown":
        tensors["diffusion_model.blocks.0.attn.q.unrecognized"] = torch.zeros(1)
    elif case == "foreign":
        tensors["foreign.weight"] = torch.zeros(1)
    elif case == "unpaired":
        tensors.pop("diffusion_model.blocks.0.attn.q.lora_B.weight")
    elif case == "unknown_target":
        model = {}
    elif case == "full_adaln":
        tensors = pair("blocks.0.adaln", rows=48, columns=4)
        model = {"blocks.0.adaln.weight": {"shape": [8, 4]}}
    elif case == "rank":
        tensors["diffusion_model.blocks.0.attn.q.lora_B.weight"] = torch.zeros(8, 3)
    elif case == "alpha":
        tensors["diffusion_model.blocks.0.attn.q.alpha"] = torch.zeros(2)
    elif case == "bias":
        tensors["diffusion_model.blocks.0.attn.q.diff_b"] = torch.zeros(9)
        model["blocks.0.attn.q.bias"] = {"shape": [8]}
    elif case == "reshape":
        tensors["diffusion_model.blocks.0.attn.q.reshape_weight"] = torch.tensor([8, 4])
    elif case == "empty":
        tensors = {"diffusion_model.blocks.0.attn.q.diff_b": torch.zeros(8)}
        model["blocks.0.attn.q.bias"] = {"shape": [8]}
    with pytest.raises(ValueError, match=match):
        restoration.validate_adapter(adapter_file(tmp_path, tensors), model)


@pytest.mark.parametrize("shape", [[3, 3], [8, 4], [8, 3, 1]])
def test_adapter_rejects_invalid_pdd_reshape(tmp_path, shape):
    tensors = pair("final_layer.video_out", rows=8, columns=3)
    tensors["diffusion_model.final_layer.video_out.reshape_weight"] = torch.tensor(shape)
    with pytest.raises(ValueError, match="Invalid PDD head reshape"):
        restoration.validate_adapter(adapter_file(tmp_path, tensors), {"final_layer.video_out.weight": {"shape": [4, 3]}})


@pytest.mark.parametrize("payload", [b"short", struct.pack("<Q", 1), struct.pack("<Q", 16*1024*1024+1)])
def test_header_read_is_bounded(tmp_path, payload):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="Invalid SafeTensors"):
        restoration.read_header(path)


def test_asset_hash_cache_does_not_accept_different_checksum_or_changed_file(tmp_path):
    path = tmp_path / "asset"
    path.write_bytes(b"tiny")
    checksum = hashlib.sha256(b"tiny").hexdigest()
    restoration.verify_asset(path, checksum)
    restoration.verify_asset(path, checksum)
    assert len(restoration._VERIFIED) == 1
    with pytest.raises(ValueError, match="checksum mismatch"):
        restoration.verify_asset(path, "0" * 64)
    path.write_bytes(b"different size")
    with pytest.raises(ValueError, match="checksum mismatch"):
        restoration.verify_asset(path, checksum)


@pytest.mark.parametrize("kind", ["restore", "restore_base"])
def test_optional_downloads_use_exact_pinned_revisions_hashes_sizes(tmp_path, monkeypatch, kind):
    fetch = Mock(side_effect=lambda url, destination, **kwargs: destination)
    monkeypatch.setattr(download, "download_url_file", fetch)
    paths = getattr(download, "download_" + kind)(tmp_path)
    folder = "loras" if kind == "restore" else "diffusion_models"
    prefix = "RESTORE" if kind == "restore" else "REF2VA_MODEL"
    filename, checksum, size = [getattr(assets, prefix + "_" + suffix)
        for suffix in ("FILE", "SHA256", "SIZE")]
    if kind == "restore":
        url = assets.RESTORE_URL
    else:
        repo, revision = assets.REF2VA_MODEL_REPO, assets.REF2VA_MODEL_REVISION
        assert len(revision) == 40 and len(checksum) == 64
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{folder}/{filename}"
    destination = tmp_path / "models" / folder / filename
    fetch.assert_called_once_with(url, destination,
        expected_sha256=checksum, expected_size=size)
    assert paths == [destination]


def test_restore_metadata_matches_exact_civitai_file():
    assert assets.RESTORE_FILE == "Restore_Enhance_Improve_rank16_v1_H3-lora.safetensors"
    assert assets.RESTORE_URL == "https://civitai.red/api/download/models/3389308?fileId=3278511"
    assert assets.RESTORE_SHA256 == "04068fccc4f7c14c3f42d0c1dc1a408eb335fa9c31149d068174e702eab47296"
    assert assets.RESTORE_SIZE == 149165944


@pytest.mark.parametrize("kind", [None, "12gb", "backend", "all", "h100"])
def test_default_and_bulk_downloads_exclude_optional_restoration(kind):
    selected = download.resolve_types(model_type=kind, base=False, loras=False, all_flag=False)
    assert not {"restore", "restore_base"} & set(selected)
    assert not any("ref2va" in name.lower() or "restoration" in name.lower() for name in assets.PRUNED_FILES)


@pytest.mark.parametrize("kind,filename", [("restore", assets.RESTORE_FILE),
                                          ("restore_base", assets.REF2VA_MODEL_FILE)])
def test_restoration_download_never_deletes_existing_invalid_weights(tmp_path, monkeypatch, kind, filename):
    folder = "loras" if kind == "restore" else "diffusion_models"
    path = tmp_path / "models" / folder / filename
    path.parent.mkdir(parents=True)
    path.write_bytes(b"retain existing checkpoint")
    fetch = Mock(side_effect=AssertionError("must not replace weights"))
    monkeypatch.setattr(download, "download_url_file", fetch)
    with pytest.raises(RuntimeError, match="retained unchanged"):
        getattr(download, "download_" + kind)(tmp_path)
    assert path.read_bytes() == b"retain existing checkpoint"
    fetch.assert_not_called()


@pytest.mark.parametrize("kind", ["restore", "restore_base"])
def test_optional_download_dispatch_is_isolated(config, tmp_path, monkeypatch, kind):
    selected = download.resolve_types(model_type=kind, base=False, loras=False, all_flag=False)
    assert selected == [kind]
    assert download._download_providers(selected, config) == {"civitai" if kind == "restore" else "huggingface"}
    monkeypatch.setattr(download, "allow_online_for_download", Mock())
    functions = {name: Mock() for name in ["restore", "restore_base", "pruned_models", "postprocess_models", "base_model", "loras"]}
    for name, mock in functions.items():
        monkeypatch.setattr(download, "download_" + name, mock)
    download.run_downloads(selected, config=config, comfy_root=tmp_path)
    functions[kind].assert_called_once_with(tmp_path)
    assert sum(mock.call_count for mock in functions.values()) == 1


def test_preflight_missing_assets_fails_without_starting_worker(config, tmp_path, monkeypatch):
    backend = restoration.RestorationBackend(config)
    backend.root = tmp_path / "worker"
    monkeypatch.setattr(backend, "ensure_worker", Mock(side_effect=AssertionError("worker started")))
    with pytest.raises(FileNotFoundError, match="Optional Restoration assets missing") as exc:
        backend.preflight()
    assert assets.REF2VA_MODEL_FILE in str(exc.value)
    assert assets.RESTORE_FILE in str(exc.value)
    assert "turbo" not in str(exc.value).lower()
    backend.ensure_worker.assert_not_called()


@pytest.mark.parametrize("case", ["valid", "busy", "wrong_directory", "low_reserve", "cached"])
def test_restoration_worker_profile_checks_live_launch_args_and_queue(config, monkeypatch, case):
    backend = restoration.RestorationBackend(config)
    argv = ["main.py", "--cache-none", "--reserve-vram", "1.0", "--input-directory",
            str(backend.input_dir), "--output-directory", str(backend.output_dir)]
    if case == "wrong_directory":
        argv[argv.index("--input-directory") + 1] = "wrong-directory"
    elif case == "low_reserve":
        argv[argv.index("--reserve-vram") + 1] = "0.1"
    elif case == "cached":
        argv.remove("--cache-none")
    def get(url, **kwargs):
        data = {"system": {"argv": argv}} if url.endswith("system_stats") else {
            "queue_running": ["job"] if case == "busy" else [], "queue_pending": []}
        return Mock(json=Mock(return_value=data))
    monkeypatch.setattr(restoration.requests, "get", get)
    if case == "valid":
        backend.validate_worker_profile()
    else:
        with pytest.raises(RuntimeError):
            backend.validate_worker_profile()


@pytest.mark.parametrize("case", ["valid", "full_adaln", "missing_pdd", "missing_dynamic_vram"])
def test_preflight_checks_curve_basis_adapter_hashes_and_loader_support(config, tmp_path, monkeypatch, case):
    backend = restoration.RestorationBackend(config)
    backend.root = tmp_path / "worker"
    required = [backend.root / "main.py", backend.root / "models/diffusion_models" / backend.model]
    monkeypatch.setattr(backend, "_required_paths", lambda: required)
    adapters = [backend.root / "models/loras" / assets.RESTORE_FILE]
    for path in required + adapters:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tiny mocked asset")
    markers = {"comfy/weight_adapter/lora.py": "reshape_weight",
        "comfy/ldm/minimax/model.py": "_pdd_head", "comfy/cli_args.py": "enables_dynamic_vram"}
    for relative, marker in markers.items():
        path = backend.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if (case == "missing_pdd" and marker == "_pdd_head") or (
                case == "missing_dynamic_vram" and marker == "enables_dynamic_vram"):
            marker = "outdated implementation"
        path.write_text(marker, encoding="utf-8")
    header = {"adaln_t_table": {"shape": [1025, 48] if case == "full_adaln" else [1025, 8]}}
    verify, validate = Mock(), Mock()
    monkeypatch.setattr(restoration, "verify_asset", verify)
    monkeypatch.setattr(restoration, "read_header", Mock(return_value=header))
    monkeypatch.setattr(restoration, "validate_adapter", validate)
    if case == "valid":
        backend.preflight()
        assert [call.args for call in verify.call_args_list] == [
            (required[1], assets.REF2VA_MODEL_SHA256), (adapters[0], assets.RESTORE_SHA256)]
        assert [call.args for call in validate.call_args_list] == [(path, header) for path in adapters]
    else:
        with pytest.raises((ValueError, RuntimeError), match="curve model|verified Restoration/PDD/DynamicVRAM"):
            backend.preflight()
        if case == "full_adaln":
            validate.assert_not_called()


@pytest.fixture
def lifecycle(config, tmp_path, monkeypatch, info):
    backend = restoration.RestorationBackend(config)
    backend.root = tmp_path / "worker"
    source = tmp_path / "source.MP4"
    source.write_bytes(b"original video and audio bytes")
    events = []
    monkeypatch.setattr(restoration, "probe_video", Mock(return_value=info))
    monkeypatch.setattr(backend, "preflight", Mock(side_effect=lambda: events.append("preflight")))
    def start(**kwargs):
        assert list(backend.input_dir.glob("*.mp4"))[0].read_bytes() == source.read_bytes()
        events.append("worker")
    monkeypatch.setattr(backend, "ensure_worker", Mock(side_effect=start))
    # Runtime argv/queue checks are separately owned by the parent implementation.
    monkeypatch.setattr(backend, "validate_worker_profile", Mock(), raising=False)
    graph = workflow(info)
    schema = {node["class_type"]: {"input": {"required": {key: [] for key in node["inputs"]},
        "optional": {"unused_optional": []}}} for node in graph.values()}
    schema["MiniMaxH3ReferenceToVideo"] = reference_schema()
    monkeypatch.setattr(restoration.requests, "get", Mock(return_value=Mock(json=Mock(return_value=schema))))
    def post(url, **kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        events.append(endpoint)
        if endpoint == "free":
            assert kwargs["json"] == {"unload_models": True, "free_memory": True}
        else:
            assert endpoint == "prompt"
            assert events.index("free") < events.index("prompt")
            prompt = kwargs["json"]["prompt"]
            if "55" in prompt:
                guide = backend.input_dir / prompt["50"]["inputs"]["file"]
                soundtrack = backend.input_dir / prompt["55"]["inputs"]["file"]
                assert guide != soundtrack and guide.read_bytes() == b"normalized guide"
                assert soundtrack.read_bytes() == source.read_bytes()
        return Mock(ok=True, json=Mock(return_value={"prompt_id": "job"}))
    monkeypatch.setattr(restoration.requests, "post", Mock(side_effect=post))
    socket = Mock()
    monkeypatch.setattr(restoration.websocket, "create_connection", Mock(return_value=socket))
    output = tmp_path / "worker_result.mp4"
    output.write_bytes(b"enhanced silent video")
    wait = Mock(return_value=(output, {}))
    monkeypatch.setattr(backend, "_wait_for_output", wait)
    def mux(enhanced, original, destination, actual_info):
        assert enhanced == output and actual_info == info
        assert original.read_bytes() == source.read_bytes()
        destination.write_bytes(b"enhanced video with original audio")
        events.append("mux")
    monkeypatch.setattr(restoration, "mux_original_audio", Mock(side_effect=mux))
    return SimpleNamespace(backend=backend, source=source, events=events, schema=schema,
        socket=socket, wait=wait, output=output)


def test_enhance_success_persists_original_releases_before_prompt_and_after(lifecycle, info):
    ctx = lifecycle
    progress = Mock()
    request = restoration.RestorationRequest(ctx.source)
    result = ctx.backend.enhance(request, progress)
    assert result.original != ctx.source and result.original.read_bytes() == ctx.source.read_bytes()
    assert result.original.parent.name == "restoration_originals"
    assert result.path.suffix == ".mkv" and result.path.is_file()
    assert result.info == info and result.elapsed_seconds >= 0
    assert ctx.events == ["preflight", "worker", "free", "prompt", "mux", "free"]
    ctx.wait.assert_called_once()
    assert ctx.wait.call_args.args == ("job",)
    assert ctx.wait.call_args.kwargs["socket"] is ctx.socket
    ctx.socket.close.assert_called_once()
    assert progress.call_args.args[0] == 1.0
    prompt = next(call.kwargs["json"]["prompt"] for call in restoration.requests.post.call_args_list
        if call.args[0].endswith("/prompt"))
    worker_source = ctx.backend.input_dir / prompt["50"]["inputs"]["file"]
    assert worker_source.read_bytes() == result.original.read_bytes()


@pytest.mark.parametrize("failure", [False, True])
def test_normalized_backend_uses_native_guide_restores_and_cleans(lifecycle, monkeypatch, info, failure):
    ctx = lifecycle
    source_info = replace(info, frames=30, fps=Fraction(30), duration=Fraction(1), audio_tracks=2)
    model, final, detailed_summary = restoration.normalization_plan(source_info, restoration.RestorationRequest(ctx.source, normalize_input=True))
    monkeypatch.setattr(restoration, "probe_video", Mock(side_effect=[source_info, final]))
    def normalize(original, destination, planned):
        assert planned == model and original.read_bytes() == ctx.source.read_bytes()
        destination.write_bytes(b"normalized guide")
        return model
    monkeypatch.setattr(restoration, "normalize_video", Mock(side_effect=normalize))
    ctx.backend.ensure_worker.side_effect = None
    def restore(enhanced, destination, actual_source, actual_model, actual_final):
        assert (enhanced, actual_source, actual_model, actual_final) == (ctx.output, source_info, model, final)
        destination.write_bytes(b"restored timeline")
    monkeypatch.setattr(restoration, "restore_video_timeline", Mock(side_effect=restore))
    def mux(enhanced, original, destination, expected, *, normalized):
        assert enhanced.read_bytes() == b"restored timeline" and expected == final and normalized
        assert original.read_bytes() == ctx.source.read_bytes()
        if failure:
            raise RuntimeError("normalized mux failed")
        destination.write_bytes(b"final")
    restoration.mux_original_audio.side_effect = mux
    request = restoration.RestorationRequest(ctx.source, normalize_input=True)
    progress = Mock()
    if failure:
        with pytest.raises(restoration.RestorationFailure, match="normalized mux failed") as exc:
            ctx.backend.enhance(request, progress)
        assert exc.value.original.read_bytes() == ctx.source.read_bytes()
    else:
        result = ctx.backend.enhance(request, progress)
        assert result.info == final and result.normalization_summary == detailed_summary
    labels = [call.args[1] for call in progress.call_args_list]
    assert "Planning native 24 fps normalization…" in labels
    assert "Normalizing guide to 24 fps and padding frames…" in labels
    assert "Restoring source timeline and trimming padded tail…" in labels
    assert "Preserving original audio and verifying output…" in labels
    assert detailed_summary not in labels
    assert all(len(label) < 100 and "\n" not in label for label in labels)
    prompt = next(call.kwargs["json"]["prompt"] for call in restoration.requests.post.call_args_list
                  if call.args[0].endswith("/prompt"))
    assert prompt["54"]["inputs"]["length"] == model.frames
    assert prompt["54"]["inputs"]["ref_videos.ref_video_0"] == ["51", 0]
    assert prompt["54"]["inputs"]["ref_video_audios.ref_video_audio_0"] == ["56", 1]
    assert prompt["54"]["inputs"]["audio_vae"] == ["4", 0]
    assert prompt["56"] == {"class_type": "GetVideoComponents", "inputs": {"video": ["55", 0]}}
    assert prompt["55"]["inputs"]["file"] != prompt["50"]["inputs"]["file"]
    assert not (ctx.backend.input_dir / prompt["50"]["inputs"]["file"]).exists()
    assert not (ctx.backend.input_dir / prompt["55"]["inputs"]["file"]).exists()
    temporary = restoration.restore_video_timeline.call_args.args[1]
    assert not temporary.parent.exists()
    assert ctx.source.read_bytes() == b"original video and audio bytes"


@pytest.mark.parametrize("stage", ["bad_source", "missing_asset", "startup", "release", "worker", "mux"])
def test_failure_preserves_durable_original_and_never_overwrites_source(lifecycle, monkeypatch, stage):
    ctx = lifecycle
    failure = RuntimeError(stage)
    if stage == "bad_source":
        monkeypatch.setattr(restoration, "probe_video", Mock(side_effect=ValueError(stage)))
    elif stage == "missing_asset":
        ctx.backend.preflight.side_effect = FileNotFoundError(stage)
    elif stage == "startup":
        ctx.backend.ensure_worker.side_effect = failure
    elif stage == "release":
        restoration.requests.post.side_effect = restoration.requests.RequestException(stage)
    elif stage == "worker":
        ctx.wait.side_effect = failure
    else:
        restoration.mux_original_audio.side_effect = failure
    with pytest.raises(restoration.RestorationFailure, match=stage) as exc:
        ctx.backend.enhance(restoration.RestorationRequest(ctx.source))
    assert exc.value.original.is_file()
    assert exc.value.original.read_bytes() == ctx.source.read_bytes() == b"original video and audio bytes"
    assert exc.value.__cause__ is not None
    if stage in {"bad_source", "missing_asset"}:
        ctx.backend.ensure_worker.assert_not_called()
        restoration.requests.post.assert_not_called()
    if stage in {"worker", "mux"}:
        ctx.socket.close.assert_called_once()
        assert ctx.events[-1] == "free"
    assert not list(ctx.backend.config.output_dir.glob("*_restoration.mkv"))


@pytest.mark.parametrize("case", ["missing_node", "unsupported_input", "missing_required"])
def test_worker_schema_requires_supported_inputs_and_all_required_keys(lifecycle, case):
    ctx = lifecycle
    if case == "missing_node":
        ctx.schema.pop("MiniMaxH3ReferenceToVideo")
    elif case == "unsupported_input":
        ctx.schema["MiniMaxH3ReferenceToVideo"]["input"]["required"].pop("ref_image_size")
    else:
        ctx.schema["MiniMaxH3ReferenceToVideo"]["input"]["required"]["new_required_argument"] = ["INT"]
    with pytest.raises(restoration.RestorationFailure, match="unavailable|incompatible"):
        ctx.backend.enhance(restoration.RestorationRequest(ctx.source))
    ctx.wait.assert_not_called()
    restoration.websocket.create_connection.assert_not_called()
    assert ctx.events == ["preflight", "worker", "free", "free"]


@pytest.mark.parametrize("group,prefix,kind", [
    ("ref_videos", "ref_video_", "IMAGE"),
    ("ref_video_audios", "ref_video_audio_", "AUDIO"),
])
@pytest.mark.parametrize("mismatch", ["prefix", "max", "type", "extra_child"])
def test_worker_rejects_dynamic_reference_schema_mismatch(lifecycle, group, prefix, kind, mismatch):
    ctx = lifecycle
    template = ctx.schema["MiniMaxH3ReferenceToVideo"]["input"]["optional"][group][1]["template"]
    if mismatch == "prefix":
        template["prefix"] = "unknown_"
    elif mismatch == "max":
        template["max"] = 0
    elif mismatch == "type":
        template["input"]["required"][prefix] = ["AUDIO" if kind == "IMAGE" else "IMAGE"]
    else:
        template["input"]["required"]["extra"] = [kind]
    with pytest.raises(restoration.RestorationFailure, match=f"incompatible: {group}") as exc:
        ctx.backend.enhance(restoration.RestorationRequest(ctx.source))
    assert exc.value.original.read_bytes() == ctx.source.read_bytes()
    ctx.wait.assert_not_called()
    restoration.websocket.create_connection.assert_not_called()
    assert ctx.events == ["preflight", "worker", "free", "free"]


@pytest.mark.parametrize("qualified", [False, True])
def test_worker_supports_explicit_typed_flat_reference_ports(lifecycle, qualified):
    ctx = lifecycle
    optional = ctx.schema["MiniMaxH3ReferenceToVideo"]["input"]["optional"]
    video = "ref_videos.ref_video_0" if qualified else "ref_video_0"
    audio = "ref_video_audios.ref_video_audio_0" if qualified else "ref_video_audio_0"
    optional.pop("ref_videos")
    optional.pop("ref_video_audios")
    optional.update({video: ["IMAGE"], audio: ["AUDIO"]})
    result = ctx.backend.enhance(restoration.RestorationRequest(ctx.source))
    assert result.path.is_file()
    prompt = next(call.kwargs["json"]["prompt"] for call in restoration.requests.post.call_args_list
                  if call.args[0].endswith("/prompt"))
    assert prompt["54"]["inputs"][video] == ["51", 0]
    assert prompt["54"]["inputs"][audio] == ["51", 1]
    ctx.wait.assert_called_once()
    ctx.socket.close.assert_called_once()


@pytest.mark.parametrize("kind", [None, "STRING", "AUDIO"])
def test_worker_rejects_untyped_or_wrongly_typed_flat_video_reference(lifecycle, kind):
    ctx = lifecycle
    optional = ctx.schema["MiniMaxH3ReferenceToVideo"]["input"]["optional"]
    optional.pop("ref_videos")
    optional["ref_video_0"] = [] if kind is None else [kind]
    with pytest.raises(restoration.RestorationFailure, match="incompatible: ref_videos"):
        ctx.backend.enhance(restoration.RestorationRequest(ctx.source))
    ctx.wait.assert_not_called()
    restoration.websocket.create_connection.assert_not_called()


def test_video_only_worker_does_not_require_soundtrack_schema(lifecycle, monkeypatch, info):
    ctx = lifecycle
    silent = replace(info, audio_tracks=0)
    monkeypatch.setattr(restoration, "probe_video", Mock(return_value=silent))
    optional = ctx.schema["MiniMaxH3ReferenceToVideo"]["input"]["optional"]
    optional.pop("audio_vae")
    optional.pop("ref_video_audios")
    def mux(enhanced, original, destination, actual_info):
        assert actual_info == silent and original.read_bytes() == ctx.source.read_bytes()
        destination.write_bytes(b"enhanced silent result")
    restoration.mux_original_audio.side_effect = mux
    result = ctx.backend.enhance(restoration.RestorationRequest(ctx.source))
    assert result.info == silent
    prompt = next(call.kwargs["json"]["prompt"] for call in restoration.requests.post.call_args_list
                  if call.args[0].endswith("/prompt"))
    assert "audio_vae" not in prompt["54"]["inputs"]
    assert "ref_video_audios.ref_video_audio_0" not in prompt["54"]["inputs"]
    assert not {"55", "56", "17"} & prompt.keys()
    assert "audio" not in prompt["18"]["inputs"]


def test_rejected_prompt_closes_socket_and_retains_original(lifecycle):
    ctx = lifecycle
    post = restoration.requests.post.side_effect
    def reject(url, **kwargs):
        response = post(url, **kwargs)
        if url.endswith("/prompt"):
            response.ok = False
            response.text = "invalid native workflow"
        return response
    restoration.requests.post.side_effect = reject
    with pytest.raises(restoration.RestorationFailure, match="workflow rejected") as exc:
        ctx.backend.enhance(restoration.RestorationRequest(ctx.source))
    assert exc.value.original.read_bytes() == ctx.source.read_bytes()
    ctx.wait.assert_not_called()
    ctx.socket.close.assert_called_once()
    assert ctx.events[-1] == "free"


def test_missing_source_is_rejected_before_persistence_or_worker(config, tmp_path, monkeypatch):
    backend = restoration.RestorationBackend(config)
    monkeypatch.setattr(backend, "ensure_worker", Mock())
    with pytest.raises(FileNotFoundError, match="source video missing"):
        backend.enhance(restoration.RestorationRequest(tmp_path / "absent.mp4"))
    backend.ensure_worker.assert_not_called()
    assert not (config.upload_dir / "restoration_originals").exists()


def test_mux_stream_copies_all_original_audio_atomically(tmp_path, monkeypatch, info):
    enhanced, original, destination = [tmp_path / name for name in ("enhanced.mp4", "original.mp4", "final.mkv")]
    monkeypatch.setattr(restoration, "probe_video", Mock(return_value=info))
    calls = []
    def run(command):
        calls.append(command)
        if command[-1] != "-":
            Path(command[-1]).write_bytes(b"muxed")
            return ""
        return "# timebase may differ\n0,a,SHA256=unchanged\n1,a,SHA256=second"
    monkeypatch.setattr(restoration, "_run", run)
    restoration.mux_original_audio(enhanced, original, destination, info)
    assert destination.read_bytes() == b"muxed"
    mux, hash_original, hash_muxed = calls
    assert mux == ["ffmpeg", "-v", "error", "-nostdin", "-y", "-copyts", "-i", str(enhanced),
        "-i", str(original), "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1", "-c", "copy",
        "-avoid_negative_ts", "disabled", str(tmp_path / "final.partial.mkv")]
    assert hash_original[hash_original.index("-i")+1] == str(original)
    assert hash_muxed[hash_muxed.index("-i")+1] == str(tmp_path / "final.partial.mkv")
    assert not (tmp_path / "final.partial.mkv").exists()


@pytest.mark.parametrize("case", ["geometry", "audio_hash"])
def test_mux_rejects_changed_geometry_or_audio_preserving_previous_result(tmp_path, monkeypatch, info, case):
    destination = tmp_path / "final.mkv"
    destination.write_bytes(b"previous result")
    monkeypatch.setattr(restoration, "probe_video", Mock(return_value=replace(info, width=96) if case == "geometry" else info))
    def run(command):
        if command[-1] != "-":
            Path(command[-1]).write_bytes(b"partial")
            return ""
        return "0,a,SHA256=" + command[command.index("-i")+1] + "\n1,a,SHA256=second"
    monkeypatch.setattr(restoration, "_run", Mock(side_effect=run))
    with pytest.raises((ValueError, RuntimeError), match="geometry/timeline|packet hashes"):
        restoration.mux_original_audio(tmp_path / "enhanced.mp4", tmp_path / "original.mp4", destination, info)
    assert destination.read_bytes() == b"previous result"
    assert not (tmp_path / "final.partial.mkv").exists()
    if case == "geometry":
        restoration._run.assert_not_called()


@pytest.fixture
def app(config, monkeypatch, info, tmp_path):
    generated_path = tmp_path / "server_generated.mp4"
    generated = SimpleNamespace(path=generated_path, width=64, height=32, frames=22,
        duration=1.0, fps=23.976, nfe=8, elapsed_seconds=1.0, lora="turbo", mode="t2va", seed=42, timings={})
    result = restoration.RestorationResult(tmp_path / "enhanced.mkv", tmp_path / "retained.mp4", info, 1.0)
    engine = SimpleNamespace(ready=True, generate=Mock(return_value=generated), enhance_restoration=Mock(return_value=result))
    monkeypatch.setattr(ui, "get_engine", lambda config: engine)
    demo = ui.build_app(config)
    enhance = next(fn for fn in demo.fns.values() if fn.fn.__name__ == "enhance_restoration")
    generate = next(fn for fn in demo.fns.values() if fn.fn.__name__ == "generate")
    yield SimpleNamespace(demo=demo, enhance=enhance, generate=generate, engine=engine,
        generated=generated, result=result)
    demo.close()


def generate_from_defaults(ctx):
    values = [component.value for component in ctx.generate.inputs]
    values[0] = "A quiet landscape"
    return ctx.generate.fn(*values, progress=lambda *args, **kwargs: None)


def test_ui_restoration_is_separate_opt_in_with_independent_defaults(app):
    ctx = app
    assert [component.value for component in ctx.enhance.inputs] == ["Generated result", None, 1.0, 30, 42, False, "native", None]
    assert isinstance(ctx.generate.outputs[-1], ui.gr.State)
    assert ctx.enhance.inputs[-1] is ctx.generate.outputs[-1]
    assert len(ctx.enhance.outputs) == 4
    assert not set(ctx.enhance.outputs) & set(ctx.generate.outputs)
    assert not set(ctx.enhance.inputs) & set(ctx.generate.inputs)
    fps = next(component for component in ctx.generate.inputs if component.label == "Output frame rate")
    assert fps.value == "23.976 fps"
    generate_from_defaults(ctx)
    ctx.engine.enhance_restoration.assert_not_called()
    assert ctx.engine.generate.call_args.args[0].target_fps == 23.976


def test_ui_restoration_has_exactly_one_dedicated_progress_target(app):
    dependencies = [dependency for dependency in app.demo.config["dependencies"]
                    if dependency["id"] == app.enhance._id]
    assert len(dependencies) == 1
    event = dependencies[0]
    status = app.enhance.outputs[-1]
    assert status.elem_id == "restoration-progress"
    assert status.label == "Restoration progress / current task"
    assert event["show_progress"] == "full"
    assert event["show_progress_on"] == [status._id]
    assert event["outputs"] == [component._id for component in app.enhance.outputs]
    assert status not in app.enhance.inputs
    assert app.enhance.outputs[2].label == "Restoration result and normalization summary"
    assert all(component._id not in event["show_progress_on"]
               for component in app.enhance.outputs[:3])


def test_ui_generated_selection_uses_server_original_not_uploaded_or_enhanced(app, tmp_path):
    ctx = app
    generated_path = generate_from_defaults(ctx)[-1]
    progress = Mock()
    outputs = ctx.enhance.fn("Generated result", str(tmp_path / "wrong_preview.mp4"), 1.0, 8, 42, generated_path=generated_path, progress=progress)
    assert ctx.engine.enhance_restoration.call_args.args[0].source == ctx.generated.path
    assert outputs[:2] == (str(ctx.result.path), str(ctx.result.original))
    assert outputs[3] == "Restoration complete; original audio preserved."
    ctx.engine.enhance_restoration.call_args.kwargs["progress_callback"](0.5, "sampling")
    progress.assert_called_once_with(0.5, desc="sampling")
    ctx.enhance.fn("Generated result", None, 0.5, 12, 123, generated_path=generated_path, progress=Mock())
    request = ctx.engine.enhance_restoration.call_args.args[0]
    assert request == restoration.RestorationRequest(ctx.generated.path, strength=0.5, steps=12, seed=123)
    assert ctx.engine.generate.call_count == 1


def test_ui_generated_selection_is_session_local(app, tmp_path):
    first_path = generate_from_defaults(app)[-1]
    app.generated.path = tmp_path / "second_session.mp4"
    second_path = generate_from_defaults(app)[-1]
    assert first_path != second_path
    for session_path in (first_path, second_path):
        app.enhance.fn("Generated result", None, 1.0, 8, 42,
                       generated_path=session_path, progress=Mock())
        assert app.engine.enhance_restoration.call_args.args[0].source == Path(session_path)
    calls = app.engine.enhance_restoration.call_count
    outputs = app.enhance.fn("Generated result", None, 1.0, 8, 42,
                             generated_path=None, progress=Mock())
    assert "generate a video first" in outputs[2]
    assert app.engine.enhance_restoration.call_count == calls


def test_ui_uploaded_selection_does_not_require_or_replace_generation(app, tmp_path):
    uploaded = tmp_path / "upload.mp4"
    app.enhance.fn("Uploaded video", str(uploaded), 1.0, 8, 42, progress=Mock())
    assert app.engine.enhance_restoration.call_args.args[0].source == uploaded
    app.engine.generate.assert_not_called()
    outputs = app.enhance.fn("Generated result", None, 1.0, 8, 42, progress=Mock())
    assert outputs[:2] == (ui.gr.update(), ui.gr.update())
    assert "generate a video first" in outputs[2]
    assert app.engine.enhance_restoration.call_count == 1


def test_ui_normalization_checkbox_and_output_radio_wired(app, tmp_path):
    source = tmp_path / "upload.mp4"
    app.result = replace(app.result, normalization_summary="Detailed normalization: pad 14 frames; temporal detail may be lost.")
    app.engine.enhance_restoration.return_value = app.result
    outputs = app.enhance.fn("Uploaded video", str(source), 0.75, 12, 123, True, "source", None, progress=Mock())
    assert app.engine.enhance_restoration.call_args.args[0] == restoration.RestorationRequest(
        source, strength=0.75, steps=12, seed=123, normalize_input=True, output_fps="source")
    assert app.result.normalization_summary in outputs[2]
    assert app.result.normalization_summary not in outputs[3]
    app.engine.generate.assert_not_called()


def test_ui_restoration_failure_preserves_previous_enhanced_result_and_returns_original(app, tmp_path):
    retained = tmp_path / "durable_original.mp4"
    app.engine.enhance_restoration.side_effect = restoration.RestorationFailure(retained, RuntimeError("worker failed"))
    outputs = app.enhance.fn("Uploaded video", str(tmp_path / "upload.mp4"), 1.0, 8, 42, progress=Mock())
    assert outputs[0] == ui.gr.update()  # No value update: keep the last enhanced result.
    assert outputs[1] == str(retained)
    assert "worker failed" in outputs[2] and "Original retained" in outputs[2]
    assert outputs[3] == "Restoration failed; original retained."
    app.engine.generate.assert_not_called()


def test_ui_restoration_rejection_keeps_files_and_source_path(app, tmp_path):
    source = tmp_path / "upload.mp4"
    app.engine.enhance_restoration.side_effect = ValueError("invalid settings")
    outputs = app.enhance.fn("Uploaded video", str(source), 1.0, 8, 42, progress=Mock())
    assert outputs[:2] == (ui.gr.update(), ui.gr.update())
    assert "invalid settings" in outputs[2] and str(source) in outputs[2]
    assert outputs[3] == "Restoration request rejected; source unchanged."


def test_engine_serializes_restoration_with_fl2va_and_releases_under_lock(config, monkeypatch):
    engine = MiniMaxH3Engine(config)
    events = []
    entered = threading.Event()
    proceed = threading.Event()
    failures = []
    def release():
        assert engine._lock.locked()
        events.append("release")
    def enhance(self, request, progress_callback=None):
        assert engine._lock.locked()
        events.append("restoration")
        entered.set()
        assert proceed.wait(5), "test did not release mocked Restoration worker"
        return "enhanced"
    def generate(self, request, spec, progress_callback=None):
        assert engine._lock.locked()
        events.append("fl2va")
        return "generated"
    monkeypatch.setattr(engine, "_release_cuda_memory", release)
    monkeypatch.setattr(restoration.RestorationBackend, "enhance", enhance)
    monkeypatch.setattr(PrunedComfyBackend, "generate", generate)
    def run(action):
        try:
            action()
        except BaseException as exc:
            failures.append(exc)
    first = threading.Thread(target=run, args=(lambda: engine.enhance_restoration(restoration.RestorationRequest(Path("source.mp4"))),))
    second = threading.Thread(target=run, args=(lambda: engine.generate(GenerationRequest(prompt="Landscape", lora_id="turbo")),))
    first.start()
    try:
        assert entered.wait(5)
        second.start()
        assert events == ["release", "restoration"]
    finally:
        proceed.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert not failures
    assert events == ["release", "restoration", "release", "fl2va"]
    assert engine.status == "ready"


def test_engine_failure_releases_lock_and_reports_original_retained(config, monkeypatch, tmp_path):
    engine = MiniMaxH3Engine(config)
    release = Mock()
    failure = restoration.RestorationFailure(tmp_path / "original.mp4", RuntimeError("failed"))
    monkeypatch.setattr(engine, "_release_cuda_memory", release)
    monkeypatch.setattr(restoration.RestorationBackend, "enhance", Mock(side_effect=failure))
    with pytest.raises(restoration.RestorationFailure) as exc:
        engine.enhance_restoration(restoration.RestorationRequest(tmp_path / "source.mp4"))
    assert exc.value is failure
    release.assert_called_once_with()
    assert not engine._lock.locked()
    assert engine.status == "Restoration enhancement failed; original retained"