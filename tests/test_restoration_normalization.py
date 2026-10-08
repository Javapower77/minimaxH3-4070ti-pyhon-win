"""CPU-only normalization contracts and real tiny ffmpeg integration."""

from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
from unittest.mock import Mock

import pytest

from minimax_h3_fl2v import restoration


@pytest.mark.parametrize("fps,cfr", [(Fraction(24000, 1001), True), (Fraction(30), True),
                                   (Fraction(60), True), (Fraction(30), False)])
@pytest.mark.parametrize("mode", ["native", "source"])
def test_plan_preserves_speed_pads_without_cutting_and_explains_loss(fps, cfr, mode):
    source = restoration.VideoInfo(64, 32, 30, fps, 1, duration=Fraction(1001, 1000), cfr=cfr)
    model, final, summary = restoration.normalization_plan(source, restoration.RestorationRequest(
        Path("source.mp4"), normalize_input=True, output_fps=mode))
    assert model.frames == 39 and model.fps == 24
    assert model.duration_seconds >= source.duration_seconds
    assert final.fps == (fps if mode == "source" else 24)
    assert 0 <= final.duration_seconds - source.duration_seconds < 1 / final.fps
    assert "pad 14" in summary and "temporal detail may be lost" in summary
    assert "Original audio timeline unchanged" in summary
    if not cfr:
        assert "average rate" in summary


@pytest.mark.parametrize("changes", [{"width": 65}, {"duration": Fraction(150)}, {"height": 16},
                                   {"start_time": Fraction(-1, 8)}])
def test_normalization_does_not_relax_model_geometry_or_maximum(changes):
    source = replace(restoration.VideoInfo(64, 32, 30, Fraction(30), 0), **changes)
    with pytest.raises(ValueError):
        restoration.normalization_plan(source, restoration.RestorationRequest(Path("source"), normalize_input=True))


@pytest.mark.parametrize("settings", [{"normalize_input": "yes"}, {"output_fps": "60"}])
def test_new_request_settings_validated(settings):
    with pytest.raises(ValueError):
        restoration.RestorationRequest(Path("source"), **settings).validate()


def test_probe_accepts_millisecond_quantized_cfr_but_rejects_vfr(monkeypatch):
    stream = {"codec_type": "video", "width": 64, "height": 32, "nb_read_frames": "22",
              "avg_frame_rate": "24/1", "time_base": "1/1000"}
    def install(times):
        monkeypatch.setattr(restoration, "_run", Mock(side_effect=[json.dumps({"streams": [stream]}),
            json.dumps({"frames": [{"best_effort_timestamp_time": str(t)} for t in times]})]))
    monkeypatch.setattr(restoration.shutil, "which", lambda name: name)
    times = [round(i / 24, 3) for i in range(22)]
    install(times)
    assert restoration.probe_video(Path("source")).cfr
    times[8] += 0.004
    install(times)
    with pytest.raises(ValueError, match="constant 24"):
        restoration.probe_video(Path("source"))
    install(times)
    assert not restoration.probe_video(Path("source"), strict=False).cfr


def test_audio_digest_compares_payload_not_headers(tmp_path, monkeypatch):
    info = restoration.VideoInfo(64, 32, 22, Fraction(24), 1)
    monkeypatch.setattr(restoration, "probe_video", Mock(return_value=info))
    hashes = iter(["#tb 0: 1/48000\n0,a,SHA256=abc\n", "#tb 0: 1/1000\n0,a,SHA256=abc\n"])
    def run(command):
        if command[-1] == "-":
            return next(hashes)
        Path(command[-1]).write_bytes(b"muxed")
        return ""
    monkeypatch.setattr(restoration, "_run", run)
    destination = tmp_path / "final.mkv"
    restoration.mux_original_audio(tmp_path / "enhanced", tmp_path / "original", destination, info)
    assert destination.read_bytes() == b"muxed"


@pytest.fixture
def ffmpeg_available():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Local ffmpeg/ffprobe unavailable")


@pytest.mark.parametrize("rate", ["24000/1001", "30", "60", "vfr"])
@pytest.mark.parametrize("mode", ["native", "source"])
@pytest.mark.parametrize("offset", ["0", "0.125"])
def test_real_normalization_audio_and_source_untouched(tmp_path, ffmpeg_available, rate, mode, offset):
    original = tmp_path / "original clip.mkv"
    command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
               f"testsrc2=size=64x32:rate={30 if rate == 'vfr' else rate}:duration=1.001",
               "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1.4",
               "-map", "0:v", "-map", "1:a"]
    if rate == "vfr":
        command += ["-vf", "select='not(eq(mod(n,5),2))'", "-fps_mode", "vfr"]
    command += ["-c:v", "ffv1", "-c:a", "pcm_s16le", "-output_ts_offset", offset, str(original)]
    restoration._run(command)
    checksum = hashlib.sha256(original.read_bytes()).hexdigest()
    source = restoration.probe_video(original, strict=False)
    with pytest.raises(ValueError):
        restoration.probe_video(original)
    if rate == "vfr":
        assert not source.cfr
    request = restoration.RestorationRequest(original, normalize_input=True, output_fps=mode)
    model, final, _ = restoration.normalization_plan(source, request)
    reference = tmp_path / "reference.mkv"
    model = restoration.normalize_video(original, reference, model)
    restored = tmp_path / "restored.mkv"
    # Identity mocked enhancement: exercise real encoding, padding, trim and remux.
    restoration.restore_video_timeline(reference, restored, source, model, final)
    result = tmp_path / "enhanced.mkv"
    restoration.mux_original_audio(restored, original, result, final, normalized=True)
    actual = restoration.probe_video(result, strict=False)
    assert actual.cfr and actual.frames == final.frames and actual.fps == final.fps
    assert abs(actual.duration_seconds - source.duration_seconds) <= 1 / final.fps + actual.time_base
    assert actual.audio_tracks == 1
    assert abs(actual.start_time - source.start_time) <= actual.time_base
    assert hashlib.sha256(original.read_bytes()).hexdigest() == checksum


def test_real_aac_mp4_negative_priming_preserved(tmp_path, ffmpeg_available):
    original = tmp_path / "original.mp4"
    restoration._run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
              "testsrc2=size=64x32:rate=60:duration=1", "-f", "lavfi", "-i",
              "sine=frequency=440:sample_rate=48000:duration=1.4", "-map", "0:v", "-map", "1:a",
              "-c:v", "libx264", "-c:a", "aac", str(original)])
    source = restoration.probe_video(original, strict=False)
    model, final, _ = restoration.normalization_plan(source, restoration.RestorationRequest(original, normalize_input=True))
    reference, restored, result = [tmp_path / name for name in ("reference.mkv", "restored.mkv", "final.mkv")]
    model = restoration.normalize_video(original, reference, model)
    restoration.restore_video_timeline(reference, restored, source, model, final)
    restoration.mux_original_audio(restored, original, result, final, normalized=True)
    restoration.verify_audio_timeline(original, result)