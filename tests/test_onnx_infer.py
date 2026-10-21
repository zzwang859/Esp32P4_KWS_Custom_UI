"""Whole-file inference and the device's keyword debounce semantics."""
import importlib
import json
import math
from pathlib import Path
import re
import runpy
import wave

import numpy as np
import pytest

from kws_custom_ui.training import TRAINING_ROOT


@pytest.fixture
def inference(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(TRAINING_ROOT))
    importlib.import_module("windows_runtime").preload_msvc()
    return runpy.run_path(str(TRAINING_ROOT / "onnx/run/infer.py"))


def logits_for(index, strength=4.0):
    logits = np.zeros(8, dtype=np.float32)
    logits[index] = strength
    return logits


def feed(filter, index, now, strength=4.0):
    detected = filter.update(index, logits_for(index, strength), now)
    if detected >= 0:
        filter.acknowledge(True, now)
    return detected


def prime(filter):
    for i in range(1, 100):
        assert feed(filter, 0, i * 10_000) == -1
    assert feed(filter, 0, 1_000_000) == 0


def test_warmup_and_held_keyword_suppression(inference):
    filter = inference["Debouncer"]()
    for i in range(1, 100):
        assert feed(filter, 2, i * 10_000) == -1
    assert feed(filter, 2, 1_000_000) == 2
    for i in range(101, 400):
        assert feed(filter, 2, i * 10_000) == -1


def test_global_cooldown_and_background_rearm(inference):
    filter = inference["Debouncer"]()
    prime(filter)
    assert [feed(filter, 2, t) for t in (1_010_000, 1_020_000, 1_030_000)] == [-1, -1, 2]
    assert [feed(filter, 1, t) for t in (1_040_000, 1_050_000, 1_060_000)] == [-1, -1, 1]
    # Even a different keyword cannot accumulate stability during cooldown.
    for now in range(1_070_000, 1_530_000, 10_000):
        assert feed(filter, 3, now) == -1
    assert [feed(filter, 3, t) for t in (1_530_000, 1_540_000, 1_550_000)] == [-1, -1, 3]
    assert [feed(filter, 0, t) for t in (1_560_000, 1_570_000, 1_580_000)] == [-1, -1, 0]
    # Stable background permits the same keyword again after cooldown.
    assert [feed(filter, 3, t) for t in (2_050_000, 2_060_000, 2_070_000)] == [-1, -1, 3]


@pytest.mark.parametrize("offset", [0, 1000, -1000])
def test_probability_threshold_and_shift_invariance(inference, offset):
    filter = inference["Debouncer"]()
    prime(filter)
    logits = np.full(8, offset, dtype=np.float32)
    logits[2] += np.log(27.9)  # below 80% among all eight classes
    for _ in range(10):
        assert filter.update(2, logits, 1_010_000) == -1
    logits[2] = offset + np.log(28.1)
    assert [filter.update(2, logits, 1_010_000) for _ in range(3)] == [-1, -1, 2]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 5.0])
def test_invalid_or_non_top_logits_break_stability(inference, invalid):
    filter = inference["Debouncer"]()
    prime(filter)
    assert feed(filter, 2, 1_010_000) == -1
    assert feed(filter, 2, 1_020_000) == -1
    logits = logits_for(2)
    logits[4] = invalid
    assert filter.update(2, logits, 1_030_000) == -1
    assert [feed(filter, 2, t) for t in (1_040_000, 1_050_000, 1_060_000)] == [-1, -1, 2]


def test_failed_ack_and_reset_do_not_bypass_cooldown(inference):
    filter = inference["Debouncer"]()
    prime(filter)
    assert feed(filter, 2, 1_010_000) == -1
    assert feed(filter, 2, 1_020_000) == -1
    assert filter.update(2, logits_for(2), 1_030_000) == 2
    filter.acknowledge(False, 1_030_000)
    assert filter.update(2, logits_for(2), 1_040_000) == 2
    filter.acknowledge(True, 1_040_000)
    filter.reset()
    for _ in range(100):
        assert filter.update(3, logits_for(3), 1_050_000) == -1
    assert [feed(filter, 3, t) for t in (1_540_000, 1_550_000, 1_560_000)] == [-1, -1, 3]


@pytest.mark.parametrize("offset", [0, 1000, -1000])
def test_softmax_probabilities_use_all_classes_and_remain_stable(inference, offset):
    logits = np.log(np.array([1, 3, 6], dtype=np.float32)) + offset
    probabilities = inference["softmax_probabilities"](logits)
    np.testing.assert_allclose(probabilities, [0.1, 0.3, 0.6], atol=1e-5)
    assert probabilities.dtype == np.float32
    np.testing.assert_allclose(inference["softmax_probabilities"]([10000, 0, -10000]), [1, 0, 0])


def test_filter_settings_are_reloaded_from_json(inference, tmp_path, monkeypatch):
    module = importlib.import_module("kws_debounce")
    path = tmp_path / "workspace.json"
    settings = dict(keyword_threshold_percent=25, cooldown_ms=1000, stable_frames=1, need_vad_checked=False)
    path.write_text(json.dumps({"kws_filter": settings}), encoding="utf-8")
    filter = module.Debouncer(module.load_filter_config(path))
    prime(filter)
    assert feed(filter, 2, 1_010_000, strength=1) == 2
    assert feed(filter, 3, 2_000_000, strength=1) == -1
    assert feed(filter, 3, 2_010_000, strength=1) == 3
    settings.update(keyword_threshold_percent=99, cooldown_ms=1, stable_frames=5)
    path.write_text(json.dumps({"kws_filter": settings}), encoding="utf-8")
    changed = module.Debouncer(module.load_filter_config(path))
    prime(changed)
    assert feed(changed, 2, 1_010_000, strength=4) == -1  # below 99%
    assert [feed(changed, 2, 1_020_000 + i * 10_000, strength=10) for i in range(5)] == [-1, -1, -1, -1, 2]


def test_live_filter_update_preserves_warmup_and_applies_new_cooldown(inference):
    filter = inference["Debouncer"]()
    prime(filter)
    assert [feed(filter, 2, now) for now in (1_010_000, 1_020_000, 1_030_000)] == [-1, -1, 2]
    settings = dict(filter.config, keyword_threshold_percent=0, cooldown_ms=1, stable_frames=1)
    filter.configure(settings)
    assert filter.processed == 100
    assert filter.cooldown_until_us == 1_031_000
    assert feed(filter, 3, 1_040_000, strength=1) == 3


@pytest.mark.parametrize("settings", [
    {}, [], dict(keyword_threshold_percent=-1, cooldown_ms=500, stable_frames=3, need_vad_checked=False),
    dict(keyword_threshold_percent=80, cooldown_ms=-1, stable_frames=3, need_vad_checked=False),
    dict(keyword_threshold_percent=80, cooldown_ms=0, stable_frames=3, need_vad_checked=False),
    dict(keyword_threshold_percent=80, cooldown_ms=1001, stable_frames=3, need_vad_checked=False),
    dict(keyword_threshold_percent=80, cooldown_ms=11, stable_frames=3, need_vad_checked=False),
    dict(keyword_threshold_percent=80, cooldown_ms=500, stable_frames=0, need_vad_checked=False),
    dict(keyword_threshold_percent=80, cooldown_ms=500, stable_frames=True, need_vad_checked=False),
    dict(keyword_threshold_percent=80, cooldown_ms=500, stable_frames=51, need_vad_checked=False),
    dict(keyword_threshold_percent=80, cooldown_ms=500, stable_frames=3, need_vad_checked="false"),
])
def test_invalid_filter_config_is_reported(inference, tmp_path, settings):
    module = importlib.import_module("kws_debounce")
    path = tmp_path / "workspace.json"
    path.write_text(json.dumps({"kws_filter": settings}), encoding="utf-8")
    with pytest.raises(ValueError, match="KWS"):
        module.load_filter_config(path)


def test_vad_gate_rejects_and_resets_keyword_candidate_only(inference):
    module = importlib.import_module("kws_debounce")
    settings = dict(module.DEFAULT_FILTER_CONFIG)
    settings["need_vad_checked"] = True
    filter = module.Debouncer(settings)
    prime(filter)  # Background needs no VAD high signal.
    for now in (1_010_000, 1_020_000):
        assert filter.update(2, logits_for(2), now, vad_high=True) == -1
    assert filter.update(2, logits_for(2), 1_030_000, vad_high=False) == -1
    assert [filter.update(2, logits_for(2), t, vad_high=True)
            for t in (1_040_000, 1_050_000, 1_060_000)] == [-1, -1, 2]


def test_audio_vad_gate_uses_only_completed_silero_frames(inference, monkeypatch):
    module = importlib.import_module("data.silero")
    import torch
    class Model:
        def reset_states(self):
            self.calls = 0
        def __call__(self, chunk, rate):
            assert len(chunk) == 512 and rate == 16000
            self.calls += 1
            return torch.tensor(0.8 if self.calls == 1 else 0.1)
    monkeypatch.setattr(module, "SileroVad", lambda: type("Vad", (), {"model": Model()})())
    flags = inference["audio_vad_flags"](torch.zeros(1101))
    assert flags.tolist() == [False, False, False, True, True, True, False]


@pytest.mark.parametrize("need_vad_checked", [False, True])
def test_entire_wav_model_path_chunk_log_and_terminal_results(inference, tmp_path, monkeypatch, capsys, need_vad_checked):
    monkeypatch.chdir(tmp_path)
    wav_path = tmp_path / "audio.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 32017)  # 2.001 seconds, including a partial last chunk
    seen = []
    module = importlib.import_module("kws_debounce")
    settings = dict(module.DEFAULT_FILTER_CONFIG)
    settings["need_vad_checked"] = need_vad_checked
    (tmp_path / "workspace.json").write_text(json.dumps({"kws_filter": settings}), encoding="utf-8")
    class Runtime:
        mode = "stream_external"
        metadata = dict(chunk_frames="1", labels=json.dumps(["silence", "unknown", "hello"]))
        def __init__(self, path):
            seen.append(path)
        def predict_chunks(self, features):
            assert features.shape == (1, 40, 201)
            for index in range(features.shape[-1]):
                values = np.full((1, 3), -1.23456, dtype=np.float32)
                values[0, 0 if index < 100 else 2] = 9.87654
                yield values
    monkeypatch.setitem(inference["infer"].__globals__, "OnnxKWS", Runtime)
    def vad_flags(audio):
        assert need_vad_checked  # Disabled gate must never run the VAD model.
        return np.arange(201) >= 150
    monkeypatch.setitem(inference["infer"].__globals__, "audio_vad_flags", vad_flags)
    config = importlib.import_module("config").load_config()
    config["_project_root"] = str(tmp_path / "must-not-be-used")
    result_path = tmp_path / "results/infer_result.md"
    inference["infer"](config, wav_path, Path("models/selected.onnx"), result_path=result_path,
                       workspace=tmp_path)
    assert seen == [tmp_path / "models/selected.onnx"]
    lines = result_path.read_text(encoding="utf-8").splitlines()
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
    assert rows[0] == ["chunk", "time_s", "0:silence", "1:unknown", "2:hello"]
    assert rows[1] == ["---:"] * 5
    assert len(rows) == math.ceil(32017 / 160) + 2
    assert rows[-1][:2] == ["201", "2.01"]
    assert all(re.fullmatch(r"-?\d+\.\d{2}\[\d+\.\d{2}%\]", re.sub(r"<[^>]+>", "", value))
               for row in rows[2:] for value in row[2:])
    for chunk_index, row in enumerate(rows[2:]):
        assert sum('<span style="color: red;">' in cell for cell in row[2:]) == 1
        assert row[2 if chunk_index < 100 else 4] == '<span style="color: red;">9.88[100.00%]</span>'
    assert rows[2][2:] == ['<span style="color: red;">9.88[100.00%]</span>', "-1.23[0.00%]", "-1.23[0.00%]"]
    assert not result_path.with_suffix(".txt").exists()
    text = capsys.readouterr().out
    assert text.count("识别结果: hello (index=2") == 1
    assert ("[1.53s]" if need_vad_checked else "[1.03s]") in text and "消抖识别结果: hello" in text
    assert inference["RESULT_PATH"] == TRAINING_ROOT.parents[1] / "workspace/test/infer_result.md"


def test_realtime_frontend_uses_consecutive_samples_and_emits_each_frame(inference, monkeypatch):
    import torch
    from data.audio import MfccExtractor
    seen = []

    class Runtime:
        mode = "stream_external"
        metadata = dict(chunk_frames="1", labels=json.dumps(["silence", "unknown", "hello"]))

        def __init__(self, _path):
            pass

        def predict(self, features):
            seen.append(features.copy())
            return np.array([[2.0, 1.0, 0.0]], dtype=np.float32)

    monkeypatch.setitem(inference["StreamingInference"].__init__.__globals__, "OnnxKWS", Runtime)
    config = importlib.import_module("config").load_config()
    engine = inference["StreamingInference"](config, Path("unused.onnx"),
                                               importlib.import_module("kws_debounce").DEFAULT_FILTER_CONFIG)
    samples = np.linspace(-0.6, 0.6, 5 * 160, dtype=np.float32)
    rows = list(engine.feed(samples[:213])) + list(engine.feed(samples[213:571])) + list(engine.feed(samples[571:]))
    expected = MfccExtractor(config["data"], config["features"]).streaming(torch.from_numpy(samples)).numpy()
    assert [row[0] for row in rows] == [1, 2, 3, 4, 5]
    np.testing.assert_allclose(np.concatenate(seen, axis=2), expected, rtol=1e-5, atol=1e-5)
    assert all(row[1].shape == (3,) and np.isclose(row[2].sum(), 1) for row in rows)
