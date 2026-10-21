"""Check augmentation rules, cache reuse/invalidation and the training input."""
import importlib
import json
from pathlib import Path
import wave
from collections import Counter

import numpy as np
import pytest

from kws_custom_ui.model import Workspace
from kws_custom_ui.training import TRAINING_ROOT


@pytest.fixture
def preprocessing(monkeypatch):
    monkeypatch.syspath_prepend(str(TRAINING_ROOT))
    return importlib.import_module("data.preprocess")


def write_audio(path, seconds, amplitude=0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.sin(np.arange(round(seconds * 16000)) * (2 * np.pi * 440 / 16000)) * amplitude
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(np.rint(samples * 32767).astype("<i2").tobytes())


@pytest.fixture
def corpus(preprocessing, tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace")
    for row in list(workspace.rows)[1:]:
        workspace.delete(row)
    workspace.rows[0].text = "test"
    workspace.save()
    write_audio(workspace.silence_dir / "background.wav", 2, 0.02)
    write_audio(workspace.unknown_dir / "other.wav", 1.04, 0.1)
    keyword = workspace.row_dir(workspace.rows[0])
    write_audio(keyword / "short.wav", 2, 0.5)
    write_audio(keyword / "long.wav", 3, 0.25)
    write_audio(keyword / "no-speech.wav", 2, 0)
    calls = []

    class FakeVad:
        def analyze(self, audio, check_stop, *, prefix, suffix):
            check_stop()
            assert prefix.shape == suffix.shape == (32000,)
            assert np.any(prefix) and np.any(suffix)
            calls.append(len(audio))
            if not np.any(audio):
                return dict(Tstart=None, Tend=None)
            return dict(Tstart=0.2, Tend=1.1) if len(audio) > 32000 else dict(Tstart=0.15, Tend=0.55)

    monkeypatch.setattr(preprocessing, "SileroVad", FakeVad)
    config = importlib.import_module("workspace_train").configure(
        workspace.path, workspace.path / "training/20261020-180000-000000", steps=1)
    return workspace, config, calls


def entries(config):
    return [(split, json.loads(line)) for split in ("training", "validation", "testing")
            for line in (Path(config["data"]["manifests_dir"]) / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]


def test_window_and_gain_rules(preprocessing):
    p = preprocessing
    windows = p.keyword_windows(1.15, 3, np.random.default_rng(5))
    assert len(windows) == 10
    for window in windows:
        a = window["position_a"]
        left = window["start_sample"] / 16000
        assert left == pytest.approx(0.15 + a * 0.1, abs=1 / 16000)
    early = p.keyword_windows(0.5, 2, np.random.default_rng(5))
    assert len(early) == 10
    assert all(window["start_sample"] < 0 for window in early)
    assert len(p.keyword_windows(1, 2, np.random.default_rng(5))) == 10
    assert len(p.keyword_windows(1.9, 2, np.random.default_rng(5))) == 10
    late = p.keyword_windows(1.94, 2, np.random.default_rng(5))
    assert len(late) == 4
    for window in late:
        assert window["start_sample"] / 16000 == pytest.approx(
            0.94 + window["position_a"] * 0.06, abs=1 / 16000)
    assert len(p.keyword_windows(1.98, 2, np.random.default_rng(5))) == 2
    assert len(p.keyword_windows(1.99, 2, np.random.default_rng(5))) == 1
    assert p.keyword_windows(2, 2, np.random.default_rng(5))[0]["start_sample"] == 16000
    with pytest.raises(ValueError, match="至少 2 秒"):
        p.keyword_windows(0.5, 1.5, np.random.default_rng(5))
    assert p.keyword_gains(1) == [0, -3, -6, -9]
    assert p.keyword_gains(0.5) == [0, 3, 6, -3, -6, -9]
    assert p.keyword_gains(0.1) == [0, 3, 6, 9, 12, 15, -3, -6, -9]
    assert p.unknown_starts(16640) == [0, 320, 640]
    assert p.unknown_starts(16641) == [0, 320, 640, 641]
    assert p.unknown_starts(8000) == [0]
    np.testing.assert_array_equal(p.crop(np.ones(16000), -8000), np.r_[np.zeros(8000), np.ones(8000)])
    audio = np.arange(32000, dtype=np.float32)
    np.testing.assert_array_equal(p.crop(audio, -1600, wrap_negative=True),
                                  np.r_[audio[-1600:], audio[:14400]])


def test_vad_ending_at_source_end_keeps_one_training_window_and_vad_clip(preprocessing, corpus, monkeypatch):
    p = preprocessing
    workspace, config, _ = corpus

    class TailVad:
        def analyze(self, audio, check_stop, *, prefix, suffix):
            if not np.any(audio):
                return dict(Tstart=None, Tend=None)
            if len(audio) > 32000:
                return dict(Tstart=0.2, Tend=len(audio) / 16000)
            return dict(Tstart=0.15, Tend=0.55)

    monkeypatch.setattr(p, "SileroVad", TailVad)
    result = p.preprocess(config)
    long_plans = [entry["augmentation"] for _, entry in entries(config)
                  if entry["path"] and Path(entry["path"]).name == "long.wav"]
    assert result["skipped_no_speech"] == 1
    assert len(long_plans) == len(p.keyword_gains(0.25))
    assert all(plan["start_sample"] == 32000 for plan in long_plans)
    assert (Path(config["data"]["preprocess_dir"]) / "keywords/keyword1/long.wav").is_file()


def test_complete_cache_rules_reuse_invalidation_and_dataset(preprocessing, corpus, monkeypatch):
    p = preprocessing
    workspace, config, calls = corpus
    original = {str(wav): p.digest(wav) for wav in workspace.path.rglob("*.wav")}
    first = p.preprocess(config)
    assert first["built_sources"] == 5 and first["reused_sources"] == 0
    assert first["skipped_no_speech"] == 1 and len(calls) == 3
    manifest = entries(config)
    assert Counter(entry["label_index"] for _, entry in manifest) == {0: 1000, 1: 103, 2: 140}
    cache = Path(config["data"]["preprocess_dir"])
    assert cache == Path(config["train"]["run_dir"]) / "preprocess"
    report = cache / "report.md"
    text = report.read_text(encoding="utf-8")
    assert "| 0 | silence | silence | 1000 |" in text
    assert "| 1 | unknown | unknown | 103 |" in text
    assert "| 2 | keywords/keyword1 | test | 140 |" in text
    assert "| 合计 | — | — | 1243 |" in text
    assert "生成白噪声：100 条" in text
    silence = [entry["augmentation"] for _, entry in manifest if entry["label_index"] == 0]
    assert sum(plan["random_gain"] for plan in silence) == 500
    assert all(-8 <= plan["gain_db"] <= 10 for plan in silence)
    assert sum(plan["gain_db"] == 0 for plan in silence) == 500
    white = [entry for _, entry in manifest if entry["path"] is None]
    assert len(white) == 100 and all(-40 <= item["augmentation"]["gain_db"] <= -10 for item in white)
    # Overlapping windows and all gain/position variants of one original stay
    # in the same split, including the continuous silence recording.
    for source in original:
        assert len({split for split, entry in manifest if entry["path"] == source}) <= 1
    dictionary = json.loads((cache / "keywords/keyword1/vad.json").read_text())
    assert dictionary["no-speech.wav"]["Tstart"] is None
    assert dictionary["no-speech.wav"]["speech_clip"] is None
    assert not (cache / "keywords/keyword1/no-speech.wav").exists()
    for name, start, end in [("short.wav", 2400, 8800), ("long.wav", 3200, 17600)]:
        clip = cache / "keywords/keyword1" / name
        assert dictionary[name]["speech_clip"] == dict(path=str(clip), start_sample=start,
                                                      end_sample=end, samples=end - start, sample_rate=16000)
        with wave.open(str(clip), "rb") as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 16000, end - start)
            actual_pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        with wave.open(str(workspace.row_dir(workspace.rows[0]) / name), "rb") as wav:
            expected_pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")[start:end]
        np.testing.assert_array_equal(actual_pcm, expected_pcm)
    assert dictionary["short.wav"]["max_amplitude"] == pytest.approx(0.5, abs=1e-4)
    sample = next(entry for _, entry in manifest if entry["path"] and Path(entry["path"]).name == "short.wav")
    original_audio = p.load_wav(sample["path"], 16000).numpy()
    clip = p.crop(original_audio, sample["augmentation"]["start_sample"], wrap_negative=True)
    expected = p.MfccExtractor(config["data"], config["features"])(p.torch.from_numpy(clip)).detach().numpy()[0]
    np.testing.assert_allclose(p.load_feature(sample), expected, atol=5e-4, rtol=1e-4)
    mtimes = {source["feature_path"]: Path(source["feature_path"]).stat().st_mtime_ns for source in first["sources"]}
    with monkeypatch.context() as no_dsp:
        report.unlink()  # A cache-only pass must also regenerate its report.
        (cache / "keywords/keyword1/short.wav").unlink()
        def fail(*args, **kwargs):
            raise AssertionError("Cached path must not execute VAD or MFCC")
        no_dsp.setattr(p, "SileroVad", fail)
        no_dsp.setattr(p.MfccExtractor, "forward", fail)
        second = p.preprocess(config)
        assert second["built_sources"] == 0 and second["reused_sources"] == 5
        assert (cache / "keywords/keyword1/short.wav").is_file()
        assert "| 2 | keywords/keyword1 | test | 140 |" in report.read_text(encoding="utf-8")
        dataset = importlib.import_module("data.dataset").SpeechCommandsDataset(config, "training", training=True)
        no_dsp.setattr("data.dataset.load_wav", fail)
        feature, label = dataset[0]
        assert feature.shape == (40, 98) and label == 0
        assert dataset.extractor is None
    assert all(Path(path).stat().st_mtime_ns == mtime for path, mtime in mtimes.items())
    assert all(p.digest(path) == value for path, value in original.items())
    write_audio(workspace.row_dir(workspace.rows[0]) / "short.wav", 2, 0.4)
    changed = p.preprocess(config)
    assert changed["built_sources"] == 1 and changed["reused_sources"] == 4
    assert len(calls) == 4  # Only the modified recording re-runs VAD.
    target = next(item for item in changed["sources"] if item["kind"] == "unknown")
    Path(target["feature_path"]).write_bytes(b"interrupted/corrupt cache")
    recovered = p.preprocess(config)
    assert recovered["built_sources"] == 1 and recovered["reused_sources"] == 4


def test_forced_preprocessing_rebuilds_unchanged_data_and_refreshes_wavs(preprocessing, corpus, monkeypatch):
    p = preprocessing
    workspace, config, calls = corpus
    p.preprocess(config)
    batches = []
    forward = p.MfccExtractor.forward
    def track_mfcc(self, audio):
        batches.append(len(audio))
        return forward(self, audio)
    monkeypatch.setattr(p.MfccExtractor, "forward", track_mfcc)
    forced = p.preprocess(config, force=True)
    assert forced["force_rebuild"]
    assert forced["built_sources"] == 5 and forced["reused_sources"] == 0
    assert len(calls) == 6 and sum(batches) == 1243
    keyword = workspace.row_dir(workspace.rows[0])
    (keyword / "long.wav").unlink()
    write_audio(keyword / "short.wav", 2, 0.4)
    write_audio(workspace.unknown_dir / "new.wav", 1.06, 0.1)
    refreshed = p.preprocess(config, force=True)
    assert refreshed["reused_sources"] == 0 and len(calls) == 8
    active = entries(config)
    assert Counter(entry["label_index"] for _, entry in active) == {0: 1000, 1: 107, 2: 60}
    assert not any(entry["path"] and Path(entry["path"]).name == "long.wav" for _, entry in active)
    dictionary = json.loads((Path(config["data"]["preprocess_dir"]) / "keywords/keyword1/vad.json").read_text())
    assert "long.wav" not in dictionary
    assert dictionary["short.wav"]["max_amplitude"] == pytest.approx(0.4, abs=1e-4)


def test_stop_before_or_during_preprocessing_does_not_publish(preprocessing, corpus, monkeypatch):
    p = preprocessing
    workspace, config, _ = corpus
    stop = Path(config["train"]["stop_file"])
    stop.parent.mkdir(exist_ok=True, parents=True)
    stop.touch()
    with pytest.raises(InterruptedError):
        p.preprocess(config)
    stop.unlink()
    forward = p.MfccExtractor.forward
    def stop_after_batch(self, audio):
        stop.touch()
        return forward(self, audio)
    monkeypatch.setattr(p.MfccExtractor, "forward", stop_after_batch)
    with pytest.raises(InterruptedError):
        p.preprocess(config)
    assert not (Path(config["data"]["preprocess_dir"]) / "metadata.json").exists()


def test_vad_exports_same_filenames_into_separate_keyword_folders(preprocessing, corpus):
    workspace, config, _ = corpus
    second = workspace.add()
    second.text = "second"
    workspace.save()
    write_audio(workspace.row_dir(second) / "short.wav", 2, 0.1)
    config = importlib.import_module("workspace_train").configure(
        workspace.path, Path(config["train"]["run_dir"]), steps=1)
    preprocessing.preprocess(config, force=True)
    cache = Path(config["data"]["preprocess_dir"])
    clips = [cache / "keywords" / row.directory / "short.wav" for row in workspace.rows]
    assert all(clip.is_file() for clip in clips)
    assert clips[0].read_bytes() != clips[1].read_bytes()
    for row, clip in zip(workspace.rows, clips):
        dictionary = json.loads((clip.parent / "vad.json").read_text())
        assert dictionary["short.wav"]["speech_clip"]["path"] == str(clip)


def test_stopped_speech_export_does_not_publish_partial_wav(preprocessing, tmp_path):
    target = tmp_path / "speech.wav"
    checks = []
    def stop_before_publish():
        checks.append(True)
        if len(checks) == 2:
            raise InterruptedError("Stopped")
    with pytest.raises(InterruptedError):
        preprocessing.export_speech(target, np.ones(16000, np.float32),
                                    dict(Tstart=0.1, Tend=0.5), stop_before_publish)
    assert not target.exists() and not target.with_suffix(".wav.tmp").exists()


def test_separate_training_runs_keep_their_own_features_and_report(preprocessing, corpus):
    p = preprocessing
    workspace, config, _ = corpus
    first = p.preprocess(config, force=True)
    old_cache = Path(config["data"]["preprocess_dir"])
    old_metadata = (old_cache / "metadata.json").read_bytes()
    old_report = (old_cache / "report.md").read_bytes()
    new_run = workspace.path / "training/20261020-190000-000000"
    new_config = importlib.import_module("workspace_train").configure(workspace.path, new_run, steps=1)
    second = p.preprocess(new_config, force=True)
    for result, cache in [(first, old_cache), (second, new_run / "preprocess")]:
        assert result["built_sources"] == 5 and result["reused_sources"] == 0
        assert all(Path(source["feature_path"]).is_relative_to(cache) for source in result["sources"])
        assert (cache / "report.md").is_file()
    assert all(Path(entry["feature_path"]).is_relative_to(new_run / "preprocess") for _, entry in entries(new_config))
    assert (old_cache / "metadata.json").read_bytes() == old_metadata
    assert (old_cache / "report.md").read_bytes() == old_report
    assert not (workspace.path / "train").exists()


@pytest.mark.parametrize("folder", ["train/preprocess", "training/another-run/preprocess"])
def test_preprocessing_cannot_write_outside_its_run(preprocessing, corpus, folder):
    workspace, config, _ = corpus
    target = workspace.path / folder
    config["data"]["preprocess_dir"] = str(target)
    with pytest.raises(ValueError, match="run_dir/preprocess"):
        preprocessing.preprocess(config)
    assert not target.exists()


def test_silero_v6_silence_and_independent_recording_state(preprocessing):
    vad = importlib.import_module("data.silero").SileroVad()
    context = np.zeros(32000, np.float32)
    result = vad.analyze(np.zeros(16000, np.float32), prefix=context, suffix=context)
    assert result["Tstart"] is None and result["Tend"] is None
    assert len(result["probabilities"]) == 157 and result["hop_samples"] == 512
    assert result["backend"] == "silero-vad" and result["version"] == "6.0"
    repeated = vad.analyze(np.zeros(16000, np.float32), prefix=context, suffix=context)
    assert repeated["probabilities"] == result["probabilities"]


def test_silence_context_wraps_short_recordings_without_zero_padding(preprocessing):
    audio = np.linspace(0.01, 0.1, 8000, dtype=np.float32)
    context, metadata = preprocessing.silence_context(
        [dict(path="background.wav", sha256="abc")], [audio], np.random.default_rng(5))
    assert len(context) == 32000 and metadata["wrapped"]
    assert metadata["source_sha256"] == "abc" and metadata["samples"] == 32000
    np.testing.assert_array_equal(context, audio[(metadata["start_sample"] + np.arange(32000)) % len(audio)])


def test_changing_silence_invalidates_cached_keyword_vad(preprocessing, corpus):
    workspace, config, calls = corpus
    preprocessing.preprocess(config)
    cache = Path(config["data"]["preprocess_dir"])
    dictionary = cache / "keywords/keyword1/vad.json"
    before = json.loads(dictionary.read_text())
    write_audio(workspace.silence_dir / "background.wav", 2, 0.01)
    preprocessing.preprocess(config)
    after = json.loads(dictionary.read_text())
    assert len(calls) == 6
    assert before["short.wav"]["vad_signature"] != after["short.wav"]["vad_signature"]
    assert before["short.wav"]["silence_context"]["prefix"]["source_sha256"] != after["short.wav"]["silence_context"]["prefix"]["source_sha256"]


def test_changing_random_seed_invalidates_context_and_vad(preprocessing, corpus):
    _, config, calls = corpus
    preprocessing.preprocess(config)
    config["project"]["seed"] += 1
    preprocessing.preprocess(config)
    assert len(calls) == 6
