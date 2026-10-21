"""Silero's context must not shift keyword timestamps or enter cropped audio."""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from kws_custom_ui.training import TRAINING_ROOT


@pytest.fixture
def silero(monkeypatch):
    monkeypatch.syspath_prepend(str(TRAINING_ROOT))
    importlib.import_module("windows_runtime").preload_msvc()
    return importlib.import_module("data.silero")


@pytest.mark.parametrize("cuda_available,expected", [(True, "cuda"), (False, "cpu")])
def test_device_selection_and_input_placement(silero, monkeypatch, capsys, cuda_available, expected):
    torch = silero.torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    resets, calls = [], []
    device_chunk = object()

    class Model:
        def reset_states(self):
            resets.append(True)

        def __call__(self, chunk, rate):
            assert chunk is device_chunk and rate == 16000
            assert not torch.is_grad_enabled()
            return torch.tensor(0.6)

    class Chunk:
        def to(self, device):
            assert device == torch.device(expected)
            return device_chunk

    model = Model()

    def load(path, *, device):
        calls.append((path, device))
        return model

    monkeypatch.setattr(silero, "upstream_utils", lambda: SimpleNamespace(init_jit_model=load))
    vad = silero.SileroVad()
    assert calls == [(str(silero.MODEL), torch.device(expected))]
    vad.model.reset_states()
    assert resets == [True]
    assert vad.model(Chunk(), 16000).item() == pytest.approx(0.6)
    assert f"SILERO_VAD_DEVICE={expected}" in capsys.readouterr().out


def test_torchscript_matches_onnx_on_real_audio_and_resets_state(silero, monkeypatch):
    monkeypatch.setattr(silero.torch.cuda, "is_available", lambda: False)
    import wave

    audio_path = silero.ROOT / "tests/data/test.wav"
    with wave.open(str(audio_path), "rb") as wav:
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
        audio = np.frombuffer(wav.readframes(8 * 16000), dtype="<i2").astype(np.float32) / 32768
    vad = silero.SileroVad()
    reference = silero.upstream_utils().OnnxWrapper(str(silero.MODEL.with_suffix(".onnx")), force_onnx_cpu=True)
    chunks = silero.torch.from_numpy(np.pad(audio, (0, (-len(audio)) % 512))).split(512)
    expected = [float(reference(chunk, 16000).item()) for chunk in chunks]
    assert max(expected) > 0.5
    for _ in range(2):
        vad.model.reset_states()
        actual = [float(vad.model(chunk, 16000).item()) for chunk in chunks]
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("context_only", [False, True])
def test_context_and_segment_coordinates_are_relative_to_original(monkeypatch, context_only):
    monkeypatch.syspath_prepend(str(TRAINING_ROOT))
    importlib.import_module("windows_runtime").preload_msvc()
    import torch
    module = importlib.import_module("data.silero")
    resets = []
    vad = module.SileroVad.__new__(module.SileroVad)
    vad.model = SimpleNamespace()  # Replaced with a callable stub below.
    class Model:
        def reset_states(self):
            resets.append(True)
        def __call__(self, chunk, rate):
            assert rate == 16000
            return torch.tensor(0.6)
    vad.model = Model()
    original = np.full(32000, 0.3, np.float32)
    prefix = np.full(32000, 0.1, np.float32)
    suffix = np.full(32000, 0.2, np.float32)
    padded_segments = [dict(start=0, end=16000), dict(start=70000, end=80000)]
    if not context_only:
        padded_segments = [padded_segments[0], dict(start=31900, end=35000),
                           dict(start=40000, end=42000), dict(start=63000, end=65000), padded_segments[-1]]
    def timestamps(audio, model, **kwargs):
        np.testing.assert_array_equal(audio.numpy(), np.concatenate([prefix, original, suffix]))
        assert kwargs["return_seconds"] is False
        model.reset_states()
        model(audio[:512], 16000)
        kwargs["progress_tracking_callback"](100)
        return padded_segments
    vad.utils = SimpleNamespace(get_speech_timestamps=timestamps)
    result = vad.analyze(original, prefix=prefix, suffix=suffix)
    assert resets == [True]
    assert result["probability_time_offset_seconds"] == -2.0
    assert result["segments_padded"] == padded_segments
    if context_only:
        assert result["segments"] == [] and result["Tstart"] is None and result["Tend"] is None
    else:
        assert result["segments"] == [dict(start=0, end=3000), dict(start=8000, end=10000),
                                       dict(start=31000, end=32000)]
        assert result["Tstart"] == 0 and result["Tend"] == 2


def test_silero_processing_can_be_cancelled(monkeypatch):
    monkeypatch.syspath_prepend(str(TRAINING_ROOT))
    importlib.import_module("windows_runtime").preload_msvc()
    module = importlib.import_module("data.silero")
    vad = module.SileroVad()
    context = np.zeros(32000, np.float32)
    checks = []
    def cancel():
        checks.append(True)
        if len(checks) > 3:
            raise InterruptedError("Stopped")
    with pytest.raises(InterruptedError):
        vad.analyze(context, cancel, prefix=context, suffix=context)
