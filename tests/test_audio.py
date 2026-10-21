import threading
import time
import wave

import numpy as np
import pytest

from kws_custom_ui import audio


class FakeInput:
    """Deliver uneven blocks, faster than real time, with a real callback thread."""

    status = None
    stopped = False

    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self.active = False
        self.stop = threading.Event()

    def __enter__(self):
        self.active = not self.stopped
        self.thread = threading.Thread(target=self.produce, daemon=True)
        self.thread.start()
        return self

    def produce(self):
        x = np.arange(17777)
        block = (12000 * np.sin(x * 0.13)).astype("<i2").tobytes()
        while not self.stop.is_set():
            if self.active:
                self.callback(block, len(x), None, self.status)
            time.sleep(0.003)

    def __exit__(self, *args):
        self.active = False
        self.stop.set()
        self.thread.join(timeout=2)


@pytest.fixture
def fake_input(monkeypatch):
    monkeypatch.setattr(audio.sd, "check_input_settings", lambda **kwargs: None)
    monkeypatch.setattr(audio.sd, "RawInputStream", FakeInput)
    monkeypatch.setattr(FakeInput, "status", None)
    monkeypatch.setattr(FakeInput, "stopped", False)


def result_of(recorder):
    # A cold scipy import on Windows can take longer than the actual conversion.
    recorder.thread.join(timeout=60)
    assert not recorder.busy
    return recorder.results.get(timeout=1)


@pytest.mark.parametrize("seconds", [audio.KEYWORD_SECONDS, audio.NOISE_SECONDS, audio.SILENCE_SECONDS])
def test_recording_has_exact_pcm_format_and_duration(tmp_path, fake_input, seconds):
    recorder = audio.Recorder()
    recorder.start(tmp_path, seconds, None, "row")
    result = result_of(recorder)
    assert result.error is None
    assert result.path is not None
    with wave.open(str(result.path), "rb") as reader:
        assert (reader.getframerate(), reader.getsampwidth(), reader.getnchannels()) == (16000, 2, 1)
        assert reader.getnframes() == seconds * 16000
        assert any(reader.readframes(100))
    assert list(tmp_path.iterdir()) == [result.path]


def test_unsupported_rate_falls_back_and_resamples(tmp_path, fake_input, monkeypatch):
    def settings(**kwargs):
        if kwargs["samplerate"] == 16000:
            raise audio.sd.PortAudioError("unsupported rate")
    monkeypatch.setattr(audio.sd, "check_input_settings", settings)
    monkeypatch.setattr(audio.sd, "query_devices", lambda *args: {"default_samplerate": 48000})
    recorder = audio.Recorder()
    recorder.start(tmp_path, 3, 7, "row")
    result = result_of(recorder)
    assert result.error is None
    with wave.open(str(result.path), "rb") as reader:
        assert reader.getframerate() == 16000
        assert reader.getnframes() == 48000
        assert reader.getsampwidth() == 2
        assert reader.getnchannels() == 1
    assert list(tmp_path.iterdir()) == [result.path]


def test_cancel_discards_partial_audio(tmp_path, fake_input):
    recorder = audio.Recorder()
    recorder.start(tmp_path, 300, None, "silence")
    time.sleep(0.03)
    recorder.cancel()
    result = result_of(recorder)
    assert result.cancelled
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["overflow", "disconnected", "unavailable"])
def test_capture_failure_never_publishes_partial_wav(tmp_path, fake_input, monkeypatch, failure):
    if failure == "overflow":
        monkeypatch.setattr(FakeInput, "status", "input overflow")
    elif failure == "disconnected":
        monkeypatch.setattr(FakeInput, "stopped", True)
    else:
        def unavailable(**kwargs):
            raise RuntimeError("device unavailable")
        monkeypatch.setattr(audio.sd, "RawInputStream", unavailable)
    recorder = audio.Recorder()
    recorder.start(tmp_path, 3, None, "row")
    result = result_of(recorder)
    assert result.error
    assert not list(tmp_path.iterdir())
