"""Offline Silero VAD v6, using the pinned GitHub source under src/silero-vad."""
from functools import lru_cache
import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2] / "silero-vad"
UTILS = ROOT / "src/silero_vad/utils_vad.py"
MODEL = ROOT / "src/silero_vad/data/silero_vad.jit"
RATE = 16000
HOP = 512
PADDING_SAMPLES = 2 * RATE
PARAMETERS = dict(threshold=0.5, min_speech_duration_ms=250,
                  min_silence_duration_ms=100, speech_pad_ms=30)


@lru_cache(maxsize=1)
def upstream_utils():
    # Load exactly the vendored source, rather than any installed pip package.
    spec = importlib.util.spec_from_file_location("kws_silero_v6_utils", UTILS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeviceModel:
    """Keep Silero's stateful interface while moving input chunks to its device."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def reset_states(self):
        self.model.reset_states()

    @torch.inference_mode()
    def __call__(self, chunk, rate):
        return self.model(chunk.to(self.device), rate)


class SileroVad:
    def __init__(self, model=MODEL):
        self.utils = upstream_utils()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeviceModel(self.utils.init_jit_model(str(model), device=self.device), self.device)
        print(f"SILERO_VAD_DEVICE={self.device}", flush=True)

    def analyze(self, audio, check_stop=lambda: None, *, prefix, suffix):
        audio = np.asarray(audio, dtype=np.float32)
        prefix, suffix = (np.asarray(value, dtype=np.float32) for value in (prefix, suffix))
        if audio.ndim != 1 or not len(audio) or prefix.shape != (PADDING_SAMPLES,) or suffix.shape != (PADDING_SAMPLES,):
            raise ValueError("Silero VAD requires mono audio and exactly 2 seconds of silence context on each side")
        check_stop()
        padded = torch.from_numpy(np.concatenate([prefix, audio, suffix]))
        probabilities = []
        model = self.model

        class RecordingModel:
            def reset_states(self):
                model.reset_states()

            def __call__(self, chunk, rate):
                value = model(chunk, rate)
                probabilities.append(float(value.item()))
                return value

        padded_segments = self.utils.get_speech_timestamps(
            padded, RecordingModel(), sampling_rate=RATE, return_seconds=False,
            progress_tracking_callback=lambda progress: check_stop(), **PARAMETERS)
        # Ignore speech in the added context. Clamp intersecting segments to the
        # original recording, and map both endpoints back to original samples.
        segments = []
        for item in padded_segments:
            start = max(0, int(item["start"]) - PADDING_SAMPLES)
            end = min(len(audio), int(item["end"]) - PADDING_SAMPLES)
            if end > start:
                segments.append(dict(start=start, end=end))
        return dict(Tstart=segments[0]["start"] / RATE if segments else None,
                    Tend=segments[-1]["end"] / RATE if segments else None,
                    segments=segments, segments_padded=padded_segments,
                    probabilities=probabilities, probability_time_offset_seconds=-2.0,
                    hop_samples=HOP, padding_samples=PADDING_SAMPLES,
                    backend="silero-vad", version="6.0", **PARAMETERS)
