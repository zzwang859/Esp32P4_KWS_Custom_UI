"""WAV loading and MFCC extraction without parent-project dependencies."""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np
import torch
import torchaudio
from device_contract import FRONTEND_ASSET, PROFILE, frontend_sha256


def load_wav(path: str | Path, sample_rate: int) -> torch.Tensor:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        width = wav_file.getsampwidth()
        source_rate = wav_file.getframerate()
        raw = wav_file.readframes(wav_file.getnframes())
    if width != 2:
        raise ValueError(f"Only PCM16 WAV is supported: {path}")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    tensor = torch.from_numpy(audio.copy())
    if source_rate != sample_rate:
        tensor = torchaudio.functional.resample(tensor, source_rate, sample_rate)
    return tensor


def fit_length(audio: torch.Tensor, samples: int, random_crop: bool = False) -> torch.Tensor:
    if audio.numel() < samples:
        return torch.nn.functional.pad(audio, (0, samples - audio.numel()))
    if audio.numel() == samples:
        return audio
    maximum = audio.numel() - samples
    start = int(torch.randint(maximum + 1, (1,)).item()) if random_crop else maximum // 2
    return audio[start:start + samples]


class MfccExtractor(torch.nn.Module):
    def __init__(self, data_config: dict, feature_config: dict):
        super().__init__()
        if int(data_config["sample_rate"]) != 16000 or feature_config.get("profile") != PROFILE:
            raise ValueError("MFCC must use the fixed ESP32-P4 frontend profile")
        self.asset_sha256 = frontend_sha256()
        self.right_padding = 0
        self.n_fft = 512
        with np.load(FRONTEND_ASSET, allow_pickle=False) as assets:
            for name in ("window", "mel_filters", "dct", "normalization"):
                self.register_buffer(name, torch.from_numpy(assets[name].astype(np.float32).copy()))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        frames = audio.unfold(-1, 480, 160) * self.window
        magnitude = torch.fft.rfft(frames, n=512, dim=-1).abs()[..., :224]
        log_mel = (magnitude @ self.mel_filters).clamp_min(1e-12).log()
        features = (log_mel @ self.dct) * self.normalization
        return features.transpose(-1, -2)

    def streaming(self, audio: torch.Tensor) -> torch.Tensor:
        """Match the device's zero-initialized 480-sample history at startup."""
        audio = torch.nn.functional.pad(audio, (320, (-audio.shape[-1]) % 160))
        return self(audio)

