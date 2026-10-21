"""Audio acquisition runs outside the UI thread; only complete WAVs are published."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import gcd
from pathlib import Path
import queue
import threading
import time
from uuid import uuid4
import wave

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 16000
KEYWORD_SECONDS = 2
NOISE_SECONDS = 3
SILENCE_SECONDS = 60
START_DISCARD_MS = 400


@dataclass
class RecordingResult:
    path: Path | None = None
    error: str | None = None
    cancelled: bool = False


def input_devices() -> list[dict]:
    hosts = sd.query_hostapis()
    return [dict(id=i, name=d["name"], host=hosts[d["hostapi"]]["name"],
                 channels=d["max_input_channels"], rate=d["default_samplerate"])
            for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


def convert_wav(source: Path, target: Path, source_rate: int, frames: int) -> None:
    """Resample only if the driver cannot capture at 16 kHz."""
    if source_rate == SAMPLE_RATE:
        source.replace(target)
        return
    from scipy.signal import resample_poly

    with wave.open(str(source), "rb") as reader:
        samples = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    factor = gcd(source_rate, SAMPLE_RATE)
    converted = resample_poly(samples.astype(np.float32), SAMPLE_RATE // factor, source_rate // factor)
    pcm = np.clip(np.rint(converted[:frames]), -32768, 32767).astype("<i2")
    if len(pcm) != frames:
        raise RuntimeError("重采样后的音频长度不正确，请重新录制")
    staged = target.with_suffix(".converting")
    try:
        with wave.open(str(staged), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(SAMPLE_RATE)
            writer.writeframes(pcm.tobytes())
        staged.replace(target)
    finally:
        staged.unlink(missing_ok=True)


class Recorder:
    def __init__(self):
        self.results: queue.SimpleQueue[RecordingResult] = queue.SimpleQueue()
        self.thread: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._state = {"elapsed": 0.0, "waveform": [0.0] * 120, "phase": "idle", "warning": ""}
        self.row_id: str | None = None
        self.duration = 0

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)

    def start(self, directory: Path, seconds: int, device: int | None, row_id: str) -> None:
        if self.busy:
            raise RuntimeError("请先等待当前录音结束")
        directory.mkdir(parents=True, exist_ok=True)
        self.cancel_event.clear()
        self.row_id, self.duration = row_id, seconds
        with self._lock:
            self._state = {"elapsed": 0.0, "waveform": [0.0] * 120, "phase": "starting", "warning": ""}
        self.thread = threading.Thread(target=self._record, args=(directory, seconds, device), daemon=True)
        self.thread.start()

    def cancel(self) -> None:
        self.cancel_event.set()

    def close(self) -> None:
        self.cancel()
        if self.thread:
            self.thread.join(timeout=5)

    def _record(self, directory: Path, seconds: int, device: int | None) -> None:
        target = directory / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}.wav"
        partial = target.with_suffix(".recording")
        blocks: queue.Queue = queue.Queue(maxsize=256)
        fault: list[str] = []
        result = RecordingResult()
        try:
            rate = SAMPLE_RATE
            try:
                sd.check_input_settings(device=device, channels=1, dtype="int16", samplerate=rate)
            except sd.PortAudioError:
                rate = round(sd.query_devices(device, "input")["default_samplerate"])
                sd.check_input_settings(device=device, channels=1, dtype="int16", samplerate=rate)

            def callback(indata, frames, timing, status):
                if status:
                    fault.append(str(status))
                try:
                    blocks.put_nowait(bytes(indata))
                except queue.Full:
                    fault.append("音频缓冲区溢出")

            limit, written = int(rate * seconds), 0
            discard = round(rate * START_DISCARD_MS / 1000)
            with wave.open(str(partial), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(rate)
                with sd.RawInputStream(samplerate=rate, channels=1, dtype="int16", device=device,
                                       blocksize=1024, callback=callback) as stream:
                    last_data = time.monotonic()
                    while written < limit and not self.cancel_event.is_set():
                        if fault:
                            raise RuntimeError("录音中断：" + fault[0] + "。请检查设备后重试。")
                        try:
                            block = blocks.get(timeout=0.15)
                        except queue.Empty:
                            if not stream.active or time.monotonic() - last_data > 3:
                                raise RuntimeError("麦克风没有返回音频，请检查设备连接和系统麦克风权限")
                            continue
                        last_data = time.monotonic()
                        if discard:
                            skipped = min(discard, len(block) // 2)
                            block = block[skipped * 2:]
                            discard -= skipped
                            if discard:
                                continue
                            with self._lock:
                                self._state["phase"] = "recording"
                        if not block:
                            continue
                        block = block[:(limit - written) * 2]
                        writer.writeframesraw(block)
                        written += len(block) // 2
                        values = np.frombuffer(block, dtype="<i2").astype(np.float32) / 32768
                        envelope = [float(np.max(np.abs(chunk))) for chunk in np.array_split(values, 12) if chunk.size]
                        with self._lock:
                            self._state["elapsed"] = written / rate
                            self._state["waveform"] = (self._state["waveform"] + envelope)[-120:]
                            self._state["warning"] = "输入接近削波，请降低麦克风增益" if np.max(np.abs(values)) >= 0.99 else ""
                    if fault:
                        raise RuntimeError("录音数据不完整，请重新录制：" + fault[0])
            if self.cancel_event.is_set():
                result.cancelled = True
            else:
                with self._lock:
                    self._state["phase"] = "saving"
                convert_wav(partial, target, rate, seconds * SAMPLE_RATE)
                if self.cancel_event.is_set():
                    target.unlink(missing_ok=True)
                    result.cancelled = True
                else:
                    result.path = target
        except Exception as exc:
            result.error = str(exc)
        finally:
            try:
                partial.unlink(missing_ok=True)
            except OSError as exc:
                result.error = result.error or f"临时文件清理失败：{exc}"
            with self._lock:
                self._state["phase"] = "idle"
            self.results.put(result)
