"""Port of USB_Audio_HS2/src/kws/kws_stream_state.hpp Debouncer.

The four Kconfig controls are stored in each workspace.json.
The device's fixed 100-chunk warmup remains separate from these controls.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_FILTER_CONFIG = dict(keyword_threshold_percent=80, cooldown_ms=500,
                             stable_frames=3, need_vad_checked=False)


def validate_filter_config(config: dict) -> dict:
    bounds = dict(keyword_threshold_percent=(0, 100), cooldown_ms=(1, 1000), stable_frames=(1, 50))
    if not isinstance(config, dict) or set(config) != {*bounds, "need_vad_checked"}:
        raise ValueError("KWS 滤波配置必须包含 keyword_threshold_percent、cooldown_ms、stable_frames、need_vad_checked 四项")
    for name, (low, high) in bounds.items():
        if type(config[name]) is not int or not low <= config[name] <= high:
            raise ValueError(f"KWS 滤波参数 {name} 必须是 {low}～{high} 的整数")
    if type(config["need_vad_checked"]) is not bool:
        raise ValueError("KWS 滤波参数 need_vad_checked 必须是 true 或 false")
    if config["cooldown_ms"] != 1 and config["cooldown_ms"] % 10:
        raise ValueError("KWS 滤波参数 cooldown_ms 必须为 1 或 10 的倍数")
    return dict(config)


def load_filter_config(path: Path) -> dict:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("工作区配置必须是 JSON 对象")
        return validate_filter_config(data.get("kws_filter", DEFAULT_FILTER_CONFIG))
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法加载 KWS 滤波配置 {path}：{exc}") from exc


def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    """Device softmax: subtract the top logit, exponentiate, normalize in float32."""
    logits = np.asarray(logits, dtype=np.float32)
    values = np.exp(logits - logits.max())
    total = np.float32(0)
    for value in values:
        total += value
    return values / total


class Debouncer:
    def __init__(self, config: dict | None = None):
        self.config = validate_filter_config(DEFAULT_FILTER_CONFIG if config is None else config)
        self.warmup_chunks = 100
        self.stable_chunks = self.config["stable_frames"]
        self.keyword_min_probability = self.config["keyword_threshold_percent"] / 100.0
        self.cooldown_us = self.config["cooldown_ms"] * 1000
        self.need_vad_checked = self.config["need_vad_checked"]
        self.cooldown_until_us = 0
        self.last_keyword_us: int | None = None
        self.reset()

    def configure(self, config: dict) -> None:
        """Apply live controls without restarting warmup or forgetting the last report."""
        updated = validate_filter_config(config)
        if updated == self.config:
            return
        self.config = updated
        self.stable_chunks = updated["stable_frames"]
        self.keyword_min_probability = updated["keyword_threshold_percent"] / 100.0
        self.cooldown_us = updated["cooldown_ms"] * 1000
        self.need_vad_checked = updated["need_vad_checked"]
        if self.last_keyword_us is not None:
            self.cooldown_until_us = self.last_keyword_us + self.cooldown_us
        self.reject_candidate()

    def reset(self) -> None:
        # A discontinuity restarts warmup, without bypassing cooldown.
        self.processed = self.stable = 0
        self.candidate = self.last_reported = -1

    def reject_candidate(self) -> int:
        self.candidate, self.stable = -1, 0
        return -1

    def update(self, index: int, logits: np.ndarray, now_us: int, vad_high: bool = False) -> int:
        logits = np.asarray(logits, dtype=np.float32)
        if (logits.ndim != 1 or not 2 <= logits.size <= 34 or not 0 <= index < logits.size
                or not np.isfinite(logits).all() or np.any(logits > logits[index])):
            return self.reject_candidate()
        if self.processed < self.warmup_chunks:
            self.processed += 1
        if index >= 2:
            if self.need_vad_checked and not vad_high:
                return self.reject_candidate()
            if now_us < self.cooldown_until_us or index == self.last_reported:
                return self.reject_candidate()
            # Float32 accumulation and comparison match the device implementation.
            if softmax_probabilities(logits)[index] < np.float32(self.keyword_min_probability):
                return self.reject_candidate()
        if self.candidate == index:
            self.stable = min(self.stable + 1, self.stable_chunks)
        else:
            self.candidate, self.stable = index, 1
        if (self.processed >= self.warmup_chunks and self.stable >= self.stable_chunks
                and self.candidate != self.last_reported):
            return self.candidate
        return -1

    def acknowledge(self, accepted: bool, now_us: int) -> None:
        if accepted:
            self.last_reported = self.candidate
            if self.candidate >= 2:
                self.last_keyword_us = now_us
                self.cooldown_until_us = now_us + self.cooldown_us
                self.reject_candidate()
        else:
            self.stable = self.stable_chunks - 1
