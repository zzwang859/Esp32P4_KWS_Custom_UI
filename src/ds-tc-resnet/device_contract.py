"""Fixed model/frontend ABI shared with the reference ESP32-P4 firmware."""
import hashlib
import json
from pathlib import Path

PROFILE = "mfcc40_16k_int16_state6_v1"
FEATURE_INPUT = "streaming/speech_features/normalizer_1/truediv:0"
LOGITS_OUTPUT = "onnx_logits:0"
STATE_INPUTS = [f"streaming/stream{'_' + str(i) if i else ''}/stream{'_' + str(i) if i else ''}/ExternalState:0"
                for i in (0, 1, 2, 3, 4, 6)]
STATE_OUTPUTS = [f"onnx_next_state_{i}:0" for i in (1, 2, 3, 4, 5, 7)]
STATE_SHAPES = [(11, 1, 40), (13, 1, 64), (15, 1, 32), (17, 1, 32), (57, 1, 32), (98, 1, 64)]
FRONTEND_ASSET = Path(__file__).resolve().parent / "data/esp32p4_frontend.npz"


def frontend_sha256():
    value = hashlib.sha256(FRONTEND_ASSET.read_bytes()).hexdigest()
    metadata = json.loads(FRONTEND_ASSET.with_suffix(".json").read_text(encoding="utf-8"))
    if value != metadata["asset_sha256"]:
        raise ValueError("固定 ESP32-P4 MFCC 系数校验失败")
    return value


def validate_contract(config):
    expected_features = dict(profile=PROFILE, window_size_ms=30.0, window_stride_ms=10.0,
                             mel_bins=80, mfcc_bins=40, mel_low_hz=20.0, mel_high_hz=7000.0)
    expected_model = dict(profile=PROFILE, pooling_frames=98, filters=[64, 32, 32, 32, 64, 64],
                          repeats=[1] * 6, kernel_sizes=[11, 13, 15, 17, 29, 1], strides=[1] * 6,
                          dilations=[1, 1, 1, 1, 2, 1], separable=[True] * 6,
                          residual=[False, True, True, True, False, False], padding=["causal"] * 6,
                          pools=[1] * 6, dropout=0.0)
    for section, expected in (("features", expected_features), ("model", expected_model),
                              ("data", dict(sample_rate=16000, clip_duration_ms=1000)),
                              ("onnx", dict(mode="stream_external", chunk_frames=1))):
        for name, value in expected.items():
            if config.get(section, {}).get(name) != value:
                raise ValueError(f"{section}.{name} 不符合固定 ESP32-P4 契约 {PROFILE}；旧版训练历史需要重新开始训练")
