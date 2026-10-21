"""Shared application settings stored alongside workspace memory."""
from __future__ import annotations

import json
from pathlib import Path


SESSION_FILE = Path(__file__).resolve().parents[2] / ".kws-studio.json"
DEFAULT_HID_DEVICE = {
    "vid": "0x303A",
    "pid": "0x8003",
    "usage_page": "0xFF00",
    "usage": "0x0001",
}
DEFAULT_RECORDING = {"silence_seconds": 60, "noise_seconds": 3}


def read_session(path: Path, *, strict: bool = False) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("配置内容必须为 JSON 对象")
        return data
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        if strict:
            raise ValueError(f"无法读取配置文件 {path}：{exc}") from exc
        return {}


def recording_durations(path: Path) -> dict[str, int]:
    values = read_session(path, strict=True).get("recording", {})
    if not isinstance(values, dict):
        raise ValueError(f"{path} 中的 recording 必须为 JSON 对象")
    result = {}
    for name, default in DEFAULT_RECORDING.items():
        value = values.get(name, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"{path} 中的 recording.{name} 必须为正整数（秒）")
        result[name] = value
    return result


def hid_search_settings(path: Path) -> dict[str, int]:
    values = read_session(path, strict=True).get("hid_device", {})
    if not isinstance(values, dict):
        raise ValueError(f"{path} 中的 hid_device 必须为 JSON 对象")
    result = {}
    for name, default in DEFAULT_HID_DEVICE.items():
        value = values.get(name, default)
        try:
            if isinstance(value, str):
                text = value.strip()
                number = int(text, 16 if text.lower().startswith("0x") else 10)
            elif type(value) is int:
                number = value
            else:
                raise ValueError()
            if not (1 if name in ("vid", "pid") else 0) <= number <= 0xFFFF:
                raise ValueError()
        except ValueError as exc:
            raise ValueError(f"{path} 中的 hid_device.{name} 无效：请填写有效的 16 位整数或 0x 十六进制字符串") from exc
        result[name] = number
    return result
