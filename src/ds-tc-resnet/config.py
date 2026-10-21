"""Configuration helpers shared by the standalone project."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from device_contract import validate_contract


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    config_path = config_path.expanduser().resolve()
    with config_path.open(encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    _validate(config)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path.resolve()


def with_overrides(config: dict[str, Any], **values: Any) -> dict[str, Any]:
    result = deepcopy(config)
    for dotted_key, value in values.items():
        if value is None:
            continue
        target = result
        parts = dotted_key.split("__")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    _validate(result)
    return result


def _validate(config: dict[str, Any]) -> None:
    validate_contract(config)
    model = config["model"]
    fields = (
        "filters", "repeats", "separable", "residual", "padding",
        "kernel_sizes", "strides", "dilations", "pools",
    )
    lengths = {field: len(model[field]) for field in fields}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Model list lengths differ: {lengths}")
    modes = {"non_stream", "stream_internal", "stream_external"}
    if model["inference_mode"] not in modes:
        raise ValueError(f"model.inference_mode must be one of {sorted(modes)}")
    if config.get("onnx", {}).get("mode") not in {"non_stream", "stream_external"}:
        raise ValueError("onnx.mode must be non_stream or stream_external")
    streaming_enabled = (
        model["inference_mode"] != "non_stream"
        or config.get("onnx", {}).get("mode") == "stream_external"
    )
    if streaming_enabled:
        if any(value != "causal" for value in model["padding"]):
            raise ValueError("Streaming modes require causal padding in every block")
        if any(value != 1 for value in model["strides"] + model["pools"]):
            raise ValueError("Streaming modes require every stride and pool value to be 1")
