#!/usr/bin/env python3
"""Export a checkpoint as the single current ONNX model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from windows_runtime import preload_msvc
preload_msvc()

import onnx
import torch
from torch import nn
from torch.nn import functional as F

from config import load_config, resolve_path  # noqa: E402
from data.dataset import load_labels  # noqa: E402
from runtime import build_model, load_checkpoint, resolve_checkpoint  # noqa: E402
from device_contract import PROFILE, FEATURE_INPUT, LOGITS_OUTPUT, STATE_INPUTS, STATE_OUTPUTS, frontend_sha256


class ExternalStreamExport(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # Freeze inference BN into pointwise projections before ONNX export.
        # Exporting shared BN constants lets onnxsim's BN fusion confuse
        # different channel counts in residual branches.
        from torch.nn.utils.fusion import fuse_conv_bn_eval
        self.pointwise = nn.ModuleList([fuse_conv_bn_eval(block.layers[0]["pointwise"], block.layers[0]["norm"])
                                        for block in model.blocks])
        self.projections = nn.ModuleDict({str(index): fuse_conv_bn_eval(block.residual[0], block.residual[1])
                                          for index, block in enumerate(model.blocks) if block.residual is not None})

    def forward(self, features: torch.Tensor, *values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Use the reference graph's four-dimensional convolution/window layout.
        # This avoids exporting Conv1D/Gather/ReduceMean variants to firmware.
        x = features.unsqueeze(2)
        states = []
        offset = 0
        for index, block in enumerate(self.model.blocks):
            residual = x.permute(0, 3, 1, 2)
            unused = x
            for layer in block.layers:
                if block.context:
                    window = torch.cat((values[offset][:, 1:, :, :], x), dim=1)
                    states.append(window)
                    offset += 1
                else:
                    window = x
                y = F.conv2d(window.permute(0, 3, 1, 2), layer["temporal"].weight.unsqueeze(-1),
                             dilation=(block.layers[0]["temporal"].dilation[0], 1), groups=layer["temporal"].groups)
                pointwise = self.pointwise[index]
                y = F.conv2d(y, pointwise.weight.unsqueeze(-1), pointwise.bias)
                if block.residual is not None:
                    projection = self.projections[str(index)]
                    y = y + F.conv2d(residual, projection.weight.unsqueeze(-1), projection.bias)
                x = F.relu(y).permute(0, 2, 3, 1)
        window = torch.cat((values[-1][:, 1:, :, :], x), dim=1)
        pooled = F.adaptive_avg_pool2d(window.permute(0, 3, 1, 2), (1, 1)).squeeze(3).squeeze(2)
        logits = self.model.classifier(pooled)
        return (logits, *states, unused, window)


def _feature_frames(config: dict) -> int:
    sample_rate = int(config["data"]["sample_rate"])
    samples = int(sample_rate * config["data"]["clip_duration_ms"] / 1000)
    window = int(sample_rate * config["features"]["window_size_ms"] / 1000)
    stride = int(sample_rate * config["features"]["window_stride_ms"] / 1000)
    return 1 + (samples - window) // stride


def export(config: dict, checkpoint_value: str | Path | None = None,
           output_value: str | Path | None = None) -> Path:
    labels = load_labels(config)
    model = build_model(config, len(labels), torch.device("cpu"))
    checkpoint_path = resolve_checkpoint(config, checkpoint_value)
    checkpoint = load_checkpoint(model, checkpoint_path, torch.device("cpu"))
    model.eval()
    output = resolve_path(config, output_value or config["onnx"]["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = config["onnx"]["mode"]
    mfcc_bins = int(config["features"]["mfcc_bins"])
    input_names: list[str]
    output_names: list[str]
    dynamic_axes: dict[str, dict[int, str]]
    wrapper = ExternalStreamExport(model)
    states = model.initial_stream_state(1, "cpu")
    inputs = (torch.zeros(1, 1, mfcc_bins), *states)
    input_names = [FEATURE_INPUT, *STATE_INPUTS]
    output_names = [LOGITS_OUTPUT, *[f"onnx_next_state_{index}:0" for index in range(1, 8)]]
    dynamic_axes = {}
    state_pairs = list(zip(STATE_INPUTS, STATE_OUTPUTS))

    torch.onnx.export(
        wrapper, inputs, output, export_params=True, opset_version=int(config["onnx"]["opset"]),
        do_constant_folding=True, input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, dynamo=False,
    )
    model_proto = onnx.load(output)
    metadata = {
        "model_name": "ds_tc_resnet",
        "mode": mode,
        "source_checkpoint": str(checkpoint_path),
        "checkpoint_step": str(checkpoint.get("step", "unknown")),
        "labels": json.dumps(labels),
        "sample_rate": str(config["data"]["sample_rate"]),
        "clip_duration_ms": str(config["data"]["clip_duration_ms"]),
        "chunk_frames": str(config["onnx"]["chunk_frames"]),
        "external_state_pairs": json.dumps(state_pairs),
        "frontend_profile": PROFILE,
        "frontend_asset_sha256": frontend_sha256(),
        "feature_input_name": FEATURE_INPUT,
        "logits_output_name": LOGITS_OUTPUT,
        "features_layout": "NTC",
    }
    onnx.helper.set_model_props(model_proto, metadata)
    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, output)
    print(f"Exported {mode} model: {output}")
    print(f"Inputs: {input_names}")
    print(f"Outputs: {output_names}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    export(load_config(args.config), args.checkpoint, args.output)


if __name__ == "__main__":
    main()

