#!/usr/bin/env python3
"""Evaluate a checkpoint on a prepared dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402
from data.dataset import SpeechCommandsDataset, load_labels  # noqa: E402
from runtime import build_model, load_checkpoint, predict_features, resolve_checkpoint, select_device  # noqa: E402


def evaluate(config: dict, checkpoint_value: str | Path | None = None) -> float:
    test_config = config["test"]
    device = select_device(test_config["device"])
    labels = load_labels(config)
    model = build_model(config, len(labels), device)
    checkpoint_path = resolve_checkpoint(config, checkpoint_value)
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    dataset = SpeechCommandsDataset(config, test_config["split"])
    loader = DataLoader(dataset, batch_size=test_config["batch_size"], shuffle=False,
                        num_workers=config["train"]["num_workers"])
    correct = 0
    count = 0
    mode = config["model"]["inference_mode"]
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        logits = predict_features(model, features, mode, config["onnx"]["chunk_frames"])
        correct += (logits.argmax(dim=1) == targets).sum().item()
        count += targets.numel()
    accuracy = correct / max(count, 1)
    print(f"checkpoint: {checkpoint_path}")
    print(f"step: {checkpoint.get('step', 'unknown')}")
    print(f"mode: {mode}, split: {test_config['split']}, samples: {count}, accuracy: {accuracy:.4f}")
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    evaluate(load_config(args.config), args.checkpoint)


if __name__ == "__main__":
    main()

