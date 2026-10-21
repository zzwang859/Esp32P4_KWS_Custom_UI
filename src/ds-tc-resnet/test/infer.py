#!/usr/bin/env python3
"""Run checkpoint inference on one WAV file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402
from data.audio import MfccExtractor, fit_length, load_wav  # noqa: E402
from data.dataset import load_labels  # noqa: E402
from runtime import build_model, load_checkpoint, predict_features, resolve_checkpoint, select_device  # noqa: E402


def infer(config: dict, wav_path: Path, checkpoint_value: str | Path | None = None) -> torch.Tensor:
    device = select_device(config["test"]["device"])
    labels = load_labels(config)
    model = build_model(config, len(labels), device)
    checkpoint_path = resolve_checkpoint(config, checkpoint_value)
    load_checkpoint(model, checkpoint_path, device)
    sample_count = int(config["data"]["sample_rate"] * config["data"]["clip_duration_ms"] / 1000)
    audio = fit_length(load_wav(wav_path, config["data"]["sample_rate"]), sample_count)
    features = MfccExtractor(config["data"], config["features"]).streaming(audio).to(device)
    logits = predict_features(
        model, features, config["model"]["inference_mode"], config["onnx"]["chunk_frames"])
    probabilities = logits.softmax(dim=1)[0].cpu()
    for index in probabilities.argsort(descending=True)[:3]:
        print(f"{labels[index]:12s} {probabilities[index].item() * 100:7.3f}%")
    return probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    infer(load_config(args.config), args.wav, args.checkpoint)


if __name__ == "__main__":
    main()

