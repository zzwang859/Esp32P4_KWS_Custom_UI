#!/usr/bin/env python3
"""Run prepare -> train -> checkpoint test -> ONNX export -> ONNX test."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from config import load_config, resolve_path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_command(arguments: list[str]) -> None:
    print("\n$ " + " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--steps", type=int, help="Override train.steps")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--wav", type=Path, help="WAV used for both inference smoke tests")
    args = parser.parse_args()

    config = load_config(args.config)
    config_path = Path(config["_config_path"])
    run_dir = args.run_dir or Path(config["train"]["run_dir"])
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    checkpoint = run_dir.resolve() / "best.pt"
    wav_path = args.wav
    if wav_path is None:
        data_root = resolve_path(config, config["data"]["root"])
        wav_path = next((data_root / config["data"]["wanted_words"][0]).glob("*.wav"))
    wav_path = wav_path.resolve()

    base = [sys.executable]
    run_command(base + ["data/prepare.py", "--config", str(config_path)])
    train_command = base + ["train/train.py", "--config", str(config_path), "--run-dir", str(run_dir)]
    if args.steps is not None:
        train_command += ["--steps", str(args.steps)]
    if args.device:
        train_command += ["--device", args.device]
    if args.resume:
        train_command += ["--resume", str(args.resume.resolve())]
    run_command(train_command)
    run_command(base + ["test/evaluate.py", "--config", str(config_path),
                        "--checkpoint", str(checkpoint)])
    run_command(base + ["test/infer.py", str(wav_path), "--config", str(config_path),
                        "--checkpoint", str(checkpoint)])
    run_command(base + ["onnx/export.py", "--config", str(config_path),
                        "--checkpoint", str(checkpoint)])
    run_command(base + ["onnx/run/infer.py", str(wav_path), "--config", str(config_path)])
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
