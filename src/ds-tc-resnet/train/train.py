#!/usr/bin/env python3
"""Train DS-TC-ResNet with step-based checkpointing and TensorBoard logs."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import shutil
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from windows_runtime import preload_msvc
preload_msvc()

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import load_config, resolve_path, with_overrides  # noqa: E402
from data.dataset import SpeechCommandsDataset, load_labels  # noqa: E402
from runtime import build_model, load_checkpoint, select_device  # noqa: E402


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             stop_file: Path | None = None) -> tuple[float, float] | None:
    model.eval()
    loss_total = 0.0
    correct = 0
    count = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for features, targets in loader:
            if stop_file is not None and stop_file.exists():
                return None
            features, targets = features.to(device), targets.to(device)
            logits = model(features)
            loss_total += criterion(logits, targets).item() * targets.numel()
            correct += (logits.argmax(dim=1) == targets).sum().item()
            count += targets.numel()
    return loss_total / max(count, 1), correct / max(count, 1)


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    step: int, best_accuracy: float, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_accuracy": best_accuracy,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
    }, temporary)
    temporary.replace(path)


def train(config: dict) -> Path | None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError("TensorBoard is required; run uv sync --locked in the KWS_Custom_UI project root") from error

    seed = int(config["project"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_config = config["train"]
    device = select_device(train_config["device"])
    print(f"TRAINING_DEVICE={device.type}", flush=True)
    stop_file = Path(train_config["stop_file"]) if train_config.get("stop_file") else None
    labels = load_labels(config)
    training_set = SpeechCommandsDataset(config, "training", training=True)
    validation_set = SpeechCommandsDataset(config, "validation")
    if not len(training_set):
        raise ValueError("No training audio; prepare the workspace recordings first")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        training_set, batch_size=train_config["batch_size"], shuffle=True,
        num_workers=train_config["num_workers"], pin_memory=device.type == "cuda",
        drop_last=False, generator=generator,
    )
    validation_loader = DataLoader(
        validation_set, batch_size=config["test"]["batch_size"], shuffle=False,
        num_workers=train_config["num_workers"], pin_memory=device.type == "cuda",
    )
    model = build_model(config, len(labels), device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )
    run_dir = resolve_path(config, train_config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    best_accuracy = 0.0
    resume = train_config.get("resume")
    if resume:
        resume_path = resolve_path(config, resume)
        checkpoint = load_checkpoint(model, resume_path, device, optimizer)
        start_step = int(checkpoint.get("step", 0))
        best_accuracy = float(checkpoint.get("best_accuracy", 0.0))
        if resume_path.parent != run_dir:
            previous_best = resume_path.parent / "best.pt"
            if previous_best.is_file():
                shutil.copy2(previous_best, run_dir / "best.pt")
            else:
                best_accuracy = 0.0
        print(f"Resumed model and optimizer at step={start_step}; target={train_config['steps']}", flush=True)

    writer = SummaryWriter(log_dir=run_dir / "tensorboard")
    criterion = torch.nn.CrossEntropyLoss()
    iterator = iter(loader)
    started = time.monotonic()
    model.train()
    last_step = start_step
    stopped = False
    for step in range(start_step + 1, int(train_config["steps"]) + 1):
        if stop_file is not None and stop_file.exists():
            stopped = True
            break
        try:
            features, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            features, targets = next(iterator)
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        last_step = step
        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/accuracy", accuracy, step)

        if step % int(train_config["log_interval"]) == 0 or step == start_step + 1:
            elapsed = time.monotonic() - started
            print(f"step={step}/{train_config['steps']} loss={loss.item():.4f} "
                  f"accuracy={accuracy:.4f} elapsed={elapsed:.1f}s", flush=True)

        if step % int(train_config["checkpoint_interval"]) == 0:
            save_checkpoint(run_dir / f"step_{step}.pt", model, optimizer, step, best_accuracy, config)
            save_checkpoint(run_dir / "latest.pt", model, optimizer, step, best_accuracy, config)

        if stop_file is not None and stop_file.exists():
            stopped = True
            break

        if step % int(train_config["eval_interval"]) == 0 or step == int(train_config["steps"]):
            if not len(validation_set):
                print("Validation skipped: no held-out recordings; best.pt uses the latest step.", flush=True)
                save_checkpoint(run_dir / "best.pt", model, optimizer, step, best_accuracy, config)
                continue
            validation = evaluate(model, validation_loader, device, stop_file)
            if validation is None:
                stopped = True
                break
            validation_loss, validation_accuracy = validation
            writer.add_scalar("validation/loss", validation_loss, step)
            writer.add_scalar("validation/accuracy", validation_accuracy, step)
            print(f"validation step={step} loss={validation_loss:.4f} "
                  f"accuracy={validation_accuracy:.4f}", flush=True)
            if validation_accuracy >= best_accuracy:
                best_accuracy = validation_accuracy
                save_checkpoint(run_dir / "best.pt", model, optimizer, step, best_accuracy, config)
            model.train()

    save_checkpoint(run_dir / "latest.pt", model, optimizer, last_step, best_accuracy, config)
    writer.close()
    if stopped:
        print(f"Training stopped on {device} at step={last_step}; checkpoint: {run_dir / 'latest.pt'}", flush=True)
        return None
    print(f"Training complete on {device}; checkpoints: {run_dir}")
    return run_dir / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = with_overrides(
        load_config(args.config), train__steps=args.steps, train__device=args.device,
        train__resume=str(args.resume) if args.resume else None,
        train__run_dir=str(args.run_dir) if args.run_dir else None,
    )
    train(config)


if __name__ == "__main__":
    main()

