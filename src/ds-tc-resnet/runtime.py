"""Runtime helpers for device selection, checkpoints, and inference modes."""

from __future__ import annotations

from pathlib import Path

import torch

from config import resolve_path
from model import DSTCResNet, InternalStreamingRunner
from device_contract import validate_contract


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def build_model(config: dict, label_count: int, device: torch.device) -> DSTCResNet:
    return DSTCResNet(config, label_count).to(device)


def load_checkpoint(
    model: DSTCResNet, path: str | Path, device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_contract(checkpoint.get("config", {}))
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def resolve_checkpoint(config: dict, value: str | Path | None) -> Path:
    configured = value or config["test"]["checkpoint"]
    return resolve_path(config, configured)


@torch.no_grad()
def predict_features(
    model: DSTCResNet, features: torch.Tensor, mode: str, chunk_frames: int = 1,
) -> torch.Tensor:
    model.eval()
    if mode == "non_stream":
        return model(features)
    if chunk_frames != 1:
        raise ValueError("Fixed ESP32-P4 streaming requires chunk_frames=1")
    if mode == "stream_internal":
        runner = InternalStreamingRunner(model, features.shape[0])
        logits = None
        for start in range(0, features.shape[2], chunk_frames):
            logits = runner.process(features[:, :, start:start + chunk_frames])
        return logits
    if mode == "stream_external":
        states = model.initial_stream_state(features.shape[0], features.device)
        logits = None
        for start in range(0, features.shape[2], chunk_frames):
            logits, states = model.stream_step(features[:, :, start:start + chunk_frames], states)
        return logits
    raise ValueError(f"Unknown inference mode: {mode}")

