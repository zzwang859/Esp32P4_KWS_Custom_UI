"""Discover training history and clear generated artifacts in one workspace."""

from dataclasses import dataclass
from pathlib import Path
import re
import shutil


@dataclass(frozen=True)
class TrainingHistory:
    present: bool
    checkpoint: Path | None = None


def _workspace_directory(workspace: Path, name: str) -> Path:
    workspace = workspace.expanduser().resolve()
    root = workspace / name
    if root.is_symlink() or root.is_junction() or root.resolve().parent != workspace:
        raise ValueError(f"{name} 目录必须位于当前工作区内，且不能是链接目录。")
    if root.exists() and not root.is_dir():
        raise ValueError(f"{name} 必须是文件夹。")
    return root


def history_root(workspace: Path) -> Path:
    return _workspace_directory(workspace, "training")


def training_history(workspace: Path) -> TrainingHistory:
    root = history_root(workspace)
    if not root.exists():
        return TrainingHistory(False)
    entries = list(root.iterdir())
    # Generated run names sort chronologically, independent of later log writes.
    runs = sorted((p for p in entries if p.is_dir() and not p.is_symlink() and not p.is_junction()),
                  key=lambda p: p.name, reverse=True)
    for run in runs:
        steps = sorted((p for p in run.glob("step_*.pt") if re.fullmatch(r"step_[0-9]+\.pt", p.name)),
                       key=lambda p: int(p.stem[5:]), reverse=True)
        for checkpoint in [run / "latest.pt", *steps, run / "best.pt"]:
            if checkpoint.is_file() and not checkpoint.is_symlink() and checkpoint.resolve().is_relative_to(root):
                return TrainingHistory(True, checkpoint)
    return TrainingHistory(bool(entries))


def clear_training_history(workspace: Path) -> None:
    """Clear training and ESP32-P4 results after the user chooses restart."""
    # Validate both roots before removing anything from either directory.
    roots = [history_root(workspace), _workspace_directory(workspace, "esp32p4-dl")]
    # Verify absolute targets before any recursive delete. Remove links themselves
    # rather than traversing them, including Windows directory junctions.
    for root in roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_symlink():
                child.unlink()
            elif child.is_junction():
                child.rmdir()
            elif child.is_dir():
                if not child.resolve().is_relative_to(root):
                    raise ValueError(f"拒绝清理工作区外的目录：{child}")
                shutil.rmtree(child)
            else:
                child.unlink()
