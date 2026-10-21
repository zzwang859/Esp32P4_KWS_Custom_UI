"""Manifest-backed speech command dataset."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset

from config import resolve_path
from data.audio import MfccExtractor, fit_length, load_wav


class SpeechCommandsDataset(Dataset):
    def __init__(self, config: dict, split: str, training: bool = False):
        manifest = resolve_path(config, config["data"]["manifests_dir"]) / f"{split}.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"Missing {manifest}; run data/prepare.py first")
        self.entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        self.project_root = Path(config["_project_root"])
        self.data_config = config["data"]
        self.samples = int(
            self.data_config["sample_rate"] * self.data_config["clip_duration_ms"] / 1000)
        self.cached = bool(self.data_config.get("preprocess_dir"))
        if self.cached and any("feature_path" not in entry for entry in self.entries):
            raise ValueError("Missing MFCC preprocessing; run data/preprocess.py before training")
        self.extractor = None if self.cached else MfccExtractor(self.data_config, config["features"])
        self.feature_arrays = {}
        self.training = training

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        entry = self.entries[index]
        if self.cached:
            path = entry["feature_path"]
            if path not in self.feature_arrays:
                self.feature_arrays[path] = np.load(path, mmap_mode="r", allow_pickle=False)
            values = self.feature_arrays[path][int(entry["feature_index"])]
            return torch.from_numpy(np.array(values, dtype=np.float32, copy=True)), int(entry["label_index"])
        if entry["path"] is None:
            audio = torch.zeros(self.samples)
        else:
            audio = load_wav(self.project_root / entry["path"], self.data_config["sample_rate"])
            audio = fit_length(audio, self.samples, random_crop=self.training)
        with torch.no_grad():
            features = self.extractor(audio).squeeze(0)
        return features, int(entry["label_index"])


def load_labels(config: dict) -> list[str]:
    return resolve_path(config, config["data"]["labels_file"]).read_text(encoding="utf-8").splitlines()

