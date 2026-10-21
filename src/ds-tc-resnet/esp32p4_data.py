"""Prepare workspace MFCC clips for the separate ESP-PPQ environment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from windows_runtime import preload_msvc
preload_msvc()

import numpy as np
import torch

from config import load_config
from data.audio import MfccExtractor, fit_length, load_wav
from data.preprocess import load_feature
from device_contract import PROFILE, FRONTEND_ASSET


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def select_entries(entries: list[dict], classes: list[dict], limit: int) -> list[dict]:
    """Bound each class equally; never traverse archived or unconfigured folders."""
    selected = []
    for item in classes:
        candidates = [entry for entry in entries if entry["label_index"] == item["index"]]
        unique = {}
        for entry in candidates:
            identity = entry["path"] or entry.get("feature_path")
            unique.setdefault(identity, entry)
        selected.extend(unique[path] for path in sorted(unique)[:limit])
    return selected


def prepare(config_path: Path, output: Path, stop_file: Path) -> None:
    def check_stop():
        if stop_file.exists():
            raise InterruptedError("ESP32-P4 conversion stopped by user")

    check_stop()
    torch.set_num_threads(4)
    config = load_config(config_path)
    source = Path(config["onnx"]["output"])
    classes = config["data"]["classes"]
    if [item["index"] for item in classes] != list(range(len(classes))):
        raise ValueError("Non-contiguous class mapping")
    if [item["label"] for item in classes[:2]] != ["silence", "unknown"]:
        raise ValueError("ESP32-P4 requires index 0 = silence, index 1 = unknown")
    models, work = output / "models", output / "work"
    models.mkdir(parents=True, exist_ok=True)
    work.mkdir(exist_ok=True)
    shutil.copy2(source, models / "ds_tc_resnet.onnx")
    shutil.copy2(config_path, work / "config.yaml")
    write_json(models / "classes.json", classes)
    (models / "labels.txt").write_text("\n".join(item["label"] for item in classes) + "\n", encoding="utf-8")
    manifests = Path(config["data"]["manifests_dir"])

    def read_entries(split):
        path = manifests / f"{split}.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []

    calibration = select_entries(read_entries("training"), classes, 4)
    if {entry["label_index"] for entry in calibration} != set(range(len(classes))):
        raise ValueError("Calibration audio is required for every configured class")
    held_out = read_entries("validation") + read_entries("testing")
    validation = []
    fallback = []
    for item in classes:
        candidates = select_entries(held_out, [item], 2)
        if not candidates:
            candidates = select_entries(read_entries("training"), [item], 2)
            fallback.append(item["index"])
        validation.extend(candidates)
    extractor = MfccExtractor(config["data"], config["features"]).eval()
    samples = int(config["data"]["sample_rate"] * config["data"]["clip_duration_ms"] / 1000)
    arrays, clips = {}, []
    # Reuse features when a small workspace has no independent holdout audio.
    cache = {}
    with torch.no_grad():
        for split, entries in [("calibration", calibration), ("validation", validation)]:
            for entry in entries:
                check_stop()
                wav = entry["path"]
                identity = (entry["feature_path"], entry["feature_index"]) if "feature_path" in entry else wav
                if identity not in cache:
                    features = (load_feature(entry)[None, ...] if "feature_path" in entry else
                                extractor(fit_length(load_wav(wav, int(config["data"]["sample_rate"])), samples)).numpy())
                    if not np.isfinite(features).all():
                        raise ValueError(f"Non-finite MFCC: {wav}")
                    cache[identity] = features.astype(np.float32)
                key = f"clip_{len(clips)}"
                arrays[key] = cache[identity]
                clips.append(dict(key=key, split=split, path=wav or f"generated:white_noise:{entry.get('feature_index')}",
                                  label_index=entry["label_index"], feature_path=entry.get("feature_path"),
                                  feature_index=entry.get("feature_index")))
    np.savez_compressed(work / "features.npz", **arrays)
    frontend = {
        "input": "MFCC features, NOT raw PCM", "data": config["data"], "features": config["features"],
        "frontend_profile": PROFILE, "frontend_asset_sha256": extractor.asset_sha256,
        "n_fft": 512, "window_samples": 480, "hop_samples": 160, "magnitude_bins": 224,
        "right_padding": 0, "center": False, "power": 1.0,
        "log": "natural", "amin": 1e-12, "normalization": "fixed reference per-coefficient scale",
    }
    write_json(models / "frontend.json", frontend)
    shutil.copy2(FRONTEND_ASSET, models / "frontend.npz")
    metadata = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    training_run=str(config_path.parent), labels=[item["label"] for item in classes],
                    mode=config["onnx"]["mode"], chunk_frames=int(config["onnx"]["chunk_frames"]),
                    clips=clips, validation_training_fallback_classes=fallback)
    check_stop()
    write_json(work / "dataset.json", metadata)
    print(f"ESP32-P4: prepared {len(calibration)} calibration and {len(validation)} validation clips: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.config, args.output, args.stop_file)
