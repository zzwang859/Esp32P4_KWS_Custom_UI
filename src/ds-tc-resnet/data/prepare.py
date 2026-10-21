#!/usr/bin/env python3
"""Create stable train/validation/test manifests from class WAV folders."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config, resolve_path  # noqa: E402


def split_for(path: Path, validation: float, testing: float) -> str:
    stable_name = re.sub(r"_nohash_.*$", "", path.name)
    value = int(hashlib.sha1(stable_name.encode()).hexdigest(), 16) % (2**27)
    percentage = value * (100.0 / (2**27 - 1))
    if percentage < validation:
        return "validation"
    if percentage < validation + testing:
        return "testing"
    return "training"


def prepare(config: dict) -> dict[str, int]:
    data = config["data"]
    if "classes" in data:
        from data.workspace import prepare_workspace
        return prepare_workspace(config)
    root = resolve_path(config, data["root"])
    wanted = list(data["wanted_words"])
    labels = ["_silence_", "_unknown_"] + wanted
    label_index = {label: index for index, label in enumerate(labels)}
    grouped: dict[str, list[dict]] = {name: [] for name in ("training", "validation", "testing")}
    unknown: dict[str, list[dict]] = {name: [] for name in grouped}

    for wav_path in sorted(root.glob("*/*.wav")):
        word = wav_path.parent.name.lower()
        split = split_for(wav_path, data["validation_percentage"], data["testing_percentage"])
        relative = wav_path.relative_to(PROJECT_ROOT).as_posix()
        entry = {"path": relative, "label": word, "label_index": label_index.get(word, 1)}
        (grouped if word in wanted else unknown)[split].append(entry)

    missing = [word for word in wanted if not (root / word).is_dir()]
    if missing:
        raise ValueError(f"Missing wanted-word directories: {missing}")

    rng = random.Random(config["project"]["seed"])
    for split, entries in grouped.items():
        rng.shuffle(unknown[split])
        known_count = len(entries)
        unknown_count = round(known_count * data["unknown_percentage"] / 100)
        silence_count = round(known_count * data["silence_percentage"] / 100)
        entries.extend(unknown[split][:unknown_count])
        entries.extend(
            {"path": None, "label": "_silence_", "label_index": 0}
            for _ in range(silence_count)
        )
        rng.shuffle(entries)

    manifest_dir = resolve_path(config, data["manifests_dir"])
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split, entries in grouped.items():
        content = "".join(json.dumps(entry, ensure_ascii=True) + "\n" for entry in entries)
        (manifest_dir / f"{split}.jsonl").write_text(content, encoding="utf-8")
    labels_path = resolve_path(config, data["labels_file"])
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    return {split: len(entries) for split, entries in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    args = parser.parse_args()
    counts = prepare(load_config(args.config))
    print("Prepared manifests: " + ", ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()

