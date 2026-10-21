"""Read the UI workspace's explicit directory/text mapping without copying audio."""

from __future__ import annotations

import json
from pathlib import Path
import re


def workspace_classes(root: Path) -> list[dict]:
    root = root.resolve()
    data = json.loads((root / "workspace.json").read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("rows"), list):
        raise ValueError("Unsupported workspace.json format")
    classes = [
        {"index": 0, "label": "silence", "directory": str(root / "silence")},
        {"index": 1, "label": "unknown", "directory": str(root / "unknown")},
    ]
    used_directories, used_labels = set(), {"silence", "unknown"}
    for index, row in enumerate(data["rows"], start=2):
        directory, text = row.get("directory"), row.get("text")
        if not isinstance(directory, str) or not re.fullmatch(r"keyword[1-9][0-9]*", directory):
            raise ValueError("Open the workspace in KWS Studio first to migrate its directory names")
        if directory in used_directories or not isinstance(text, str):
            raise ValueError("Invalid or duplicate workspace directory/text mapping")
        label = text.strip() or directory
        if label in used_labels or any(c in label for c in "\r\n"):
            raise ValueError(f"Keyword labels must be unique single-line text: {label!r}")
        classes.append({"index": index, "label": label,
                        "directory": str(root / "keywords" / directory), "row_id": row["id"], "text": text})
        used_directories.add(directory)
        used_labels.add(label)
    if len(classes) == 2:
        raise ValueError("The workspace must contain at least one keyword")
    for item in classes:
        directory = Path(item["directory"])
        if not directory.resolve().is_relative_to(root) or not directory.is_dir():
            raise ValueError(f"Missing or unsafe class directory: {directory}")
        if not any(p.is_file() and p.suffix.lower() == ".wav" for p in directory.rglob("*")):
            raise ValueError(f"No WAV recordings for class {item['index']} ({item['label']}): {directory}")
    return classes


def prepare_workspace(config: dict) -> dict[str, int]:
    data = config["data"]
    classes = data["classes"]
    if [item["index"] for item in classes] != list(range(len(classes))):
        raise ValueError("Class indices must be contiguous from zero")
    if [item["label"] for item in classes[:2]] != ["silence", "unknown"]:
        raise ValueError("Class 0 must be silence and class 1 must be unknown")
    grouped = {name: [] for name in ("training", "validation", "testing")}
    root = Path(data["root"]).resolve()
    for item in classes:
        directory = Path(item["directory"])
        found = False
        for wav in sorted(directory.rglob("*")):
            if not wav.is_file() or wav.suffix.lower() != ".wav":
                continue
            if not wav.resolve().is_relative_to(root):
                raise ValueError(f"Audio must stay inside the workspace: {wav}")
            grouped["training"].append({"path": str(wav.resolve()), "label": item["label"],
                                        "label_index": item["index"]})
            found = True
        if not found:
            raise ValueError(f"No WAV recordings for {item['label']}")
    destination = Path(data["manifests_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    for name, entries in grouped.items():
        (destination / f"{name}.jsonl").write_text(
            "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries), encoding="utf-8")
    Path(data["labels_file"]).write_text("\n".join(item["label"] for item in classes) + "\n", encoding="utf-8")
    (destination.parent / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    return {name: len(entries) for name, entries in grouped.items()}
