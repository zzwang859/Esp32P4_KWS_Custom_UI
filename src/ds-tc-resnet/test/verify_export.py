"""Check checkpoint/ONNX label order and numerical agreement on real workspace audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from windows_runtime import preload_msvc
preload_msvc()

import numpy as np
import torch

from config import load_config, resolve_path
from data.audio import MfccExtractor, fit_length, load_wav
from data.dataset import load_labels
from data.preprocess import load_feature
from runtime import build_model, load_checkpoint, predict_features


def verify(config: dict) -> dict:
    labels = load_labels(config)
    classes = config["data"]["classes"]
    assert labels == [item["label"] for item in classes]
    assert labels[:2] == ["silence", "unknown"]
    model = build_model(config, len(labels), torch.device("cpu"))
    checkpoint = load_checkpoint(model, resolve_path(config, config["test"]["checkpoint"]), torch.device("cpu"))
    latest = torch.load(resolve_path(config, config["train"]["run_dir"]) / "latest.pt", map_location="cpu", weights_only=False)
    assert latest["step"] == config["train"]["steps"]
    namespace = runpy.run_path(str(PROJECT_ROOT / "onnx" / "run" / "infer.py"))
    onnx_model = namespace["OnnxKWS"](resolve_path(config, config["onnx"]["output"]))
    assert json.loads(onnx_model.metadata["labels"]) == labels
    assert int(onnx_model.metadata["checkpoint_step"]) == checkpoint["step"]
    entries = [json.loads(line) for line in (Path(config["data"]["manifests_dir"]) / "training.jsonl").read_text(encoding="utf-8").splitlines()]
    entry = next(entry for entry in entries if entry["label_index"] == 2)
    wav = Path(entry["path"])
    samples = int(config["data"]["sample_rate"] * config["data"]["clip_duration_ms"] / 1000)
    features = (torch.from_numpy(load_feature(entry)).unsqueeze(0) if "feature_path" in entry else
                MfccExtractor(config["data"], config["features"])(fit_length(load_wav(wav, config["data"]["sample_rate"]), samples)))
    with torch.no_grad():
        expected = predict_features(model, features, "non_stream").numpy()
    actual = onnx_model.predict(features.numpy())
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    result = {"steps": config["train"]["steps"], "checkpoint_step": checkpoint["step"], "latest_step": latest["step"],
              "resume_step": config["train"].get("resume_step", 0),
              "additional_steps": config["train"].get("additional_steps", config["train"]["steps"]),
              "labels": labels, "wav": str(wav), "onnx_max_absolute_error": float(np.abs(actual - expected).max()),
              "onnx": config["onnx"]["output"], "success": True}
    (Path(config["train"]["run_dir"]) / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Verified checkpoint/ONNX: " + json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    verify(load_config(parser.parse_args().config))
