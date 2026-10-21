#!/usr/bin/env python3
"""Train and export a UI workspace, with its own immutable run configuration."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser

import yaml

from config import load_config, PROJECT_ROOT
from data.prepare import prepare
from data.workspace import workspace_classes
from device_contract import validate_contract


class TrainingStopped(Exception):
    pass


def join_owner_job(name: str) -> None:
    """Join before launching children; only the UI retains a job handle."""
    if os.name != "nt":
        raise RuntimeError("Windows job ownership is only supported on Windows")
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.OpenJobObjectW.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    job = kernel.OpenJobObjectW(1, False, name)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(job)


def check_stop(run_dir: Path) -> None:
    if (run_dir / "stop.requested").exists():
        raise TrainingStopped("Training stopped by user")


def load_resume_checkpoint(checkpoint: Path) -> dict:
    from windows_runtime import preload_msvc
    preload_msvc()
    import torch
    return torch.load(checkpoint, map_location="cpu", weights_only=False)


def class_signature(classes: list[dict]) -> list[tuple]:
    return [(item["index"], item["label"], item.get("row_id"), Path(item["directory"]).name) for item in classes]


def configure(workspace: Path, run_dir: Path, steps: int | None = None, resume: Path | None = None) -> dict:
    run_dir = run_dir.expanduser().resolve()
    config = load_config()
    requested_steps = config["train"]["steps"] if steps is None else steps
    if type(requested_steps) is not int or requested_steps < 1:
        raise ValueError("Training steps must be positive")
    classes = workspace_classes(workspace)
    resume_step = 0
    if resume is not None:
        checkpoint = load_resume_checkpoint(resume)
        prior_config = checkpoint["config"]
        validate_contract(prior_config)
        if class_signature(prior_config["data"].get("classes", [])) != class_signature(classes):
            raise ValueError("历史模型的类别顺序、文本或目录与当前工作区不一致，请重新开始训练。")
        if not checkpoint.get("optimizer"):
            raise ValueError("历史权重缺少优化器状态，无法接续训练。")
        resume_step = checkpoint["step"]
        if type(resume_step) is not int or resume_step < 0:
            raise ValueError("历史训练步数无效。")
        metadata = {key: value for key, value in config.items() if key.startswith("_")}
        config = deepcopy(prior_config)
        config.update(metadata)
    config["data"].update(root=str(workspace.resolve()), classes=classes,
                          manifests_dir=str(run_dir / "manifests"), labels_file=str(run_dir / "labels.txt"),
                          preprocess_dir=str(run_dir / "preprocess"))
    config["train"].update(run_dir=str(run_dir), num_workers=0, stop_file=str(run_dir / "stop.requested"),
                           resume=str(resume.resolve()) if resume is not None else None,
                           resume_step=resume_step, additional_steps=requested_steps,
                           steps=resume_step + requested_steps)
    config["test"]["checkpoint"] = str(run_dir / "best.pt")
    config["onnx"]["output"] = str(run_dir / "onnx" / "ds_tc_resnet.onnx")
    config["train"]["log_interval"] = min(config["train"]["log_interval"], requested_steps)
    config["train"]["eval_interval"] = min(config["train"]["eval_interval"], requested_steps)
    return config


def start_tensorboard(run_dir: Path, open_browser: bool = True) -> str:
    (run_dir / "tensorboard").mkdir(exist_ok=True)
    port = 6006
    while True:
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
                break
            except OSError:
                port += 1
                if port > 6106:
                    raise RuntimeError("No free local TensorBoard port")
    url = f"http://127.0.0.1:{port}/"
    with (run_dir / "tensorboard-server.log").open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main", "--logdir", str(run_dir / "tensorboard"),
             "--host", "127.0.0.1", "--port", str(port)], stdout=output, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        for _ in range(240):
            check_stop(run_dir)
            if process.poll() is not None:
                raise RuntimeError("TensorBoard stopped; see tensorboard-server.log")
            try:
                with urllib.request.urlopen(url, timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.25)
        else:
            raise RuntimeError("TensorBoard did not become ready")
        check_stop(run_dir)
        opened = webbrowser.open(url, new=2) if open_browser else False
    except Exception:
        process.terminate()
        process.wait(timeout=10)
        raise
    (run_dir / "tensorboard.json").write_text(
        json.dumps({"url": url, "pid": process.pid, "browser_opened": opened}, indent=2), encoding="utf-8")
    print(f"TensorBoard: {url} (browser_opened={opened})", flush=True)
    return url


def run_stage(command: list[str], run_dir: Path, environment: dict, *,
              python: Path | None = None, log_path: Path | None = None) -> None:
    check_stop(run_dir)
    print("Running: " + " ".join(command), flush=True)
    output = log_path.open("w", encoding="utf-8") if log_path else None
    try:
        child = subprocess.Popen([str(python or sys.executable), "-u", *command], cwd=PROJECT_ROOT,
                                 env=environment, stdout=output, stderr=subprocess.STDOUT if output else None)
        stop_started = None
        while child.poll() is None:
            if (run_dir / "stop.requested").exists():
                if stop_started is None:
                    stop_started = time.monotonic()
                # Save between batches; bound native code and conversion waits.
                if time.monotonic() - stop_started >= 5:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    break
            time.sleep(0.1)
    finally:
        if output:
            output.close()
    check_stop(run_dir)
    if child.returncode:
        if log_path:
            print(f"Stage failed; log: {log_path}\n" + log_path.read_text(encoding="utf-8", errors="replace")[-2000:], flush=True)
        raise subprocess.CalledProcessError(child.returncode, command)


def conversion_runtime() -> tuple[Path, Path]:
    toolchain = PROJECT_ROOT.parent / "espdl_toolchain"
    python = Path(sys.executable)
    script = toolchain / "esp32p4-dl/scripts/convert_workspace.py"
    if not python.is_file():
        raise ValueError(f"Training Python environment not found: {python}; run uv sync in the project root")
    if not script.is_file():
        raise ValueError(f"ESP32-P4 converter not found: {script}")
    return python.resolve(), script.resolve()


def publish_conversion(bundle: Path, run_dir: Path) -> dict:
    """Only complete, verified models can replace the workspace latest pointer."""
    check_stop(run_dir)
    dataset = json.loads((bundle / "work/dataset.json").read_text(encoding="utf-8"))
    source_hash = hashlib.sha256((bundle / "models/ds_tc_resnet.onnx").read_bytes()).hexdigest()
    if source_hash != dataset["source_sha256"]:
        raise ValueError("ESP32-P4 source provenance mismatch")
    models = {}
    for scheme in ("w8a8", "w16a16"):
        check_stop(run_dir)
        folder = bundle / "output" / scheme
        model = folder / f"ds_tc_resnet_backend_{scheme}_esp32p4.espdl"
        report = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        if (not model.stat().st_size or report["model_sha256"] != digest
                or report["source_sha256"] != source_hash or report["target"] != "esp32p4"):
            raise ValueError(f"ESP32-P4 model provenance mismatch: {scheme}")
        for suffix in (".info", ".json"):
            if not model.with_suffix(suffix).is_file():
                raise ValueError(f"Missing ESP32-P4 model metadata: {model.with_suffix(suffix)}")
        shutil.copy2(model, bundle / "output" / model.name)
        models[scheme] = dict(path=str(model.relative_to(bundle.parent)), sha256=digest,
                              bytes=model.stat().st_size, final_top1_agreement=report["final_top1_agreement"],
                              validation_clips=report["clips"])
    result = dict(target="esp32p4", selected="w16a16", bundle=bundle.name,
                  training_run=str(run_dir), source_onnx=dataset["source"], source_sha256=source_hash,
                  labels=dataset["labels"], models=models, model=models["w16a16"]["path"],
                  io_mapping=str((bundle / "models/io_mapping.json").relative_to(bundle.parent)),
                  hardware_tested=False, success=True)
    report = ("# ESP32-P4 conversion\n\n"
              f"Training run: `{run_dir}`\n\nSource ONNX SHA-256: `{source_hash}`\n\n"
              "Target: ESP32-P4. Selected precision: **w16a16**. Both INT8 and INT16 are exported.\n\n"
              "Inputs are MFCC features plus external stream state, not raw PCM. Use models/io_mapping.json "
              "for state feedback, models/frontend.json and frontend.npz for the exact training frontend. "
              "ESP-DL tensor layout and embedded test vectors are recorded in each .info file.\n\n"
              "Calibration uses the current training manifest's preprocessed MFCC samples. Numerical comparisons and any reuse of "
              "training audio for validation are recorded in output/w*/comparison.json. These are conversion "
              "checks, not a measure of generalization. No firmware build or hardware test is performed.\n")
    (bundle / "REPORT.md").write_text(report, encoding="utf-8")
    (bundle / "conversion.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    check_stop(run_dir)
    temporary = bundle.parent / "latest.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    check_stop(run_dir)
    temporary.replace(bundle.parent / "latest.json")
    return result


def convert_esp32p4(workspace: Path, run_dir: Path, config_path: Path, environment: dict) -> Path:
    check_stop(run_dir)
    python, script = conversion_runtime()
    root = workspace.expanduser().resolve() / "esp32p4-dl"
    if root.is_symlink() or root.resolve() != root:
        raise ValueError("ESP32-P4 output must be a real directory inside the current workspace")
    root.mkdir(exist_ok=True)
    bundle = root / run_dir.name
    bundle.mkdir(exist_ok=False)
    stop = str(run_dir / "stop.requested")
    (bundle / "output").mkdir()
    print(f"ESP32-P4: preparing workspace conversion: {bundle}", flush=True)
    run_stage(["esp32p4_data.py", "--config", str(config_path), "--output", str(bundle), "--stop-file", stop],
              run_dir, environment, log_path=bundle / "output/prepare_data.log")
    base = [str(script), "--bundle", str(bundle), "--stop-file", stop]
    run_stage(base, run_dir, environment, python=python, log_path=bundle / "output/prepare_model.log")
    for bits in (8, 16):
        print(f"ESP32-P4: converting w{bits}a{bits}; log: {bundle / 'output' / f'conversion_w{bits}a{bits}.log'}", flush=True)
        run_stage([*base, "--bits", str(bits)], run_dir, environment, python=python,
                  log_path=bundle / "output" / f"conversion_w{bits}a{bits}.log")
    publish_conversion(bundle, run_dir)
    print(f"ESP32-P4 conversion completed: {bundle}", flush=True)
    return bundle


def run(workspace: Path, run_dir: Path, steps: int | None = None, open_browser: bool = True,
        resume: Path | None = None) -> Path:
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    check_stop(run_dir)
    config = configure(workspace, run_dir, steps, resume)
    counts = prepare(config)
    check_stop(run_dir)
    conversion_runtime()
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump({k: v for k, v in config.items() if not k.startswith("_")},
                                         allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Prepared workspace: {counts}", flush=True)
    if resume is not None:
        print(f"Resume: {resume} | step {config['train']['resume_step']} -> {config['train']['steps']}", flush=True)
    for item in config["data"]["classes"]:
        print(f"class {item['index']}: {item['label']} <- {item['directory']}", flush=True)
    url = start_tensorboard(run_dir, open_browser)
    environment = os.environ.copy()
    environment.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    environment.setdefault("OMP_NUM_THREADS", "4")
    environment.setdefault("MKL_NUM_THREADS", "4")
    commands = [
        ["data/preprocess.py", "--config", str(config_path), "--force"],
        ["train/train.py", "--config", str(config_path)],
        ["onnx/export.py", "--config", str(config_path), "--checkpoint", str(run_dir / "best.pt")],
        ["test/verify_export.py", "--config", str(config_path)],
    ]
    for command in commands:
        run_stage(command, run_dir, environment)
    bundle = convert_esp32p4(workspace, run_dir, config_path, environment)
    print(f"Completed training, ONNX export and ESP32-P4 conversion: {bundle} | TensorBoard: {url}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume", type=Path, help="Restore checkpoint and train --steps additional steps")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--job-name", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.job_name:
        join_owner_job(args.job_name)
    run_dir = args.run_dir or args.workspace / "training" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    try:
        run(args.workspace, run_dir, args.steps, open_browser=not args.no_browser, resume=args.resume)
    except TrainingStopped:
        print("Training stopped by user; remaining pipeline stages skipped.", flush=True)


if __name__ == "__main__":
    main()
