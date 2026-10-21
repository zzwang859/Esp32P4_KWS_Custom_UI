"""Convert the current MFCC-input KWS ONNX, without an external firmware checkout.

Run in the project root .venv managed by uv. Every write stays in --bundle; the source example models
and the training ONNX are never modified. INT8 and INT16 use separate processes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "scripts"))
from runtime_bootstrap import preload_msvc_runtime
preload_msvc_runtime()

# PyTorch 2.8.0+cu126's Windows MKL path crashes on the CPU float64 Conv
# used by INT16 debug simulation. Select MKL's compatible branch before any
# native imports initialize it. This affects only the conversion subprocess.
if sys.platform == "win32":
    os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import numpy as np
import onnx
import onnxruntime as ort
from onnxsim import simplify
import torch


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def initial_state(runtime, pairs):
    specs = {item.name: item for item in runtime.get_inputs()}
    return {name: np.zeros([size if isinstance(size, int) else 1 for size in specs[name].shape], np.float32)
            for name, _ in pairs}


def frames(features, chunk_frames):
    for offset in range(0, features.shape[2], chunk_frames):
        chunk = features[:, :, offset:offset + chunk_frames]
        yield np.pad(chunk, ((0, 0), (0, 0), (0, chunk_frames - chunk.shape[2])))


def feature_feed(runtime, chunk):
    metadata = runtime.get_modelmeta().custom_metadata_map
    name = metadata.get("feature_input_name", "features")
    values = chunk.transpose(0, 2, 1) if metadata.get("features_layout") == "NTC" else chunk
    return {name: values}


def prepare(bundle: Path, check_stop):
    check_stop()
    metadata = json.loads((bundle / "work/dataset.json").read_text(encoding="utf-8"))
    source = bundle / "models/ds_tc_resnet.onnx"
    if digest(source) != metadata["source_sha256"]:
        raise ValueError("Source ONNX hash does not match training run")
    model = onnx.load(source)
    runtime = session(source)
    properties = runtime.get_modelmeta().custom_metadata_map
    pairs = json.loads(properties.get("external_state_pairs", "[]"))
    mode = metadata["mode"]
    if json.loads(properties["labels"]) != metadata["labels"]:
        raise ValueError("ONNX labels do not match workspace class indices")
    shapes = {}
    for info in runtime.get_inputs():
        shape = [size if isinstance(size, int) and size > 0 else 1 for size in info.shape]
        if info.name == "features" and mode == "non_stream":
            with np.load(bundle / "work/features.npz") as features:
                shape = list(features[metadata["clips"][0]["key"]].shape)
        shapes[info.name] = shape
    for info in model.graph.input:
        for dimension, size in zip(info.type.tensor_type.shape.dim, shapes[info.name]):
            dimension.ClearField("dim_param")
            dimension.dim_value = size
    model = onnx.shape_inference.infer_shapes(model)
    model, valid = simplify(model, check_n=1)
    if not valid:
        raise ValueError("Static ONNX simplification changed model outputs")
    onnx.checker.check_model(model)
    prepared = bundle / "models/ds_tc_resnet_esp32p4_prepared.onnx"
    onnx.save(model, prepared)
    after = session(prepared)
    input_names = [info.name for info in after.get_inputs()]
    # Simplification may discard state inputs that have no consumers.
    used_pairs = [pair for pair in pairs if pair[0] in input_names]
    calibration, max_error = [], 0.0
    arrays = {}
    with np.load(bundle / "work/features.npz") as features:
        for clip_index, clip in enumerate(metadata["clips"]):
            check_stop()
            state = initial_state(runtime, pairs)
            new_state = initial_state(after, used_pairs)
            chunks = [features[clip["key"]]] if mode == "non_stream" else frames(features[clip["key"]], metadata["chunk_frames"])
            samples = []
            for chunk_index, chunk in enumerate(chunks):
                check_stop()
                feed = {**feature_feed(runtime, chunk), **state}
                prepared_feed = {name: value for name, value in {**feature_feed(after, chunk), **new_state}.items() if name in input_names}
                expected = runtime.run(None, feed)
                actual = after.run(None, prepared_feed)
                error = max(float(np.max(np.abs(a - b))) for a, b in zip(expected, actual))
                max_error = max(max_error, error)
                if not all(np.isfinite(value).all() and np.allclose(value, reference, atol=1e-5, rtol=1e-4)
                           for value, reference in zip(actual, expected)):
                    raise ValueError(f"Prepared ONNX stream changed: {clip['path']}, chunk {chunk_index}, error={error}")
                if clip["split"] == "calibration" and chunk_index % 4 == 0:
                    samples.append([prepared_feed[name].copy() for name in input_names])
                state = {name: expected[[info.name for info in runtime.get_outputs()].index(output)] for name, output in pairs}
                new_state = {name: actual[[info.name for info in after.get_outputs()].index(output)] for name, output in used_pairs}
            if samples:
                calibration.append(samples)
            arrays[f"reference_{clip_index}"] = expected[0]
    # Sample across all classes and positions in the stream, including warmup.
    selected = []
    for index in range(max(map(len, calibration), default=0)):
        for samples in calibration:
            if index < len(samples):
                selected.append(samples[index])
    if not selected:
        raise ValueError("No real calibration inputs were produced")
    if len(selected) > 512:
        selected = [selected[index] for index in np.linspace(0, len(selected) - 1, 512, dtype=int)]
    for input_index, name in enumerate(input_names):
        arrays[f"input_{input_index}"] = np.stack([sample[input_index] for sample in selected])
    np.savez_compressed(bundle / "work/calibration.npz", **arrays)
    mapping = dict(inputs={info.name: info.shape for info in after.get_inputs()},
                   outputs={info.name: info.shape for info in after.get_outputs()},
                   state_pairs=used_pairs, mode=mode, chunk_frames=metadata["chunk_frames"],
                   input_layout="MFCC NTC; states NT1C, matching fixed device ABI",
                   frontend_profile=properties.get("frontend_profile"),
                   frontend_asset_sha256=properties.get("frontend_asset_sha256"),
                   calibration_steps=max(8, len(selected)), fp32_max_absolute_error=max_error,
                   prepared_sha256=digest(prepared), source_sha256=digest(source))
    write_json(bundle / "models/io_mapping.json", mapping)
    print(f"ESP32-P4: static ONNX verified, calibration steps={mapping['calibration_steps']}, max error={max_error}", flush=True)


def quantize(bundle: Path, bits: int, check_stop):
    check_stop()
    from esp_ppq.api import espdl_quantize_onnx
    from esp_ppq import TorchExecutor
    torch.set_num_threads(4)
    torch.manual_seed(20261020)
    metadata = json.loads((bundle / "work/dataset.json").read_text(encoding="utf-8"))
    mapping = json.loads((bundle / "models/io_mapping.json").read_text(encoding="utf-8"))
    scheme = f"w{bits}a{bits}"
    output = bundle / "output" / scheme
    output.mkdir(parents=True, exist_ok=True)
    backend = output / "backend_fp32.onnx"
    shutil.copy2(bundle / "models/ds_tc_resnet_esp32p4_prepared.onnx", backend)
    model = output / f"ds_tc_resnet_backend_{scheme}_esp32p4.espdl"
    with np.load(bundle / "work/calibration.npz") as arrays:
        count = len(arrays["input_0"])
        loader = [{name: torch.from_numpy(arrays[f"input_{index}"][step].copy())
                   for index, name in enumerate(mapping["inputs"])} for step in range(count)]
    print(f"ESP32-P4: quantizing {scheme} using {count} real stream states", flush=True)
    graph = espdl_quantize_onnx(str(backend), str(model), loader, mapping["calibration_steps"],
                               input_shape=None, inputs=loader[0], target="esp32p4", quant_type=scheme,
                               device="cpu", error_report=False, export_test_values=True,
                               collate_fn=lambda sample: sample, debug=bits == 16)
    check_stop()
    executor = TorchExecutor(graph=graph, device="cpu")
    runtime = session(bundle / "models/ds_tc_resnet_esp32p4_prepared.onnx")
    pairs = mapping["state_pairs"]
    rows = []
    with np.load(bundle / "work/features.npz") as features:
        for clip in metadata["clips"]:
            if clip["split"] != "validation":
                continue
            check_stop()
            state = initial_state(runtime, pairs)
            quantized_state = {name: torch.from_numpy(value.copy()) for name, value in state.items()}
            errors, agreements = [], []
            chunks = [features[clip["key"]]] if metadata["mode"] == "non_stream" else frames(features[clip["key"]], metadata["chunk_frames"])
            with torch.no_grad():
                for chunk in chunks:
                    check_stop()
                    feed = feature_feed(runtime, chunk)
                    expected = runtime.run(None, {**feed, **state})
                    outputs = list(mapping["outputs"])
                    actual = executor.forward({**{name: torch.from_numpy(value.copy()) for name, value in feed.items()}, **quantized_state}, output_names=outputs)
                    result = actual[0].detach().cpu().numpy()
                    if not all(torch.isfinite(value).all() for value in actual):
                        raise ValueError(f"Non-finite {scheme} streaming output: {clip['path']}")
                    errors.append(float(np.mean(np.abs(result - expected[0]))))
                    agreements.append(int(result.argmax() == expected[0].argmax()))
                    state = {name: expected[outputs.index(next_name)] for name, next_name in pairs}
                    quantized_state = {name: actual[outputs.index(next_name)] for name, next_name in pairs}
            rows.append(dict(path=clip["path"], label_index=clip["label_index"],
                             onnx_top1=int(expected[0].argmax()), quantized_top1=int(result.argmax()),
                             chunk_top1_agreement=float(np.mean(agreements)), logits_mae=float(np.mean(errors))))
    check_stop()
    report = dict(target="esp32p4", scheme=scheme, source_sha256=metadata["source_sha256"],
                  model_sha256=digest(model), bytes=model.stat().st_size, calibration_steps=mapping["calibration_steps"],
                  calibration_files=[clip["path"] for clip in metadata["clips"] if clip["split"] == "calibration"],
                  validation_training_fallback_classes=metadata["validation_training_fallback_classes"],
                  clips=len(rows), final_top1_agreement=sum(row["onnx_top1"] == row["quantized_top1"] for row in rows),
                  onnx_correct=sum(row["onnx_top1"] == row["label_index"] for row in rows),
                  quantized_correct=sum(row["quantized_top1"] == row["label_index"] for row in rows), rows=rows)
    write_json(output / "comparison.json", report)
    print(f"ESP32-P4: {scheme} exported {model.stat().st_size} bytes, final top1 agreement {report['final_top1_agreement']}/{len(rows)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=[8, 16])
    args = parser.parse_args()

    def check_stop():
        if args.stop_file.exists():
            raise InterruptedError("ESP32-P4 conversion stopped by user")

    if args.bits:
        quantize(args.bundle, args.bits, check_stop)
    else:
        prepare(args.bundle, check_stop)
