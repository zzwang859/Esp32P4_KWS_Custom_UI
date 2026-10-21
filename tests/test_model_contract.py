"""Regression against golden outputs from the reference project's ONNX model."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from kws_custom_ui.training import TRAINING_ROOT


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(TRAINING_ROOT))
    importlib.import_module("windows_runtime").preload_msvc()
    import torch
    torch.set_num_threads(4)
    return torch, importlib.import_module("config"), importlib.import_module("device_contract")


@pytest.fixture
def reference_model(modules):
    torch, config, contract = modules
    model = importlib.import_module("model").DSTCResNet(config.load_config(), 8).eval()
    with np.load(Path(__file__).parent / "fixtures/esp32p4_reference.npz", allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    model.load_state_dict({name.removeprefix("weight__"): torch.from_numpy(value)
                           for name, value in arrays.items() if name.startswith("weight__")})
    return model, arrays


@pytest.mark.parametrize("section,name,value", [
    ("model", "pooling_frames", 99), ("model", "profile", "legacy"),
    ("model", "residual", [False] * 6), ("features", "profile", None),
    ("features", "mel_bins", 64), ("data", "sample_rate", 8000),
    ("onnx", "chunk_frames", 2), ("onnx", "mode", "non_stream"),
])
def test_reject_incompatible_model_and_frontend_settings(modules, section, name, value):
    _, config, contract = modules
    settings = config.load_config()
    settings[section][name] = value
    with pytest.raises(ValueError, match="ESP32-P4"):
        contract.validate_contract(settings)


def test_mfcc_matches_reference_onnx_golden(modules, reference_model):
    torch, config, _ = modules
    _, data = reference_model
    extractor = importlib.import_module("data.audio").MfccExtractor(config.load_config()["data"], config.load_config()["features"])
    actual = extractor.streaming(torch.from_numpy(data["audio"])).numpy()
    np.testing.assert_allclose(actual, data["expected_mfcc"], rtol=1e-5, atol=1e-5)
    valid = extractor(torch.from_numpy(data["audio"])).numpy()
    assert valid.shape == (1, 40, 98)
    np.testing.assert_allclose(valid, data["expected_mfcc"][:, :, 2:], rtol=1e-5, atol=1e-5)


def test_backend_matches_reference_onnx_through_pool_window_boundary(modules, reference_model):
    torch, _, contract = modules
    model, data = reference_model
    features = torch.from_numpy(data["input_features"])
    states = model.initial_stream_state(1, "cpu")
    assert [tuple(state.shape[1:]) for state in states] == contract.STATE_SHAPES
    with torch.no_grad():
        for index in range(features.shape[-1]):
            actual, states = model.stream_step(features[:, :, index:index+1], states)
            np.testing.assert_allclose(actual.numpy(), data["expected_logits"][index], rtol=1e-4, atol=1e-5)
            if index in (16, 97, 98, 179):
                np.testing.assert_allclose(model(features[:, :, :index+1]).numpy(), data["expected_logits"][index], rtol=1e-4, atol=1e-5)


def test_exported_onnx_uses_exact_device_ports_and_six_feedback_pairs(modules, reference_model, tmp_path):
    torch, config, contract = modules
    model, data = reference_model
    settings = config.load_config()
    labels = tmp_path / "labels.txt"
    labels.write_text("silence\nunknown\na\nb\nc\nd\ne\nf\n", encoding="utf-8")
    settings["data"]["labels_file"] = str(labels)
    settings["onnx"]["output"] = str(tmp_path / "model.onnx")
    checkpoint = tmp_path / "model.pt"
    torch.save(dict(model=model.state_dict(), config=settings, step=1), checkpoint)
    exporter = runpy.run_path(str(TRAINING_ROOT / "onnx/export.py"))
    output = exporter["export"](settings, checkpoint)
    import onnx
    graph = onnx.load(output).graph
    assert "GlobalAveragePool" in {node.op_type for node in graph.node}
    assert not {"Gather", "ReduceMean"} & {node.op_type for node in graph.node}
    assert all(len(next(attr.ints for attr in node.attribute if attr.name == "kernel_shape")) == 2
               for node in graph.node if node.op_type == "Conv")
    runtime = runpy.run_path(str(TRAINING_ROOT / "onnx/run/infer.py"))["OnnxKWS"](output)
    assert [item.name for item in runtime.inputs] == [contract.FEATURE_INPUT, *contract.STATE_INPUTS]
    assert runtime.state_pairs == [list(pair) for pair in zip(contract.STATE_INPUTS, contract.STATE_OUTPUTS)]
    assert len(runtime.outputs) == 8 and len(runtime.states) == 6
    assert "onnx_next_state_6:0" in [item.name for item in runtime.outputs]
    np.testing.assert_allclose(runtime.predict(data["input_features"]), data["expected_logits"][-1], rtol=1e-4, atol=1e-5)
    runtime.reset()
    all_logits = np.stack(list(runtime.predict_chunks(data["input_features"])))
    np.testing.assert_allclose(all_logits, data["expected_logits"], rtol=1e-4, atol=1e-5)


def test_old_checkpoint_is_rejected_before_loading_weights(modules, tmp_path):
    torch, config, _ = modules
    settings = config.load_config()
    model = importlib.import_module("model").DSTCResNet(settings, 8)
    old = deepcopy(settings)
    old["model"].pop("profile")
    checkpoint = tmp_path / "old.pt"
    torch.save(dict(model=model.state_dict(), config=old), checkpoint)
    with pytest.raises(ValueError, match="旧版训练历史"):
        importlib.import_module("runtime").load_checkpoint(model, checkpoint, torch.device("cpu"))
