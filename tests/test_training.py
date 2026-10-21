import importlib
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock
import wave

import pytest

from kws_custom_ui.model import Workspace
from kws_custom_ui.training import Trainer, TRAINING_ROOT


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.syspath_prepend(str(TRAINING_ROOT))
    return importlib.import_module("workspace_train")


def test_conversion_uses_training_interpreter_despite_legacy_override(pipeline, monkeypatch):
    monkeypatch.setenv("KWS_ESPDL_PYTHON", "missing-legacy-environment/python.exe")
    python, script = pipeline.conversion_runtime()
    assert python == Path(pipeline.sys.executable).resolve()
    assert script == (TRAINING_ROOT.parent / "espdl_toolchain/esp32p4-dl/scripts/convert_workspace.py").resolve()


def test_conversion_subprocesses_share_training_interpreter(pipeline, tmp_path, monkeypatch):
    run_dir = tmp_path / "training/new-run"
    run_dir.mkdir(parents=True)
    stages = []
    monkeypatch.setattr(pipeline, "run_stage", lambda command, *args, **kwargs: stages.append((command, kwargs)))
    monkeypatch.setattr(pipeline, "publish_conversion", lambda *args: {"model": "verified.espdl"})
    pipeline.convert_esp32p4(tmp_path, run_dir, run_dir / "config.yaml", {})
    assert len(stages) == 4
    assert all(Path(stage[1].get("python", pipeline.sys.executable)).resolve() == Path(pipeline.sys.executable).resolve()
               for stage in stages)


def write_wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(b"\0\0" * 16000)


@pytest.fixture
def training_workspace(tmp_path):
    workspace = Workspace(tmp_path / "中文语料")
    workspace.delete(workspace.rows[1])
    workspace.rows[0].text = "打开台灯"
    workspace.rows[1].text = "close"
    workspace.rows[2].text = "next"
    workspace.save()
    for directory in [workspace.silence_dir, workspace.unknown_dir, *(workspace.row_dir(row) for row in workspace.rows)]:
        for name in ["speaker_a_nohash_0.wav", "speaker_b_nohash_0.WAV", "speaker_c_nohash_0.wav"]:
            write_wav(directory / name)
    write_wav(workspace.path / "keywords" / "delete-123-keyword2" / "archived.wav")
    write_wav(workspace.path / "keywords" / "unconfigured" / "unconfigured.wav")
    return workspace


def test_workspace_class_indices_and_manifests_use_only_configured_audio(pipeline, training_workspace, tmp_path):
    training_workspace.rows[0].flags = 0
    training_workspace.save()
    run_dir = tmp_path / "run"
    config = pipeline.configure(training_workspace.path, run_dir, steps=10)
    counts = pipeline.prepare(config)
    assert config["train"]["steps"] == 10
    assert config["train"]["num_workers"] == 0
    assert Path(config["data"]["preprocess_dir"]) == run_dir / "preprocess"
    classes = json.loads((run_dir / "classes.json").read_text(encoding="utf-8"))
    assert [(item["index"], item["label"]) for item in classes] == [
        (0, "silence"), (1, "unknown"), (2, "打开台灯"), (3, "close"), (4, "next")]
    assert Path(classes[3]["directory"]).name == "keyword3"
    entries = [json.loads(line) for manifest in (run_dir / "manifests").glob("*.jsonl")
               for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert sum(counts.values()) == 15
    assert {entry["label_index"] for entry in entries} == set(range(5))
    for entry in entries:
        assert entry["path"] is not None
        assert Path(entry["path"]).is_file()
        assert Path(entry["path"]).parent == Path(classes[entry["label_index"]]["directory"])
    assert (run_dir / "labels.txt").read_text(encoding="utf-8").splitlines() == [item["label"] for item in classes]
    training_entries = [json.loads(line) for line in (run_dir / "manifests" / "training.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {entry["label_index"] for entry in training_entries} == set(range(5))


def test_missing_required_class_audio_fails_before_tensorboard(pipeline, training_workspace, tmp_path, monkeypatch):
    for wav in training_workspace.unknown_dir.iterdir():
        wav.unlink()
    tensorboard = Mock()
    monkeypatch.setattr(pipeline, "start_tensorboard", tensorboard)
    with pytest.raises(ValueError, match="No WAV recordings for class 1"):
        pipeline.run(training_workspace.path, tmp_path / "run", steps=10)
    tensorboard.assert_not_called()


def test_duplicate_keyword_names_are_rejected(pipeline, training_workspace, tmp_path):
    training_workspace.rows[1].text = training_workspace.rows[0].text
    training_workspace.save()
    with pytest.raises(ValueError, match="unique"):
        pipeline.configure(training_workspace.path, tmp_path / "run")


def test_training_launches_background_process_and_reports_failure(tmp_path, monkeypatch):
    process = Mock()
    process.poll.return_value = None
    launch = Mock(return_value=process)
    monkeypatch.setattr("kws_custom_ui.training.subprocess.Popen", launch)
    trainer = Trainer()
    run_dir = trainer.start(tmp_path, steps=10)
    assert trainer.busy
    command = launch.call_args.args[0]
    assert command[command.index("--workspace") + 1] == str(tmp_path)
    assert command[command.index("--steps") + 1] == "10"
    with pytest.raises(RuntimeError):
        trainer.start(tmp_path)
    with (run_dir / "training.log").open("a", encoding="utf-8") as output:
        output.write("ValueError: no audio\n")
    process.poll.return_value = 1
    message, error = trainer.update()
    assert error and "no audio" in message and "training.log" in message
    assert trainer.update() is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_actual_device_from_child_log_is_in_status_and_stop_is_not_failure(tmp_path, monkeypatch, device):
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr("kws_custom_ui.training.subprocess.Popen", Mock(return_value=process))
    trainer = Trainer()
    run_dir = trainer.start(tmp_path, steps=10)
    with (run_dir / "training.log").open("a", encoding="utf-8") as output:
        output.write(f"TRAINING_DEVICE={device}\nstep=1/10 loss=2.0\n")
    message, error = trainer.update()
    assert not error
    assert message == f"训练中({device}):step=1/10 loss=2.0"
    trainer.stop()
    assert (run_dir / "stop.requested").is_file()
    assert trainer.stopping
    process.poll.return_value = 0
    message, error = trainer.update()
    assert not error and "训练已停止" in message
    assert "导出完成" not in message


def test_cancelled_pipeline_never_launches_next_stage(pipeline, tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "stop.requested").touch()
    launch = Mock()
    monkeypatch.setattr(pipeline.subprocess, "Popen", launch)
    with pytest.raises(pipeline.TrainingStopped):
        pipeline.run_stage(["onnx/export.py"], run_dir, {})
    launch.assert_not_called()


def test_resume_configuration_uses_saved_model_and_additional_steps(pipeline, training_workspace, tmp_path, monkeypatch):
    prior = pipeline.configure(training_workspace.path, tmp_path / "old", steps=100)
    prior["train"]["learning_rate"] = 0.0025
    monkeypatch.setattr(pipeline, "load_resume_checkpoint", lambda path: {
        "config": prior, "optimizer": {"state": {}}, "step": 50})
    resumed = pipeline.configure(training_workspace.path, tmp_path / "new", steps=10, resume=tmp_path / "old/latest.pt")
    assert resumed["train"]["resume_step"] == 50
    assert resumed["train"]["steps"] == 60
    assert resumed["train"]["additional_steps"] == 10
    assert resumed["train"]["learning_rate"] == 0.0025
    assert Path(resumed["data"]["preprocess_dir"]) == tmp_path / "new/preprocess"
    assert resumed["data"]["preprocess_dir"] != prior["data"]["preprocess_dir"]
    training_workspace.rows[0].text = "changed meaning"
    training_workspace.save()
    with pytest.raises(ValueError, match="类别顺序、文本或目录"):
        pipeline.configure(training_workspace.path, tmp_path / "new", steps=10, resume=tmp_path / "old/latest.pt")


def test_restart_closes_old_services_before_clearing_history(tmp_path, monkeypatch):
    old = tmp_path / "training" / "old"
    old.mkdir(parents=True)
    (old / "latest.pt").write_bytes(b"old")
    conversion = tmp_path / "esp32p4-dl"
    conversion.mkdir()
    (conversion / "latest.json").write_text("old model")
    cache = old / "preprocess/feature.npy"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"old MFCC")
    trainer = Trainer()
    released = []
    group = Mock()
    group.close.side_effect = lambda: released.append(True)
    trainer._groups.append(group)
    process = Mock()
    process.poll.return_value = 0
    def launch(*args, **kwargs):
        assert released == [True]
        assert not old.exists()
        assert not list(conversion.iterdir())
        assert not cache.exists()
        return process
    monkeypatch.setattr("kws_custom_ui.training.subprocess.Popen", launch)
    trainer.start(tmp_path, steps=1, restart=True)
    trainer.close()


def test_resuming_training_preserves_conversion_history(tmp_path, monkeypatch):
    old = tmp_path / "training/old"
    old.mkdir(parents=True)
    checkpoint = old / "latest.pt"
    checkpoint.write_bytes(b"old")
    conversion = tmp_path / "esp32p4-dl"
    conversion.mkdir()
    latest = conversion / "latest.json"
    latest.write_text("old model")
    process = Mock()
    process.poll.return_value = 0
    monkeypatch.setattr("kws_custom_ui.training.subprocess.Popen", Mock(return_value=process))
    trainer = Trainer()
    try:
        trainer.start(tmp_path, steps=1, resume=checkpoint)
        assert checkpoint.read_bytes() == b"old"
        assert latest.read_text() == "old model"
    finally:
        trainer.close()


@pytest.mark.parametrize("stop_after", [0, 1, 2, 3])
def test_stop_during_training_or_export_skips_esp32p4(pipeline, training_workspace, tmp_path, monkeypatch, stop_after):
    run_dir = tmp_path / "run"
    monkeypatch.setattr(pipeline, "conversion_runtime", lambda: (Path("python"), Path("convert.py")))
    monkeypatch.setattr(pipeline, "start_tensorboard", lambda *args: "http://127.0.0.1:6006")
    commands = []

    def stage(command, run, env):
        pipeline.check_stop(run)
        commands.append(command)
        if len(commands) == stop_after + 1:
            (run / "stop.requested").touch()
        pipeline.check_stop(run)

    conversion = Mock()
    monkeypatch.setattr(pipeline, "run_stage", stage)
    monkeypatch.setattr(pipeline, "convert_esp32p4", conversion)
    with pytest.raises(pipeline.TrainingStopped):
        pipeline.run(training_workspace.path, run_dir, steps=10, open_browser=False)
    assert len(commands) == stop_after + 1
    conversion.assert_not_called()
    assert not (training_workspace.path / "esp32p4-dl").exists()


@pytest.mark.parametrize("resuming", [False, True])
def test_esp32p4_only_runs_after_training_export_and_verification(pipeline, training_workspace, tmp_path, monkeypatch, resuming):
    run_dir = tmp_path / "run"
    monkeypatch.setattr(pipeline, "conversion_runtime", lambda: (Path("python"), Path("convert.py")))
    monkeypatch.setattr(pipeline, "start_tensorboard", lambda *args: "http://127.0.0.1:6006")
    commands = []
    def stage(command, *args):
        commands.append(command[0])
        if command[0] == "data/preprocess.py":
            assert "--force" in command
    monkeypatch.setattr(pipeline, "run_stage", stage)
    checkpoint = None
    if resuming:
        prior = pipeline.configure(training_workspace.path, tmp_path / "old", steps=10)
        checkpoint = tmp_path / "old/latest.pt"
        monkeypatch.setattr(pipeline, "load_resume_checkpoint", lambda path: {
            "config": prior, "optimizer": {"state": {}}, "step": 10})

    def convert(*args):
        assert commands == ["data/preprocess.py", "train/train.py", "onnx/export.py", "test/verify_export.py"]
        assert args[0] == training_workspace.path
        return training_workspace.path / "esp32p4-dl" / "run"

    conversion = Mock(side_effect=convert)
    monkeypatch.setattr(pipeline, "convert_esp32p4", conversion)
    assert pipeline.run(training_workspace.path, run_dir, steps=10, open_browser=False, resume=checkpoint) == run_dir
    conversion.assert_called_once()


@pytest.mark.parametrize("stop_after", [0, 1, 2, 3])
def test_cancelled_conversion_never_launches_later_precision_or_updates_latest(pipeline, tmp_path, monkeypatch, stop_after):
    run_dir = tmp_path / "training/new-run"
    run_dir.mkdir(parents=True)
    root = tmp_path / "esp32p4-dl"
    root.mkdir()
    (root / "latest.json").write_text("previous successful build")
    monkeypatch.setattr(pipeline, "conversion_runtime", lambda: (Path("python"), Path("convert.py")))
    stages = []

    def stage(command, run, environment, **kwargs):
        pipeline.check_stop(run)
        stages.append(command)
        assert Path(command[command.index("--stop-file") + 1]) == run_dir / "stop.requested"
        if len(stages) == stop_after + 1:
            (run_dir / "stop.requested").touch()
        pipeline.check_stop(run)

    monkeypatch.setattr(pipeline, "run_stage", stage)
    with pytest.raises(pipeline.TrainingStopped):
        pipeline.convert_esp32p4(tmp_path, run_dir, run_dir / "config.yaml", {})
    assert len(stages) == stop_after + 1
    assert (root / "latest.json").read_text() == "previous successful build"
    assert not (run_dir / "esp32p4.json").exists()


def make_converted_bundle(tmp_path):
    run_dir = tmp_path / "training/new-run"
    run_dir.mkdir(parents=True)
    bundle = tmp_path / "esp32p4-dl/new-run"
    (bundle / "work").mkdir(parents=True)
    (bundle / "models").mkdir()
    source = b"current training ONNX"
    (bundle / "models/ds_tc_resnet.onnx").write_bytes(source)
    source_hash = hashlib.sha256(source).hexdigest()
    (bundle / "work/dataset.json").write_text(json.dumps(dict(
        source="training/new-run/onnx/ds_tc_resnet.onnx", source_sha256=source_hash,
        labels=["silence", "unknown", "keyword1"])))
    for scheme in ("w8a8", "w16a16"):
        folder = bundle / "output" / scheme
        folder.mkdir(parents=True)
        model = folder / f"ds_tc_resnet_backend_{scheme}_esp32p4.espdl"
        model.write_bytes(scheme.encode())
        model.with_suffix(".json").write_text("quantization config")
        model.with_suffix(".info").write_text("ESP-DL model info")
        (folder / "comparison.json").write_text(json.dumps(dict(
            model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(), source_sha256=source_hash,
            target="esp32p4", clips=3, final_top1_agreement=3)))
    return run_dir, bundle


def test_successful_conversion_publishes_both_models_and_selects_int16(pipeline, tmp_path):
    run_dir, bundle = make_converted_bundle(tmp_path)
    result = pipeline.publish_conversion(bundle, run_dir)
    assert result["selected"] == "w16a16"
    assert result["labels"][:2] == ["silence", "unknown"]
    assert (bundle.parent / result["model"]).read_bytes() == b"w16a16"
    assert (bundle / "output/ds_tc_resnet_backend_w8a8_esp32p4.espdl").read_bytes() == b"w8a8"
    assert json.loads((bundle.parent / "latest.json").read_text()) == result
    assert json.loads((bundle / "conversion.json").read_text()) == result
    # Publishing conversion artifacts must not write into the training folder.
    assert not list(run_dir.iterdir())


@pytest.mark.parametrize("tamper", ["model", "source", "metadata", "stop"])
def test_invalid_or_stopped_conversion_preserves_previous_success(pipeline, tmp_path, tamper):
    run_dir, bundle = make_converted_bundle(tmp_path)
    latest = bundle.parent / "latest.json"
    latest.write_text("previous build")
    model = bundle / "output/w16a16/ds_tc_resnet_backend_w16a16_esp32p4.espdl"
    if tamper == "model":
        model.write_bytes(b"wrong model")
    elif tamper == "source":
        (bundle / "models/ds_tc_resnet.onnx").write_bytes(b"wrong source")
    elif tamper == "metadata":
        model.with_suffix(".info").unlink()
    else:
        (run_dir / "stop.requested").touch()
    with pytest.raises((ValueError, pipeline.TrainingStopped)):
        pipeline.publish_conversion(bundle, run_dir)
    assert latest.read_text() == "previous build"
    assert not (run_dir / "esp32p4.json").exists()


def test_gui_completion_names_esp32p4_output(tmp_path, monkeypatch):
    process = Mock()
    process.poll.return_value = 0
    monkeypatch.setattr("kws_custom_ui.training.subprocess.Popen", Mock(return_value=process))
    trainer = Trainer()
    run_dir = trainer.start(tmp_path, steps=10)
    message, error = trainer.update()
    assert not error and "ESP32-P4 转换完成" in message
    assert str(tmp_path / "esp32p4-dl" / run_dir.name) in message
    trainer.close()
