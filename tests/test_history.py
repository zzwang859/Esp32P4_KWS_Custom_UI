import os
from pathlib import Path
import subprocess

import pytest

from kws_custom_ui.history import clear_training_history, training_history


def history_run(workspace, stamp, file="latest.pt"):
    run = workspace / "training" / stamp
    run.mkdir(parents=True)
    (run / file).write_bytes(b"checkpoint")
    return run


def test_empty_history_and_latest_checkpoint_selection(tmp_path):
    assert not training_history(tmp_path).present
    older = history_run(tmp_path, "20261019-100000-000000")
    latest = history_run(tmp_path, "20261019-110000-000000")
    incomplete = history_run(tmp_path, "20261019-120000-000000", "training.log")
    assert training_history(tmp_path).checkpoint == latest / "latest.pt"
    (latest / "latest.pt").unlink()
    (latest / "step_2.pt").write_bytes(b"two")
    (latest / "step_10.pt").write_bytes(b"ten")
    (latest / "best.pt").write_bytes(b"best")
    assert training_history(tmp_path).checkpoint == latest / "step_10.pt"
    assert older.exists() and incomplete.exists()


def test_log_only_history_is_present_but_not_resumable(tmp_path):
    history_run(tmp_path, "20261019-100000-000000", "training.log")
    history = training_history(tmp_path)
    assert history.present and history.checkpoint is None


def test_clear_removes_all_history_and_preserves_workspace_data(tmp_path):
    history_run(tmp_path, "20261019-100000-000000")
    history_run(tmp_path, "20261019-110000-000000")
    (tmp_path / "training" / "extra.log").write_text("log")
    conversion = tmp_path / "esp32p4-dl"
    output = conversion / "old-run" / "output"
    output.mkdir(parents=True)
    (output / "model.espdl").write_bytes(b"model")
    (conversion / "latest.json").write_text("previous conversion")
    (conversion / "latest.json.tmp").write_text("unfinished conversion")
    (tmp_path / "workspace.json").write_text("config")
    audio = tmp_path / "keywords" / "keyword1" / "keep.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"audio")
    clear_training_history(tmp_path)
    assert (tmp_path / "training").is_dir()
    assert list((tmp_path / "training").iterdir()) == []
    assert conversion.is_dir() and not list(conversion.iterdir())
    assert audio.read_bytes() == b"audio"
    assert (tmp_path / "workspace.json").read_text() == "config"
    assert not training_history(tmp_path).present


def test_conversion_history_is_cleared_when_training_directory_is_missing(tmp_path):
    conversion = tmp_path / "esp32p4-dl"
    conversion.mkdir()
    (conversion / "latest.json").write_text("previous conversion")
    clear_training_history(tmp_path)
    assert not list(conversion.iterdir())
    assert not (tmp_path / "training").exists()


def test_invalid_conversion_root_is_rejected_before_training_history_is_removed(tmp_path):
    old = history_run(tmp_path, "old-run")
    (tmp_path / "esp32p4-dl").write_text("not a directory")
    with pytest.raises(ValueError, match="esp32p4-dl 必须是文件夹"):
        clear_training_history(tmp_path)
    assert (old / "latest.pt").read_bytes() == b"checkpoint"


@pytest.mark.skipif(os.name != "nt", reason="Windows directory junction behavior")
@pytest.mark.parametrize("root_is_link", [False, True])
@pytest.mark.parametrize("directory", ["training", "esp32p4-dl"])
def test_history_cleanup_never_follows_junctions_outside_workspace(tmp_path, root_is_link, directory):
    workspace, outside = tmp_path / "workspace", tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("keep")
    if root_is_link:
        link = workspace / directory
    else:
        old_run = workspace / directory / "20261019-100000-000000"
        old_run.mkdir(parents=True)
        link = old_run / "linked"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                   creationflags=subprocess.CREATE_NO_WINDOW, capture_output=True, check=True)
    if root_is_link:
        with pytest.raises(ValueError):
            clear_training_history(workspace)
        if directory == "training":
            with pytest.raises(ValueError):
                training_history(workspace)
    else:
        clear_training_history(workspace)
        assert not list((workspace / directory).iterdir())
    assert marker.read_text() == "keep"
