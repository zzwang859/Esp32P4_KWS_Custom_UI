import json
from datetime import datetime
from unittest.mock import patch

import pytest

from kws_custom_ui.model import Workspace


def test_kws_filter_is_independent_per_workspace(tmp_path):
    first = Workspace(tmp_path / "first")
    second = Workspace(tmp_path / "second")
    first.kws_filter.update(keyword_threshold_percent=0, cooldown_ms=1000,
                            stable_frames=50, need_vad_checked=True)
    first.save()
    assert Workspace(first.path).kws_filter == first.kws_filter
    assert Workspace(second.path).kws_filter != first.kws_filter
    assert "kws_filter" in json.loads((second.path / "workspace.json").read_text(encoding="utf-8"))


def test_workspace_persists_text_colors_and_stable_directories(tmp_path):
    workspace = Workspace(tmp_path / "中文项目" / "新建")
    assert len(workspace.rows) == 4
    assert [workspace.row_dir(row).name for row in workspace.rows] == [f"keyword{i}" for i in range(1, 5)]
    row = workspace.rows[2]
    original_directory = workspace.row_dir(row)
    row.text, row.color = "打开台灯", (12, 56, 200)
    workspace.delete(workspace.rows[0])
    stored_rows = json.loads((workspace.path / "workspace.json").read_text(encoding="utf-8"))["rows"]
    assert {item["directory"]: item["text"] for item in stored_rows} == {
        "keyword2": "", "keyword3": "打开台灯", "keyword4": ""
    }
    restored = Workspace(workspace.path)
    assert restored.rows[1].text == "打开台灯"
    assert restored.rows[1].color == (12, 56, 200)
    assert restored.row_dir(restored.rows[1]) == original_directory
    assert workspace.silence_dir.is_dir()


def test_unknown_directory_is_created_and_existing_contents_are_preserved(tmp_path):
    workspace = Workspace(tmp_path)
    assert workspace.unknown_dir == tmp_path / "unknown"
    assert workspace.unknown_dir.is_dir()
    sample = workspace.unknown_dir / "sample.wav"
    sample.write_bytes(b"audio")
    restored = Workspace(tmp_path)
    assert (restored.unknown_dir / sample.name).read_bytes() == b"audio"
    sample.unlink()
    restored.unknown_dir.rmdir()
    assert Workspace(tmp_path).unknown_dir.is_dir()


def test_removed_rows_keep_recordings_and_counts_include_external_wavs(tmp_path):
    workspace = Workspace(tmp_path)
    directory = workspace.row_dir(workspace.rows[0])
    (directory / "external.WAV").write_bytes(b"test")
    (directory / "pending.recording").write_bytes(b"test")
    (directory / "nested.wav").mkdir()
    assert Workspace.wav_count(directory) == 1
    archived = workspace.delete(workspace.rows[0])
    for row in list(workspace.rows):
        workspace.delete(row)
    assert Workspace(tmp_path).rows == []
    assert not directory.exists()
    assert archived.parent == directory.parent
    assert archived.name.startswith("delete-") and archived.name.endswith("-keyword1")
    assert (archived / "external.WAV").read_bytes() == b"test"
    assert (archived / "pending.recording").read_bytes() == b"test"
    assert (archived / "nested.wav").is_dir()


def test_delete_empty_directory_and_add_without_renaming_survivors(tmp_path):
    workspace = Workspace(tmp_path)
    deleted = workspace.rows[1]
    directory = workspace.row_dir(deleted)
    assert workspace.delete(deleted) is None
    assert not directory.exists()
    assert not list(directory.parent.glob("delete-*"))
    added = workspace.add()
    workspace.save()
    assert workspace.row_dir(added).name == "keyword5"
    restored = Workspace(tmp_path)
    assert [row.directory for row in restored.rows] == ["keyword1", "keyword3", "keyword4", "keyword5"]


def test_legacy_directories_migrate_and_recordings_survive_reopen(tmp_path):
    rows = [{"id": f"{i:012x}", "text": f"text {i}", "color": [1, 2, 3]} for i in range(1, 3)]
    (tmp_path / "workspace.json").write_text(json.dumps({"version": 1, "rows": rows}))
    for row in rows:
        directory = tmp_path / "keywords" / row["id"]
        directory.mkdir(parents=True)
        (directory / "keep.wav").write_bytes(row["id"].encode())
    workspace = Workspace(tmp_path)
    assert [row.directory for row in workspace.rows] == ["keyword1", "keyword2"]
    assert all(row.enabled for row in workspace.rows)
    for row in Workspace(tmp_path).rows:
        assert (workspace.row_dir(row) / "keep.wav").read_bytes() == row.id.encode()
        assert not (tmp_path / "keywords" / row.id).exists()


def test_existing_unassigned_numbered_directory_is_not_reused(tmp_path):
    orphan = tmp_path / "keywords" / "keyword1"
    orphan.mkdir(parents=True)
    (orphan / "keep.txt").write_text("keep")
    workspace = Workspace(tmp_path)
    assert workspace.rows[0].directory == "keyword2"
    assert (orphan / "keep.txt").read_text() == "keep"


@pytest.mark.parametrize("has_contents", [False, True])
def test_delete_save_failure_restores_row_and_directory(tmp_path, monkeypatch, has_contents):
    workspace = Workspace(tmp_path)
    original = (tmp_path / "workspace.json").read_bytes()
    rows = list(workspace.rows)
    row = rows[1]
    directory = workspace.row_dir(row)
    if has_contents:
        (directory / "keep.txt").write_text("keep")
    def fail_save():
        raise OSError("disk full")
    monkeypatch.setattr(workspace, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        workspace.delete(row)
    assert workspace.rows == rows
    assert directory.is_dir()
    assert not list(directory.parent.glob("delete-*"))
    assert (tmp_path / "workspace.json").read_bytes() == original
    if has_contents:
        assert (directory / "keep.txt").read_text() == "keep"


def test_legacy_migration_save_failure_restores_old_directory(tmp_path, monkeypatch):
    row = {"id": "a" * 12, "text": "old", "color": [1, 2, 3]}
    config = tmp_path / "workspace.json"
    original = json.dumps({"version": 1, "rows": [row]})
    config.write_text(original)
    directory = tmp_path / "keywords" / row["id"]
    directory.mkdir(parents=True)
    (directory / "keep.wav").write_bytes(b"audio")
    def fail_save(self):
        raise OSError("disk full")
    monkeypatch.setattr(Workspace, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        Workspace(tmp_path)
    assert config.read_text() == original
    assert (directory / "keep.wav").read_bytes() == b"audio"


def test_archive_timestamp_collision_keeps_both_directories(tmp_path):
    workspace = Workspace(tmp_path)
    row = workspace.rows[0]
    directory = workspace.row_dir(row)
    (directory / "keep.txt").write_text("new")
    instant = datetime(2026, 10, 19, 15, 30)
    existing = directory.with_name("delete-20261019-153000-000000-keyword1")
    existing.mkdir()
    (existing / "keep.txt").write_text("old")
    with patch("kws_custom_ui.model.datetime") as clock:
        clock.now.return_value = instant
        archived = workspace.delete(row)
    assert archived != existing
    assert (existing / "keep.txt").read_text() == "old"
    assert (archived / "keep.txt").read_text() == "new"


@pytest.mark.parametrize("directory", ["../outside", "/absolute", "keyword0", "keyword01", "delete-keyword1", None])
def test_config_rejects_unsafe_directory_names(tmp_path, directory):
    row = {"id": "a" * 12, "text": "x", "color": [1, 2, 3], "directory": directory}
    config = tmp_path / "workspace.json"
    original = json.dumps({"version": 1, "rows": [row]})
    config.write_text(original)
    with pytest.raises(ValueError, match="目录无效"):
        Workspace(tmp_path)
    assert config.read_text() == original


def test_config_rejects_duplicate_directory_names(tmp_path):
    rows = [{"id": f"{i:012x}", "text": "x", "color": [1, 2, 3], "directory": "keyword1"} for i in (1, 2)]
    (tmp_path / "workspace.json").write_text(json.dumps({"version": 1, "rows": rows}))
    with pytest.raises(ValueError, match="目录无效或重复"):
        Workspace(tmp_path)


@pytest.mark.parametrize("bad_value", ["../outside", "/absolute", "a" * 11, "g" * 12])
def test_config_cannot_route_recordings_outside_workspace(tmp_path, bad_value):
    config = {"version": 1, "rows": [{"id": bad_value, "text": "x", "color": [1, 2, 3]}]}
    file = tmp_path / "workspace.json"
    original = json.dumps(config)
    file.write_text(original)
    with pytest.raises(ValueError):
        Workspace(tmp_path)
    assert file.read_text() == original


def test_corrupt_config_is_preserved(tmp_path):
    config = tmp_path / "workspace.json"
    config.write_text("{bad json")
    with pytest.raises(ValueError, match="原文件已保留"):
        Workspace(tmp_path)
    assert config.read_text() == "{bad json"


def test_invalid_shortcut_enabled_state_is_rejected_without_changing_config(tmp_path):
    workspace = Workspace(tmp_path)
    config = tmp_path / "workspace.json"
    data = json.loads(config.read_text(encoding="utf-8"))
    data["rows"][0].pop("flags")
    data["rows"][0]["enabled"] = "false"
    config.write_text(json.dumps(data), encoding="utf-8")
    original = config.read_bytes()
    with pytest.raises(ValueError, match="启用状态必须为布尔值"):
        Workspace(tmp_path)
    assert config.read_bytes() == original


def test_flags_migrate_legacy_booleans_and_override_them_when_present(tmp_path):
    Workspace(tmp_path)
    config = tmp_path / "workspace.json"
    data = json.loads(config.read_text(encoding="utf-8"))
    for row in data["rows"]:
        row.pop("flags")
    data["rows"][0]["enabled"] = True
    data["rows"][1]["enabled"] = False
    data["rows"][3].update(enabled=True, flags=0xA5)
    config.write_text(json.dumps(data), encoding="utf-8")
    restored = Workspace(tmp_path)
    assert [row.flags for row in restored.rows] == [7, 0, 7, 0xA5]
    stored = json.loads(config.read_text(encoding="utf-8"))
    assert all("enabled" not in row for row in stored["rows"])
    assert [row["flags"] for row in stored["rows"]] == [7, 0, 7, 0xA5]


@pytest.mark.parametrize("flags", [True, -1, 256, "7", None])
def test_flags_reject_invalid_values_without_modifying_config(tmp_path, flags):
    Workspace(tmp_path)
    config = tmp_path / "workspace.json"
    data = json.loads(config.read_text(encoding="utf-8"))
    data["rows"][0]["flags"] = flags
    config.write_text(json.dumps(data), encoding="utf-8")
    original = config.read_bytes()
    with pytest.raises(ValueError, match="flags 必须为"):
        Workspace(tmp_path)
    assert config.read_bytes() == original
