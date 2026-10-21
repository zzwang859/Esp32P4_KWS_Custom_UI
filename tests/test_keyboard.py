import json

import pytest

from kws_custom_ui.keyboard import keyboard_report
from kws_custom_ui.model import Workspace


@pytest.mark.parametrize("text, expected", [
    ("a", "00 00 04 00 00 00 00 00"),
    ("B", "00 00 05 00 00 00 00 00"),
    ("z", "00 00 1d 00 00 00 00 00"),
    ("1", "00 00 1e 00 00 00 00 00"),
    ("0", "00 00 27 00 00 00 00 00"),
    ("{F12}", "00 00 45 00 00 00 00 00"),
    ("{F13}", "00 00 68 00 00 00 00 00"),
    ("{F24}", "00 00 73 00 00 00 00 00"),
    ("{CTRL}+a", "01 00 04 00 00 00 00 00"),
    ("{ctrl} + A", "01 00 04 00 00 00 00 00"),
    ("{CTRL}+{ALT}+{DELETE}", "05 00 4c 00 00 00 00 00"),
    ("{RCTRL}+{RSHIFT}+{RALT}+{RGUI}+{ENTER}", "f0 00 28 00 00 00 00 00"),
    ("{WIN}+{SHIFT}+s", "0a 00 16 00 00 00 00 00"),
    ("{CTRL}", "01 00 00 00 00 00 00 00"),
    ("{+}", "02 00 2e 00 00 00 00 00"),
    ("{CTRL}+{+}", "03 00 2e 00 00 00 00 00"),
    ("!", "02 00 1e 00 00 00 00 00"),
    ("/", "00 00 38 00 00 00 00 00"),
    ("{KP_ADD}", "00 00 57 00 00 00 00 00"),
    ("{SPACE}+{TAB}", "00 00 2b 2c 00 00 00 00"),
    ("a+b+c+d+e+f", "00 00 04 05 06 07 08 09"),
    ("{CTRL}+c+A+a+{LCTRL}", "01 00 04 06 00 00 00 00"),
])
def test_usb_keyboard_reports(text, expected):
    assert keyboard_report(text) == bytes.fromhex(expected)


@pytest.mark.parametrize("text", [
    "+", "a+", "+a", "a++b", "a +   ", "{}", "{CTRL", "CTRL}",
    "{F25}", "{NOSUCHKEY}", "{音量}", "中文", "ab", "a b c",
    "CTRL+a", "{CTRL}a", "{CTRL}{F12}", "{{CTRL}}", "a+b+c+d+e+f+g",
])
def test_invalid_or_unrepresentable_chords_are_rejected(text):
    with pytest.raises(ValueError):
        keyboard_report(text)


@pytest.mark.parametrize("text", ["", "  \t "])
def test_empty_text_is_unbound(text):
    assert keyboard_report(text) is None


def test_workspace_restores_chords_and_recomputes_reports_instead_of_trusting_json(tmp_path):
    workspace = Workspace(tmp_path)
    row = workspace.rows[1]
    row.shortcut = "{ctrl}+A"
    row.text = "打开台灯"
    workspace.save()
    config = tmp_path / "workspace.json"
    data = json.loads(config.read_text(encoding="utf-8"))
    stored = data["rows"][1]
    assert stored["directory"] == row.directory
    assert stored["text"] == "打开台灯"
    assert stored["shortcut"] == "{ctrl}+A"
    assert stored["hid_report"] == [1, 0, 4, 0, 0, 0, 0, 0]
    assert "hid_release_report" not in stored
    stored["hid_report"] = [255] * 8
    # Older workspaces may still include the redundant release report.
    stored["hid_release_report"] = [0] * 8
    config.write_text(json.dumps(data), encoding="utf-8")
    restored = Workspace(tmp_path)
    assert restored.rows[1].shortcut == "{ctrl}+A"
    data = json.loads(config.read_text(encoding="utf-8"))
    assert data["rows"][1]["hid_report"] == [1, 0, 4, 0, 0, 0, 0, 0]
    assert "hid_release_report" not in data["rows"][1]
    assert data["rows"][0]["hid_report"] is None


def test_invalid_draft_clears_previous_report_and_survives_reopen(tmp_path):
    workspace = Workspace(tmp_path)
    row = workspace.rows[0]
    row.shortcut = "{CTRL}+a"
    workspace.save()
    row.shortcut = "{CTRL}+"
    workspace.save()
    data = json.loads((tmp_path / "workspace.json").read_text(encoding="utf-8"))
    assert data["rows"][0]["hid_report"] is None
    assert "hid_release_report" not in data["rows"][0]
    assert Workspace(tmp_path).rows[0].shortcut == "{CTRL}+"


def test_legacy_config_without_shortcuts_opens_without_changing_labels_or_audio(tmp_path):
    workspace = Workspace(tmp_path)
    config = tmp_path / "workspace.json"
    data = json.loads(config.read_text(encoding="utf-8"))
    for row in data["rows"]:
        for key in ("shortcut", "hid_report"):
            row.pop(key)
    data["rows"][0]["text"] = "旧工作区中已有的超过十六个字符的完整关键词文本"
    config.write_text(json.dumps(data), encoding="utf-8")
    audio = workspace.row_dir(workspace.rows[0]) / "keep.wav"
    audio.write_bytes(b"original audio")
    restored = Workspace(tmp_path)
    assert all(row.shortcut == "" for row in restored.rows)
    assert restored.rows[0].text == data["rows"][0]["text"]
    assert audio.read_bytes() == b"original audio"


def test_malformed_shortcut_type_preserves_original_config(tmp_path):
    Workspace(tmp_path)
    config = tmp_path / "workspace.json"
    data = json.loads(config.read_text(encoding="utf-8"))
    data["rows"][0]["shortcut"] = [1, 2, 3]
    config.write_text(json.dumps(data), encoding="utf-8")
    original = config.read_bytes()
    with pytest.raises(ValueError, match="组合键必须为文本"):
        Workspace(tmp_path)
    assert config.read_bytes() == original
