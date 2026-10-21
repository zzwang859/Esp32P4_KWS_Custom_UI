"""Configurable HID discovery, without opening a real device."""
import json
import sys
from types import SimpleNamespace

import pytest

from kws_custom_ui.device import open_device
from kws_custom_ui.settings import hid_search_settings, recording_durations


def test_missing_settings_keep_default_search(tmp_path):
    assert hid_search_settings(tmp_path / "missing.json") == dict(
        vid=0x303A, pid=0x8003, usage_page=0xFF00, usage=1)


def test_recording_durations_default_and_custom_values(tmp_path):
    session = tmp_path / "session.json"
    assert recording_durations(session) == {"silence_seconds": 60, "noise_seconds": 3}
    session.write_text(json.dumps({"recording": {"silence_seconds": 75, "noise_seconds": 8}}), encoding="utf-8")
    assert recording_durations(session) == {"silence_seconds": 75, "noise_seconds": 8}


@pytest.mark.parametrize("recording", [None, [], {"silence_seconds": 0},
                                       {"noise_seconds": -1}, {"noise_seconds": 1.5},
                                       {"silence_seconds": True}, {"noise_seconds": "3"}])
def test_invalid_recording_duration_is_rejected(tmp_path, recording):
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"recording": recording}), encoding="utf-8")
    with pytest.raises(ValueError, match="recording"):
        recording_durations(session)


def test_discovery_uses_all_configured_fields_and_rereads_on_connect(tmp_path, monkeypatch):
    session = tmp_path / "session.json"
    calls, paths = [], []
    device = SimpleNamespace(open_path=paths.append, close=lambda: None)
    candidates = [dict(path=b"other-page", usage_page=0xFF00, usage=2),
                  dict(path=b"other-usage", usage_page=0xFF10, usage=1),
                  dict(path=b"match", usage_page=0xFF10, usage=2)]
    def enumerate_hid(vid, pid):
        calls.append((vid, pid))
        return candidates
    monkeypatch.setitem(sys.modules, "hid", SimpleNamespace(enumerate=enumerate_hid, device=lambda: device))
    config = dict(vid="0x1234", pid=22136, usage_page="0xff10", usage="2")
    session.write_text(json.dumps(dict(hid_device=config)), encoding="utf-8-sig")
    assert open_device(session) is device
    config["pid"] = "0x5679"
    session.write_text(json.dumps(dict(hid_device=config)), encoding="utf-8")
    assert open_device(session) is device
    assert calls == [(0x1234, 0x5678), (0x1234, 0x5679)]
    assert paths == [b"match", b"match"]


@pytest.mark.parametrize("config", [None, [], dict(vid=True), dict(pid=0), dict(vid=-1),
                                    dict(pid=65536), dict(usage=1.5), dict(usage_page="FF00")])
def test_invalid_search_is_rejected_before_enumeration(tmp_path, monkeypatch, config):
    session = tmp_path / "session.json"
    session.write_text(json.dumps(dict(hid_device=config)), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "hid", SimpleNamespace())
    with pytest.raises(ValueError, match="hid_device"):
        open_device(session)


@pytest.mark.parametrize("contents", ["{broken", "[]"])
def test_invalid_config_file_is_reported(tmp_path, contents):
    session = tmp_path / "session.json"
    session.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="无法读取配置文件"):
        hid_search_settings(session)


def test_device_not_found_reports_configured_ids(tmp_path, monkeypatch):
    session = tmp_path / "session.json"
    session.write_text(json.dumps(dict(hid_device=dict(vid="0x1234", pid="0x5678"))), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "hid", SimpleNamespace(enumerate=lambda vid, pid: []))
    with pytest.raises(OSError, match="1234:5678"):
        open_device(session)
