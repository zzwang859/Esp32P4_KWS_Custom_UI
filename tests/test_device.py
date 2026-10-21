"""Exercise the host against a stateful HID device, without changing a board."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame as pg
import pytest

from kws_custom_ui.app import App, RED
from kws_custom_ui.device import (DeviceController, CompatibilityError, PROFILE, FEATURE_INPUT,
                                  STATE_INPUTS, STATE_OUTPUTS, STATE_SHAPES, deployment_package, row_binding)
from kws_custom_ui.hid_protocol import *
from kws_custom_ui.keyboard import keyboard_report, shortcut_from_report
from kws_custom_ui.model import Workspace


def snapshot():
    version = dict(boot_id=1, model_epoch=2, config_revision=3)
    package = "ab" * 32
    classes = [dict(class_index=index, text=text) for index, text in enumerate(("silence", "unknown", "开灯", "关闭"))]
    bindings = [dict(class_index=index, actions_enabled=False,
                     led=dict(enabled=False, effect="flash", rgb=[index, 20, 30], duration_ms=678),
                     keyboard=dict(enabled=False, modifier=1, keycodes=[4, 0, 0, 0, 0, 0], hold_ms=31))
                for index in (2, 3)]
    return dict(info=dict(**version, device_id="TEST", protocol_version=1, firmware_version="test"),
                caps=dict(**version, frontend_profile=PROFILE, max_class_count=34, max_model_bytes=960 * 1024,
                          max_manifest_bytes=8192, supports_runtime_kws_filter=True),
                status=dict(**version, active_model=dict(package_sha256=package),
                            runtime=dict(save_pending=False, config_dirty=False),
                            kws_filter=dict(revision=1, keyword_threshold_percent=80, cooldown_ms=500,
                                            stable_frames=3, need_vad_checked=False)),
                classes=dict(**version, package_sha256=package, classes=classes),
                bindings=dict(**version, package_sha256=package, bindings=bindings))


class FakeHID:
    def __init__(self):
        self.state = snapshot()
        self.writes, self.responses, self.replay = [], [], {}
        self.closed = False
        self.drop_once = False
        self.fail = False
        self.on_write = lambda command: None
        self.token = 8
        self.transaction = 0
        self.objects = {}
        self.save_statuses = []

    def close(self):
        self.closed = True

    def read(self, size, timeout):
        if self.fail:
            raise OSError("USB unplugged")
        return self.responses.pop(0) if self.responses else []

    def write(self, wire):
        assert wire[0] == 0 and len(wire) == 65
        request = decode(wire[1:])
        assert request["kind"] == 0 and request["status"] == 0
        self.writes.append(bytes(wire))
        sequence, command = request["sequence"], request["command"]
        if sequence in self.replay:
            previous, response = self.replay[sequence]
            assert previous == wire
        else:
            response = self.execute(request)
            self.replay[sequence] = (bytes(wire), response)
        self.on_write(command)
        if self.drop_once:
            self.drop_once = False
        else:
            self.responses.append(response)
        return 65

    def execute(self, request):
        cmd, payload = request["command"], request["payload"]
        transaction, offset = request["transaction"], request["offset"]
        response, status = b"", 0
        if cmd == OPEN_SESSION:
            response = payload
        elif cmd == READ_BEGIN:
            object_id, = struct.unpack("<H", payload)
            name = next(name for name, value in OBJECTS.items() if value == object_id)
            if name == "status" and self.save_statuses:
                self.state["status"]["runtime"].update(self.save_statuses.pop(0))
            version = self.state["status"]
            data = deepcopy(self.state[name])
            data.update({key: version[key] for key in ("boot_id", "model_epoch", "config_revision")})
            self.data = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.token += 1
            transaction = self.token
            response = struct.pack("<IIII", len(self.data), version["boot_id"], version["model_epoch"], version["config_revision"])
        elif cmd == READ_CHUNK:
            assert transaction == self.token
            response = self.data[offset:offset + 40]
        elif cmd == READ_END:
            assert transaction == self.token
        elif cmd == SET_BINDING:
            assert len(payload) == 29
            values = struct.unpack("<IIIHB3BB6BHH", payload)
            current = self.state["status"]
            if values[:2] != (current["boot_id"], current["model_epoch"]):
                status = 16
            elif values[2] != current["config_revision"]:
                status = 14
            else:
                binding = next(item for item in self.state["bindings"]["bindings"] if item["class_index"] == values[3])
                flags = values[4]
                binding.update(actions_enabled=bool(flags & 1))
                binding["led"].update(enabled=bool(flags & 2), rgb=list(values[5:8]), duration_ms=values[-2])
                binding["keyboard"].update(enabled=bool(flags & 4), modifier=values[8], keycodes=list(values[9:15]), hold_ms=values[-1])
                current["config_revision"] += 1
                response = struct.pack("<I", current["config_revision"])
        elif cmd == SAVE_CONFIG:
            assert len(payload) == 12
            self.state["status"]["runtime"].update(save_pending=False, config_dirty=False)
            status = 1
        elif cmd == SET_KWS_FILTER:
            assert len(payload) == 17 and not transaction and not offset
            boot, revision, threshold, cooldown, stable, vad = struct.unpack("<IIHIHB", payload)
            current = self.state["status"]
            if boot != current["boot_id"]:
                status = 16
            elif revision != current["kws_filter"]["revision"]:
                status = 14
            elif not (0 <= threshold <= 100 and 1 <= cooldown <= 60000 and
                      1 <= stable <= 1000 and vad in (0, 1)):
                status = 7
            else:
                settings = dict(keyword_threshold_percent=threshold, cooldown_ms=cooldown,
                                stable_frames=stable, need_vad_checked=bool(vad))
                if settings != device_filter_values(current["kws_filter"]):
                    current["kws_filter"].update(settings)
                    current["kws_filter"]["revision"] += 1
                response = struct.pack("<I", current["kws_filter"]["revision"])
        elif cmd == UPDATE_BEGIN:
            self.transaction = transaction = 42
            response = bytes([1])
        elif cmd == OBJECT_BEGIN:
            assert transaction == self.transaction
            self.object_id, self.object_size = struct.unpack_from("<HI", payload)
            self.object_sha = payload[6:]
            self.object_data = bytearray()
        elif cmd == OBJECT_DATA:
            assert transaction == self.transaction and offset == len(self.object_data)
            self.object_data.extend(payload)
            response = struct.pack("<I", len(self.object_data))
        elif cmd == OBJECT_END:
            assert len(self.object_data) == self.object_size
            assert hashlib.sha256(self.object_data).digest() == self.object_sha
            self.objects[self.object_id] = bytes(self.object_data)
        elif cmd == UPDATE_COMMIT:
            assert set(self.objects) == {0x100, 0x101}
        elif cmd == UPDATE_ABORT:
            self.transaction = 0
        elif cmd == ACTIVATE:
            assert payload == bytes([1])
            manifest = json.loads(self.objects[0x100])
            assert manifest["classes"] == self.state["classes"]["classes"]
            package = hashlib.sha256(self.objects[0x101] + self.objects[0x100]).hexdigest()
            self.state["classes"]["package_sha256"] = self.state["bindings"]["package_sha256"] = package
            self.state["status"]["active_model"]["package_sha256"] = package
            self.state["status"]["model_epoch"] += 1
        else:
            raise AssertionError(cmd)
        return encode(cmd, request["sequence"], response, transaction, offset, kind=1, status=status)


def commands(device):
    return [decode(wire[1:])["command"] for wire in device.writes]


def finish(controller):
    controller.thread.join(timeout=5)
    assert not controller.busy
    return [controller.results.get_nowait() for _ in range(controller.results.qsize())]


def test_codec_crc_and_replay_identical_frame():
    wire = encode(SET_BINDING, 65535, b"hello", transaction=123, offset=78)
    assert decode(wire)["payload"] == b"hello"
    corrupt = bytearray(wire)
    corrupt[21] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        decode(corrupt)
    device = FakeHID()
    device.drop_once = True
    client = Client(device)
    client.request(OPEN_SESSION, client.session, timeout=0.002)
    assert device.writes[0] == device.writes[1]


def test_snapshot_reads_utf8_without_writing_device_configuration():
    device = FakeHID()
    result = Client(device).read_snapshot()
    assert result["classes"]["classes"][2]["text"] == "开灯"
    assert set(commands(device)) == {OPEN_SESSION, READ_BEGIN, READ_CHUNK, READ_END}


def test_kws_filter_command_uses_device_revision_and_reads_back_without_save_config():
    device = FakeHID()
    client = Client(device)
    current = client.read_object("status")
    desired = dict(keyword_threshold_percent=61, cooldown_ms=1000, stable_frames=7,
                   need_vad_checked=True)
    updated = client.set_kws_filter(desired, current)
    command = next(decode(wire[1:]) for wire in device.writes
                   if decode(wire[1:])["command"] == SET_KWS_FILTER)
    assert command["payload"] == struct.pack("<IIHIHB", 1, 1, 61, 1000, 7, 1)
    assert updated["kws_filter"]["revision"] == 2
    assert device_filter_values(updated["kws_filter"]) == desired
    assert SAVE_CONFIG not in commands(device)


def test_pending_save_times_out_without_claiming_completion(monkeypatch):
    client = Client(FakeHID())
    current = snapshot()["status"]
    pending = deepcopy(current)
    pending["runtime"].update(save_pending=True, config_dirty=True)
    monkeypatch.setattr(client, "request", lambda *args, **kwargs: dict(status=1))
    monkeypatch.setattr(client, "read_object", lambda name: pending)
    with pytest.raises(TimeoutError, match="未确认保存完成"):
        client.save(current, timeout=0.001)


@pytest.mark.parametrize("field,error", [("boot_id", "MODEL_CHANGED"),
                                         ("model_epoch", "MODEL_CHANGED"),
                                         ("config_revision", "CONFLICT")])
def test_save_completion_must_match_requested_version(monkeypatch, field, error):
    client = Client(FakeHID())
    current = snapshot()["status"]
    changed = deepcopy(current)
    changed[field] += 1
    monkeypatch.setattr(client, "request", lambda *args, **kwargs: dict(status=1))
    monkeypatch.setattr(client, "read_object", lambda name: changed)
    with pytest.raises(ProtocolError, match=error):
        client.save(current)


def test_download_cancel_aborts_transaction():
    device = FakeHID()
    cancelled = [False]
    device.on_write = lambda cmd: cancelled.__setitem__(0, True) if cmd == OBJECT_DATA else None
    client = Client(device, lambda: cancelled[0])
    with pytest.raises(DeviceCancelled):
        client.download(b"manifest", b"model")
    assert UPDATE_ABORT in commands(device) and device.transaction == 0
    assert ACTIVATE not in commands(device)


@pytest.mark.parametrize("usage", range(4, 0xE0))
def test_all_device_keyboard_usages_roundtrip(usage):
    report = bytes([0xA5, 0, usage, 0, 0, 0, 0, 0])
    assert keyboard_report(shortcut_from_report(report)) == report


def package_workspace(tmp_path, compatible=True):
    workspace = Workspace(tmp_path / "local")
    for row in list(workspace.rows)[2:]:
        workspace.delete(row)
    for row, text in zip(workspace.rows, ("开灯", "关闭")):
        row.text, row.shortcut = text, "{CTRL}+a"
    workspace.save()
    bundle = workspace.path / "esp32p4-dl/run"
    (bundle / "models").mkdir(parents=True)
    model = b"EDL2" + bytes(4) + struct.pack("<I", 16) + bytes(20)
    (bundle / "model.espdl").write_bytes(model)
    root = bundle.parent
    (root / "latest.json").write_text(json.dumps(dict(success=True, selected="w16a16", model="run/model.espdl", bundle="run",
        labels=["silence", "unknown", "开灯", "关闭"], models=dict(w16a16=dict(sha256=hashlib.sha256(model).hexdigest())))), encoding="utf-8")
    (bundle / "models/classes.json").write_text(json.dumps([
        dict(index=0, label="silence"), dict(index=1, label="unknown"),
        *[dict(index=index, label=row.text, row_id=row.id, directory=str(workspace.row_dir(row)))
          for index, row in enumerate(workspace.rows, 2)]]), encoding="utf-8")
    (bundle / "models/io_mapping.json").write_text(json.dumps(dict(
        state_pairs=list(zip(STATE_INPUTS, STATE_OUTPUTS)) if compatible else [["running_sum", "next_running_sum"]],
        inputs={FEATURE_INPUT: [1, 1, 40], **dict(zip(STATE_INPUTS, STATE_SHAPES))},
        outputs={"onnx_logits:0": [1, 4], **dict(zip(STATE_OUTPUTS, STATE_SHAPES))})), encoding="utf-8")
    asset = Path(__file__).resolve().parents[1] / "src/ds-tc-resnet/data/esp32p4_frontend.npz"
    (bundle / "models/frontend.npz").write_bytes(asset.read_bytes())
    (bundle / "models/frontend.json").write_text(json.dumps(dict(frontend_profile=PROFILE,
        frontend_asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest())), encoding="utf-8")
    return workspace


def test_deploy_upload_activate_then_complete_bindings_and_save(tmp_path):
    workspace = package_workspace(tmp_path)
    workspace.kws_filter.update(keyword_threshold_percent=70, cooldown_ms=800,
                                stable_frames=5, need_vad_checked=True)
    workspace.save()
    device = FakeHID()
    controller = DeviceController(lambda: device)
    try:
        assert controller.submit("deploy", workspace)
        result = finish(controller)[-1]
        assert result["kind"] == "deploy", result
        sent = commands(device)
        assert sent.index(UPDATE_COMMIT) < sent.index(ACTIVATE) < sent.index(SET_BINDING) < sent.index(SAVE_CONFIG) < sent.index(SET_KWS_FILTER)
        assert all(item["actions_enabled"] and item["keyboard"]["enabled"] for item in result["snapshot"]["bindings"]["bindings"])
        assert device.state["status"]["kws_filter"]["keyword_threshold_percent"] == 70
        assert device.state["status"]["kws_filter"]["need_vad_checked"] is True
    finally:
        controller.close()
    assert device.closed


def test_incompatible_model_is_rejected_before_opening_or_writing_hid(tmp_path):
    workspace = package_workspace(tmp_path, compatible=False)
    opened = []
    controller = DeviceController(lambda: opened.append(True))
    try:
        controller.submit("deploy", workspace)
        result = finish(controller)[-1]
        assert result["kind"] == "error" and "6 对状态" in result["message"]
        assert opened == []
    finally:
        controller.close()


def test_pc_only_zero_filter_value_blocks_deploy_before_hid_open(tmp_path):
    workspace = package_workspace(tmp_path)
    workspace.kws_filter["cooldown_ms"] = 0
    opened = []
    controller = DeviceController(lambda: opened.append(True))
    try:
        controller.submit("deploy", workspace)
        result = finish(controller)[-1]
        assert result["kind"] == "error" and "0 ms" in result["message"]
        assert not opened
    finally:
        controller.close()


def test_revision_conflict_refreshes_device_instead_of_overwriting_changes():
    device = FakeHID()
    controller = DeviceController(lambda: device)
    try:
        controller.submit("connect")
        finish(controller)
        device.state["status"]["config_revision"] += 1
        desired = deepcopy(controller.snapshot["bindings"]["bindings"][:1])
        desired[0]["actions_enabled"] = True
        controller.submit("bindings", desired)
        result = finish(controller)[-1]
        assert result["kind"] == "error" and "CONFLICT" in result["message"]
        assert not result["snapshot"]["bindings"]["bindings"][0]["actions_enabled"]
        assert SAVE_CONFIG not in commands(device) and controller.connected
    finally:
        controller.close()


def test_kws_filter_revision_conflict_refreshes_device_value():
    device = FakeHID()
    controller = DeviceController(lambda: device)
    try:
        controller.submit("connect")
        finish(controller)
        device.state["status"]["kws_filter"].update(revision=2, keyword_threshold_percent=55)
        desired = dict(keyword_threshold_percent=70, cooldown_ms=500, stable_frames=3,
                       need_vad_checked=False)
        controller.submit("filter", desired)
        result = finish(controller)[-1]
        assert result["kind"] == "error" and "CONFLICT" in result["message"]
        assert result["snapshot"]["status"]["kws_filter"]["keyword_threshold_percent"] == 55
        assert device.state["status"]["kws_filter"]["keyword_threshold_percent"] == 55
    finally:
        controller.close()


@pytest.fixture
def device_app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = App(tmp_path / "local", session_file=tmp_path / "session.json")
    devices = []
    def opener():
        device = FakeHID()
        devices.append(device)
        return device
    app.hid = DeviceController(opener)
    app.update()
    app.draw()
    yield app, devices
    app.hid.close()
    app.trainer.close()
    app.recorder.close()
    pg.quit()


def connect_app(app):
    app.set_mode("device")
    app.hid.thread.join(timeout=5)
    app.update()
    app.draw()
    assert app.connection_state == "connected"


def test_device_workspace_and_connection_cards_share_two_to_one_row(device_app, monkeypatch):
    app, _ = device_app
    connect_app(app)
    original_panel = app.panel
    panels = []

    def capture(rect, *args, **kwargs):
        panels.append(pg.Rect(rect))
        return original_panel(rect, *args, **kwargs)

    monkeypatch.setattr(app, "panel", capture)
    for width in (1100, 1280):
        app.screen = pg.display.set_mode((width, 740), pg.RESIZABLE)
        panels.clear()
        app.draw()
        cards = [rect for rect in panels if rect.y == 110 and rect.h == 106]
        assert len(cards) == 2
        workspace, status = cards
        assert workspace.x == 32 and status.right == width - 32
        assert status.x - workspace.right == 12
        assert abs(workspace.w - 2 * status.w) <= 2
        assert app.viewport.y == 452 and app.viewport.bottom < 740 - 110


def test_modes_isolate_workspaces_hide_local_actions_and_restore_local_memory(device_app, monkeypatch):
    app, devices = device_app
    local = app.workspace.path
    before = (local / "workspace.json").read_bytes()
    connect_app(app)
    assert app.workspace.path == Path.cwd() / "workspace/device"
    assert [row.text for row in app.workspace.rows] == ["开灯", "关闭"]
    assert app.workspace.rows[0].color == (2, 20, 30)
    assert app.workspace.rows[0].shortcut == "{CTRL}+a"
    assert all(row.flags == 0 for row in app.workspace.rows)
    assert not any(callback in (app.record, app.start_training, app.import_model, app.choose_device, app.add_row) for _, callback in app.buttons)
    app.record()
    app.record(app.workspace.rows[0])
    app.record(noise=True)
    app.start_training()
    app.launch_training(10)
    app.import_model()
    app.add_row()
    app.choose_directory()
    assert not app.recorder.busy and not app.trainer.busy and len(app.workspace.rows) == 2
    app.remember_workspace()
    assert json.loads(app.session_file.read_text())["last_workspace"] == str(local)
    assert set(commands(devices[0])) == {OPEN_SESSION, READ_BEGIN, READ_CHUNK, READ_END}
    app.set_mode("local")
    assert app.workspace.path == local and (local / "workspace.json").read_bytes() == before
    assert devices[0].closed


def test_device_edits_send_bindings_preserve_timing_and_save_after_ack(device_app):
    app, devices = device_app
    connect_app(app)
    app.toggle_all_shortcuts()
    app.hid.thread.join(timeout=5)
    app.update()
    assert [row.flags for row in app.workspace.rows] == [1, 1]
    app.toggle_shortcut(app.workspace.rows[0])
    app.hid.thread.join(timeout=5)
    app.update()
    row = app.workspace.rows[0]
    assert row.flags == 7
    app.shortcut_fields[row.id].value = "{CTRL}+{BACKSPACE}"
    row.color = (10, 11, 12)
    app.save()
    app.hid.thread.join(timeout=5)
    app.update()
    binding = app.hid.snapshot["bindings"]["bindings"][0]
    assert binding["led"]["rgb"] == [10, 11, 12] and binding["led"]["duration_ms"] == 678
    assert binding["keyboard"]["keycodes"][0] == 0x2A and binding["keyboard"]["hold_ms"] == 31
    assert SAVE_CONFIG in commands(devices[0])
    assert app.notice == "写入完成" and not app.notice_error
    data = json.loads((app.workspace.path / "workspace.json").read_text(encoding="utf-8"))
    assert data["rows"][0]["flags"] == 7 and data["rows"][0]["hid_report"] == [1, 0, 42, 0, 0, 0, 0, 0]


def test_device_filter_controls_read_write_and_poll_without_changing_local_workspace(device_app):
    app, devices = device_app
    local = app.workspace.path
    local_config = (local / "workspace.json").read_bytes()
    connect_app(app)
    assert app.workspace.kws_filter["keyword_threshold_percent"] == 80
    track = app.device_filter_rects["keyword_threshold_percent"]
    position = (track.x + track.w // 2, track.centery)
    app.event(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=position))
    app.event(pg.event.Event(pg.MOUSEBUTTONUP, button=1, pos=position))
    app.hid.thread.join(timeout=5)
    app.update()
    assert devices[0].state["status"]["kws_filter"]["keyword_threshold_percent"] == 50
    assert app.workspace.kws_filter["keyword_threshold_percent"] == 50
    assert SET_KWS_FILTER in commands(devices[0])
    assert SAVE_CONFIG not in commands(devices[0])
    app.toggle_device_vad()
    app.hid.thread.join(timeout=5)
    app.update()
    assert app.workspace.kws_filter["need_vad_checked"] is True
    devices[0].state["status"]["kws_filter"].update(revision=4, cooldown_ms=1500)
    app.hid.submit("poll")
    app.hid.thread.join(timeout=5)
    app.update()
    assert app.workspace.kws_filter["cooldown_ms"] == 1500
    assert Workspace(app.workspace.path).kws_filter["cooldown_ms"] == 1500
    assert (local / "workspace.json").read_bytes() == local_config


def test_device_filter_mirror_accepts_full_firmware_ranges(device_app):
    app, devices = device_app
    original = app.hid.opener
    def opener():
        device = original()
        device.state["status"]["kws_filter"].update(cooldown_ms=12345, stable_frames=789)
        return device
    app.hid.opener = opener
    connect_app(app)
    assert app.workspace.kws_filter["cooldown_ms"] == 12345
    assert app.workspace.kws_filter["stable_frames"] == 789
    assert Workspace(app.workspace.path).kws_filter == app.workspace.kws_filter


def test_pending_shortcut_edit_is_saved_before_device_filter_change(device_app):
    app, devices = device_app
    connect_app(app)
    row = app.workspace.rows[0]
    app.shortcut_fields[row.id].value = "b"
    app.save_due = 1
    desired = dict(app.workspace.kws_filter, keyword_threshold_percent=65)
    app.send_device_filter(desired)
    app.hid.thread.join(timeout=5)
    app.update()
    app.hid.thread.join(timeout=5)
    app.update()
    sent = commands(devices[0])
    assert sent.index(SET_BINDING) < sent.index(SAVE_CONFIG) < sent.index(SET_KWS_FILTER)
    assert app.workspace.kws_filter["keyword_threshold_percent"] == 65
    assert app.workspace.rows[0].shortcut == "b"


def test_device_write_waits_for_actual_save_completion(device_app):
    app, devices = device_app
    connect_app(app)
    devices[0].save_statuses = [dict(save_pending=True, config_dirty=True),
                                dict(save_pending=False, config_dirty=False)]
    row = app.workspace.rows[0]
    app.shortcut_fields[row.id].value = "b"
    app.save()
    app.hid.thread.join(timeout=5)
    app.update()
    assert not devices[0].save_statuses
    assert app.notice == "写入完成" and not app.notice_error


def test_device_save_failure_is_not_reported_as_completed(device_app):
    app, devices = device_app
    connect_app(app)
    devices[0].save_statuses = [dict(save_pending=False, config_dirty=True)]
    app.shortcut_fields[app.workspace.rows[0].id].value = "b"
    app.save()
    app.hid.thread.join(timeout=5)
    app.update()
    assert app.notice_error and "写入失败" in app.notice


def test_device_footer_has_no_audio_idle_wait_message(device_app, monkeypatch):
    app, _ = device_app
    connect_app(app)
    app.hid.snapshot["status"]["runtime"]["save_pending"] = True
    labels = []
    original = app.text
    def capture(text, *args, **kwargs):
        labels.append(text)
        return original(text, *args, **kwargs)
    monkeypatch.setattr(app, "text", capture)
    app.draw()
    assert not any("等待" in text and "音频" in text for text in labels)


def test_device_workspace_cannot_replace_the_same_local_workspace(device_app):
    app, devices = device_app
    app.device_workspace_path = app.workspace.path
    app.set_mode("device")
    assert app.mode == "local" and not devices
    assert "独立" in app.notice and len(app.workspace.rows) == 4


def test_exit_flushes_device_edit_then_automatically_closes(device_app):
    app, devices = device_app
    connect_app(app)
    row = app.workspace.rows[0]
    app.shortcut_fields[row.id].value = "b"
    app.save_due = 1
    app.exit()
    if app.hid.thread is not None:
        app.hid.thread.join(timeout=5)
    app.update()
    assert not app.running and devices[0].closed
    assert devices[0].state["bindings"]["bindings"][0]["keyboard"]["keycodes"][0] == 5
    assert json.loads((app.workspace.path / "workspace.json").read_text(encoding="utf-8"))["rows"][0]["shortcut"] == "b"


def test_connection_failure_is_red_retryable_and_disconnection_detected(device_app, monkeypatch):
    app, devices = device_app
    def missing():
        raise OSError("未找到设备")
    opener = app.hid.opener
    app.hid.opener = missing
    app.set_mode("device")
    app.hid.thread.join(timeout=5)
    app.update()
    assert app.connection_state == "failed"
    labels = []
    original = app.text
    def record(text, pos, size=16, color=None, *args, **kwargs):
        labels.append((text, color))
        if color is None:
            return original(text, pos, size, *args, **kwargs)
        return original(text, pos, size, color, *args, **kwargs)
    monkeypatch.setattr(app, "text", record)
    app.draw()
    assert ("连接失败", RED) in labels
    app.hid.opener = opener
    app.set_mode("device")  # Clicking the failed switch retries.
    app.hid.thread.join(timeout=5)
    app.update()
    assert app.connection_state == "connected"
    devices[-1].fail = True
    app.hid.submit("poll")
    app.hid.thread.join(timeout=5)
    app.update()
    assert app.connection_state == "failed" and devices[-1].closed
