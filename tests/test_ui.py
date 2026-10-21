import json
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame as pg
import pytest

from kws_custom_ui.app import App
from kws_custom_ui.model import Workspace


@pytest.fixture
def app(tmp_path):
    instance = App(tmp_path / "workspace", session_file=tmp_path / "session.json")
    instance.update()
    instance.draw()
    yield instance
    instance.recorder.close()
    pg.quit()


def click(app, pos):
    app.event(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=pos))
    app.event(pg.event.Event(pg.MOUSEBUTTONUP, button=1, pos=pos))
    app.draw()


def test_recording_uses_session_durations_on_next_start(app, monkeypatch):
    calls = []
    monkeypatch.setattr(app.recorder, "start", lambda directory, seconds, device, identity:
                        calls.append((seconds, identity)))
    app.session_file.write_text(json.dumps({"recording": {
        "silence_seconds": 75, "noise_seconds": 8}}), encoding="utf-8")

    app.record(noise=True)
    app.record()

    assert calls == [(8, "unknown"), (75, "silence")]
    assert app.recording == {"silence_seconds": 75, "noise_seconds": 8}
    assert "75 秒后" in app.notice
    texts, buttons = [], []
    monkeypatch.setattr(app, "text", lambda value, *args, **kwargs: texts.append(value))
    monkeypatch.setattr(app, "button", lambda label, *args, **kwargs: buttons.append(label))
    app.draw()
    assert "75 秒 · 自动保存" in texts
    assert "00:00 / 01:15" in texts
    assert "录音 8 秒" in buttons
    assert "00:00 / 00:08" in texts


def master_color_pos(app):
    return app.shortcut_master_rect.x + 10, app.shortcut_master_rect.y + 10


def first_row_color_pos(app):
    return app.screen.get_width() - 152, app.fields[app.workspace.rows[0].id].rect.top + 11


def test_default_rows_scroll_and_chinese_text_persists(app):
    assert app.max_scroll > 0
    assert len(app.row_rects) == 3
    app.scroll = app.max_scroll
    app.draw()
    assert app.workspace.rows[-1].id in [identity for _, identity in app.row_rects]
    app.scroll = 0
    app.draw()
    first = app.workspace.rows[0]
    field = app.fields[first.id]
    click(app, field.rect.center)
    app.event(pg.event.Event(pg.TEXTINPUT, text="打开台灯"))
    app.save()
    data = json.loads((app.workspace.path / "workspace.json").read_text(encoding="utf-8"))
    stored = next(row for row in data["rows"] if row["id"] == first.id)
    assert stored["directory"] == app.workspace.row_dir(first).name
    assert stored["text"] == "打开台灯"
    assert Workspace(app.workspace.path).rows[0].text == "打开台灯"


def test_shortcut_row_and_master_buttons_sync_and_restore_without_losing_binding(app):
    from kws_custom_ui.app import OFF_BG, TEAL

    first = app.workspace.rows[0]
    app.shortcut_fields[first.id].value = "{CTRL}+a"
    toggle_pos = (app.screen.get_width() - 110, app.fields[first.id].rect.centery)
    master_pos = app.shortcut_master_rect.center
    assert all(row.enabled for row in app.workspace.rows)
    click(app, toggle_pos)
    assert not first.enabled
    assert all(row.enabled for row in app.workspace.rows[1:])
    data = json.loads((app.workspace.path / "workspace.json").read_text(encoding="utf-8"))
    assert data["rows"][0]["flags"] == 0x01
    assert "enabled" not in data["rows"][0]
    assert data["rows"][0]["hid_report"] == [1, 0, 4, 0, 0, 0, 0, 0]
    # Master state uses bit0 only: disabling one row's outputs leaves it on.
    click(app, master_pos)
    assert not any(row.enabled for row in app.workspace.rows)
    assert [row.flags for row in app.workspace.rows] == [0, 6, 6, 6]
    assert app.screen.get_at(master_color_pos(app))[:3] == OFF_BG
    assert app.screen.get_at(first_row_color_pos(app))[:3] == OFF_BG
    assert not any(row.enabled for row in Workspace(app.workspace.path).rows)
    click(app, master_pos)
    assert [row.flags for row in app.workspace.rows] == [1, 7, 7, 7]
    assert app.screen.get_at(master_color_pos(app))[:3] == TEAL
    app.change_workspace()
    assert app.workspace.rows[0].enabled is False
    assert app.shortcut_fields[first.id].value == "{CTRL}+a"
    assert all(row.enabled for row in app.workspace.rows[1:])
    assert app.modal is None


def test_new_row_inherits_master_bit_and_row_button_cannot_override_it(app):
    master_pos = app.shortcut_master_rect.center
    click(app, master_pos)
    app.add_row()
    app.draw()
    added = app.workspace.rows[-1]
    assert added.flags == 0x06
    row_pos = (app.screen.get_width() - 110, app.fields[added.id].rect.centery)
    click(app, row_pos)
    assert added.flags == 0x06 and not added.enabled
    # Only the master can enable bit0, preserving each row's output switches.
    click(app, master_pos)
    assert all(row.flags == 7 for row in app.workspace.rows)
    click(app, row_pos)
    assert added.flags == 1
    click(app, master_pos)
    assert [row.flags for row in app.workspace.rows] == [6, 6, 6, 6, 0]
    click(app, master_pos)
    assert [row.flags for row in app.workspace.rows] == [7, 7, 7, 7, 1]


@pytest.mark.parametrize("flags", [0, 1, 2, 3, 4, 5, 6, 7, 0xA7])
def test_row_flag_button_changes_only_output_bits_and_color_matches_all_three(app, flags):
    from kws_custom_ui.app import OFF_BG, TEAL
    row = app.workspace.rows[0]
    row.flags = flags
    app.draw()
    color_pos = first_row_color_pos(app)
    assert app.screen.get_at(color_pos)[:3] == (TEAL if flags & 7 == 7 else OFF_BG)
    click(app, (app.screen.get_width() - 110, app.fields[row.id].rect.centery))
    assert row.flags == (flags & ~6 if flags & 7 == 7 else flags | 6)
    assert row.flags & ~6 == flags & ~6
    assert app.screen.get_at(color_pos)[:3] == (TEAL if row.flags & 7 == 7 else OFF_BG)


def test_master_uses_and_of_bit0_and_preserves_output_and_reserved_bits(app):
    from kws_custom_ui.app import OFF_BG, TEAL
    initial = [0xA7, 0xA0, 0xA3, 0xA5]
    for row, flags in zip(app.workspace.rows, initial):
        row.flags = flags
    app.draw()
    color_pos = master_color_pos(app)
    master_pos = app.shortcut_master_rect.center
    assert app.screen.get_at(color_pos)[:3] == OFF_BG
    click(app, master_pos)
    assert [row.flags for row in app.workspace.rows] == [0xA7, 0xA1, 0xA3, 0xA5]
    assert app.screen.get_at(color_pos)[:3] == TEAL
    click(app, master_pos)
    assert [row.flags for row in app.workspace.rows] == [0xA6, 0xA0, 0xA2, 0xA4]
    assert app.screen.get_at(color_pos)[:3] == OFF_BG
    assert [row.flags & ~1 for row in app.workspace.rows] == [flags & ~1 for flags in initial]


@pytest.mark.parametrize("all_rows", [False, True])
def test_shortcut_toggle_save_failure_keeps_previous_states(app, monkeypatch, all_rows):
    previous = [0xA7, 0xA0, 0xA3, 0xA5]
    for row, flags in zip(app.workspace.rows, previous):
        row.flags = flags
    app.draw()
    def fail_save():
        raise OSError("disk full")
    monkeypatch.setattr(app.workspace, "save", fail_save)
    position = app.shortcut_master_rect.center if all_rows else (
        app.screen.get_width() - 110, app.fields[app.workspace.rows[0].id].rect.centery)
    click(app, position)
    assert [row.flags for row in app.workspace.rows] == previous
    assert app.notice_error and "disk full" in app.notice


def test_keyword_limit_and_shortcut_autosave_restore(app):
    first = app.workspace.rows[0]
    keyword, shortcut = app.fields[first.id], app.shortcut_fields[first.id]
    click(app, keyword.rect.center)
    app.event(pg.event.Event(pg.TEXTINPUT, text="一二三四五六七八九十一二三四五六七八九十"))
    assert keyword.value == "一二三四五六七八九十一二三四五六"
    click(app, shortcut.rect.center)
    app.event(pg.event.Event(pg.TEXTINPUT, text="{CTRL}+{+}"))
    assert not keyword.focus and shortcut.focus
    app.save_due = 1
    app.update()
    data = json.loads((app.workspace.path / "workspace.json").read_text(encoding="utf-8"))
    assert data["rows"][0]["text"] == keyword.value
    assert data["rows"][0]["hid_report"] == [3, 0, 46, 0, 0, 0, 0, 0]
    app.change_workspace()
    assert app.shortcut_fields[first.id].value == "{CTRL}+{+}"
    assert app.fields[first.id].limit == 16


def test_invalid_shortcut_is_marked_and_saved_without_stale_report(app):
    from kws_custom_ui.app import RED

    first = app.workspace.rows[0]
    field = app.shortcut_fields[first.id]
    click(app, field.rect.center)
    app.event(pg.event.Event(pg.TEXTINPUT, text="{CTRL}+"))
    app.draw()
    assert field.validation_error()
    assert app.screen.get_at((field.rect.left, field.rect.centery))[:3] == RED
    app.save()
    assert Workspace(app.workspace.path).rows[0].shortcut == "{CTRL}+"
    data = json.loads((app.workspace.path / "workspace.json").read_text(encoding="utf-8"))
    assert data["rows"][0]["hid_report"] is None
    app.event(pg.event.Event(pg.TEXTINPUT, text="a"))
    assert not field.validation_error()


def test_tab_moves_through_keyword_and_shortcut_in_row_order(app):
    first, second = app.workspace.rows[:2]
    click(app, app.fields[first.id].rect.center)
    app.event(pg.event.Event(pg.KEYDOWN, key=pg.K_TAB, mod=0))
    assert app.shortcut_fields[first.id].focus
    app.event(pg.event.Event(pg.KEYDOWN, key=pg.K_TAB, mod=0))
    assert app.fields[second.id].focus
    app.event(pg.event.Event(pg.KEYDOWN, key=pg.K_TAB, mod=pg.KMOD_SHIFT))
    assert app.shortcut_fields[first.id].focus


@pytest.mark.parametrize("width", [1100, 1280, 1600])
def test_shortcut_and_row_controls_fit_without_overlap(app, width):
    app.event(pg.event.Event(pg.VIDEORESIZE, w=width, h=900))
    app.draw()
    first = app.workspace.rows[0]
    keyword, shortcut = app.fields[first.id], app.shortcut_fields[first.id]
    assert keyword.rect.width <= 280
    assert shortcut.rect.width >= 180
    controls = [keyword.rect, shortcut.rect] + [
        rect for rect, _ in app.buttons if rect.centery == keyword.rect.centery
    ]
    controls.sort(key=lambda rect: rect.x)
    assert all(a.right < b.left for a, b in zip(controls, controls[1:]))
    assert controls[-1].right <= width - 48


def test_shortcut_fields_follow_add_delete_and_workspace_switch(app, tmp_path):
    first = app.workspace.rows[0]
    app.shortcut_fields[first.id].value = "{F12}"
    app.add_row()
    added = app.workspace.rows[-1]
    assert app.shortcut_fields[added.id].value == ""
    app.shortcut_fields[added.id].value = "{ALT}+a"
    app.request_delete()
    app.delete_row()
    assert added.id not in app.shortcut_fields
    assert Workspace(app.workspace.path).rows[0].shortcut == "{F12}"
    previous = app.workspace.path
    app.path_field.value = str(tmp_path / "other")
    app.change_workspace()
    assert first.id not in app.shortcut_fields
    app.path_field.value = str(previous)
    app.change_workspace()
    assert app.shortcut_fields[first.id].value == "{F12}"


def test_add_scroll_select_color_delete_and_preserve_audio(app):
    for _ in range(8):
        app.add_row()
    app.draw()
    assert app.max_scroll > 0
    row = app.workspace.rows[-1]
    assert app.selected == row.id
    directory = app.workspace.row_dir(row)
    (directory / "keep.wav").write_bytes(b"audio")
    app.choose_color(row)
    app.draw()
    app.preset_color((128, 42, 190))
    app.apply_color()
    assert Workspace(app.workspace.path).rows[-1].color == (128, 42, 190)
    app.request_delete()
    app.delete_row()
    assert not directory.exists()
    archives = list(directory.parent.glob(f"delete-*-{directory.name}"))
    assert len(archives) == 1
    assert archives[0].joinpath("keep.wav").read_bytes() == b"audio"
    assert len(Workspace(app.workspace.path).rows) == 11


def test_clipped_row_controls_cannot_receive_clicks_over_header(app):
    for _ in range(4):
        app.add_row()
    app.scroll = 32
    app.draw()
    first = app.workspace.rows[0]
    field = app.fields[first.id]
    click(app, (field.rect.centerx, app.viewport.top - 2))
    assert not field.focus
    # Every row callback is clipped, including a row partially above the viewport.
    for rect, _ in app.buttons:
        if app.viewport.top - 40 < rect.y < app.viewport.top + 75:
            assert rect.top >= app.viewport.top


def test_invalid_workspace_keeps_current_project(app, tmp_path):
    previous = app.workspace.path
    app.remember_workspace()
    previous_session = app.session_file.read_bytes()
    blocked = tmp_path / "file.txt"
    blocked.write_text("not a folder")
    app.path_field.value = str(blocked)
    app.change_workspace()
    assert app.workspace.path == previous
    assert app.notice_error
    assert app.session_file.read_bytes() == previous_session


def test_minimum_window_and_placeholder_dialog(app):
    app.event(pg.event.Event(pg.VIDEORESIZE, w=1100, h=740))
    app.draw()
    app.placeholder("开始训练")
    app.draw()
    assert app.modal["type"] == "info"
    app.close_modal()
    app.choose_color(app.workspace.rows[0])
    app.draw()
    assert app.sv_rect.bottom < app.screen.get_height()


@pytest.mark.parametrize("noise,seconds", [(False, 2), (True, 3)])
def test_record_buttons_save_expected_wav_duration_and_directory(app, monkeypatch, noise, seconds):
    import wave
    from kws_custom_ui import audio

    class Input:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]
            self.active = True

        def __enter__(self):
            for _ in range(30):
                self.callback(b"\x00\x20" * 1600, 1600, None, None)
            return self

        def __exit__(self, *args):
            self.active = False

    monkeypatch.setattr(audio.sd, "check_input_settings", lambda **kwargs: None)
    monkeypatch.setattr(audio.sd, "RawInputStream", Input)
    row = app.workspace.rows[0]
    directory = app.workspace.unknown_dir if noise else app.workspace.row_dir(row)
    click(app, (402, 391) if noise else (app.screen.get_width() - 586, app.fields[row.id].rect.centery))
    app.recorder.thread.join(timeout=5)
    assert not app.recorder.busy
    assert app.recorder.row_id == ("unknown" if noise else row.id)
    assert app.recorder.duration == seconds
    files = list(directory.glob("*.wav"))
    assert len(files) == 1
    with wave.open(str(files[0]), "rb") as reader:
        assert (reader.getframerate(), reader.getnchannels(), reader.getsampwidth(), reader.getnframes()) == (16000, 1, 2, seconds * 16000)
        assert any(reader.readframes(100))
    app.update()
    assert app.counts["unknown"] == int(noise)
    assert app.counts["silence"] == 0
    assert app.counts[row.id] == int(not noise)
    assert all(app.counts[item.id] == 0 for item in app.workspace.rows[1:])
    rendered = []
    original = app.text

    def capture(text, *args, **kwargs):
        rendered.append(str(text))
        return original(text, *args, **kwargs)

    monkeypatch.setattr(app, "text", capture)
    app.draw()
    assert f"{len(app.workspace.rows)} 个关键词   /   {int(not noise)} 份录音" in rendered
    folders = []
    monkeypatch.setattr(app, "open_folder", folders.append)
    click(app, (app.screen.get_width() - 154, 391) if noise else (
        app.screen.get_width() - 214, app.fields[row.id].rect.centery))
    assert folders == [directory]


def test_noise_recording_activates_only_its_waveform_and_can_cancel(app, monkeypatch):
    monkeypatch.setattr(type(app.recorder), "busy", property(lambda recorder: True))
    app.recorder.row_id = "unknown"
    state = dict(elapsed=1.5, waveform=[0.2] * 120, phase="recording", warning="")
    monkeypatch.setattr(app.recorder, "snapshot", lambda: state)
    waves = []
    waveform = app.waveform

    def capture(rect, values, active, progress=0):
        waves.append((pg.Rect(rect), active, progress))
        return waveform(rect, values, active, progress)

    monkeypatch.setattr(app, "waveform", capture)
    cancelled = []
    monkeypatch.setattr(app.recorder, "cancel", lambda: cancelled.append(True))
    app.draw()
    active = [(rect, progress) for rect, is_active, progress in waves if is_active]
    assert len(active) == 1 and active[0][0].y == 364 and active[0][1] == 0.5
    click(app, (402, 391))
    assert cancelled == [True]


def test_noise_and_first_keyword_row_fit_in_minimum_window(app):
    app.event(pg.event.Event(pg.VIDEORESIZE, w=1100, h=740))
    app.draw()
    field = app.fields[app.workspace.rows[0].id]
    assert app.viewport.contains(field.rect)
    assert app.viewport.bottom < app.screen.get_height() - 110
    assert app.shortcut_master_rect.top > 428


def test_training_button_saves_current_workspace_and_calls_trainer(app, monkeypatch, tmp_path):
    calls = []
    first = app.workspace.rows[0]
    app.fields[first.id].value = "打开台灯"
    def start(workspace, steps=None):
        stored = Workspace(workspace)
        calls.append((stored.path, stored.rows[0].text, steps))
        return tmp_path / "run"
    monkeypatch.setattr(app.trainer, "start", start)
    click(app, app.steps_field.rect.center)
    app.event(pg.event.Event(pg.KEYDOWN, key=pg.K_a, mod=pg.KMOD_CTRL))
    app.event(pg.event.Event(pg.TEXTINPUT, text="10"))
    click(app, (app.screen.get_width() - 438, app.screen.get_height() - 65))
    assert calls == [(app.workspace.path, "打开台灯", 10)]
    assert app.modal is None


@pytest.mark.parametrize("resume", [True, False])
def test_history_dialog_selects_latest_checkpoint_or_confirmed_restart(app, monkeypatch, tmp_path, resume):
    from kws_custom_ui.history import training_history
    for name in ["20261019-100000-000000", "20261019-110000-000000"]:
        run = app.workspace.path / "training" / name
        run.mkdir(parents=True)
        (run / "latest.pt").write_bytes(b"history")
    calls = []
    def start(workspace, steps=None, **options):
        calls.append((workspace, steps, options))
        return tmp_path / "new-run"
    monkeypatch.setattr(app.trainer, "start", start)
    app.steps_field.value = "10"
    click(app, (app.screen.get_width() - 438, app.screen.get_height() - 65))
    assert app.modal["type"] == "training_history"
    assert calls == []
    checkpoint = training_history(app.workspace.path).checkpoint
    assert app.modal["checkpoint"] == checkpoint
    # Exercise the real modal button, including its registered hit target.
    x = (app.screen.get_width() - 530) // 2 + 28
    y = (app.screen.get_height() - 310) // 2 + 240 + 21
    click(app, (x + 184 if resume else x + 380, y))
    assert app.modal is None
    assert calls == [(app.workspace.path, 10, {"resume": checkpoint if resume else None, "restart": not resume})]


def test_cancel_history_dialog_does_not_launch_or_delete(app, monkeypatch):
    run = app.workspace.path / "training" / "old"
    run.mkdir(parents=True)
    checkpoint = run / "latest.pt"
    checkpoint.write_bytes(b"keep")
    def unexpected_start(*args, **kwargs):
        pytest.fail("Closing the dialog launched training")
    monkeypatch.setattr(app.trainer, "start", unexpected_start)
    app.start_training()
    app.draw()
    app.event(pg.event.Event(pg.KEYDOWN, key=pg.K_ESCAPE))
    assert app.modal is None
    assert checkpoint.read_bytes() == b"keep"


def test_log_only_history_disables_resume(app, monkeypatch):
    run = app.workspace.path / "training" / "old"
    run.mkdir(parents=True)
    (run / "training.log").write_text("log")
    app.start_training()
    buttons = []
    original = app.button
    def capture(label, rect, action, *args, **kwargs):
        buttons.append((label, kwargs.get("enabled", True)))
        return original(label, rect, action, *args, **kwargs)
    monkeypatch.setattr(app, "button", capture)
    app.draw()
    assert ("接续最近训练", False) in buttons
    assert ("清空并重新训练", True) in buttons


def test_training_locks_data_controls_without_marking_previous_recording_active(app, monkeypatch):
    monkeypatch.setattr(type(app.trainer), "busy", property(lambda trainer: True))
    app.recorder.row_id = app.workspace.rows[0].id
    buttons = []
    original = app.button
    def capture(label, rect, action, *args, **kwargs):
        buttons.append((label, kwargs.get("enabled", True)))
        return original(label, rect, action, *args, **kwargs)
    monkeypatch.setattr(app, "button", capture)
    app.draw()
    assert ("停止训练", True) in buttons
    assert not any(label == "处理素材" for label, _ in buttons)
    assert ("添加行", False) in buttons
    assert ("删除行", False) in buttons
    assert all(not enabled for label, enabled in buttons if label in ("录音 2 秒", "录音 3 秒"))
    assert not any(label.startswith("取消") for label, _ in buttons)


@pytest.mark.parametrize("value", ["", "0", "-1", "1.5", "abc"])
def test_invalid_training_steps_cannot_start_a_process(app, monkeypatch, value):
    def unexpected_start(*args, **kwargs):
        pytest.fail("Invalid training steps launched training")
    monkeypatch.setattr(app.trainer, "start", unexpected_start)
    app.steps_field.value = value
    app.start_training()
    assert app.notice_error and "正整数" in app.notice


def test_stop_training_button_requests_stop_and_restores_start_button(app, monkeypatch):
    busy = [True]
    monkeypatch.setattr(type(app.trainer), "busy", property(lambda trainer: busy[0]))
    stopped = []
    def stop():
        stopped.append(True)
        app.trainer.stopping = True
    monkeypatch.setattr(app.trainer, "stop", stop)
    buttons = []
    original = app.button
    def capture(label, rect, action, *args, **kwargs):
        buttons.append((label, kwargs.get("enabled", True)))
        return original(label, rect, action, *args, **kwargs)
    monkeypatch.setattr(app, "button", capture)
    app.draw()
    click(app, (app.screen.get_width() - 438, app.screen.get_height() - 65))
    assert stopped == [True]
    assert ("正在停止…", False) in buttons
    busy[0] = False
    buttons.clear()
    app.draw()
    assert ("开始训练", True) in buttons


def test_workspace_memory_preserves_hid_settings_and_other_preferences(app, tmp_path):
    settings = dict(hid_device=dict(vid="0x1234", pid=22136), custom_preference="keep")
    app.session_file.write_text(json.dumps(settings), encoding="utf-8")
    app.path_field.value = str(tmp_path / "new-workspace")
    app.change_workspace()
    app.exit()
    saved = json.loads(app.session_file.read_text(encoding="utf-8"))
    assert saved["last_workspace"] == str(tmp_path / "new-workspace")
    assert saved["hid_device"] == dict(vid="0x1234", pid=22136, usage_page="0xFF00", usage="0x0001")
    assert saved["custom_preference"] == "keep"


def test_restart_restores_last_workspace_and_pending_text(app, tmp_path, monkeypatch):
    target = tmp_path / "中文工作区"
    app.path_field.value = str(target)
    app.change_workspace()
    assert json.loads(app.session_file.read_text(encoding="utf-8"))["last_workspace"] == str(target)
    first = app.workspace.rows[0]
    app.fields[first.id].value = "打开台灯"
    app.path_field.value = str(tmp_path / "未应用的路径")
    app.exit()
    assert not app.running
    other_start = tmp_path / "different-start-directory"
    other_start.mkdir()
    monkeypatch.chdir(other_start)
    reopened = App(session_file=app.session_file)
    try:
        assert reopened.workspace.path == target
        assert reopened.fields[first.id].value == "打开台灯"
        assert reopened.workspace.unknown_dir.is_dir()
        assert not (tmp_path / "未应用的路径").exists()
    finally:
        reopened.recorder.close()


@pytest.mark.parametrize("contents", [None, "{broken", "[]", '{}', '{"last_workspace": 123}', '{"last_workspace": "relative"}'])
def test_first_start_and_invalid_session_use_default_workspace(tmp_path, monkeypatch, contents):
    monkeypatch.chdir(tmp_path)
    session = tmp_path / "session.json"
    if contents is not None:
        session.write_text(contents, encoding="utf-8")
    instance = App(session_file=session)
    try:
        assert instance.workspace.path == tmp_path / "workspace" / "test"
        assert instance.workspace.unknown_dir.is_dir()
    finally:
        instance.recorder.close()
        pg.quit()


@pytest.mark.parametrize("missing", [False, True])
def test_unavailable_remembered_workspace_falls_back_without_overwriting_config(tmp_path, monkeypatch, missing):
    monkeypatch.chdir(tmp_path)
    old = tmp_path / "old"
    if not missing:
        old.mkdir()
        (old / "workspace.json").write_text("{broken", encoding="utf-8")
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"last_workspace": str(old)}), encoding="utf-8")
    instance = App(session_file=session)
    try:
        assert instance.workspace.path == tmp_path / "workspace" / "test"
        if missing:
            assert not old.exists()
        else:
            assert (old / "workspace.json").read_text(encoding="utf-8") == "{broken"
    finally:
        instance.recorder.close()
        pg.quit()


def test_cli_workspace_overrides_memory_then_is_remembered(tmp_path, monkeypatch):
    from kws_custom_ui import app as app_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_module, "SESSION_FILE", tmp_path / "session.json")
    opened = []
    def close_after_open(instance):
        opened.append(instance.workspace.path)
        instance.exit()
        instance.recorder.close()
        pg.quit()
    monkeypatch.setattr(App, "run", close_after_open)
    for arguments in [["kws-studio"], ["kws-studio", "--workspace", "custom"], ["kws-studio"]]:
        monkeypatch.setattr("sys.argv", arguments)
        app_module.main()
    assert opened == [tmp_path / "workspace" / "test", tmp_path / "custom", tmp_path / "custom"]


def test_exit_closes_trainer_before_finishing(app, monkeypatch):
    closed = []
    monkeypatch.setattr(app.trainer, "close", lambda: closed.append(app.running))
    app.event(pg.event.Event(pg.QUIT))
    assert closed == [True]
    assert not app.running


def test_run_cleans_training_and_recorder_if_initial_update_fails(app, monkeypatch):
    closed = []
    def fail_update():
        raise RuntimeError("unexpected update failure")
    monkeypatch.setattr(app, "update", fail_update)
    monkeypatch.setattr(app.trainer, "close", lambda: closed.append("training"))
    monkeypatch.setattr(app.recorder, "close", lambda: closed.append("recorder"))
    with pytest.raises(RuntimeError, match="unexpected update failure"):
        app.run()
    assert closed == ["training", "recorder"]
