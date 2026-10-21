"""Pygame desktop interface, with native folder selection and Chinese IME support."""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame as pg

from .audio import KEYWORD_SECONDS, Recorder, input_devices
from .model import FLAG_ACTION, FLAG_OUTPUTS, PALETTE, Workspace
from .keyboard import keyboard_report
from .training import Trainer, default_steps
from .history import training_history
from .device import DeviceController, open_device, row_binding, sync_workspace
from .hid_protocol import device_filter_values
from .settings import SESSION_FILE, DEFAULT_HID_DEVICE, DEFAULT_RECORDING, read_session, recording_durations


BG = (245, 247, 251)
WHITE = (255, 255, 255)
INK = (33, 45, 65)
MUTED = (132, 143, 160)
LINE = (229, 234, 242)
TEAL = (24, 156, 137)
TEAL_BG = (231, 246, 241)
OFF_BG = (234, 237, 242)
PURPLE = (104, 113, 217)
RED = (210, 80, 91)
DEVICE_POLL_INTERVAL = 5.0


def last_workspace(session_file: Path) -> Path | None:
    try:
        data = json.loads(session_file.read_text(encoding="utf-8"))
        value = data.get("last_workspace")
        if isinstance(value, str) and value.strip():
            path = Path(value)
            if path.is_absolute() and path.is_dir():
                return path
    except (OSError, ValueError, AttributeError):
        pass
    return None


def clock_text(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class TextField:
    def __init__(self, value: str = "", placeholder: str = "", limit: int = 4096):
        self.value, self.placeholder, self.limit = value, placeholder, limit
        self.cursor = len(value)
        self.anchor = self.cursor
        self.focus = False
        self.rect = pg.Rect(0, 0, 0, 0)
        self.offset = 0
        self.composition = ""

    def replace_selection(self, text: str) -> None:
        start, end = sorted((self.cursor, self.anchor))
        text = text.replace("\r", "").replace("\n", "")[:max(0, self.limit - len(self.value) + end - start)]
        self.value = self.value[:start] + text + self.value[end:]
        self.cursor = self.anchor = start + len(text)

    def event(self, event, app) -> bool:
        before = self.value
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            self.focus = self.rect.collidepoint(event.pos) and (self in (app.path_field, app.steps_field) or app.viewport.collidepoint(event.pos))
            self.composition = ""
            if self.focus:
                x = event.pos[0] - self.rect.x - 12 + self.offset
                self.cursor = min(range(len(self.value) + 1), key=lambda i: abs(app.font(16).size(self.value[:i])[0] - x))
                self.anchor = self.cursor
                pg.key.set_text_input_rect(self.rect)
        if not self.focus:
            return False
        if event.type == pg.TEXTEDITING:
            self.composition = event.text
        elif event.type == pg.TEXTINPUT:
            self.composition = ""
            self.replace_selection(event.text)
        elif event.type == pg.KEYDOWN:
            ctrl, shift = bool(event.mod & pg.KMOD_CTRL), bool(event.mod & pg.KMOD_SHIFT)
            if ctrl and event.key == pg.K_a:
                self.anchor, self.cursor = 0, len(self.value)
            elif ctrl and event.key in (pg.K_c, pg.K_x):
                a, b = sorted((self.cursor, self.anchor))
                if a != b:
                    app.clipboard_set(self.value[a:b])
                    if event.key == pg.K_x:
                        self.replace_selection("")
            elif ctrl and event.key == pg.K_v:
                self.replace_selection(app.clipboard_get())
            elif event.key in (pg.K_LEFT, pg.K_RIGHT, pg.K_HOME, pg.K_END):
                if event.key == pg.K_HOME:
                    self.cursor = 0
                elif event.key == pg.K_END:
                    self.cursor = len(self.value)
                elif self.cursor != self.anchor and not shift:
                    self.cursor = (min if event.key == pg.K_LEFT else max)(self.cursor, self.anchor)
                else:
                    self.cursor = max(0, min(len(self.value), self.cursor + (-1 if event.key == pg.K_LEFT else 1)))
                if not shift:
                    self.anchor = self.cursor
            elif event.key in (pg.K_BACKSPACE, pg.K_DELETE) and not self.composition:
                if self.anchor == self.cursor:
                    self.anchor = max(0, self.cursor - 1) if event.key == pg.K_BACKSPACE else min(len(self.value), self.cursor + 1)
                self.replace_selection("")
            elif event.key in (pg.K_RETURN, pg.K_KP_ENTER) and not self.composition:
                self.focus = False
                if self is app.path_field:
                    app.change_workspace()
            elif event.key == pg.K_ESCAPE:
                self.focus = False
                self.composition = ""
        return self.value != before

    def validation_error(self):
        return ""

    def draw(self, app, rect):
        self.rect = pg.Rect(rect)
        if self.focus:
            pg.key.set_text_input_rect(self.rect)
        pg.draw.rect(app.screen, WHITE if self.focus else (250, 251, 253), self.rect, border_radius=8)
        border = RED if self.validation_error() else TEAL if self.focus else LINE
        pg.draw.rect(app.screen, border, self.rect, 1, border_radius=8)
        inner = self.rect.inflate(-24, -8)
        cursor_x = app.font(16).size(self.value[:self.cursor])[0]
        self.offset = max(0, min(self.offset, max(0, app.font(16).size(self.value)[0] - inner.w + 8)))
        if self.focus:
            if cursor_x - self.offset > inner.w - 6:
                self.offset = cursor_x - inner.w + 6
            if cursor_x < self.offset:
                self.offset = cursor_x
        old_clip = app.screen.get_clip()
        app.screen.set_clip(inner.clip(old_clip))
        if self.focus and self.cursor != self.anchor:
            a, b = sorted((self.cursor, self.anchor))
            xa = app.font(16).size(self.value[:a])[0]
            xb = app.font(16).size(self.value[:b])[0]
            pg.draw.rect(app.screen, (205, 235, 230), (inner.x + xa - self.offset, inner.y, xb - xa, inner.h))
        app.text(self.value or self.placeholder, (inner.x - self.offset, self.rect.centery), 16,
                 INK if self.value else MUTED, center_y=True)
        if self.focus and (time.monotonic() % 1 < 0.55 or self.composition):
            x = inner.x + cursor_x - self.offset
            pg.draw.line(app.screen, TEAL, (x, self.rect.y + 10), (x, self.rect.bottom - 10), 2)
            if self.composition:
                app.text(self.composition, (x + 3, self.rect.centery), 16, TEAL, center_y=True)
        app.screen.set_clip(old_clip)


class ShortcutField(TextField):
    def __init__(self, value=""):
        super().__init__(value, "例如 {CTRL}+a", limit=256)

    def validation_error(self):
        try:
            keyboard_report(self.value)
        except ValueError as exc:
            return str(exc)
        return ""


class App:
    def __init__(self, workspace: Path | None = None, size=(1280, 900), *, session_file: Path | None = None):
        self.session_file = session_file if session_file is not None else SESSION_FILE
        try:
            self.recording = recording_durations(self.session_file)
            recording_error = None
        except ValueError as exc:
            self.recording = DEFAULT_RECORDING.copy()
            recording_error = str(exc)
        remembered = last_workspace(self.session_file) if workspace is None else None
        default_workspace = Path.cwd() / "workspace" / "test"
        workspace = workspace if workspace is not None else remembered or default_workspace
        pg.init()
        pg.display.set_caption("KWS Studio · 关键词采集工作台")
        icon_path = Path(__file__).resolve().parents[1] / "icon" / "whys_logo.png"
        pg.display.set_icon(pg.image.load(str(icon_path)))
        self.screen = pg.display.set_mode(size, pg.RESIZABLE)
        self.font_path = next((p for p in [
            "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhl.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/PingFang.ttc"] if Path(p).exists()), None)
        if self.font_path is None:
            self.font_path = pg.font.match_font("microsoftyahei,notosanscjk,wenquanyimicrohei,arial")
        self.fonts = {}
        self.workspace: Workspace | None = None
        self.fields: dict[str, TextField] = {}
        self.shortcut_fields: dict[str, ShortcutField] = {}
        self.path_field = TextField(str(workspace), "输入工作区路径，按 Enter 应用")
        self.steps_field = TextField(str(default_steps()), "步数", limit=9)
        self.recorder = Recorder()
        self.trainer = Trainer()
        self.simulator: subprocess.Popen | None = None
        self.simulator_error_log: Path | None = None
        self.hid = DeviceController(lambda: open_device(self.session_file))
        self.device_workspace_path = Path.cwd() / "workspace" / "device"
        self.local_workspace_path = None
        self.connection_state = "idle"
        self.next_device_poll = 0.0
        self.exit_after_device_save = False
        self.mode = "local"
        self.mode_switch_rect = pg.Rect(0, 0, 0, 0)
        self.mode_drag = False
        self.mode_thumb = 0.0
        self.mode_anim_from = 0.0
        self.mode_anim_started = time.monotonic()
        self.device = None
        self.device_name = "系统默认麦克风"
        self.devices = []
        self.modal: dict | None = None
        self.buttons = []
        self.selected: str | None = None
        self.scroll = 0
        self.device_scroll = 0
        self.running = True
        self.notice = "准备就绪，选择一个关键词开始采集。"
        self.notice_error = False
        self.counts = {}
        self.last_count = 0
        self.save_due = 0.0
        self.tk_root = None
        self.drag_scroll = False
        self.color_drag: str | None = None
        self.filter_drag: str | None = None
        self.device_filter_rects = {}
        self.device_filter_pending: dict | None = None
        self.queued_device_filter: dict | None = None
        pg.key.start_text_input()
        self.change_workspace(initial=True)
        if self.workspace is None and remembered is not None:
            self.path_field.value = str(default_workspace)
            self.change_workspace(initial=True)
            if self.workspace is not None:
                self.tell("上次工作区无法打开，已打开默认工作区。", True)
        if recording_error:
            self.tell(recording_error, True)

    def font(self, size: int, bold=False):
        key = size, bold
        if key not in self.fonts:
            font = pg.font.Font(self.font_path, size)
            font.set_bold(bold)
            self.fonts[key] = font
        return self.fonts[key]

    def text(self, text, pos, size=16, color=INK, bold=False, center_y=False, max_width=None):
        font = self.font(size, bold)
        text = str(text)
        if max_width is not None and font.size(text)[0] > max_width:
            while text and font.size(text + "…")[0] > max_width:
                text = text[:-1]
            text += "…"
        surface = font.render(text, True, color)
        self.screen.blit(surface, (pos[0], pos[1] - surface.get_height() // 2 if center_y else pos[1]))

    def panel(self, rect, color=WHITE, border=LINE, radius=14):
        rect = pg.Rect(rect)
        pg.draw.rect(self.screen, color, rect, border_radius=radius)
        if border:
            pg.draw.rect(self.screen, border, rect, 1, border_radius=radius)

    def icon(self, name, center, color=TEAL, size=18):
        x, y = center
        if name == "record":
            pg.draw.circle(self.screen, color, (int(x), int(y)), size // 3)
        elif name == "cancel":
            pg.draw.rect(self.screen, color, (x - 5, y - 5, 10, 10), border_radius=2)
        elif name == "folder":
            pg.draw.lines(self.screen, color, False, [(x - 9, y + 6), (x - 9, y - 6), (x - 2, y - 6), (x + 1, y - 3), (x + 9, y - 3), (x + 9, y + 6), (x - 9, y + 6)], 2)
        elif name == "plus":
            pg.draw.line(self.screen, color, (x - 5, y), (x + 5, y), 2)
            pg.draw.line(self.screen, color, (x, y - 5), (x, y + 5), 2)
        elif name == "mic":
            pg.draw.rect(self.screen, color, (x - 3, y - 9, 6, 12), 2, border_radius=3)
            pg.draw.arc(self.screen, color, (x - 7, y - 8, 14, 16), 3.14, 6.28, 2)
            pg.draw.line(self.screen, color, (x, y + 7), (x, y + 10), 2)
            pg.draw.line(self.screen, color, (x - 4, y + 10), (x + 4, y + 10), 2)
        elif name == "arrow":
            pg.draw.lines(self.screen, color, False, [(x - 5, y - 4), (x, y + 1), (x + 5, y - 4)], 2)

    def button(self, label, rect, action, style="plain", enabled=True, icon=None):
        rect = pg.Rect(rect)
        hover = rect.collidepoint(pg.mouse.get_pos()) and (not self.modal or self._drawing_modal)
        fg = INK
        bg, border = WHITE, LINE
        if style == "primary":
            bg, fg, border = TEAL, WHITE, None
        elif style == "soft":
            bg, fg, border = TEAL_BG, TEAL, None
        elif style == "danger":
            bg, fg, border = (255, 240, 241), RED, None
        elif style == "purple":
            bg, fg, border = (238, 239, 253), PURPLE, None
        elif style == "off":
            bg, fg, border = OFF_BG, MUTED, None
        if not enabled:
            bg, fg, border = (242, 244, 248), (168, 177, 190), LINE
        elif hover:
            bg = tuple(max(0, c - 9) for c in bg)
        self.panel(rect, bg, border, 8)
        width = self.font(15).size(label)[0] + (25 if icon else 0)
        left = rect.centerx - width // 2
        if icon:
            self.icon(icon, (left + 8, rect.centery), fg)
            left += 25
        self.text(label, (left, rect.centery), 15, fg, center_y=True)
        if enabled:
            self.buttons.append((rect, action))

    def tell(self, message, error=False):
        self.notice, self.notice_error = message, error

    def set_mode(self, mode):
        if self.simulator is not None and self.simulator.poll() is None:
            return
        if mode == self.mode:
            if mode == "device" and self.connection_state == "failed":
                self.connect_device()
            return
        if self.recorder.busy or self.trainer.busy or (self.hid.busy and self.hid.operation not in ("connect", "poll")):
            self.tell("录音、训练或设备传输期间无法切换模式。", True)
            return
        if self.mode == "local" and not self.save():
            return
        if mode == "device" and self.local_workspace_path.resolve() == self.device_workspace_path.resolve():
            self.tell("设备模式需要独立的 workspace/device，请先选择其他本地训练工作区。", True)
            return
        if self.mode == "device" and self.save_due:
            if not self.save():
                return
            if self.hid.busy or not self.hid.results.empty():
                self.tell("正在下发设备参数，请完成后切换模式。")
                return
        try:
            self.hid.close()
            destination = self.device_workspace_path if mode == "device" else self.local_workspace_path
            new_workspace = Workspace(destination)
        except (OSError, RuntimeError, ValueError) as exc:
            self.tell(f"无法切换模式：{exc}", True)
            return
        self.mode_anim_from = self.mode_thumb
        self.mode_anim_started = time.monotonic()
        self.mode = mode
        self.modal = None
        self.save_due = 0
        self.queued_device_filter = None
        self.use_workspace(new_workspace)
        if mode == "device":
            self.connect_device()
        else:
            self.connection_state = "idle"
            self.tell("当前模式：本地训练，所有修改保存在 PC。")

    def connect_device(self):
        if self.mode != "device" or self.hid.busy:
            return
        self.connection_state = "connecting"
        self.hid.submit("connect")
        self.tell("正在连接控制 HID 并读取设备参数…")

    def import_model(self):
        if self.mode != "local" or self.recorder.busy or self.trainer.busy or self.hid.busy or self.simulator is not None or not self.workspace:
            return
        if self.save() and self.hid.submit("deploy", self.workspace):
            self.tell("正在检查模型兼容性并连接设备…")

    def save_device(self):
        if not self.hid.connected:
            self.save_due = 0
            return True
        if self.hid.busy:
            self.save_due = time.monotonic() + 0.45
            return True
        try:
            known = {item["class_index"]: item for item in self.hid.snapshot["bindings"]["bindings"]}
            changed = []
            for index, row in enumerate(self.workspace.rows, 2):
                row.shortcut = self.shortcut_fields[row.id].value
                binding = row_binding(row, index, known[index])
                if binding != known[index]:
                    changed.append(binding)
            if changed:
                self.hid.submit("bindings", changed)
                self.tell("正在下发颜色、快捷键和开关状态…")
            self.save_due = 0
            return True
        except (ValueError, KeyError) as exc:
            self.save_due = 0
            self.tell(f"设备参数未下发：{exc}", True)
            return False

    def apply_device_snapshot(self, snapshot):
        sync_workspace(self.workspace, snapshot)
        self.device_filter_pending = None
        self.filter_drag = None
        self.use_workspace(self.workspace)
        self.connection_state = "connected"
        self.next_device_poll = time.monotonic() + DEVICE_POLL_INTERVAL

    def toggle_shortcut(self, row):
        if self.hid.busy or (self.mode == "device" and not self.hid.connected):
            return
        previous = row.flags
        row.flags = row.flags & ~FLAG_OUTPUTS if row.enabled else row.flags | FLAG_OUTPUTS
        if not self.save():
            row.flags = previous
            return
        if not row.flags & FLAG_ACTION:
            self.tell("该行灯效与键盘开关已开启；动作总开关关闭，暂不执行。")
        else:
            self.tell(f"已{'启用' if row.enabled else '禁用'}「{row.text or row.directory}」的灯效与键盘动作。")

    def toggle_all_shortcuts(self):
        if self.hid.busy or (self.mode == "device" and not self.hid.connected):
            return
        if not self.workspace or not self.workspace.rows:
            return
        rows = self.workspace.rows
        previous = [row.flags for row in rows]
        enabled = not self.workspace.actions_enabled
        for row in rows:
            row.flags = row.flags | FLAG_ACTION if enabled else row.flags & ~FLAG_ACTION
        if not self.save():
            for row, state in zip(rows, previous):
                row.flags = state
            return
        self.tell("已打开全部行的动作总开关。" if enabled else "已关闭全部行的动作总开关。")

    def draw_mode_switch(self, rect):
        self.mode_switch_rect = pg.Rect(rect)
        rect = self.mode_switch_rect
        self.panel(rect, (233, 238, 245), LINE, 21)
        progress = min(1.0, (time.monotonic() - self.mode_anim_started) / 0.16)
        eased = progress * progress * (3 - 2 * progress)
        target = 0.0 if self.mode == "local" else 1.0
        self.mode_thumb = self.mode_anim_from + (target - self.mode_anim_from) * eased
        thumb = pg.Rect(rect.x + 4 + round((rect.w // 2) * self.mode_thumb), rect.y + 4,
                        rect.w // 2 - 8, rect.h - 8)
        pg.draw.rect(self.screen, (214, 223, 232), thumb.move(0, 1), border_radius=17)
        pg.draw.rect(self.screen, WHITE, thumb, border_radius=17)
        for index, (mode, label) in enumerate((("local", "本地训练"), ("device", "连接设备"))):
            if mode == "device" and self.mode == "device":
                label = "连接失败" if self.connection_state == "failed" else "连接中…" if self.connection_state == "connecting" else label
            half = pg.Rect(rect.x + index * (rect.w // 2), rect.y, rect.w // 2, rect.h)
            selected = self.mode == mode
            width = self.font(14, selected).size(label)[0]
            self.text(label, (half.centerx - width // 2, half.centery), 14,
                      RED if mode == "device" and self.connection_state == "failed" else TEAL if selected else MUTED,
                      selected, center_y=True)
            self.buttons.append((half, lambda selected_mode=mode: self.set_mode(selected_mode)))

    def native_root(self):
        if self.tk_root is None:
            import tkinter as tk
            self.tk_root = tk.Tk()
            self.tk_root.withdraw()
        return self.tk_root

    def clipboard_get(self):
        try:
            return self.native_root().clipboard_get()
        except Exception:
            return ""

    def clipboard_set(self, text):
        try:
            root = self.native_root()
            root.clipboard_clear()
            root.clipboard_append(text)
            root.update()
        except Exception as exc:
            self.tell(f"无法访问剪贴板：{exc}", True)

    def choose_directory(self):
        if self.mode != "local" or self.recorder.busy or self.trainer.busy or self.hid.busy:
            return
        try:
            from tkinter import filedialog
            chosen = filedialog.askdirectory(parent=self.native_root(), title="选择工作区文件夹",
                                             initialdir=str(self.workspace.path if self.workspace else Path.cwd()))
            if chosen:
                self.path_field.value = chosen
                self.path_field.cursor = self.path_field.anchor = len(chosen)
                self.change_workspace()
        except Exception as exc:
            self.tell(f"无法打开文件夹选择器：{exc}", True)

    def save(self):
        if not self.workspace:
            return True
        if self.mode == "device":
            return self.save_device()
        try:
            for row in self.workspace.rows:
                row.text = self.fields[row.id].value
                row.shortcut = self.shortcut_fields[row.id].value
            self.workspace.save()
            self.save_due = 0
            return True
        except OSError as exc:
            self.save_due = time.monotonic() + 3
            self.tell(f"配置保存失败：{exc}", True)
            return False

    def change_workspace(self, initial=False):
        if self.mode != "local" or self.recorder.busy or self.trainer.busy or self.hid.busy or self.simulator is not None:
            self.tell("录音或训练期间无法切换工作区。", True)
            return
        value = self.path_field.value.strip().strip('"')
        if not value:
            self.tell("请输入工作区路径。", True)
            return
        if not initial and not self.save():
            return
        try:
            new_workspace = Workspace(value)
        except (OSError, ValueError, RuntimeError) as exc:
            self.tell(f"工作区打开失败：{exc}", True)
            return
        self.use_workspace(new_workspace)
        self.local_workspace_path = new_workspace.path
        self.tell("工作区已就绪，文本、组合键与颜色会自动保存。")
        if not initial:
            self.remember_workspace()

    def use_workspace(self, new_workspace):
        self.workspace = new_workspace
        self.path_field.value = str(new_workspace.path)
        self.path_field.cursor = self.path_field.anchor = len(self.path_field.value)
        self.fields = {row.id: TextField(row.text, "关键词，最多 16 字符", 16) for row in new_workspace.rows}
        self.shortcut_fields = {row.id: ShortcutField(row.shortcut) for row in new_workspace.rows}
        self.selected = new_workspace.rows[0].id if new_workspace.rows else None
        self.scroll = 0
        self.last_count = 0
        self.path_field.focus = self.steps_field.focus = False
        self.device_filter_pending = None
        self.filter_drag = None

    @staticmethod
    def filter_fraction(name, value):
        if name == "keyword_threshold_percent":
            return value / 100
        knee, maximum = (1000, 60000) if name == "cooldown_ms" else (50, 1000)
        if value <= knee:
            return .8 * (value - 1) / (knee - 1)
        return .8 + .2 * math.log(value / knee) / math.log(maximum / knee)

    @staticmethod
    def filter_value(name, fraction):
        fraction = max(0.0, min(1.0, fraction))
        if name == "keyword_threshold_percent":
            return round(fraction * 100)
        knee, maximum = (1000, 60000) if name == "cooldown_ms" else (50, 1000)
        value = (1 + fraction / .8 * (knee - 1) if fraction <= .8 else
                 knee * (maximum / knee) ** ((fraction - .8) / .2))
        if name == "cooldown_ms" and value >= 10:
            return min(maximum, round(value / 10) * 10)
        return max(1, min(maximum, round(value)))

    def send_device_filter(self, settings):
        if self.mode != "device" or not self.hid.connected:
            return
        try:
            settings = device_filter_values(settings)
            current = device_filter_values(self.hid.snapshot["status"]["kws_filter"])
            if settings == current:
                self.device_filter_pending = None
                return
            if self.save_due and not self.save():
                self.device_filter_pending = None
                return
            if self.hid.busy or not self.hid.results.empty():
                self.queued_device_filter = settings
                self.device_filter_pending = settings
                self.tell("正在先保存快捷键，再写入 KWS 参数…")
                return
            if not self.hid.submit("filter", settings):
                return
            self.device_filter_pending = settings
            self.tell("正在写入设备 KWS 参数…")
        except (ValueError, KeyError) as exc:
            self.device_filter_pending = None
            self.tell(f"KWS 参数无效：{exc}", True)

    def toggle_device_vad(self):
        if not self.hid.connected or self.hid.busy:
            return
        settings = device_filter_values(self.hid.snapshot["status"]["kws_filter"])
        settings["need_vad_checked"] = not settings["need_vad_checked"]
        self.send_device_filter(settings)

    def update_device_filter_drag(self, position):
        if self.filter_drag is None:
            return
        rect = self.device_filter_rects[self.filter_drag]
        fraction = (position[0] - rect.x) / max(1, rect.w - 1)
        self.device_filter_pending[self.filter_drag] = self.filter_value(self.filter_drag, fraction)

    def draw_device_filter_slider(self, name, label, x, y, width, enabled, settings):
        value = settings[name] if settings else None
        color = INK if enabled else MUTED
        self.text(label, (x, y + 17), 14, color, center_y=True)
        track = pg.Rect(x + 145, y + 13, width, 8)
        pg.draw.rect(self.screen, OFF_BG, track, border_radius=4)
        if value is not None:
            fraction = self.filter_fraction(name, value)
            pg.draw.rect(self.screen, TEAL if enabled else MUTED,
                         (track.x, track.y, max(1, round(track.w * fraction)), track.h), border_radius=4)
            pg.draw.circle(self.screen, TEAL if enabled else MUTED,
                           (track.x + round((track.w - 1) * fraction), track.centery), 8)
        self.text("--" if value is None else str(value), (track.right + 13, y + 17), 14, color, center_y=True)
        if enabled:
            self.device_filter_rects[name] = track

    def remember_workspace(self):
        if self.workspace is None:
            return
        try:
            self.session_file.parent.mkdir(parents=True, exist_ok=True)
            settings = read_session(self.session_file)
            settings["last_workspace"] = str(self.local_workspace_path)
            settings.setdefault("recording", DEFAULT_RECORDING.copy())
            hid_settings = settings.setdefault("hid_device", {})
            if isinstance(hid_settings, dict):
                for name, value in DEFAULT_HID_DEVICE.items():
                    hid_settings.setdefault(name, value)
            temporary = self.session_file.with_name(self.session_file.name + ".tmp")
            temporary.write_text(json.dumps(settings,
                                            ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.session_file)
        except OSError as exc:
            self.tell(f"工作区记忆保存失败：{exc}", True)

    def open_folder(self, directory):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(str(directory))
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(directory)])
        except OSError as exc:
            self.tell(f"无法打开目录：{exc}", True)

    def add_row(self):
        if self.mode != "local" or not self.workspace or self.recorder.busy or self.trainer.busy or self.hid.busy:
            return
        try:
            row = self.workspace.add()
            self.fields[row.id] = TextField("", "关键词，最多 16 字符", 16)
            self.shortcut_fields[row.id] = ShortcutField()
            self.selected = row.id
            self.scroll = max(0, len(self.workspace.rows) * 84 - 10 - self.viewport.h)
            self.save()
            self.tell("已添加一行。")
        except OSError as exc:
            self.tell(f"添加失败：{exc}", True)

    def request_delete(self):
        if self.mode != "local" or not self.selected or not self.workspace or self.recorder.busy or self.trainer.busy or self.hid.busy:
            return
        self.modal = {"type": "delete", "id": self.selected}

    def delete_row(self):
        if self.mode != "local" or self.hid.busy or self.recorder.busy or self.trainer.busy or not self.workspace or not self.save():
            return
        identity = self.modal["id"]
        row = next(row for row in self.workspace.rows if row.id == identity)
        try:
            archived = self.workspace.delete(row)
        except (OSError, ValueError) as exc:
            self.modal = None
            self.tell(f"删除失败：{exc}", True)
            return
        self.fields.pop(identity, None)
        self.shortcut_fields.pop(identity, None)
        self.selected = self.workspace.rows[0].id if self.workspace.rows else None
        self.modal = None
        self.last_count = 0
        self.tell(f"已删除该行，原目录已归档为 {archived.name}。" if archived else "已删除该行，空目录已清理。")

    def record(self, row=None, *, noise=False):
        if self.mode != "local" or self.hid.busy or self.recorder.busy or self.trainer.busy or not self.workspace or not self.save():
            return
        try:
            self.recording = recording_durations(self.session_file)
            if noise:
                directory, seconds, identity = self.workspace.unknown_dir, self.recording["noise_seconds"], "unknown"
            elif row:
                directory, seconds, identity = self.workspace.row_dir(row), KEYWORD_SECONDS, row.id
            else:
                directory, seconds, identity = self.workspace.silence_dir, self.recording["silence_seconds"], "silence"
            self.recorder.start(directory, seconds, self.device, identity)
            if row:
                self.selected = row.id
            if noise:
                self.tell(f"正在采集非静音噪声，{seconds} 秒后自动保存。")
            else:
                self.tell(f"正在采集关键词，请自然说话；{KEYWORD_SECONDS} 秒后自动保存。" if row else f"正在采集背景静音，请保持环境安静；{seconds} 秒后自动保存。")
        except (OSError, RuntimeError, ValueError) as exc:
            self.tell(str(exc), True)

    def start_training(self, steps=None):
        if self.mode != "local" or self.hid.busy or self.recorder.busy or self.trainer.busy or not self.workspace:
            return
        try:
            if steps is None:
                value = self.steps_field.value.strip()
                if not value.isascii() or not value.isdecimal():
                    raise ValueError("训练步数必须为正整数。")
                steps = int(value)
            if type(steps) is not int or steps < 1:
                raise ValueError("训练步数必须为正整数。")
            if not self.save():
                return
            history = training_history(self.workspace.path)
            if history.present:
                self.modal = {"type": "training_history", "steps": steps,
                              "checkpoint": history.checkpoint, "workspace": self.workspace.path}
                self.steps_field.focus = self.path_field.focus = False
                return
            self.launch_training(steps)
        except (OSError, RuntimeError, ValueError) as exc:
            self.tell(f"无法启动训练：{exc}", True)

    def launch_training(self, steps, *, resume=None, restart=False):
        if self.mode != "local" or self.hid.busy:
            return
        try:
            if resume is None and not restart:
                run_dir = self.trainer.start(self.workspace.path, steps=steps)
            else:
                run_dir = self.trainer.start(self.workspace.path, steps=steps, resume=resume, restart=restart)
            self.steps_field.focus = self.path_field.focus = False
            self.tell(f"正在准备训练数据并打开 TensorBoard；日志：{run_dir / 'training.log'}")
        except (OSError, RuntimeError, ValueError) as exc:
            self.tell(f"无法启动训练：{exc}", True)

    def confirm_training(self, resume: bool):
        selection = self.modal
        if not selection or selection["type"] != "training_history" or self.trainer.busy or self.recorder.busy:
            return
        if selection["workspace"] != self.workspace.path:
            return
        if resume and selection["checkpoint"] is None:
            return
        self.close_modal()
        self.launch_training(selection["steps"], resume=selection["checkpoint"] if resume else None, restart=not resume)

    def stop_training(self):
        try:
            self.trainer.stop()
            self.tell(self.trainer.status)
        except OSError as exc:
            self.tell(f"无法停止训练：{exc}", True)

    def open_simulation(self):
        if (self.mode != "local" or not self.workspace or self.recorder.busy or self.trainer.busy
                or self.hid.busy or self.simulator is not None or not self.save()):
            return
        try:
            from .simulation import latest_model
            latest_model(self.workspace.path)
            self.simulator_error_log = self.workspace.path / "simulation" / "startup.log"
            self.simulator_error_log.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-m", "kws_custom_ui.simulation", "--workspace", str(self.workspace.path)]
            if self.device is not None:
                command.extend(("--device", str(self.device)))
            environment = os.environ.copy()
            environment["PYTHONUTF8"] = "1"
            source_root = str(Path(__file__).resolve().parents[1])
            environment["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, environment.get("PYTHONPATH"))))
            with self.simulator_error_log.open("w", encoding="utf-8") as errors:
                self.simulator = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                                                  env=environment, stdout=errors, stderr=subprocess.STDOUT,
                                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self.tell("PC端仿真窗口已启动，参数与逐帧日志保存在当前工作区。")
        except (OSError, ValueError, KeyError) as exc:
            self.tell(f"无法启动 PC端仿真：{exc}", True)

    def stop_simulation(self):
        if self.simulator is None:
            return
        if self.simulator.poll() is None:
            self.simulator.terminate()
            try:
                self.simulator.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.simulator.kill()
                self.simulator.wait()
        self.simulator = None
        if self.workspace:
            try:
                data = json.loads((self.workspace.path / self.workspace.CONFIG).read_text(encoding="utf-8"))
                self.workspace.kws_filter = data["kws_filter"]
            except (OSError, ValueError, KeyError):
                pass

    def choose_device(self):
        if self.mode != "local" or self.hid.busy:
            return
        try:
            self.devices = input_devices()
            self.device_scroll = 0
            self.modal = {"type": "devices"}
        except Exception as exc:
            self.tell(f"读取麦克风失败：{exc}", True)

    def set_device(self, device):
        self.device = device["id"] if device else None
        self.device_name = device["name"] if device else "系统默认麦克风"
        self.modal = None
        self.tell(f"已选择：{self.device_name}")

    def choose_color(self, row):
        if self.hid.busy or (self.mode == "device" and not self.hid.connected):
            return
        self.selected = row.id
        h, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in row.color))
        self.modal = {"type": "color", "row": row, "h": h, "s": s, "v": v, "surface_h": None}

    def apply_color(self):
        self.modal["row"].color = self.modal_color()
        self.modal = None
        self.save()

    def modal_color(self):
        m = self.modal
        return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(m["h"], m["s"], m["v"]))

    def waveform(self, rect, values, active=False, progress=0):
        rect = pg.Rect(rect)
        self.panel(rect, (245, 250, 249) if active else (248, 250, 252), None, 7)
        if active and progress:
            fill = rect.copy()
            fill.w = max(1, int(rect.w * min(1, progress)))
            pg.draw.rect(self.screen, (226, 243, 237), fill, border_radius=7)
        mid = rect.centery
        pg.draw.line(self.screen, (210, 226, 224) if active else LINE, (rect.x + 8, mid), (rect.right - 8, mid))
        for i in range(min(65, max(1, rect.w // 5))):
            amp = values[int(i * len(values) / min(65, max(1, rect.w // 5)))] if active else 0
            height = max(2, int(min(1, amp ** 0.55 * 1.6) * (rect.h - 12)))
            x = rect.x + 9 + i * (rect.w - 18) / min(65, max(1, rect.w // 5))
            pg.draw.line(self.screen, TEAL if active else (211, 219, 230), (x, mid - height // 2), (x, mid + height // 2), 2)

    def draw(self):
        if self.mode == "device":
            self.draw_device()
            return
        w, h = self.screen.get_size()
        self.screen.fill(BG)
        self.buttons = []
        self._drawing_modal = False
        busy, state = self.recorder.busy or self.trainer.busy or self.hid.busy or self.simulator is not None, self.recorder.snapshot()
        active_id = self.recorder.row_id if self.recorder.busy else None
        self.panel((32, 30, 44, 44), TEAL, None, 12)
        for i, length in enumerate([10, 20, 29, 17, 10]):
            pg.draw.line(self.screen, WHITE, (43 + i * 5, 52 - length // 2), (43 + i * 5, 52 + length // 2), 3)
        self.text("关键词采集工作台", (90, 25), 26, INK, True)
        self.text("KWS STUDIO   /   让每一次唤醒，都有回应", (92, 64), 13, MUTED)
        self.draw_mode_switch((w - 474, 30, 244, 42))
        self.button("选择麦克风", (w - 218, 30, 186, 42), self.choose_device, icon="mic", enabled=not busy)
        self.text(self.device_name, (w - 340, 82), 12, MUTED, max_width=308)

        self.panel((32, 110, w - 64, 68))
        self.text("工作区", (52, 144), 15, INK, True, center_y=True)
        self.path_field.draw(self, (116, 124, w - 402, 40))
        self.button("选择文件夹", (w - 274, 124, 130, 40), self.choose_directory, icon="folder", enabled=not busy)
        self.button("应用", (w - 132, 124, 80, 40), self.change_workspace, "soft", enabled=not busy)

        self.panel((32, 196, w - 64, 142))
        self.panel((52, 216, 34, 34), TEAL_BG, None, 9)
        self.icon("mic", (69, 233))
        self.text("背景静音素材", (98, 214), 18, INK, True)
        self.text("采集真实环境底噪，帮助模型区分语音", (98, 245), 13, MUTED)
        silence_active = active_id == "silence"
        self.button("取消本次录音" if silence_active else "录制静音素材", (52, 281, 170, 38),
                    self.recorder.cancel if silence_active else self.record, "danger" if silence_active else "primary",
                    enabled=silence_active or (not busy and self.workspace is not None), icon="cancel" if silence_active else "record")
        silence_seconds = self.recording["silence_seconds"]
        noise_seconds = self.recording["noise_seconds"]
        self.text(f"{silence_seconds} 秒 · 自动保存", (236, 300), 13, MUTED, center_y=True)
        wave_rect = pg.Rect(426, 225, w - 724, 55)
        self.waveform(wave_rect, state["waveform"], silence_active, state["elapsed"] / silence_seconds)
        silence_time = state["elapsed"] if silence_active else 0
        self.text(f"{clock_text(silence_time)} / {clock_text(silence_seconds)}", (wave_rect.x, 296), 14, TEAL if silence_active else MUTED)
        self.text("录制中" if silence_active else "等待采集", (wave_rect.right - 63, 298), 12, TEAL if silence_active else MUTED)
        self.text(str(self.counts.get("silence", 0)), (w - 255, 212), 29, INK, True, max_width=203)
        self.text("份静音素材", (w - 255, 255), 12, MUTED)
        self.button("打开目录", (w - 256, 281, 204, 38),
                    lambda: self.open_folder(self.workspace.silence_dir), icon="folder", enabled=self.workspace is not None)

        self.panel((32, 354, w - 64, 74))
        self.text("非静音噪声素材", (52, 364), 18, INK, True)
        self.text("采集环境噪声或非关键词声音", (52, 397), 12, MUTED)
        noise_active = active_id == "unknown"
        self.button("取消录音" if noise_active else f"录音 {noise_seconds} 秒", (344, 371, 116, 40),
                    self.recorder.cancel if noise_active else lambda: self.record(noise=True),
                    "danger" if noise_active else "soft", enabled=noise_active or (not busy and self.workspace is not None),
                    icon="cancel" if noise_active else "record")
        noise_wave = pg.Rect(476, 364, w - 866, 28)
        self.waveform(noise_wave, state["waveform"], noise_active, state["elapsed"] / noise_seconds)
        self.text(f"{state['elapsed']:04.1f} / {noise_seconds:.1f} s" if noise_active else f"00:00 / {clock_text(noise_seconds)}",
                  (noise_wave.x + 3, 404), 11, TEAL if noise_active else MUTED)
        self.text(str(self.counts.get("unknown", 0)), (w - 362, 361), 26, INK, True, max_width=90)
        self.text("份噪声素材", (w - 362, 400), 12, MUTED)
        self.button("打开目录", (w - 256, 371, 204, 40),
                    lambda: self.open_folder(self.workspace.unknown_dir), icon="folder", enabled=self.workspace is not None)

        self.text("关键词素材", (34, 455), 21, INK, True)
        total = sum(self.counts.get(row.id, 0) for row in self.workspace.rows) if self.workspace else 0
        self.text(f"{len(self.fields)} 个关键词   /   {total} 份录音", (173, 464), 13, MUTED)
        self.text("组合键：{CTRL}+a / {F12} / {+}", (345, 464), 12, MUTED, max_width=w - 817)
        rows = self.workspace.rows if self.workspace else []
        shortcuts_on = bool(self.workspace and self.workspace.actions_enabled)
        self.shortcut_master_rect = pg.Rect(w - 460, 446, 188, 38)
        self.button("禁用全部快捷键" if shortcuts_on or not rows else "启用全部快捷键",
                    self.shortcut_master_rect, self.toggle_all_shortcuts,
                    "primary" if shortcuts_on else "off", enabled=bool(rows))
        self.button("删除行", (w - 260, 446, 104, 38), self.request_delete, enabled=not busy and bool(self.selected))
        self.button("添加行", (w - 144, 446, 112, 38), self.add_row, "soft", enabled=not busy and self.workspace is not None, icon="plus")

        rec_x, count_x, led_x, folder_x, toggle_x = w - 636, w - 404, w - 330, w - 254, w - 162
        keyword_width = min(280, 160 + (w - 1100) // 2)
        shortcut_x, wave_x = 112 + keyword_width, rec_x + 112
        for label, x in [("序号", 53), ("关键词（16 字符）", 110), ("输入组合键", shortcut_x + 10),
                         ("录音操作", rec_x + 14), ("波形 / 时长", wave_x + 3),
                         ("WAV 数量", count_x), ("LED 颜色", led_x), ("素材目录", folder_x + 10),
                         ("快捷键", toggle_x + 20)]:
            self.text(label, (x, 503), 12, MUTED)
        self.viewport = pg.Rect(32, 530, w - 64, max(74, h - 658))
        self.max_scroll = max(0, len(rows) * 84 - 10 - self.viewport.h)
        self.scroll = min(max(0, self.scroll), self.max_scroll)
        old_clip = self.screen.get_clip()
        self.screen.set_clip(self.viewport)
        self.row_rects = []
        # Hidden text fields must not remain clickable at their old positions.
        for field in [*self.fields.values(), *self.shortcut_fields.values()]:
            field.rect = pg.Rect(0, 0, 0, 0)
        for index, row in enumerate(rows):
            y = self.viewport.y + index * 84 - self.scroll
            rect = pg.Rect(33, y, w - 80, 74)
            if not rect.colliderect(self.viewport):
                continue
            is_active, is_selected = active_id == row.id, self.selected == row.id
            self.panel(rect, WHITE, (152, 207, 195) if is_selected else LINE, 10)
            if is_selected:
                pg.draw.rect(self.screen, TEAL, (34, y + 19, 3, 36), border_radius=2)
            self.row_rects.append((rect.clip(self.viewport), row.id))
            button_start = len(self.buttons)
            self.text(f"{index + 1:02d}", (53, y + 36), 17, TEAL if is_selected else MUTED, True, center_y=True)
            self.fields[row.id].draw(self, (100, y + 16, keyword_width, 42))
            self.shortcut_fields[row.id].draw(self, (shortcut_x, y + 16, rec_x - shortcut_x - 12, 42))
            self.button("取消录音" if is_active else f"录音 {KEYWORD_SECONDS} 秒", (rec_x, y + 17, 100, 40),
                        self.recorder.cancel if is_active else lambda r=row: self.record(r),
                        "danger" if is_active else "soft", enabled=is_active or not busy, icon="cancel" if is_active else "record")
            self.waveform((wave_x, y + 13, count_x - wave_x - 14, 28), state["waveform"], is_active,
                          state["elapsed"] / KEYWORD_SECONDS)
            elapsed = state["elapsed"] if is_active else 0
            self.text(f"{elapsed:04.1f} / {KEYWORD_SECONDS:.1f} s" if is_active else f"00:00 / {clock_text(KEYWORD_SECONDS)}", (wave_x + 3, y + 45), 11, TEAL if is_active else MUTED)
            self.panel((count_x, y + 21, 62, 32), (244, 246, 250), None, 7)
            self.text(f"{self.counts.get(row.id, 0)}  份", (count_x + 9, y + 37), 14, INK, center_y=True, max_width=46)
            color_rect = pg.Rect(led_x, y + 20, 64, 34)
            self.panel(color_rect, row.color, None, 8)
            pg.draw.circle(self.screen, WHITE, (color_rect.right - 14, color_rect.centery), 3)
            self.buttons.append((color_rect, lambda r=row: self.choose_color(r)))
            self.button("打开", (folder_x, y + 17, 80, 40), lambda r=row: self.open_folder(self.workspace.row_dir(r)), icon="folder")
            self.button("禁用" if row.enabled else "启用", (toggle_x, y + 17, 104, 40),
                        lambda r=row: self.toggle_shortcut(r), "primary" if row.enabled else "off")
            self.buttons[button_start:] = [(r.clip(self.viewport), fn) for r, fn in self.buttons[button_start:]]
        if not rows:
            self.text("点击「添加行」，创建你的第一个关键词。", (w // 2 - 170, self.viewport.centery), 17, MUTED)
        self.screen.set_clip(old_clip)
        self.scroll_track = pg.Rect(w - 39, self.viewport.y, 5, self.viewport.h)
        if self.max_scroll:
            pg.draw.rect(self.screen, LINE, self.scroll_track, border_radius=3)
            thumb_h = max(32, self.viewport.h * self.viewport.h / (len(rows) * 84))
            thumb_y = self.viewport.y + (self.viewport.h - thumb_h) * self.scroll / self.max_scroll
            pg.draw.rect(self.screen, (179, 195, 203), (w - 39, thumb_y, 5, thumb_h), border_radius=3)

        footer_y = h - 110
        pg.draw.line(self.screen, LINE, (32, footer_y), (w - 32, footer_y))
        self.text("采集格式", (34, footer_y + 24), 12, MUTED)
        self.text("16 kHz  /  16 bit PCM  /  单声道 WAV", (34, footer_y + 48), 14, INK)
        self.text("训练步数", (w - 692, footer_y + 45), 14, MUTED, center_y=True)
        self.steps_field.draw(self, (w - 620, footer_y + 23, 88, 44))
        training = self.trainer.busy
        self.button("正在停止…" if training and self.trainer.stopping else "停止训练" if training else "开始训练",
                    (w - 518, footer_y + 23, 160, 44),
                    self.stop_training if training else self.start_training, "danger" if training else "primary",
                    enabled=(not self.trainer.stopping if training else not busy and self.workspace is not None))
        self.button("导入模型到设备", (w - 344, footer_y + 23, 160, 44),
                    self.import_model, "purple", enabled=not busy)
        self.button("PC端仿真", (w - 170, footer_y + 23, 138, 44),
                    self.open_simulation, "soft", enabled=not busy and self.workspace is not None)
        self.text("●", (34, h - 32), 12, RED if self.notice_error else TEAL)
        status = state["warning"] or ("正在保存音频…" if busy and state["phase"] == "saving" else self.notice)
        shortcut_error = next((f.validation_error() for f in self.shortcut_fields.values()
                               if f.focus and f.validation_error()), "")
        if shortcut_error:
            status = f"组合键无效：{shortcut_error}"
        if training:
            status = self.trainer.status
        self.text(status, (53, h - 33), 12, RED if self.notice_error or state["warning"] or shortcut_error else MUTED, max_width=w - 290)
        self.text("所有素材保存在本地", (w - 177, h - 33), 12, MUTED)
        if self.modal:
            self.draw_modal()
        pg.display.flip()

    def draw_device(self):
        w, h = self.screen.get_size()
        self.screen.fill(BG)
        self.buttons = []
        self._drawing_modal = False
        connected = self.hid.connected and self.connection_state == "connected"
        editable = connected and not self.hid.busy
        snapshot = self.hid.snapshot if connected else None
        rows = self.workspace.rows if connected else []
        self.panel((32, 30, 44, 44), TEAL, None, 12)
        self.text("HID", (38, 52), 16, WHITE, True, center_y=True)
        self.text("关键词设备控制台", (90, 25), 26, INK, True)
        self.text("KWS STUDIO   /   设备模型、KWS 参数和动作开关", (92, 64), 13, MUTED)
        self.draw_mode_switch((w - 474, 30, 244, 42))
        self.button("刷新设备参数" if connected else "重新连接", (w - 218, 30, 186, 42),
                    lambda: self.hid.submit("refresh") if connected else self.connect_device(),
                    "soft", enabled=not self.hid.busy)
        card_gap = 12
        workspace_width = (w - 64 - card_gap) * 2 // 3
        status_x = 32 + workspace_width + card_gap
        status_width = w - 32 - status_x
        self.panel((32, 110, workspace_width, 106))
        self.text("设备工作区", (52, 128), 15, INK, True)
        self.text(str(self.workspace.path), (52, 166), 15, MUTED, max_width=workspace_width - 40)
        self.path_field.rect = self.steps_field.rect = pg.Rect(0, 0, 0, 0)
        self.panel((status_x, 110, status_width, 106))
        title = "设备已连接" if connected else "正在连接设备…" if self.connection_state == "connecting" else "连接失败"
        self.text(title, (status_x + 20, 123), 21, TEAL if connected else RED if self.connection_state == "failed" else INK, True)
        if snapshot:
            info = snapshot["info"]
            self.text(f"设备 {info['device_id']}", (status_x + 20, 158), 13, MUTED,
                      max_width=status_width - 40)
            self.text(f"固件 {info.get('firmware_version', '')}   ·   {len(rows)} 个关键词",
                      (status_x + 20, 183), 13, MUTED, max_width=status_width - 40)
        else:
            self.text("请连接 USB 数据口后重试。", (status_x + 20, 170), 13, MUTED,
                      max_width=status_width - 40)
        filter_supported = connected and bool(snapshot["caps"].get("supports_runtime_kws_filter"))
        filter_editable = filter_supported and not self.hid.busy
        self.device_filter_rects = {}
        settings = (self.device_filter_pending or device_filter_values(snapshot["status"]["kws_filter"])
                    if filter_supported else None)
        self.panel((32, 228, w - 64, 132))
        self.text("KWS-Filter", (52, 239), 18, INK, True)
        self.text("设备参数 · 松开滑块后下发" if filter_supported else "连接设备后读取参数", (193, 249), 12, MUTED)
        slider_width = max(150, w // 2 - 270)
        self.draw_device_filter_slider("keyword_threshold_percent", "关键词阈值 (%)", 52, 258,
                                       slider_width, filter_editable, settings)
        self.draw_device_filter_slider("cooldown_ms", "冷却时间 (ms)", w // 2 + 16, 258,
                                       slider_width, filter_editable, settings)
        self.draw_device_filter_slider("stable_frames", "稳定帧数", 52, 301,
                                       slider_width, filter_editable, settings)
        self.text("需要 VAD 检查", (w // 2 + 16, 319), 14, INK if filter_editable else MUTED, center_y=True)
        vad = bool(settings and settings["need_vad_checked"])
        self.button("开启" if vad else "关闭", (w // 2 + 161, 300, 94, 38), self.toggle_device_vad,
                    "primary" if vad else "off", enabled=filter_editable)
        self.text("关键词参数", (34, 381), 21, INK, True)
        self.text("参数修改后自动下发到设备", (184, 391), 13, MUTED)
        master = bool(rows) and self.workspace.actions_enabled
        self.shortcut_master_rect = pg.Rect(w - 252, 373, 220, 38)
        self.button("禁用全部快捷键" if master else "启用全部快捷键", self.shortcut_master_rect,
                    self.toggle_all_shortcuts, "primary" if master else "off", enabled=editable and bool(rows))
        shortcut_x, led_x, toggle_x = 322, w - 282, w - 162
        for text, x in (("类别 index", 52), ("关键词（来自设备）", 132), ("输入组合键", shortcut_x + 10),
                        ("LED 颜色", led_x), ("快捷键", toggle_x + 20)):
            self.text(text, (x, 426), 12, MUTED)
        self.viewport = pg.Rect(32, 452, w - 64, max(74, h - 580))
        self.max_scroll = max(0, len(rows) * 84 - 10 - self.viewport.h)
        self.scroll = max(0, min(self.scroll, self.max_scroll))
        old_clip = self.screen.get_clip()
        self.screen.set_clip(self.viewport)
        self.row_rects = []
        for field in [*self.fields.values(), *self.shortcut_fields.values()]:
            field.rect = pg.Rect(0, 0, 0, 0)
        for index, row in enumerate(rows, 2):
            y = self.viewport.y + (index - 2) * 84 - self.scroll
            rect = pg.Rect(33, y, w - 80, 74)
            if not rect.colliderect(self.viewport):
                continue
            self.panel(rect, WHITE, (152, 207, 195) if self.selected == row.id else LINE, 10)
            self.row_rects.append((rect.clip(self.viewport), row.id))
            start = len(self.buttons)
            self.text(f"{index:02d}", (53, y + 36), 17, TEAL, True, center_y=True)
            self.text(row.text, (132, y + 36), 16, INK, center_y=True, max_width=174)
            self.shortcut_fields[row.id].draw(self, (shortcut_x, y + 16, led_x - shortcut_x - 24, 42))
            color_rect = pg.Rect(led_x, y + 20, 78, 34)
            self.panel(color_rect, row.color, None, 8)
            if editable:
                self.buttons.append((color_rect, lambda r=row: self.choose_color(r)))
            self.button("禁用" if row.enabled else "启用", (toggle_x, y + 17, 104, 40),
                        lambda r=row: self.toggle_shortcut(r), "primary" if row.enabled else "off", enabled=editable)
            self.buttons[start:] = [(rect.clip(self.viewport), fn) for rect, fn in self.buttons[start:]]
        if not rows:
            self.text("连接成功后显示设备中的关键词、颜色和快捷键。", (w // 2 - 210, self.viewport.centery), 17, MUTED)
        self.screen.set_clip(old_clip)
        self.scroll_track = pg.Rect(w - 39, self.viewport.y, 5, self.viewport.h)
        if self.max_scroll:
            pg.draw.rect(self.screen, LINE, self.scroll_track, border_radius=3)
            thumb_h = max(32, self.viewport.h * self.viewport.h / (len(rows) * 84))
            thumb_y = self.viewport.y + (self.viewport.h - thumb_h) * self.scroll / self.max_scroll
            pg.draw.rect(self.screen, (179, 195, 203), (w - 39, thumb_y, 5, thumb_h), border_radius=3)
        pg.draw.line(self.screen, LINE, (32, h - 110), (w - 32, h - 110))
        self.text("录音、训练和模型导入请切换到「本地训练」模式", (34, h - 86), 14, MUTED)
        self.text("●", (34, h - 32), 12, RED if self.notice_error else TEAL)
        self.text(self.notice, (53, h - 33), 12, RED if self.notice_error else MUTED, max_width=w - 90)
        if self.modal:
            self.draw_modal()
        pg.display.flip()

    def placeholder(self, name):
        self.modal = {"type": "info", "title": name, "message": "该功能待补充，当前版本已预留入口。"}

    def draw_modal(self):
        self.buttons = []
        self._drawing_modal = True
        overlay = pg.Surface(self.screen.get_size(), pg.SRCALPHA)
        overlay.fill((25, 38, 55, 80))
        self.screen.blit(overlay, (0, 0))
        kind = self.modal["type"]
        width = 680 if kind == "devices" else 530
        height = 524 if kind in ("devices", "color") else 310 if kind == "training_history" else 260
        box = pg.Rect(0, 0, width, height)
        box.center = self.screen.get_rect().center
        self.panel(box, WHITE, LINE, 18)
        x, y = box.x + 28, box.y + 26
        title = {"devices": "选择麦克风", "color": "LED 响应颜色", "delete": "删除选中行", "info": self.modal.get("title", ""), "quit": "退出工作台", "training_history": "发现历史训练"}[kind]
        self.text(title, (x, y), 22, INK, True)
        self.button("关闭", (box.right - 88, y - 3, 60, 34), self.close_modal)
        if kind == "devices":
            self.text("选择一个音频输入设备；建议优先使用 MME 或 WASAPI。", (x, y + 43), 13, MUTED)
            self.button("使用系统默认麦克风", (x, y + 80, width - 56, 40), lambda: self.set_device(None), "soft")
            self.device_viewport = pg.Rect(x, y + 136, width - 56, 302)
            clip = self.screen.get_clip()
            self.screen.set_clip(self.device_viewport)
            maximum = max(0, len(self.devices) * 60 - self.device_viewport.h)
            self.device_scroll = max(0, min(self.device_scroll, maximum))
            if not self.devices:
                self.text("未发现麦克风，请检查设备连接和系统权限。", (x + 12, y + 160), 15, MUTED)
            for i, device in enumerate(self.devices):
                r = pg.Rect(x, self.device_viewport.y + i * 60 - self.device_scroll, width - 56, 54)
                if not r.colliderect(self.device_viewport):
                    continue
                selected = self.device == device["id"]
                self.panel(r, TEAL_BG if selected else BG, None, 8)
                self.icon("mic", (r.x + 21, r.y + 25), TEAL if selected else MUTED)
                self.text(device["name"], (r.x + 44, r.y + 7), 14, INK, max_width=r.w - 60)
                self.text(f"{device['host']}   ·   输入通道 {device['channels']}", (r.x + 44, r.y + 31), 11, MUTED)
                self.buttons.append((r.clip(self.device_viewport), lambda d=device: self.set_device(d)))
            self.screen.set_clip(clip)
            self.text("滚动查看全部设备", (x, box.bottom - 36), 12, MUTED)
        elif kind == "color":
            self.text("点击色板或拖动色条，为关键词选择灯光响应。", (x, y + 43), 13, MUTED)
            self.sv_rect = pg.Rect(x, y + 82, width - 56, 235)
            self.hue_rect = pg.Rect(x, y + 333, width - 56, 20)
            m = self.modal
            if m.get("surface_h") != m["h"]:
                surface = pg.Surface((240, 120))
                for sx in range(240):
                    for sy in range(120):
                        color = tuple(round(c * 255) for c in colorsys.hsv_to_rgb(m["h"], sx / 239, 1 - sy / 119))
                        surface.set_at((sx, sy), color)
                m["sv_surface"] = pg.transform.smoothscale(surface, self.sv_rect.size)
                m["surface_h"] = m["h"]
            self.screen.blit(m["sv_surface"], self.sv_rect)
            for i in range(self.hue_rect.w):
                color = tuple(round(c * 255) for c in colorsys.hsv_to_rgb(i / (self.hue_rect.w - 1), 1, 1))
                pg.draw.line(self.screen, color, (self.hue_rect.x + i, self.hue_rect.y), (self.hue_rect.x + i, self.hue_rect.bottom - 1))
            point = (round(self.sv_rect.x + m["s"] * (self.sv_rect.w - 1)), round(self.sv_rect.y + (1 - m["v"]) * (self.sv_rect.h - 1)))
            pg.draw.circle(self.screen, INK, point, 8, 2)
            pg.draw.circle(self.screen, WHITE, point, 6, 2)
            hue_x = self.hue_rect.x + round(m["h"] * (self.hue_rect.w - 1))
            pg.draw.rect(self.screen, WHITE, (hue_x - 3, self.hue_rect.y - 2, 6, self.hue_rect.h + 4), 2, border_radius=2)
            for i, color in enumerate(PALETTE + [(242, 92, 83), (76, 169, 235), (255, 255, 255), (30, 39, 54)]):
                rect = pg.Rect(x + i * 38, y + 377, 27, 27)
                self.panel(rect, color, LINE, 6)
                self.buttons.append((rect, lambda c=color: self.preset_color(c)))
            self.panel((x, box.bottom - 67, 52, 36), self.modal_color(), LINE, 8)
            self.text("颜色预览", (x + 64, box.bottom - 58), 13, MUTED)
            self.button("应用颜色", (box.right - 155, box.bottom - 70, 127, 42), self.apply_color, "primary")
        elif kind == "training_history":
            checkpoint = self.modal["checkpoint"]
            self.text(f"本次训练步数：{self.modal['steps']}，请选择训练方式。", (x, y + 58), 15, INK)
            self.text("接续将恢复最近一次历史的权重和优化器，再训练指定步数。" if checkpoint else "历史中没有可接续的权重，请选择重新开始。",
                      (x, y + 93), 14, MUTED, max_width=width - 56)
            if checkpoint:
                self.text(f"最近历史：{checkpoint.parent.name}", (x, y + 124), 13, MUTED, max_width=width - 56)
            self.text("重新开始会清空 training 和 esp32p4-dl 的全部内容。", (x, y + 158), 14, RED,
                      max_width=width - 56)
            self.button("取消", (x, box.bottom - 70, 82, 42), self.close_modal)
            self.button("接续最近训练", (x + 96, box.bottom - 70, 176, 42),
                        lambda: self.confirm_training(True), "primary", enabled=checkpoint is not None)
            self.button("清空并重新训练", (x + 286, box.bottom - 70, 188, 42),
                        lambda: self.confirm_training(False), "danger")
        else:
            message = {"delete": "删除后空目录会移除，有内容的目录会加时间戳归档。",
                       "quit": "当前录音尚未完成，退出将丢弃本次录音。"}.get(kind, self.modal.get("message", ""))
            self.text(message, (x, y + 68), 15, MUTED, max_width=width - 56)
            if kind == "delete":
                self.button("保留此行", (box.right - 270, box.bottom - 72, 112, 42), self.close_modal)
                self.button("删除行", (box.right - 140, box.bottom - 72, 112, 42), self.delete_row, "danger")
            elif kind == "quit":
                self.button("继续录制", (box.right - 270, box.bottom - 72, 112, 42), self.close_modal)
                self.button("丢弃并退出", (box.right - 140, box.bottom - 72, 112, 42), self.exit, "danger")
            else:
                self.button("知道了", (box.right - 140, box.bottom - 72, 112, 42), self.close_modal, "soft")

    def preset_color(self, color):
        self.modal["h"], self.modal["s"], self.modal["v"] = colorsys.rgb_to_hsv(*(c / 255 for c in color))

    def close_modal(self):
        self.modal = None
        self.color_drag = None

    def exit(self):
        self.stop_simulation()
        if self.save():
            if (self.mode == "device" and self.hid.operation in ("bindings", "filter")
                    and (self.hid.busy or not self.hid.results.empty())):
                self.exit_after_device_save = True
                self.tell("正在保存设备参数，完成后将自动退出。")
                return
            self.remember_workspace()
            try:
                self.hid.close()
                self.trainer.close()
            except (OSError, RuntimeError) as exc:
                self.tell(f"退出清理失败：{exc}", True)
                return
            self.running = False

    def color_at(self, pos):
        if self.color_drag == "sv":
            self.modal["s"] = min(1, max(0, (pos[0] - self.sv_rect.x) / (self.sv_rect.w - 1)))
            self.modal["v"] = 1 - min(1, max(0, (pos[1] - self.sv_rect.y) / (self.sv_rect.h - 1)))
        elif self.color_drag == "hue":
            self.modal["h"] = min(1, max(0, (pos[0] - self.hue_rect.x) / (self.hue_rect.w - 1)))

    def event(self, event):
        if event.type == pg.QUIT:
            if self.recorder.busy:
                self.modal = {"type": "quit"}
            else:
                self.exit()
            return
        if event.type == pg.VIDEORESIZE:
            self.screen = pg.display.set_mode((max(1100, event.w), max(740, event.h)), pg.RESIZABLE)
            return
        if self.simulator is not None:
            return
        if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE and self.modal:
            self.close_modal()
            return
        if event.type == pg.MOUSEWHEEL:
            if self.modal and self.modal["type"] == "devices":
                self.device_scroll -= event.y * 45
            elif not self.modal and self.viewport.collidepoint(pg.mouse.get_pos()):
                self.scroll = max(0, min(self.max_scroll, self.scroll - event.y * 40))
            return
        if event.type == pg.MOUSEBUTTONUP and event.button == 1:
            if self.filter_drag is not None:
                self.update_device_filter_drag(event.pos)
                self.filter_drag = None
                self.send_device_filter(self.device_filter_pending)
                return
            self.drag_scroll = False
            self.mode_drag = False
            self.color_drag = None
        if event.type == pg.MOUSEMOTION and self.mode_drag and not self.modal:
            self.set_mode("local" if event.pos[0] < self.mode_switch_rect.centerx else "device")
        if event.type == pg.MOUSEMOTION and self.filter_drag is not None:
            self.update_device_filter_drag(event.pos)
            return
        if (event.type == pg.MOUSEBUTTONDOWN and event.button == 1 and self.mode == "device"
                and self.hid.connected and not self.hid.busy and not self.modal):
            for name, rect in self.device_filter_rects.items():
                if rect.inflate(6, 24).collidepoint(event.pos):
                    self.filter_drag = name
                    self.device_filter_pending = device_filter_values(self.hid.snapshot["status"]["kws_filter"])
                    self.update_device_filter_drag(event.pos)
                    return
        if self.modal and self.modal["type"] == "color":
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                if self.sv_rect.collidepoint(event.pos):
                    self.color_drag = "sv"
                elif self.hue_rect.inflate(0, 10).collidepoint(event.pos):
                    self.color_drag = "hue"
                self.color_at(event.pos)
            elif event.type == pg.MOUSEMOTION and self.color_drag:
                self.color_at(event.pos)
        if event.type == pg.MOUSEMOTION and self.drag_scroll and self.max_scroll:
            self.scroll = int(min(1, max(0, (event.pos[1] - self.viewport.y) / self.viewport.h)) * self.max_scroll)
        if not self.modal:
            # Keep a stable list: pressing Enter can replace the current workspace.
            if self.mode == "local" and not self.recorder.busy and not self.trainer.busy and not self.hid.busy:
                self.path_field.event(event, self)
                self.steps_field.event(event, self)
            row_fields = ([self.shortcut_fields[identity] for identity in self.fields] if self.mode == "device" else
                          [field for identity in self.fields for field in (self.fields[identity], self.shortcut_fields[identity])])
            if self.hid.busy or (self.mode == "device" and not self.hid.connected):
                row_fields = []
            for field in row_fields:
                if field.event(event, self):
                    self.save_due = time.monotonic() + 0.45
            if event.type == pg.KEYDOWN and event.key == pg.K_TAB:
                all_fields = row_fields
                if self.mode == "local" and not self.recorder.busy and not self.trainer.busy and not self.hid.busy:
                    all_fields = [self.path_field] + all_fields + [self.steps_field]
                if not all_fields:
                    return
                current = next((i for i, f in enumerate(all_fields) if f.focus), -1)
                for field in all_fields:
                    field.focus = False
                target = all_fields[(current + (-1 if event.mod & pg.KMOD_SHIFT else 1)) % len(all_fields)]
                target.focus = True
                if target in row_fields:
                    index = row_fields.index(target) // (1 if self.mode == "device" else 2)
                    self.scroll = max(0, min(self.max_scroll, index * 84 - self.viewport.h // 2 + 42))
                pg.key.set_text_input_rect(target.rect)
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                self.mode_drag = self.mode_switch_rect.collidepoint(event.pos)
                if self.scroll_track.inflate(12, 0).collidepoint(event.pos):
                    self.drag_scroll = True
                for rect, identity in self.row_rects:
                    if rect.collidepoint(event.pos):
                        self.selected = identity
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            for rect, action in reversed(self.buttons):
                if rect.collidepoint(event.pos):
                    # A dialog or workspace change invalidates this frame's hit targets.
                    self.buttons = []
                    action()
                    break

    def update(self):
        if self.simulator is not None and self.simulator.poll() is not None:
            code = self.simulator.returncode
            self.stop_simulation()
            if code:
                self.tell(f"PC端仿真异常退出，详情见 {self.simulator_error_log}", True)
            else:
                self.tell("PC端仿真已结束，参数已保存到工作区。")
        self.update_hid()
        training_notice = self.trainer.update()
        if training_notice is not None:
            self.tell(*training_notice)
        while True:
            try:
                result = self.recorder.results.get_nowait()
            except queue.Empty:
                break
            if result.error:
                self.tell(f"录音失败：{result.error}", True)
            elif result.cancelled:
                self.tell("本次录音已取消，未保存不完整素材。")
            else:
                self.tell(f"录音已保存：{result.path.name}")
            self.last_count = 0
        now = time.monotonic()
        if self.save_due and now >= self.save_due:
            self.save()
        if self.workspace and now - self.last_count > 1:
            try:
                counts = {"silence": Workspace.wav_count(self.workspace.silence_dir),
                          "unknown": Workspace.wav_count(self.workspace.unknown_dir)}
                counts.update({row.id: Workspace.wav_count(self.workspace.row_dir(row)) for row in self.workspace.rows})
                self.counts = counts
            except OSError as exc:
                self.tell(f"无法统计文件：{exc}", True)
            self.last_count = now

        if (self.mode == "device" and self.hid.connected and not self.hid.busy and now >= self.next_device_poll
                and not self.save_due and not self.modal and self.filter_drag is None
                and not any(field.focus for field in self.shortcut_fields.values())):
            self.next_device_poll = now + DEVICE_POLL_INTERVAL
            self.hid.submit("poll")

    def update_hid(self):
        while True:
            try:
                result = self.hid.results.get_nowait()
            except queue.Empty:
                break
            kind = result["kind"]
            if kind == "progress":
                self.tell(result["message"])
                continue
            snapshot = result.get("snapshot")
            if snapshot is not None and self.mode == "device":
                try:
                    self.apply_device_snapshot(snapshot)
                except (OSError, ValueError, KeyError) as exc:
                    self.tell(f"设备参数已读取，本地镜像保存失败：{exc}", True)
            if kind == "error":
                if result.get("operation") == "filter":
                    self.device_filter_pending = None
                if result.get("operation") == "bindings":
                    self.queued_device_filter = None
                self.exit_after_device_save = False
                if self.mode == "device" and result.get("disconnected"):
                    self.connection_state = "failed"
                self.tell(f"设备操作失败：{result['message']}", True)
            elif kind in ("connect", "refresh"):
                self.tell("设备已连接，已读取颜色、快捷键、开关和 KWS 参数。")
            elif kind == "bindings":
                self.tell("写入完成")
            elif kind == "filter":
                self.tell("设备 KWS 参数已更新并保存。")
            elif kind == "deploy":
                self.tell("模型已激活，颜色、快捷键、开关和 KWS 参数已写入设备。")
        if self.queued_device_filter is not None and self.mode == "device" and self.hid.connected and not self.hid.busy:
            settings, self.queued_device_filter = self.queued_device_filter, None
            self.send_device_filter(settings)
        if self.exit_after_device_save and not self.hid.busy:
            self.exit_after_device_save = False
            self.exit()

    def run(self):
        timer = pg.time.Clock()
        try:
            self.update()
            self.draw()
            while self.running:
                for event in pg.event.get():
                    self.event(event)
                self.update()
                self.draw()
                timer.tick(30)
        finally:
            try:
                try:
                    self.stop_simulation()
                    self.hid.close()
                finally:
                    self.trainer.close()
            finally:
                self.recorder.close()
                if self.tk_root:
                    self.tk_root.destroy()
                pg.quit()


def main():
    parser = argparse.ArgumentParser(description="KWS Studio · 关键词音频采集工作台")
    parser.add_argument("--workspace", type=Path, help="指定工作区路径；未指定时恢复上次工作区，首次默认 workspace/test")
    parser.add_argument("--screenshot", type=Path, help="渲染一帧到 PNG 后退出，用于布局检查")
    args = parser.parse_args()
    app = App(args.workspace)
    if args.screenshot:
        app.update()
        app.draw()
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        pg.image.save(app.screen, str(args.screenshot))
        pg.quit()
    else:
        app.run()
