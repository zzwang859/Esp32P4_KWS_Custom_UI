"""Secondary PC simulation window for a workspace's latest ONNX model."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from math import gcd
import importlib.util
import json
from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from .model import DEFAULT_KWS_FILTER


TRAINING_ROOT = Path(__file__).resolve().parents[1] / "ds-tc-resnet"
INFER_SCRIPT = TRAINING_ROOT / "onnx/run/infer.py"


def latest_model(workspace: Path) -> Path:
    latest = json.loads((workspace / "esp32p4-dl/latest.json").read_text(encoding="utf-8"))
    path = Path(latest["source_onnx"])
    if not path.is_file():
        path = workspace / "training" / latest["bundle"] / "onnx/ds_tc_resnet.onnx"
    path = path.resolve()
    if not latest.get("success") or not path.is_file() or not path.is_relative_to(workspace.resolve()):
        raise ValueError("当前工作区没有可用的 ONNX 模型，请先完成训练")
    if hashlib.sha256(path.read_bytes()).hexdigest() != latest["source_sha256"]:
        raise ValueError("ONNX 模型校验失败，请重新训练")
    return path


def load_inference_module():
    spec = importlib.util.spec_from_file_location("kws_live_infer", INFER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_frame(frame: int, logits: np.ndarray, probabilities: np.ndarray,
                 detected: int) -> tuple[str, tuple[int, int]]:
    """Return the plain log line and the character span of its most probable index."""
    prefix = f"[{frame * .01:8.2f}s] "
    parts = []
    cursor = len(prefix)
    highlight = (0, 0)
    top_index = int(np.argmax(probabilities))
    for index, (value, probability) in enumerate(zip(logits, probabilities)):
        if parts:
            cursor += 2
        cell = f"{index}:{value:.2f}[{probability * 100:.2f}%]"
        if index == top_index:
            highlight = cursor, cursor + len(cell)
        parts.append(cell)
        cursor += len(cell)
    suffix = f"  >>> 识别：{detected}" if detected >= 2 else ""
    return prefix + "  ".join(parts) + suffix + "\n", highlight


def cooldown_from_slider(position: str | int) -> int:
    """Expose the device's 1 ms minimum alongside the existing 10 ms steps."""
    step = int(float(position))
    return 1 if step == 0 else step * 10


class SimulationWindow:
    def __init__(self, workspace: Path, device: int | None):
        self.workspace = workspace
        self.device = device
        self.model_path = latest_model(workspace)
        self.config_path = workspace / "workspace.json"
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        settings = dict(data.get("kws_filter", DEFAULT_KWS_FILTER))
        old_zero_values = any(settings[name] == 0 for name in ("cooldown_ms", "stable_frames"))
        for name in ("cooldown_ms", "stable_frames"):
            settings[name] = max(1, settings[name])
        self.root = tk.Tk()
        self.root.title("PC端仿真 · " + workspace.name)
        self.root.geometry("1180x820")
        self.root.minsize(900, 650)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.stop_event = threading.Event()
        self.messages: queue.Queue = queue.Queue()
        self.settings: queue.Queue = queue.Queue()
        self.pending_save = None
        self.log_path = workspace / "simulation" / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".log")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.values = {name: tk.IntVar(value=settings[name]) for name in
                       ("keyword_threshold_percent", "cooldown_ms", "stable_frames")}
        self.vad = tk.BooleanVar(value=settings["need_vad_checked"])
        self.initial_settings = dict(settings)
        self.status = tk.StringVar(value="正在初始化模型与麦克风…")
        self.wave = [0.0] * 100
        self.build()
        if old_zero_values:
            self.save_settings()
        self.worker = threading.Thread(target=self.run_audio, daemon=True)
        self.worker.start()
        self.root.after(30, self.poll)

    def build(self):
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="PC端仿真", font=("Microsoft YaHei UI", 19, "bold")).pack(anchor="w")
        device_text = f"录音端口：{'系统默认麦克风' if self.device is None else sd.query_devices(self.device)['name']}"
        ttk.Label(outer, text=device_text).pack(anchor="w", pady=(2, 12))
        controls = ttk.LabelFrame(outer, text="KWS-Filter 参数", padding=12)
        controls.pack(fill="x")
        labels = (("keyword_threshold_percent", "关键词阈值 (%)", 0, 100),
                  ("cooldown_ms", "冷却时间 (ms)", 0, 100),
                  ("stable_frames", "稳定帧数", 1, 50))
        for row, (name, label, minimum, maximum) in enumerate(labels):
            ttk.Label(controls, text=label, width=18).grid(row=row, column=0, sticky="w")
            if name == "cooldown_ms":
                slider_position = 0 if self.values[name].get() == 1 else self.values[name].get() // 10
                self.cooldown_slider_position = tk.IntVar(value=slider_position)
                scale = tk.Scale(controls, from_=minimum, to=maximum, resolution=1, orient="horizontal",
                                 showvalue=False, variable=self.cooldown_slider_position,
                                 command=lambda position: self.set_cooldown_from_slider(position))
            else:
                scale = tk.Scale(controls, from_=minimum, to=maximum, resolution=1, orient="horizontal",
                                 showvalue=False, variable=self.values[name], command=lambda _v: self.schedule_save())
            scale.grid(row=row, column=1, sticky="ew")
            ttk.Label(controls, textvariable=self.values[name], width=6).grid(row=row, column=2)
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="需要 VAD 检查", width=18).grid(row=3, column=0, sticky="w")
        self.vad_switch = tk.Canvas(controls, width=60, height=30, highlightthickness=0, cursor="hand2")
        self.vad_switch.grid(row=3, column=1, sticky="w", pady=5)
        self.vad_switch.bind("<Button-1>", self.toggle_vad)
        self.draw_vad_switch()
        ttk.Label(outer, textvariable=self.status).pack(anchor="w", pady=(12, 4))
        self.canvas = tk.Canvas(outer, height=125, background="#f5faf9", highlightthickness=1,
                                highlightbackground="#d9e7e3")
        self.canvas.pack(fill="x", pady=(0, 12))
        logs = ttk.Frame(outer)
        logs.pack(fill="both", expand=True)
        logs.columnconfigure(0, weight=1)
        logs.rowconfigure(0, weight=3)
        logs.rowconfigure(1, weight=1)
        output = ttk.LabelFrame(logs, text="逐帧 logits[概率] · 自动写入日志", padding=8)
        output.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        self.text = tk.Text(output, wrap="none", height=1, font=("Consolas", 10), state="disabled")
        self.text.tag_configure("max_probability", foreground="#2563eb")
        vertical = ttk.Scrollbar(output, orient="vertical", command=self.text.yview)
        horizontal = ttk.Scrollbar(output, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        output.rowconfigure(0, weight=1)
        output.columnconfigure(0, weight=1)
        events = ttk.LabelFrame(logs, text="消抖触发结果（index ≥ 2）", padding=8)
        events.grid(row=1, column=0, sticky="nsew")
        self.events_text = tk.Text(events, wrap="word", height=1, font=("Consolas", 10), state="disabled")
        events_scroll = ttk.Scrollbar(events, orient="vertical", command=self.events_text.yview)
        self.events_text.configure(yscrollcommand=events_scroll.set)
        self.events_text.grid(row=0, column=0, sticky="nsew")
        events_scroll.grid(row=0, column=1, sticky="ns")
        events.rowconfigure(0, weight=1)
        events.columnconfigure(0, weight=1)
        ttk.Label(outer, text=f"日志：{self.log_path}").pack(anchor="w", pady=(8, 0))
        self.canvas.bind("<Configure>", lambda _event: self.draw_wave())

    def draw_vad_switch(self):
        canvas = self.vad_switch
        canvas.delete("all")
        enabled = self.vad.get()
        color = "#189c89" if enabled else "#cbd3dd"
        canvas.create_oval(2, 2, 30, 28, fill=color, outline=color)
        canvas.create_oval(30, 2, 58, 28, fill=color, outline=color)
        canvas.create_rectangle(16, 2, 44, 28, fill=color, outline=color)
        x = 43 if enabled else 17
        canvas.create_oval(x - 10, 5, x + 10, 25, fill="white", outline="white")

    def toggle_vad(self, _event):
        self.vad.set(not self.vad.get())
        self.draw_vad_switch()
        self.schedule_save()

    def set_cooldown_from_slider(self, position):
        self.values["cooldown_ms"].set(cooldown_from_slider(position))
        self.schedule_save()

    def current_settings(self) -> dict:
        return {**{key: value.get() for key, value in self.values.items()},
                "need_vad_checked": self.vad.get()}

    def schedule_save(self):
        self.settings.put(self.current_settings())
        if self.pending_save is not None:
            self.root.after_cancel(self.pending_save)
        self.pending_save = self.root.after(150, self.save_settings)

    def save_settings(self):
        self.pending_save = None
        try:
            config = self.current_settings()
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            data["kws_filter"] = config
            temporary = self.config_path.with_name("workspace.simulation.tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.config_path)
        except (OSError, ValueError) as exc:
            self.status.set(f"参数保存失败：{exc}")

    def run_audio(self):
        try:
            module = load_inference_module()
            config = module.load_config()
            engine = module.StreamingInference(config, self.model_path, self.initial_settings)
            try:
                sd.check_input_settings(device=self.device, channels=1, dtype="int16", samplerate=16000)
                rate = 16000
            except sd.PortAudioError:
                rate = round(sd.query_devices(self.device, "input")["default_samplerate"])
                sd.check_input_settings(device=self.device, channels=1, dtype="int16", samplerate=rate)
            blocks: queue.Queue = queue.Queue(maxsize=100)

            def callback(indata, _frames, _time, status):
                if status:
                    self.messages.put(("error", f"音频设备状态：{status}"))
                try:
                    blocks.put_nowait(bytes(indata))
                except queue.Full:
                    self.messages.put(("error", "音频处理落后，已丢失采样"))

            self.messages.put(("status", f"实时运行中 · {len(engine.labels)} 类 · {rate} Hz 输入"))
            source = np.empty(0, dtype=np.float32)
            source_history = np.zeros(rate // 100, dtype=np.float32)
            source_count = rate // 100
            factor = gcd(rate, 16000)
            with self.log_path.open("w", encoding="utf-8", buffering=1) as log:
                log.write(f"model: {self.model_path}\ndevice: {self.device}\nindices: 0..{len(engine.labels) - 1}\n")
                with sd.RawInputStream(samplerate=rate, channels=1, dtype="int16", device=self.device,
                                       callback=callback):
                    while not self.stop_event.is_set():
                        try:
                            raw = blocks.get(timeout=0.2)
                        except queue.Empty:
                            continue
                        source = np.concatenate((source, np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768))
                        while len(source) >= source_count:
                            updated = None
                            while not self.settings.empty():
                                updated = self.settings.get_nowait()
                            if updated is not None:
                                engine.set_filter(updated)
                            segment, source = source[:source_count], source[source_count:]
                            if rate == 16000:
                                samples = segment
                            else:
                                combined = np.concatenate((source_history, segment))
                                samples = resample_poly(combined, 16000 // factor, rate // factor)[160:]
                                source_history = segment[-rate // 100:].copy()
                            self.messages.put(("wave", [float(np.max(np.abs(samples)))]))
                            for frame, logits, probabilities, detected in engine.feed(samples):
                                line, highlight = format_frame(frame, logits, probabilities, detected)
                                log.write(line)
                                self.messages.put(("line", (line, highlight)))
                                if detected >= 2:
                                    self.messages.put(("event", f"[{frame * .01:8.2f}s] index={detected} "
                                                                 f"{engine.labels[detected]}  "
                                                                 f"{probabilities[detected] * 100:.2f}%\n"))
        except Exception as exc:
            self.messages.put(("error", f"仿真停止：{exc}"))

    def draw_wave(self):
        self.canvas.delete("all")
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        middle = height // 2
        self.canvas.create_line(0, middle, width, middle, fill="#c6dcd6")
        for index, amplitude in enumerate(self.wave):
            x = index * width / len(self.wave)
            size = max(2, min(middle - 4, int(amplitude ** .55 * (middle - 4))))
            self.canvas.create_line(x, middle - size, x, middle + size, fill="#189c89", width=2)

    def poll(self):
        lines = []
        events = []
        redraw = False
        for _ in range(2000):
            try:
                kind, value = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                lines.append(value)
            elif kind == "event":
                events.append(value)
            elif kind == "wave":
                self.wave = (self.wave + value)[-100:]
                redraw = True
            else:
                self.status.set(value)
        if lines:
            self.text.configure(state="normal")
            for line, (start, finish) in lines:
                position = self.text.index("end-1c")
                self.text.insert("end", line)
                self.text.tag_add("max_probability", f"{position}+{start}c", f"{position}+{finish}c")
            if int(self.text.index("end-1c").split(".")[0]) > 2500:
                self.text.delete("1.0", "500.0")
            self.text.see("end")
            self.text.configure(state="disabled")
        if events:
            self.events_text.configure(state="normal")
            self.events_text.insert("end", "".join(events))
            if int(self.events_text.index("end-1c").split(".")[0]) > 1500:
                self.events_text.delete("1.0", "300.0")
            self.events_text.see("end")
            self.events_text.configure(state="disabled")
        if redraw:
            self.draw_wave()
        self.root.after(30, self.poll)

    def close(self):
        if self.pending_save is not None:
            self.root.after_cancel(self.pending_save)
            self.save_settings()
        self.stop_event.set()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    window = SimulationWindow(args.workspace.resolve(), args.device)
    window.root.mainloop()


if __name__ == "__main__":
    main()
