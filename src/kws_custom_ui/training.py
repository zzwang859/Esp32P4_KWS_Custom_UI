"""Launch the standalone trainer without blocking pygame or importing torch into it."""

from __future__ import annotations

from datetime import datetime
import os
import json
from pathlib import Path
import subprocess
import sys
import webbrowser
from urllib.parse import urlsplit

import yaml

from .processes import ProcessGroup, python_executable
from .history import clear_training_history, history_root


TRAINING_ROOT = Path(__file__).resolve().parents[1] / "ds-tc-resnet"


def default_steps() -> int:
    try:
        config = yaml.safe_load((TRAINING_ROOT / "config.yaml").read_text(encoding="utf-8"))
        steps = int(config["train"]["steps"])
        return steps if steps > 0 else 3000
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError):
        return 3000


class Trainer:
    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.run_dir: Path | None = None
        self._offset = 0
        self._reported = True
        self._last_line = ""
        self.device: str | None = None
        self.stopping = False
        self._groups: list[ProcessGroup] = []
        self._browser_opened = False

    @property
    def busy(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def status(self) -> str:
        message = "正在停止训练…" if self.stopping else self._last_line or "正在准备训练数据并打开 TensorBoard…"
        prefix = f"训练中({self.device}):" if self.device else "准备训练:"
        return prefix + message

    def start(self, workspace: Path, steps: int | None = None, *, resume: Path | None = None, restart: bool = False) -> Path:
        if self.busy:
            raise RuntimeError("已有训练任务正在运行。")
        if steps is not None and (type(steps) is not int or steps < 1):
            raise ValueError("训练步数必须为正整数。")
        script = TRAINING_ROOT / "workspace_train.py"
        if not script.is_file():
            raise RuntimeError(f"训练入口不存在：{script}")
        root = history_root(workspace)
        if restart and resume is not None:
            raise ValueError("不能同时选择接续训练和重新开始。")
        if resume is not None:
            resume = resume.resolve()
            if not resume.is_file() or not resume.is_relative_to(root):
                raise ValueError("接续权重不存在或不属于当前工作区。")
        if restart:
            self.close()
            clear_training_history(workspace)
        self.run_dir = root / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.run_dir.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-u", str(script), "--workspace", str(workspace), "--run-dir", str(self.run_dir)]
        group = ProcessGroup()
        if group.name:
            command.extend(["--job-name", group.name])
        command.append("--no-browser")
        if steps is not None:
            command.extend(["--steps", str(steps)])
        if resume is not None:
            command.extend(["--resume", str(resume)])
        environment = os.environ.copy()
        environment.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        command[0] = python_executable(environment)
        try:
            with (self.run_dir / "training.log").open("w", encoding="utf-8") as output:
                self.process = subprocess.Popen(command, cwd=TRAINING_ROOT, env=environment,
                                                stdout=output, stderr=subprocess.STDOUT,
                                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                                start_new_session=os.name != "nt")
            group.pid = self.process.pid
            self._groups.append(group)
        except BaseException:
            group.close()
            raise
        self._offset, self._reported, self._last_line = 0, False, ""
        self.device, self.stopping = None, False
        self._browser_opened = False
        return self.run_dir

    def close(self) -> None:
        """Save active training, then release every service launched by this UI."""
        try:
            if self.busy:
                try:
                    self.stop()
                except OSError:
                    pass  # The owned process group is still terminated below.
                try:
                    self.process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            errors = []
            try:
                for group in self._groups:
                    try:
                        group.close()
                    except (OSError, RuntimeError) as exc:
                        errors.append(exc)
            finally:
                self._groups.clear()
                if self.process is not None:
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.terminate()
                        self.process.wait(timeout=5)
            if errors:
                raise RuntimeError("; ".join(str(error) for error in errors))
            self._reported = True

    def _open_tensorboard(self):
        if self._browser_opened or self.stopping:
            return
        metadata = self.run_dir / "tensorboard.json"
        try:
            data = json.loads(metadata.read_text(encoding="utf-8"))
            url = urlsplit(data["url"])
            if url.scheme != "http" or url.hostname != "127.0.0.1":
                return
            self._browser_opened = True
            # Launch from the UI so the user's browser is outside the training job.
            data["browser_opened"] = webbrowser.open(data["url"], new=2)
            metadata.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except (OSError, ValueError, KeyError, TypeError, webbrowser.Error):
            pass

    def stop(self) -> None:
        if not self.busy or self.stopping:
            return
        (self.run_dir / "stop.requested").touch()
        self.stopping = True

    def update(self) -> tuple[str, bool] | None:
        if self.process is None or self._reported:
            return None
        self._open_tensorboard()
        with (self.run_dir / "training.log").open("r", encoding="utf-8", errors="replace") as output:
            output.seek(self._offset)
            lines = output.read().splitlines()
            self._offset = output.tell()
        for line in lines:
            if line.startswith("TRAINING_DEVICE="):
                device = line.partition("=")[2].strip()
                if device in ("cpu", "cuda"):
                    self.device = device
                    self._last_line = "已选择训练设备，正在加载语料…"
            elif line.strip():
                self._last_line = line.strip()
        code = self.process.poll()
        if code is not None:
            self._reported = True
            if self.stopping or (self.run_dir / "stop.requested").exists():
                return f"训练已停止；已完成的权重和日志保存在 {self.run_dir}", False
            if code:
                return f"训练失败：{self._last_line}；日志：{self.run_dir / 'training.log'}", True
            return f"训练、ONNX 导出及 ESP32-P4 转换完成：{self.run_dir.parent.parent / 'esp32p4-dl' / self.run_dir.name}", False
        if lines:
            return self.status, False
        return None
