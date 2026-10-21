"""Workspace storage. Row identities remain stable when rows are reordered/deleted."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
from uuid import uuid4

from .keyboard import saved_reports


PALETTE = [(45, 184, 163), (102, 119, 232), (239, 166, 75), (220, 112, 160)]
FLAG_ACTION = 0x01
FLAG_LIGHT = 0x02
FLAG_KEYBOARD = 0x04
FLAG_OUTPUTS = FLAG_LIGHT | FLAG_KEYBOARD
DEFAULT_FLAGS = FLAG_ACTION | FLAG_OUTPUTS
DEFAULT_KWS_FILTER = dict(keyword_threshold_percent=80, cooldown_ms=500,
                          stable_frames=3, need_vad_checked=False)


@dataclass
class Keyword:
    id: str
    text: str = ""
    color: tuple[int, int, int] = PALETTE[0]
    directory: str = ""
    shortcut: str = ""
    flags: int = DEFAULT_FLAGS

    @property
    def enabled(self) -> bool:
        return self.flags & DEFAULT_FLAGS == DEFAULT_FLAGS

    @classmethod
    def create(cls, index: int = 0) -> Keyword:
        return cls(uuid4().hex[:12], color=PALETTE[index % len(PALETTE)])


class Workspace:
    CONFIG = "workspace.json"

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.rows: list[Keyword] = []
        self.kws_filter = dict(DEFAULT_KWS_FILTER)
        config = self.path / self.CONFIG
        if config.exists():
            try:
                data = json.loads(config.read_text(encoding="utf-8"))
                if data.get("version") != 1 or not isinstance(data.get("rows"), list):
                    raise ValueError("不支持的工作区配置格式")
                settings = data.get("kws_filter", DEFAULT_KWS_FILTER)
                # Device workspaces may contain values above the PC simulation sliders.
                bounds = {"keyword_threshold_percent": (0, 100), "cooldown_ms": (0, 60000),
                          "stable_frames": (0, 1000)}
                if not isinstance(settings, dict) or set(settings) != {*bounds, "need_vad_checked"}:
                    raise ValueError("KWS 滤波配置格式无效")
                for name, (low, high) in bounds.items():
                    if type(settings[name]) is not int or not low <= settings[name] <= high:
                        raise ValueError(f"KWS 滤波参数 {name} 必须是 {low}～{high} 的整数")
                if type(settings["need_vad_checked"]) is not bool:
                    raise ValueError("KWS 滤波参数 need_vad_checked 无效")
                self.kws_filter = dict(settings)
                seen = set()
                seen_directories = set()
                for item in data["rows"]:
                    identity, color = item["id"], item["color"]
                    if not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{12}", identity) or identity in seen:
                        raise ValueError("行标识无效或重复")
                    if not isinstance(color, (list, tuple)) or len(color) != 3 or any(type(c) is not int or not 0 <= c <= 255 for c in color):
                        raise ValueError("颜色配置无效")
                    if not isinstance(item["text"], str):
                        raise ValueError("关键词必须为文本")
                    directory = item.get("directory", "")
                    if "directory" in item:
                        if not isinstance(directory, str) or not re.fullmatch(r"keyword[1-9][0-9]*", directory) or directory in seen_directories:
                            raise ValueError("关键词目录无效或重复")
                        seen_directories.add(directory)
                    shortcut = item.get("shortcut", "")
                    if not isinstance(shortcut, str):
                        raise ValueError("组合键必须为文本")
                    if "flags" in item:
                        flags = item["flags"]
                        if type(flags) is not int or not 0 <= flags <= 0xFF:
                            raise ValueError("flags 必须为 0～255 的整数")
                    else:
                        enabled = item.get("enabled", True)
                        if type(enabled) is not bool:
                            raise ValueError("快捷键启用状态必须为布尔值")
                        flags = DEFAULT_FLAGS if enabled else 0
                    self.rows.append(Keyword(identity, item["text"], tuple(color), directory, shortcut, flags))
                    seen.add(identity)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise ValueError(f"无法读取 {config.name}：{exc}；原文件已保留。") from exc
        else:
            self.rows = [Keyword.create(i) for i in range(4)]
        self.silence_dir.mkdir(exist_ok=True)
        self.unknown_dir.mkdir(exist_ok=True)
        self._check_directory(self.path / "keywords")
        (self.path / "keywords").mkdir(exist_ok=True)
        migrated = []
        try:
            for row in self.rows:
                if not row.directory:
                    row.directory = self._next_directory()
                    legacy = self.path / "keywords" / row.id
                    self._check_directory(legacy)
                    if legacy.exists():
                        destination = self.row_dir(row)
                        self._check_directory(destination)
                        legacy.rename(destination)
                        migrated.append((legacy, destination))
                self._check_directory(self.row_dir(row))
                self.row_dir(row).mkdir(exist_ok=True)
            self.save()
        except (OSError, ValueError):
            for legacy, destination in reversed(migrated):
                destination.rename(legacy)
            raise

    @property
    def silence_dir(self) -> Path:
        return self.path / "silence"

    @property
    def unknown_dir(self) -> Path:
        return self.path / "unknown"

    @property
    def actions_enabled(self) -> bool:
        return bool(self.rows) and all(row.flags & FLAG_ACTION for row in self.rows)

    def row_dir(self, row: Keyword) -> Path:
        return self.path / "keywords" / row.directory

    def _check_directory(self, directory: Path) -> None:
        if directory.is_symlink() or not directory.resolve().is_relative_to(self.path):
            raise ValueError(f"目录必须位于当前工作区内，且不能是符号链接：{directory}")

    def _next_directory(self) -> str:
        names = {row.directory for row in self.rows}
        names.update(path.name for path in (self.path / "keywords").iterdir())
        numbers = [int(name[7:]) for name in names if re.fullmatch(r"keyword[1-9][0-9]*", name)]
        return f"keyword{max(numbers, default=0) + 1}"

    def save(self) -> None:
        data = {"version": 1, "audio": {"sample_rate": 16000, "sample_width": 2, "channels": 1},
                "kws_filter": self.kws_filter,
                "rows": [{"id": row.id, "directory": row.directory, "text": row.text, "color": row.color,
                          "shortcut": row.shortcut, "flags": row.flags, **saved_reports(row.shortcut)}
                         for row in self.rows]}
        temporary = self.path / (self.CONFIG + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path / self.CONFIG)

    def add(self) -> Keyword:
        row = Keyword.create(len(self.rows))
        row.flags = DEFAULT_FLAGS if not self.rows or self.actions_enabled else FLAG_OUTPUTS
        row.directory = self._next_directory()
        self._check_directory(self.row_dir(row))
        self.row_dir(row).mkdir(parents=True, exist_ok=True)
        self.rows.append(row)
        return row

    def delete(self, row: Keyword) -> Path | None:
        index = self.rows.index(row)
        directory = self.row_dir(row)
        self._check_directory(directory)
        archived = None
        existed = directory.exists()
        if existed:
            if any(directory.iterdir()):
                instant = datetime.now()
                timestamp = instant.strftime("%Y%m%d-%H%M%S-%f")
                archived = directory.with_name(f"delete-{timestamp}-{directory.name}")
                while archived.exists():
                    instant += timedelta(microseconds=1)
                    timestamp = instant.strftime("%Y%m%d-%H%M%S-%f")
                    archived = directory.with_name(f"delete-{timestamp}-{directory.name}")
                self._check_directory(archived)
                directory.rename(archived)
            else:
                directory.rmdir()
        self.rows.pop(index)
        try:
            self.save()
        except OSError:
            self.rows.insert(index, row)
            if archived is not None:
                archived.rename(directory)
            elif existed:
                directory.mkdir(exist_ok=True)
            raise
        return archived

    @staticmethod
    def wav_count(directory: Path) -> int:
        return sum(p.is_file() and p.suffix.lower() == ".wav" for p in directory.iterdir()) if directory.exists() else 0
