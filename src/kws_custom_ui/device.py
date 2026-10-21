"""Device snapshots, deployment validation and asynchronous HID operations."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import queue
import struct
import threading

from .hid_protocol import Client, DeviceCancelled, ProtocolError, device_filter_values
from .keyboard import keyboard_report, shortcut_from_report
from .model import Keyword
from .settings import SESSION_FILE, hid_search_settings

PROFILE = "mfcc40_16k_int16_state6_v1"
FEATURE_INPUT = "streaming/speech_features/normalizer_1/truediv:0"
STATE_INPUTS = [f"streaming/stream{'_' + str(i) if i else ''}/stream{'_' + str(i) if i else ''}/ExternalState:0"
                for i in (0, 1, 2, 3, 4, 6)]
STATE_OUTPUTS = [f"onnx_next_state_{i}:0" for i in (1, 2, 3, 4, 5, 7)]
STATE_SHAPES = [[1, 11, 1, 40], [1, 13, 1, 64], [1, 15, 1, 32],
                [1, 17, 1, 32], [1, 57, 1, 32], [1, 98, 1, 64]]


class CompatibilityError(ValueError):
    pass


def open_device(session_file: Path | None = None):
    import hid
    search = hid_search_settings(session_file if session_file is not None else SESSION_FILE)
    matches = [item for item in hid.enumerate(search["vid"], search["pid"])
               if item.get("usage_page") == search["usage_page"] and item.get("usage") == search["usage"]]
    if not matches:
        raise OSError(f"未找到 KWS 控制 HID（{search['vid']:04X}:{search['pid']:04X}，"
                      f"Usage Page=0x{search['usage_page']:04X}，Usage=0x{search['usage']:04X}），请检查配置并连接设备 USB 数据口")
    if len(matches) != 1:
        raise OSError("发现多个 KWS 控制 HID，请只连接一台待配置设备")
    device = hid.device()
    try:
        device.open_path(matches[0]["path"])
    except BaseException:
        device.close()
        raise
    return device


def validate_snapshot(snapshot):
    if snapshot["info"].get("protocol_version") != 1:
        raise ValueError("设备 HID 协议版本不受支持")
    classes = snapshot["classes"]["classes"]
    bindings = snapshot["bindings"]["bindings"]
    if not 3 <= len(classes) <= 34 or [item["class_index"] for item in classes] != list(range(len(classes))):
        raise ValueError("设备类别表索引无效")
    if [item["text"] for item in classes[:2]] != ["silence", "unknown"]:
        raise ValueError("设备背景类别必须为 silence、unknown")
    if {item["class_index"] for item in bindings} != set(range(2, len(classes))) or len(bindings) != len(classes) - 2:
        raise ValueError("设备关键词绑定不完整")
    for binding in bindings:
        validate_binding(binding)
    if snapshot["caps"].get("supports_runtime_kws_filter"):
        device_filter_values(snapshot["status"]["kws_filter"])
        if type(snapshot["status"]["kws_filter"].get("revision")) is not int:
            raise ValueError("设备 KWS 参数 revision 无效")
    return snapshot


def validate_binding(binding):
    led, key = binding["led"], binding["keyboard"]
    rgb, keys = led["rgb"], key["keycodes"]
    if len(rgb) != 3 or any(type(value) is not int or not 0 <= value <= 255 for value in rgb):
        raise ValueError("RGB 必须为三个 0～255 整数")
    if len(keys) != 6 or type(key["modifier"]) is not int or not 0 <= key["modifier"] <= 255:
        raise ValueError("设备键盘绑定格式无效")
    if any(type(value) is not int or (value != 0 and not 4 <= value <= 0xDF) for value in keys):
        raise ValueError("设备仅支持 HID 0x04～0xDF 的键位")
    if len([value for value in keys if value]) != len({value for value in keys if value}):
        raise ValueError("快捷键不能包含重复键位")
    if key["enabled"] and not any(keys):
        raise ValueError(f"类别 {binding['class_index']} 键盘已启用但没有普通键位；请填写快捷键或禁用该行")
    if led.get("effect", "flash") != "flash" or not 0 <= led["duration_ms"] <= 60000 or not 0 <= key["hold_ms"] <= 1000:
        raise ValueError("设备灯效/按键时长超出协议范围")


def row_binding(row, index, previous=None):
    report = keyboard_report(row.shortcut) or bytes(8)
    if row.flags & ~7:
        raise ValueError(f"「{row.text or row.directory}」的 flags 含设备不支持的保留位")
    previous = previous or {}
    if previous:
        old_key = previous["keyboard"]
        original = bytes([old_key["modifier"], 0, *old_key["keycodes"]])
        if row.shortcut == shortcut_from_report(original):
            report = original
    result = dict(class_index=index, actions_enabled=bool(row.flags & 1),
                  led=dict(enabled=bool(row.flags & 2), effect="flash", rgb=list(row.color),
                           duration_ms=previous.get("led", {}).get("duration_ms", 600)),
                  keyboard=dict(enabled=bool(row.flags & 4), modifier=report[0], keycodes=list(report[2:]),
                                hold_ms=previous.get("keyboard", {}).get("hold_ms", 0)))
    validate_binding(result)
    return result


def sync_workspace(workspace, snapshot):
    validate_snapshot(snapshot)
    bindings = {item["class_index"]: item for item in snapshot["bindings"]["bindings"]}
    rows = []
    for item in snapshot["classes"]["classes"][2:]:
        index = item["class_index"]
        binding = bindings[index]
        led, key = binding["led"], binding["keyboard"]
        flags = int(binding["actions_enabled"]) | (int(led["enabled"]) << 1) | (int(key["enabled"]) << 2)
        identity = hashlib.sha256(f"{snapshot['info']['device_id']}:{index}".encode()).hexdigest()[:12]
        shortcut = shortcut_from_report(bytes([key["modifier"], 0, *key["keycodes"]]))
        row = Keyword(identity, item["text"], tuple(led["rgb"]), f"keyword{index - 1}", shortcut, flags)
        workspace.row_dir(row).mkdir(exist_ok=True)
        rows.append(row)
    workspace.rows = rows
    if snapshot["caps"].get("supports_runtime_kws_filter"):
        workspace.kws_filter = device_filter_values(snapshot["status"]["kws_filter"])
    workspace.save()
    temporary = workspace.path / "device.json.tmp"
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(workspace.path / "device.json")


def deployment_package(workspace):
    root = workspace.path / "esp32p4-dl"
    if not (root / "latest.json").is_file():
        raise CompatibilityError("当前工作区没有成功转换的 ESP32-P4 模型，请先完成本地训练和转换")
    latest = json.loads((root / "latest.json").read_text(encoding="utf-8"))
    if not latest.get("success") or latest.get("selected") != "w16a16":
        raise CompatibilityError("设备部署需要成功转换的 INT16 ESP32-P4 模型")
    model_path = (root / latest["model"]).resolve()
    bundle = (root / latest["bundle"]).resolve()
    if not model_path.is_relative_to(root.resolve()) or not bundle.is_relative_to(root.resolve()):
        raise ValueError("模型路径必须位于当前工作区 esp32p4-dl 内")
    model = model_path.read_bytes()
    if hashlib.sha256(model).hexdigest() != latest["models"]["w16a16"]["sha256"]:
        raise ValueError("模型校验和不匹配，请重新转换")
    if (len(model) < 32 or model[:4] != b"EDL2" or model[4:8] != bytes(4) or len(model) > 960 * 1024
            or not 16 <= struct.unpack_from("<I", model, 8)[0] <= len(model) - 16):
        raise CompatibilityError("设备仅接收未加密、单模型 EDL2 .espdl，最大 960 KiB")
    labels = ["silence", "unknown"] + [row.text.strip() or row.directory for row in workspace.rows]
    trained = json.loads((bundle / "models/classes.json").read_text(encoding="utf-8"))
    if latest["labels"] != labels or [item["label"] for item in trained] != labels:
        raise CompatibilityError("关键词文本或顺序已变化，请先重新训练模型")
    if [(item.get("row_id"), Path(item["directory"]).name) for item in trained[2:]] != [(row.id, row.directory) for row in workspace.rows]:
        raise CompatibilityError("关键词目录或行映射已变化，请先重新训练模型")
    mapping = json.loads((bundle / "models/io_mapping.json").read_text(encoding="utf-8"))
    if mapping.get("state_pairs") != [list(pair) for pair in zip(STATE_INPUTS, STATE_OUTPUTS)] or FEATURE_INPUT not in mapping.get("inputs", {}) or "onnx_logits:0" not in mapping.get("outputs", {}):
        raise CompatibilityError("模型接口不兼容：设备要求固定名称的 6 对状态；请使用新契约重新训练和转换。详见 docs/device-protocol.md")
    if (mapping["inputs"][FEATURE_INPUT] != [1, 1, 40] or mapping["outputs"]["onnx_logits:0"] != [1, len(labels)]
            or any(mapping["inputs"].get(name) != shape or mapping["outputs"].get(output) != shape
                   for name, output, shape in zip(STATE_INPUTS, STATE_OUTPUTS, STATE_SHAPES))):
        raise CompatibilityError("模型张量形状不符合固定 ESP32-P4 契约，请重新训练和转换")
    frontend = json.loads((bundle / "models/frontend.json").read_text(encoding="utf-8"))
    if frontend.get("frontend_profile") != PROFILE:
        raise CompatibilityError("MFCC 前端不兼容：请使用固定自然对数幅度谱前端重新训练，不能仅修改 profile 名称")
    asset = Path(__file__).resolve().parents[1] / "ds-tc-resnet/data/esp32p4_frontend.npz"
    expected_frontend = hashlib.sha256(asset.read_bytes()).hexdigest()
    if (frontend.get("frontend_asset_sha256") != expected_frontend
            or hashlib.sha256((bundle / "models/frontend.npz").read_bytes()).hexdigest() != expected_frontend):
        raise CompatibilityError("模型包的 MFCC 系数与固定设备前端不匹配")
    if not 3 <= len(labels) <= 34 or any(not text or len(text.encode("utf-8")) > 64 for text in labels):
        raise CompatibilityError("设备要求 3～34 类，每个文本不超过 64 个 UTF-8 字节")
    bindings = [row_binding(row, index) for index, row in enumerate(workspace.rows, 2)]
    manifest = json.dumps(dict(format_version=1, frontend_profile=PROFILE,
                               classes=[dict(class_index=index, text=text) for index, text in enumerate(labels)]),
                          ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(manifest) > 8192:
        raise CompatibilityError("设备类别清单超过 8192 字节")
    return dict(model=model, manifest=manifest, bindings=bindings, labels=labels,
                kws_filter=device_filter_values(workspace.kws_filter), model_path=str(model_path))


class DeviceController:
    """One HID operation at a time; worker results are consumed by pygame."""
    def __init__(self, opener=open_device):
        self.opener = opener
        self.results = queue.Queue()
        self.snapshot = None
        self.client = None
        self.thread = None
        self.cancelled = threading.Event()
        self.operation = None

    @property
    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    @property
    def connected(self):
        return self.client is not None and self.snapshot is not None

    def close(self):
        self.cancelled.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                raise RuntimeError("正在等待 HID 操作结束，请稍后重试")
        self._disconnect()
        self.thread = None
        self.operation = None
        while not self.results.empty():
            self.results.get_nowait()

    def _disconnect(self):
        client, self.client = self.client, None
        self.snapshot = None
        if client is not None:
            client.device.close()

    def submit(self, operation, value=None):
        if self.busy:
            return False
        self.cancelled.clear()
        self.operation = operation
        self.thread = threading.Thread(target=self._work, args=(operation, value), daemon=True, name="KWS-HID")
        self.thread.start()
        return True

    def _read(self):
        self.snapshot = validate_snapshot(self.client.read_snapshot())
        return self.snapshot

    def _connect(self):
        if self.client is None:
            self.client = Client(self.opener(), self.cancelled.is_set)
        return self._read()

    def _write(self, bindings):
        current = deepcopy(self.snapshot["status"])
        known = {item["class_index"] for item in self.snapshot["classes"]["classes"][2:]}
        for binding in bindings:
            validate_binding(binding)
            if binding["class_index"] not in known:
                raise ValueError("关键词不属于当前设备模型")
        for binding in bindings:
            self.client.set_binding(binding, current)
        self.client.save(current)
        return self._read()

    def _work(self, operation, value):
        try:
            if operation in ("connect", "refresh"):
                result = self._connect()
            elif operation == "bindings":
                result = self._write(value)
            elif operation == "filter":
                if not self.snapshot["caps"].get("supports_runtime_kws_filter"):
                    raise CompatibilityError("设备固件不支持运行时 KWS 参数设置")
                self.client.set_kws_filter(value, self.snapshot["status"])
                result = self._read()
            elif operation == "poll":
                status = self.client.read_object("status")
                before = self.snapshot["status"]
                if (any(status[key] != before[key] for key in ("boot_id", "model_epoch", "config_revision"))
                        or status.get("kws_filter") != before.get("kws_filter")):
                    result = self._read()
                else:
                    self.snapshot["status"] = status
                    result = None
            elif operation == "deploy":
                package = deployment_package(value)
                self._connect()
                if self.snapshot["caps"]["frontend_profile"] != PROFILE:
                    raise CompatibilityError("设备前端 profile 不受支持")
                caps = self.snapshot["caps"]
                if not caps.get("supports_runtime_kws_filter"):
                    raise CompatibilityError("设备固件不支持运行时 KWS 参数设置")
                if (len(package["model"]) > caps["max_model_bytes"] or len(package["manifest"]) > caps["max_manifest_bytes"]
                        or len(package["labels"]) > caps["max_class_count"]):
                    raise CompatibilityError("模型或类别清单超过设备报告的容量")
                self.client.download(package["manifest"], package["model"],
                                     lambda message: self.results.put(dict(kind="progress", message=message)))
                self._read()
                expected = hashlib.sha256(package["model"] + package["manifest"]).hexdigest()
                if self.snapshot["status"]["active_model"]["package_sha256"] != expected:
                    raise ValueError("设备当前模型包校验和不匹配，未下发绑定")
                result = self._write(package["bindings"])
                self.results.put(dict(kind="progress", message="模型与绑定已写入，正在下发 KWS 参数…"))
                self.client.set_kws_filter(package["kws_filter"], result["status"])
                result = self._read()
            else:
                raise ValueError(f"Unknown device operation: {operation}")
            self.results.put(dict(kind=operation, snapshot=result))
        except DeviceCancelled:
            pass
        except Exception as exc:
            snapshot = None
            if (isinstance(exc, ProtocolError) or operation == "filter") and self.client is not None:
                try:
                    snapshot = self._read()
                except Exception:
                    self._disconnect()
            elif isinstance(exc, (OSError, TimeoutError)) or operation == "connect":
                self._disconnect()
            self.results.put(dict(kind="error", operation=operation, message=str(exc), snapshot=snapshot,
                                  disconnected=not self.connected))
