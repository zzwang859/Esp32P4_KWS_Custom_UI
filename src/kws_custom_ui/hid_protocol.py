"""Custom KWS HID v1, matching USB_Audio_HS2/src/custom (no firmware dependency)."""
from __future__ import annotations

import hashlib
import json
import secrets
import struct
import time
import zlib

HEADER = struct.Struct("<2sBBHHIIHH")
OPEN_SESSION = 0x04
READ_BEGIN, READ_CHUNK, READ_END = 0x10, 0x11, 0x12
SET_BINDING, SAVE_CONFIG, SET_KWS_FILTER = 0x20, 0x21, 0x23
UPDATE_BEGIN, OBJECT_BEGIN, OBJECT_DATA, OBJECT_END = 0x30, 0x31, 0x32, 0x33
UPDATE_COMMIT, UPDATE_ABORT, ACTIVATE = 0x34, 0x35, 0x36
OBJECTS = dict(info=1, caps=2, status=3, classes=4, bindings=5, stats=6)
ERRORS = {2: "BUSY", 3: "BAD_VERSION", 4: "BAD_LENGTH", 5: "BAD_CRC",
          6: "INVALID_STATE", 7: "OUT_OF_RANGE", 8: "OFFSET_MISMATCH", 9: "HASH_MISMATCH",
          10: "MODEL_INCOMPATIBLE", 11: "LOAD_FAILED", 12: "NO_SPACE", 13: "FLASH_ERROR",
          14: "CONFLICT", 15: "NOT_FOUND", 16: "MODEL_CHANGED", 17: "QUEUE_FULL", 18: "BAD_COMMAND"}


class ProtocolError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"设备返回 {ERRORS.get(status, str(status))}")


class DeviceCancelled(Exception):
    pass


FILTER_FIELDS = ("keyword_threshold_percent", "cooldown_ms", "stable_frames", "need_vad_checked")


def device_filter_values(settings):
    """Validate the firmware's HID ranges, which differ from PC simulation ranges."""
    if not isinstance(settings, dict) or any(name not in settings for name in FILTER_FIELDS):
        raise ValueError("缺少 KWS 滤波参数")
    result = {name: settings[name] for name in FILTER_FIELDS}
    threshold, cooldown, stable, vad = (result[name] for name in FILTER_FIELDS)
    if (type(threshold) is not int or not 0 <= threshold <= 100 or
            type(cooldown) is not int or not 1 <= cooldown <= 60000 or
            type(stable) is not int or not 1 <= stable <= 1000 or type(vad) is not bool):
        raise ValueError("设备 KWS 参数要求：阈值 0～100，冷却 1～60000 ms，稳定帧 1～1000，VAD 为开关；旧配置中的 0 ms/0 帧不能下发")
    return result


def encode(command, sequence, payload=b"", transaction=0, offset=0, kind=0, status=0):
    if len(payload) > 40:
        raise ValueError("HID payload exceeds 40 bytes")
    body = HEADER.pack(b"KC", 1, kind, command, sequence, transaction, offset, len(payload), status)
    body += payload.ljust(40, b"\0")
    return body + struct.pack("<I", zlib.crc32(body))


def decode(wire):
    wire = bytes(wire)
    if len(wire) != 64:
        raise ValueError("HID report must contain 64 bytes")
    magic, version, kind, command, sequence, transaction, offset, size, status = HEADER.unpack_from(wire)
    if magic != b"KC" or version != 1 or kind not in (0, 1, 2) or size > 40:
        raise ValueError("Invalid HID report header")
    if zlib.crc32(wire[:60]) != struct.unpack_from("<I", wire, 60)[0]:
        raise ValueError("Invalid HID CRC32")
    if any(wire[20 + size:60]):
        raise ValueError("Nonzero HID padding")
    return dict(kind=kind, command=command, sequence=sequence, transaction=transaction,
                offset=offset, status=status, payload=wire[20:20 + size])


class Client:
    def __init__(self, device, cancelled=lambda: False):
        self.device, self.cancelled = device, cancelled
        self.sequence = 0
        self.session = secrets.token_bytes(16)
        self.session_open = False

    def check_cancel(self):
        if self.cancelled():
            raise DeviceCancelled("设备操作已取消")

    def request(self, command, payload=b"", transaction=0, offset=0, timeout=15.0):
        self.check_cancel()
        if not self.session_open and command != OPEN_SESSION:
            response = self.request(OPEN_SESSION, self.session, timeout=timeout)
            if response["payload"] != self.session:
                raise ValueError("HID session nonce mismatch")
            self.session_open = True
        self.sequence = (self.sequence + 1) & 0xFFFF
        wire = encode(command, self.sequence, payload, transaction, offset)
        for _ in range(3):
            self.check_cancel()
            if self.device.write(b"\0" + wire) != 65:
                raise OSError("HID 报文未完整写入")
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                self.check_cancel()
                data = self.device.read(64, 100)
                if not data:
                    continue
                response = decode(data)
                if response["kind"] != 1 or response["sequence"] != self.sequence or response["command"] != command:
                    continue
                if command == OPEN_SESSION and response["status"] == 0 and response["payload"] != payload:
                    continue
                if response["status"] > 1:
                    raise ProtocolError(response["status"])
                return response
        raise TimeoutError(f"HID 命令 0x{command:04X} 超时，请检查 USB 连接")

    def read_object(self, name):
        response = self.request(READ_BEGIN, struct.pack("<H", OBJECTS[name]))
        token = response["transaction"]
        if len(response["payload"]) != 16:
            raise ValueError("Invalid snapshot header")
        size, boot, epoch, revision = struct.unpack("<IIII", response["payload"])
        if not token or not 0 < size <= 256 * 1024:
            raise ValueError("Invalid snapshot size/ID")
        data = bytearray()
        try:
            while len(data) < size:
                chunk = self.request(READ_CHUNK, transaction=token, offset=len(data))
                if chunk["transaction"] != token or chunk["offset"] != len(data):
                    raise ValueError("HID snapshot offset/ID mismatch")
                if not chunk["payload"] or len(data) + len(chunk["payload"]) > size:
                    raise ValueError("Invalid snapshot chunk")
                data.extend(chunk["payload"])
        except BaseException:
            if not self.cancelled():
                try:
                    self.request(READ_END, transaction=token, timeout=0.5)
                except Exception:
                    pass
            raise
        else:
            self.request(READ_END, transaction=token)
        result = json.loads(data.decode("utf-8"))
        if (result["boot_id"], result["model_epoch"], result["config_revision"]) != (boot, epoch, revision):
            raise ValueError("Snapshot version mismatch")
        return result

    def read_snapshot(self, info=None, caps=None):
        info = info or self.read_object("info")
        caps = caps or self.read_object("caps")
        for _ in range(3):
            classes = self.read_object("classes")
            bindings = self.read_object("bindings")
            status = self.read_object("status")
            versions = {(item["boot_id"], item["model_epoch"], item["config_revision"])
                        for item in (classes, bindings, status)}
            package = status["active_model"]["package_sha256"]
            if len(versions) == 1 and classes["package_sha256"] == bindings["package_sha256"] == package:
                return dict(info=info, caps=caps, status=status, classes=classes, bindings=bindings)
        raise RuntimeError("设备模型或参数正在变化，请重新读取")

    def set_binding(self, binding, current):
        led, key = binding["led"], binding["keyboard"]
        flags = int(binding["actions_enabled"]) | (int(led["enabled"]) << 1) | (int(key["enabled"]) << 2)
        payload = struct.pack("<IIIHB3BB6BHH", current["boot_id"], current["model_epoch"], current["config_revision"],
                              binding["class_index"], flags, *led["rgb"], key["modifier"], *key["keycodes"],
                              led["duration_ms"], key["hold_ms"])
        response = self.request(SET_BINDING, payload)
        if len(response["payload"]) != 4:
            raise ValueError("Invalid binding revision response")
        current["config_revision"] = struct.unpack("<I", response["payload"])[0]

    def set_kws_filter(self, settings, current):
        desired = device_filter_values(settings)
        existing = current["kws_filter"]
        device_filter_values(existing)
        payload = struct.pack("<IIHIHB", current["boot_id"], existing["revision"],
                              desired["keyword_threshold_percent"], desired["cooldown_ms"],
                              desired["stable_frames"], int(desired["need_vad_checked"]))
        response = self.request(SET_KWS_FILTER, payload)
        if len(response["payload"]) != 4:
            raise ValueError("设备 KWS 参数应答长度无效")
        revision, = struct.unpack("<I", response["payload"])
        status = self.read_object("status")
        if status["boot_id"] != current["boot_id"]:
            raise ProtocolError(16)
        if status["kws_filter"]["revision"] != revision or device_filter_values(status["kws_filter"]) != desired:
            raise RuntimeError("设备 KWS 参数回读与下发值不一致")
        return status

    def save(self, current, timeout=15.0):
        response = self.request(SAVE_CONFIG, struct.pack("<III", current["boot_id"], current["model_epoch"], current["config_revision"]))
        if response["status"] == 0:
            return response
        # ACCEPTED fixes the snapshot; completion is confirmed by device status.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.check_cancel()
            status = self.read_object("status")
            if any(status[key] != current[key] for key in ("boot_id", "model_epoch")):
                raise ProtocolError(16)
            if status["config_revision"] != current["config_revision"]:
                raise ProtocolError(14)
            runtime = status["runtime"]
            if runtime["save_pending"] is False:
                if runtime["config_dirty"] is False:
                    return response
                raise RuntimeError("设备参数写入失败，配置尚未保存")
            time.sleep(0.1)
        raise TimeoutError("设备参数写入超时，未确认保存完成")

    def download(self, manifest, model, progress=lambda message: None):
        response = self.request(UPDATE_BEGIN)
        transaction, slot = response["transaction"], response["payload"][0]
        if not transaction or slot not in (0, 1):
            raise ValueError("Invalid update transaction/slot")
        try:
            for object_id, data, title in ((0x100, manifest, "类别表"), (0x101, model, "模型")):
                self.request(OBJECT_BEGIN, struct.pack("<HI", object_id, len(data)) + hashlib.sha256(data).digest(), transaction)
                for offset in range(0, len(data), 40):
                    ack = self.request(OBJECT_DATA, data[offset:offset + 40], transaction, offset)
                    if len(ack["payload"]) != 4 or struct.unpack("<I", ack["payload"])[0] != min(offset + 40, len(data)):
                        raise ValueError("Device acknowledged an unexpected upload offset")
                    if offset % 4000 == 0 or offset + 40 >= len(data):
                        progress(f"传输{title}：{min(offset + 40, len(data))}/{len(data)} 字节")
                self.request(OBJECT_END, transaction=transaction)
            self.request(UPDATE_COMMIT, transaction=transaction)
        except BaseException:
            cancelled = self.cancelled
            self.cancelled = lambda: False
            try:
                self.request(UPDATE_ABORT, transaction=transaction, timeout=0.5)
            except Exception:
                pass
            finally:
                self.cancelled = cancelled
            raise
        progress("模型已提交，正在激活并执行设备自检…")
        self.request(ACTIVATE, bytes([slot]))
        return slot
