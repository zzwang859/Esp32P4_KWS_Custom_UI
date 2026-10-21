"""Parse simultaneous key chords into USB HID boot keyboard input reports.

Keyboard/Keypad usages are from USB HID Usage Tables, page 0x07. The report
is modifier bitmap, reserved zero, then six usages (HID 1.12 appendix B.1).
Printable punctuation follows the US keyboard layout; letter case is ignored.
"""

from __future__ import annotations

import re


# Bit order: left Ctrl/Shift/Alt/GUI, right Ctrl/Shift/Alt/GUI.
MODIFIERS = {
    "CTRL": 0x01, "CONTROL": 0x01, "LCTRL": 0x01, "LEFTCTRL": 0x01,
    "SHIFT": 0x02, "LSHIFT": 0x02, "LEFTSHIFT": 0x02,
    "ALT": 0x04, "LALT": 0x04, "LEFTALT": 0x04,
    "WIN": 0x08, "GUI": 0x08, "META": 0x08, "LWIN": 0x08, "LGUI": 0x08,
    "RCTRL": 0x10, "RIGHTCTRL": 0x10,
    "RSHIFT": 0x20, "RIGHTSHIFT": 0x20,
    "RALT": 0x40, "RIGHTALT": 0x40,
    "RWIN": 0x80, "RGUI": 0x80, "RIGHTGUI": 0x80,
}

NAMED_KEYS = {
    "ENTER": 0x28, "RETURN": 0x28, "ESC": 0x29, "ESCAPE": 0x29,
    "BACKSPACE": 0x2A, "TAB": 0x2B, "SPACE": 0x2C,
    "CAPSLOCK": 0x39, "PRINTSCREEN": 0x46, "PRTSC": 0x46,
    "SCROLLLOCK": 0x47, "PAUSE": 0x48, "INSERT": 0x49, "INS": 0x49,
    "HOME": 0x4A, "PAGEUP": 0x4B, "PGUP": 0x4B,
    "DELETE": 0x4C, "DEL": 0x4C, "END": 0x4D,
    "PAGEDOWN": 0x4E, "PGDN": 0x4E,
    "RIGHT": 0x4F, "LEFT": 0x50, "DOWN": 0x51, "UP": 0x52,
    "NUMLOCK": 0x53, "KP_DIVIDE": 0x54, "KP_MULTIPLY": 0x55,
    "KP_SUBTRACT": 0x56, "KP_ADD": 0x57, "KP_ENTER": 0x58,
    "KP_0": 0x62, "KP_DECIMAL": 0x63, "MENU": 0x65, "APPLICATION": 0x65,
    "MINUS": 0x2D, "EQUAL": 0x2E, "LEFTBRACKET": 0x2F,
    "RIGHTBRACKET": 0x30, "BACKSLASH": 0x31, "SEMICOLON": 0x33,
    "QUOTE": 0x34, "GRAVE": 0x35, "COMMA": 0x36,
    "PERIOD": 0x37, "SLASH": 0x38,
}
NAMED_KEYS.update({f"F{i}": 0x39 + i for i in range(1, 13)})
NAMED_KEYS.update({f"F{i}": 0x68 + i - 13 for i in range(13, 25)})
NAMED_KEYS.update({f"KP_{i}": 0x58 + i for i in range(1, 10)})

PRINTABLE_KEYS = {chr(ord("a") + i): (0, 0x04 + i) for i in range(26)}
PRINTABLE_KEYS.update({str(i): (0, 0x1D + i) for i in range(1, 10)})
PRINTABLE_KEYS["0"] = (0, 0x27)
for plain, shifted, usage in [
    ("-", "_", 0x2D), ("=", "+", 0x2E), ("[", "{", 0x2F),
    ("]", "}", 0x30), ("\\", "|", 0x31), (";", ":", 0x33),
    ("'", '"', 0x34), ("`", "~", 0x35), (",", "<", 0x36),
    (".", ">", 0x37), ("/", "?", 0x38),
]:
    PRINTABLE_KEYS[plain] = (0, usage)
    PRINTABLE_KEYS[shifted] = (0x02, usage)
for digit, symbol in zip("1234567890", "!@#$%^&*()"):
    PRINTABLE_KEYS[symbol] = (0x02, PRINTABLE_KEYS[digit][1])


def _tokens(text: str):
    """Split at '+' outside braces, rejecting missing keys and trailing input."""
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text) or text[position] == "+":
            raise ValueError("+ 两侧必须有按键；加号键请写为 {+}")
        if text[position] == "{":
            end = text.find("}", position + 1)
            if end == -1:
                raise ValueError("控制键缺少右花括号 }")
            token = text[position + 1:end].strip()
            if not token or "{" in token:
                raise ValueError("花括号内必须是有效键名，例如 {CTRL}、{F12}")
            position = end + 1
            yield token, True
        else:
            token = text[position]
            if token == "}":
                raise ValueError("出现多余的右花括号 }")
            position += 1
            yield token, False
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            return
        if text[position] != "+":
            raise ValueError("多个按键请用 + 连接，控制键请用 {} 包住")
        position += 1
        if position == len(text):
            raise ValueError("+ 后缺少按键；加号键请写为 {+}")


def keyboard_report(shortcut: str) -> bytes | None:
    """Return an eight-byte press report; blank means unbound, invalid raises.

    A chord is one simultaneous key state, not a sequence of text. Duplicate
    usages collapse and sort so equivalent chords serialize identically.
    """
    text = shortcut.strip()
    if not text:
        return None
    modifiers = 0
    keys = set()
    for token, braced in _tokens(text):
        name = token.upper()
        if braced and name in MODIFIERS:
            modifiers |= MODIFIERS[name]
            continue
        if braced and name in NAMED_KEYS:
            usage = NAMED_KEYS[name]
        elif braced and re.fullmatch(r"HID_[0-9A-F]{2}", name) and 0x04 <= int(name[4:], 16) <= 0xDF:
            usage = int(name[4:], 16)
        elif len(token) == 1 and token.lower() in PRINTABLE_KEYS:
            shifted, usage = PRINTABLE_KEYS[token.lower()]
            modifiers |= shifted
        else:
            raise ValueError(f"不支持的键名：{{{token}}}；例如 {{CTRL}}、{{F12}}、a")
        keys.add(usage)
        if len(keys) > 6:
            raise ValueError("一个组合最多包含 6 个不同的普通键（不含 CTRL 等修饰键）")
    return bytes([modifiers, 0, *sorted(keys), *([0] * (6 - len(keys)))])


def shortcut_from_report(report: bytes) -> str:
    """Lossless text for device bindings, including otherwise unnamed usages."""
    if len(report) != 8 or report[1] or any(key and not 0x04 <= key <= 0xDF for key in report[2:]):
        raise ValueError("设备键盘报文无效")
    keys = [key for key in report[2:] if key]
    if len(set(keys)) != len(keys):
        raise ValueError("设备键盘报文有重复键位")
    tokens = ["{" + name + "}" for bit, name in enumerate(("CTRL", "SHIFT", "ALT", "WIN", "RCTRL", "RSHIFT", "RALT", "RWIN"))
              if report[0] & (1 << bit)]
    named = {}
    for name, key in NAMED_KEYS.items():
        named.setdefault(key, "{" + name + "}")
    printable = {key: text for text, (modifier, key) in PRINTABLE_KEYS.items()
                 if modifier == 0 and text.isalnum()}
    tokens.extend(printable.get(key, named.get(key, f"{{HID_{key:02X}}}")) for key in keys)
    return "+".join(tokens)


def saved_reports(shortcut: str) -> dict:
    """Preserve unfinished text without persisting a stale or invalid report."""
    try:
        report = keyboard_report(shortcut)
    except ValueError:
        report = None
    return {
        "hid_report": list(report) if report is not None else None,
    }
