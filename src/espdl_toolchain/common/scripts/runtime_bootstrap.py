"""Load a compatible private MSVC runtime before ONNX native extensions.

Preload private DLLs when present beside the current interpreter. The unified uv
environment installs msvc-runtime; no separate ESP-IDF Python environment is used.
"""

from __future__ import annotations

import ctypes
import os
import sys


def preload_msvc_runtime() -> None:
    if os.name != "nt":
        return
    runtime_dir = os.path.dirname(os.path.abspath(sys.executable))
    for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        path = os.path.join(runtime_dir, name)
        if os.path.exists(path):
            ctypes.WinDLL(path)

