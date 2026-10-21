"""Load the environment's C++ runtime before native ML modules on Windows."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys


_runtime = None


def preload_msvc() -> None:
    global _runtime
    if sys.platform != "win32" or _runtime is not None:
        return
    library = Path(sys.executable).parent / "msvcp140.dll"
    if not library.is_file():
        raise RuntimeError("Missing local MSVC runtime; install msvc-runtime in this Python environment")
    _runtime = ctypes.WinDLL(str(library))
