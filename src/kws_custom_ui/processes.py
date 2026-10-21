"""Own a training process tree, including services left alive after training."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import signal
import sys
import time
from uuid import uuid4
import weakref


def python_executable(environment: dict) -> str:
    # Match multiprocessing's Windows venv launch strategy. The venv launcher
    # owns an extra Job which would kill background services when it exits.
    if os.name == "nt" and sys.prefix != sys.base_prefix:
        environment["__PYVENV_LAUNCHER__"] = sys.executable
        return sys._base_executable
    return sys.executable


class ProcessGroup:
    def __init__(self):
        self.name = None
        self.pid = None
        self._closed = False
        if os.name != "nt":
            return
        size_t = ctypes.c_size_t
        class Limits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                        ("flags", wintypes.DWORD), ("min_working_set", size_t), ("max_working_set", size_t),
                        ("active_limit", wintypes.DWORD), ("affinity", size_t),
                        ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]
        class Counters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", Limits), ("io", Counters), ("process_memory", size_t),
                        ("job_memory", size_t), ("peak_process_memory", size_t), ("peak_job_memory", size_t)]
        class Accounting(ctypes.Structure):
            _fields_ = [(name, ctypes.c_longlong) for name in ("user", "kernel", "period_user", "period_kernel")] + [
                (name, wintypes.DWORD) for name in ("faults", "total", "active", "terminated")]
        self._accounting_type = Accounting
        self._kernel = kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD, wintypes.LPVOID]
        kernel.QueryInformationJobObject.restype = wintypes.BOOL
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.TerminateJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.name = "Local\\KwsTraining-" + uuid4().hex
        self._handle = kernel.CreateJobObjectW(None, self.name)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self._finalize = weakref.finalize(self, kernel.CloseHandle, self._handle)
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(self._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self._finalize()
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self._closed:
            return
        self._closed = True
        if os.name == "nt":
            handles = []
            try:
                # Accounting can reach zero before terminating processes finish
                # closing their file handles. Wait on the process objects too.
                capacity = 32
                while True:
                    buffer = ctypes.create_string_buffer(8 + capacity * ctypes.sizeof(ctypes.c_size_t))
                    if self._kernel.QueryInformationJobObject(self._handle, 3, buffer, ctypes.sizeof(buffer), None):
                        break
                    if ctypes.get_last_error() != 234:  # ERROR_MORE_DATA
                        raise ctypes.WinError(ctypes.get_last_error())
                    capacity *= 2
                count = wintypes.DWORD.from_buffer(buffer, 4).value
                for index in range(count):
                    pid = ctypes.c_size_t.from_buffer(buffer, 8 + index * ctypes.sizeof(ctypes.c_size_t)).value
                    handle = self._kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
                    if handle:
                        handles.append(handle)
                if not self._kernel.TerminateJobObject(self._handle, 1):
                    raise ctypes.WinError(ctypes.get_last_error())
                deadline = time.monotonic() + 5
                for handle in handles:
                    remaining = max(0, int((deadline - time.monotonic()) * 1000))
                    if self._kernel.WaitForSingleObject(handle, remaining) != 0:
                        raise RuntimeError("后台进程尚未释放文件句柄")
                while True:
                    accounting = self._accounting_type()
                    if not self._kernel.QueryInformationJobObject(self._handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None):
                        raise ctypes.WinError(ctypes.get_last_error())
                    if not accounting.active:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("训练进程组未能及时退出")
                    time.sleep(0.02)
            finally:
                for handle in handles:
                    self._kernel.CloseHandle(handle)
                self._finalize()
        elif self.pid is not None:
            try:
                os.killpg(self.pid, signal.SIGTERM)
                time.sleep(0.1)
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
