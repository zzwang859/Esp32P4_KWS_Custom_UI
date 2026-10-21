import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from kws_custom_ui.processes import ProcessGroup, python_executable
from kws_custom_ui.training import TRAINING_ROOT


def wait_for_file(path, timeout=10):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            pytest.fail(f"Timed out waiting for {path}")
        time.sleep(0.02)


def worker_script(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(
        "import sys,subprocess,time\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(TRAINING_ROOT)!r})\n"
        "from workspace_train import join_owner_job\n"
        "join_owner_job(sys.argv[1])\n"
        "folder=Path(sys.argv[2])\n"
        "code=\"from pathlib import Path; import sys,time; p=Path(sys.argv[1]); f=(p/'locked.log').open('w'); f.write('log'); f.flush(); (p/'ready').touch(); time.sleep(60)\"\n"
        "subprocess.Popen([sys.executable,'-c',code,str(folder)])\n"
        "while not (folder/'ready').exists(): time.sleep(.02)\n"
        "if sys.argv[3]=='stay': time.sleep(60)\n", encoding="utf-8")
    return script


@pytest.mark.skipif(os.name != "nt", reason="Windows file locking and Job Objects")
@pytest.mark.parametrize("parent_stays_alive", [False, True])
def test_job_closes_logging_descendants_even_after_launcher_exits(tmp_path, parent_stays_alive):
    group = ProcessGroup()
    environment = os.environ.copy()
    executable = python_executable(environment)
    process = subprocess.Popen([executable, str(worker_script(tmp_path)), group.name, str(tmp_path),
                                "stay" if parent_stays_alive else "exit"],
                               creationflags=subprocess.CREATE_NO_WINDOW, env=environment)
    try:
        wait_for_file(tmp_path / "ready")
        if not parent_stays_alive:
            process.wait(timeout=5)
        with pytest.raises(PermissionError):
            (tmp_path / "locked.log").unlink()
        group.close()
        process.wait(timeout=5)
        (tmp_path / "locked.log").unlink()
        group.close()  # Cleanup may be invoked from both exit() and finally.
    finally:
        group.close()
        process.wait(timeout=5)


@pytest.mark.skipif(os.name != "nt", reason="Windows kernel closes Job handle on owner death")
def test_abrupt_owner_exit_also_releases_logging_descendants(tmp_path):
    worker = worker_script(tmp_path)
    owner = tmp_path / "owner.py"
    source_root = TRAINING_ROOT.parent
    owner.write_text(
        "import os,sys,subprocess,time\nfrom pathlib import Path\n"
        f"sys.path.insert(0,{str(source_root)!r})\n"
        "from kws_custom_ui.processes import ProcessGroup,python_executable\n"
        "group=ProcessGroup()\n"
        "environment=os.environ.copy()\nexecutable=python_executable(environment)\n"
        f"subprocess.Popen([executable,{str(worker)!r},group.name,{str(tmp_path)!r},'exit'],env=environment)\n"
        f"abort=Path({str(tmp_path / 'abort')!r})\n"
        "while not abort.exists(): time.sleep(.02)\n"
        "os._exit(7)\n", encoding="utf-8")
    environment = os.environ.copy()
    process = subprocess.Popen([python_executable(environment), str(owner)], creationflags=subprocess.CREATE_NO_WINDOW, env=environment)
    try:
        wait_for_file(tmp_path / "ready")
        with pytest.raises(PermissionError):
            (tmp_path / "locked.log").unlink()
    finally:
        (tmp_path / "abort").touch()
        process.wait(timeout=10)
    deadline = time.monotonic() + 5
    while True:
        try:
            (tmp_path / "locked.log").unlink()
            break
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)
