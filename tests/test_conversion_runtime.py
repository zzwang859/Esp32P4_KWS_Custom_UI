"""Exercise the native float64 kernels used by ESP-PPQ's INT16 simulation."""
import os
from pathlib import Path
import subprocess
import sys


CONVERTER = (Path(__file__).resolve().parents[1] /
             "src/espdl_toolchain/esp32p4-dl/scripts/convert_workspace.py")


def test_converter_runtime_supports_cpu_float64_conv_and_gemm():
    # A child process turns a native access violation into a useful test failure.
    # Import the real entry point first: MKL must be configured before torch.
    script = """
import runpy, sys
namespace = runpy.run_path(sys.argv[1])
torch = namespace['torch']
torch.set_num_threads(4)
torch.manual_seed(42)
x = torch.randn(1, 11, 1, 40, dtype=torch.float64).permute(0, 3, 1, 2)
w = torch.randn(40, 1, 11, 1, dtype=torch.float64)
actual = torch.nn.functional.conv2d(x, w, groups=40)
expected = (x * w[:, 0].unsqueeze(0)).sum(dim=2, keepdim=True)
torch.testing.assert_close(actual, expected)
# The streaming classifier consumes one frame with 64 channels and 6 labels.
a = torch.randn(1, 64, dtype=torch.float64)
b = torch.randn(64, 6, dtype=torch.float64)
torch.testing.assert_close(a @ b, (a[:, :, None] * b[None, :, :]).sum(dim=1))
assert actual.dtype == torch.float64
print('FLOAT64_KERNELS_OK')
"""
    environment = os.environ.copy()
    environment.pop("MKL_CBWR", None)
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script, str(CONVERTER)],
        env=environment, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FLOAT64_KERNELS_OK" in result.stdout
