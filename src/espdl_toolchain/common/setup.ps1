param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "../../..")).Path

# UI, training, ONNX and ESP-PPQ share the root uv project.
& uv sync --project $ProjectRoot --locked
if ($LASTEXITCODE -ne 0) { throw "Failed to sync the project environment." }

$ProjectPython = Join-Path $ProjectRoot ".venv/Scripts/python.exe"
& $ProjectPython -c "import sys; sys.path.insert(0, sys.argv[1]); from runtime_bootstrap import preload_msvc_runtime; preload_msvc_runtime(); import onnx, onnxruntime, torch, esp_ppq; from onnxsim import simplify; print('Unified training and ESP32-P4 toolchain OK')" (Join-Path $PSScriptRoot "scripts")
if ($LASTEXITCODE -ne 0) { throw "Model toolchain import check failed." }
