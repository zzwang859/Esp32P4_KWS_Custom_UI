# ESP32-P4 模型工具链

| 目录 | 内容 |
|---|---|
| `common/` | ESP-PPQ 1.3.11 wheel、根目录 uv 环境同步入口及 MSVC 运行库辅助脚本 |
| `esp32p4-dl/scripts/convert_workspace.py` | 当前 KWS Studio ONNX 的静态准备、INT8/INT16 量化和流式对照入口 |
| `esp32p4-dl/models/`、`output/`、`work/`、`REPORT.md` | 原始转换例程与历史产物，作为参考保留 |

界面的“开始训练”以及 `ds-tc-resnet/workspace_train.py` 已自动串联：

```text
训练 → 导出最终 ONNX → PyTorch/ONNX 核对
     → 用本次真实训练录音生成 MFCC
     → 静态 ONNX 准备及浮点逐块核对
     → ESP32-P4 w8a8 → ESP32-P4 w16a16 → PC 流式对照 → 更新成功结果
```

界面、预处理、训练、ONNX 导出和量化共用项目根目录 `.venv`（Python 3.12、PyTorch 2.8、ONNX 1.17）。转换子进程沿用训练的 `sys.executable`，不再读取 `KWS_ESPDL_PYTHON`，也不再创建 `common/.venv`。依赖统一维护在项目根目录 `pyproject.toml` 和 `uv.lock`，ESP-PPQ 使用 `common/packages/esp_ppq-1.3.11-py3-none-any.whl`。

在项目根目录执行：

```powershell
uv sync --locked
uv run --locked python run.py
```

`common/setup.ps1` 兼容入口也只同步根目录环境并检查工具链导入。旧 `common/.venv` 可删除，无需保留第二份环境。

转换方式参照 [Espressif ESP-DL 量化接口](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_quantize_model.html)：目标 `esp32p4`，分别输出 INT8、INT16，INT16 使用 `debug=True` 进行高精度 PC 仿真，默认选择 INT16。当前训练导出的 ONNX 自身是 MFCC 后端；原例程的 ONNX 含不同的原始音频前端，旧 `build.ps1` / `scripts/build.py` 还依赖未包含在当前工程中的板端源码，因此工作区流程使用新的 `convert_workspace.py`，不调用或改写旧例程产物。

Windows 转换入口在导入原生库前默认设置 `MKL_CBWR=COMPATIBLE`，避免当前 `torch 2.8.0+cu126` 在模型的 CPU 双精度卷积中触发 `0xC0000005`，使 INT16 高精度仿真正常完成。该设置仅作用于转换子进程，保留双精度计算；CUDA 训练不受影响。显式设置的 `MKL_CBWR` 会保留。

输出全部在所选工作区：

```text
<工作区>/esp32p4-dl/
├── latest.json                       # 只指向最后成功的一次，默认 w16a16
└── <训练时间戳>/
    ├── conversion.json
    ├── REPORT.md
    ├── models/
    │   ├── ds_tc_resnet.onnx          # 本次训练的最终 ONNX 副本
    │   ├── ds_tc_resnet_esp32p4_prepared.onnx
    │   ├── labels.txt / classes.json
    │   ├── frontend.json / frontend.npz
    │   └── io_mapping.json
    ├── output/
    │   ├── prepare_data.log / prepare_model.log
    │   ├── conversion_w8a8.log / conversion_w16a16.log
    │   ├── ds_tc_resnet_backend_w8a8_esp32p4.espdl
    │   ├── ds_tc_resnet_backend_w16a16_esp32p4.espdl
    │   ├── w8a8/                     # .espdl / .info / .json / comparison.json
    │   └── w16a16/
    └── work/                         # 配置、校准/验证特征和完整流式状态
```

`latest.json` 的 `model` 为相对于 `esp32p4-dl/` 的默认模型路径；`models` 同时记录两个精度的路径、大小与 SHA-256。本次转换记录保存在独立转换目录的 `conversion.json`，转换产物不写入 `training/`。成功前不会更新指针；停止、错误的中间结果不能覆盖之前成功的模型。

界面选择“清空并重新训练”时，会先回收相关后台进程，再清空当前工作区 `training/` 和 `esp32p4-dl/` 中的全部内容，包括旧模型、转换历史、日志、`latest.json` 及临时文件；保留这两个目录本身。接续训练保留原有结果。

类别顺序保持 `silence=0`、`unknown=1`、工作区关键词从 2 开始。校准使用本次训练清单、各类别最多 4 个不同 WAV，以真实历史状态抽取最多 512 个校准步；验证优先采用 validation/testing 清单，各类别最多 2 个 WAV。缺少独立验证样本的类别会在 `comparison.json` 中明确记录回用训练样本，不能把这些结果视作泛化准确率。

板端输入为 MFCC 和显式状态，参照 `models/io_mapping.json` 回灌状态，并按训练的单段音频长度重置状态。`frontend.json` 和 `frontend.npz` 记录与训练一致的窗、Mel、DCT 和 dB 参数。ESP-DL 导出时可能改变张量布局，实际模型布局及内嵌测试输入输出以 `.info` 为准。转换不涉及固件构建、烧录或实际硬件验证。

点击“停止训练”在整个流水线均有效：尚未开始的阶段不执行；当前量化进程若未及时响应会在等待 5 秒后终止。所有子进程由界面已有的进程作业管理，退出时一并回收。
