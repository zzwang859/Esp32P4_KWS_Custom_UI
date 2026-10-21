# 独立 DS-TC-ResNet 关键词识别项目

该目录包含数据、预处理、模型、训练、PyTorch 推理、ONNX 导出和 ONNX Runtime 推理所需的全部代码。模型与前端固定为参考项目 `E:/esp32-p4-onnx/model_toolchain/ds_tc_resnet` 的设备契约 `mfcc40_16k_int16_state6_v1`；运行时不读取外部参考工程。网络为 6 个时序深度可分离卷积块，通道数 `64,32,32,32,64,64`，8 类时可训练参数为 21,472，分类头随工作区类别数变化。

## 目录

```text
ds-tc-resnet/
├── config.yaml                 # 数据、模型、训练、推理和 ONNX 配置
├── data/
│   ├── mini_speech_commands/   # 8,000 条本地训练音频
│   ├── prepare.py              # 稳定划分并生成 manifest/labels
│   ├── audio.py                # WAV 与 MFCC 预处理
│   └── dataset.py
├── model/ds_tc_resnet.py       # 网络及内部/外部状态流式实现
├── train/train.py              # 训练、恢复、TensorBoard、checkpoint
├── test/evaluate.py            # 数据集准确率测试
├── test/infer.py               # 指定 WAV 的 checkpoint 推理
├── logs/                       # TensorBoard 日志与 checkpoint
├── onnx/export.py              # 导出最新 ONNX
├── onnx/run/infer.py           # ONNX Runtime WAV 推理
└── run.py                      # 完整流水线
```

## 环境

统一使用 KWS_Custom_UI 根目录由 uv 管理的 Python 3.12 环境，在项目根目录同步依赖后进入本目录：

```bash
uv sync --locked
source .venv/Scripts/activate
cd src/ds-tc-resnet
```

依赖仅维护在根目录 `pyproject.toml` 与 `uv.lock`，不再在本目录创建虚拟环境。若需特定 CUDA 构建，在根目录配置 uv 的 PyTorch 来源并同步锁文件；训练时 `device: auto` 优先使用 CUDA，没有可用 CUDA 时自动使用 CPU。

## 配置

所有主要参数都在 `config.yaml`：

- `model.filters`、`kernel_sizes`、`dilations` 等拓扑参数固定为参考模型，修改为其他数值会明确报错。
- `model.inference_mode` 可设为 `non_stream`、`stream_internal` 或 `stream_external`。
- `train.steps` 控制总训练步数，`train.resume` 可指定本项目产生的 `.pt` checkpoint。
- `onnx.mode` 固定为 `stream_external`、`chunk_frames=1`；ONNX 使用设备的固定端口名称与形状。

卷积固定使用 causal padding、stride=1、pool=1，残差相加后再 ReLU，最终采用 98 帧滑动窗口平均池化。旧版累计 sum/count 模型及旧 MFCC 配置的 checkpoint 不允许接续，需要重新开始训练；旧历史文件保留。兼容新契约的 checkpoint 仍支持接续训练。

## 分步运行

先生成稳定的数据划分。替换 `data/mini_speech_commands/<类别>/*.wav` 后需要重新执行这一步：

```bash
python data/prepare.py
```

训练并观察 TensorBoard：

```bash
python train/train.py --steps 3000 --device auto
tensorboard --logdir logs/default/tensorboard
```

从 checkpoint 继续训练时，`--steps` 表示目标总步数。例如从第 1,000 步继续到第 3,000 步：

```bash
python train/train.py --resume logs/default/latest.pt --steps 3000
```

测试和单条音频推理：

```bash
python test/evaluate.py --checkpoint logs/default/best.pt
python test/infer.py data/mini_speech_commands/yes/004ae714_nohash_0.wav \
  --checkpoint logs/default/best.pt
```

导出写入配置的目标 ONNX 路径；不会删除同目录的其他模型文件：

```bash
python onnx/export.py --checkpoint logs/default/best.pt
python onnx/run/infer.py data/mini_speech_commands/yes/004ae714_nohash_0.wav
```

ONNX 流式模型输入的是 MFCC 帧和显式状态；`onnx/run/infer.py` 负责完整 WAV 预处理、状态初始化及逐帧回灌，不截取为 1 秒。显式 `--model` 路径直接使用，相对路径以终端当前目录为准；省略时沿用配置的导出路径。标签、模式、chunk 大小和状态映射均保存在 ONNX metadata 中。

每个 10ms chunk 的原始 logits 与概率写入项目根目录下的 `workspace/test/infer_result.md`，按类别顺序输出 Markdown 表格，类别单元格格式为 `logits[概率%]`，两者均保留两位小数，例如 `0.14[18.93%]`；每行概率最大的单元格使用 HTML 红色文字高亮，在 Markdown 预览中查看，并列时取索引最小的类别。每次运行覆盖。概率采用设备端稳定 softmax：原始 logits 减最大值、取指数、float32 累加后归一化，与消抖使用同一函数。音频末尾不足一个 chunk 时补零。终端只报告消抖通过的关键词及最终汇总。该入口要求 `stream_external`、`chunk_frames=1` 模型。

每次运行从工作区的 `workspace.json` 读取 `kws_filter` 四项参数：`keyword_threshold_percent`（默认 80，范围 0～100）、`cooldown_ms`（默认 500，最小值 1，其后为 10～1000、步长 10）、`stable_frames`（默认 3，范围 1～50）、`need_vad_checked`（默认 false）。可用 `--workspace` 选择工作区；格式错误会报错。

消抖按设备端 `kws_stream_state.hpp` 执行：固定预热 100 帧，然后要求相同 top-1 关键词连续达到概率门限，开启 VAD 条件时还要求每帧 VAD 为语音，达到稳定帧数后确认。条件失败或类别切换会重置计数。确认关键词后启动全局冷却，期间不累积关键词稳定帧；冷却依据音频时间计算。持续同类只报告一次，稳定背景或其他已确认类别可重新允许同词触发；背景类别也需连续稳定，但不受概率、VAD 和冷却条件限制，不输出为关键词。预热 100 帧是设备固定常量，不放入四项配置。

设备 VAD 来自 AFE；PC 开启 `need_vad_checked` 时使用本地 Silero VAD v6 的逐帧概率≥0.5，按已完成的 32ms VAD 帧映射到 10ms KWS 帧，可能与设备 AFE 门控结果不同；关闭时跳过 VAD 计算。JSON 只控制 PC 推理，不会修改设备固件参数。字段表与完整流程见[项目 README](../../README.md#验证)。

ONNX 端口和 ESP-DL 布局与设备一致：特征 `[1,1,40]`，6 个状态为 `[1,11,1,40]`、`[1,13,1,64]`、`[1,15,1,32]`、`[1,17,1,32]`、`[1,57,1,32]`、`[1,98,1,64]`，对应 `onnx_next_state_1/2/3/4/5/7:0`。额外保留 `onnx_next_state_6:0` 输出，不回灌。分类输出为 `onnx_logits:0`。详细端口与前端契约见 [设备协议说明](../../docs/device-protocol.md#固定模型与前端契约)。

MFCC 固定使用 `data/esp32p4_frontend.npz` 中从参考 ONNX 提取的窗函数、Mel、DCT 和归一化常量，采用幅度谱、自然对数及固定系数缩放。普通 1 秒训练样本产生 98 帧；WAV 流式推理加入启动零历史后产生 100 帧。常量来源/校验和见同名 JSON，运行不依赖参考项目。需要重新提取相同版本资产时，在项目根目录执行：

```powershell
.venv/Scripts/python.exe src/ds-tc-resnet/tools/import_reference_frontend.py --source E:/esp32-p4-onnx/model_toolchain/ds_tc_resnet/models/ds_tc_resnet_stream_external.onnx
```

## 一键流水线

以下命令依次执行数据预处理、训练、checkpoint 测试、单音频推理、ONNX 导出和 ONNX 推理：

```bash
python run.py --steps 3000 --device auto
```

快速检查完整链路可使用较少步数和独立日志目录：

```bash
python run.py --steps 2 --device cpu --run-dir logs/smoke
```

结构及流式一致性测试：

```bash
python -m unittest test.smoke_test
```

## KWS Studio 工作区

界面“开始训练”调用本目录的 `workspace_train.py`，也可独立使用：

```powershell
python workspace_train.py --workspace ../../workspace/mini_speech_commands --steps 10
```

界面发现历史训练后会询问接续或清空重训。接续恢复最近一次历史的 `latest.pt` 及优化器，保留旧训练与转换历史并生成新的运行目录；清空重训会关闭相关 TensorBoard 和转换进程后，同时清空所选工作区 `training/` 与 `esp32p4-dl/` 的全部内容，包括转换模型、日志和最新成功指针。录音及工作区配置保持不变，不会清理源码的 `train/`。类别映射变化时禁止接续。

工作区入口追加 10 步的示例：

```powershell
python workspace_train.py --workspace ../../workspace/mini_speech_commands --resume ../../workspace/mini_speech_commands/training/<历史时间戳>/latest.pt --steps 10
```

此处 `--steps` 表示本次追加步数；底层独立入口 `train/train.py --steps` 仍表示目标累计步数。

类别顺序始终为 `silence=0`、`unknown=1`，随后按 `workspace.json` 中的 `rows` 顺序排列关键词（从 2 开始）。使用真实的 `silence/`、`unknown/`、`keywords/keywordn/` WAV，不混入其他目录。每次运行在工作区 `training/<时间戳>/` 保存配置快照、类别映射、清单、checkpoint、ONNX 及输出一致性验证报告。训练前自动启动 TensorBoard 并打开浏览器。

工作区入口随后自动调用相邻的 `espdl_toolchain`，使用本次最终 ONNX 与训练清单中的 MFCC 转换 ESP32-P4 的 INT8、INT16 模型（默认 INT16）。转换复用预处理特征，预处理、训练、ONNX 导出和量化均使用项目根目录的 uv `.venv`，量化子进程沿用训练解释器，不再读取 `KWS_ESPDL_PYTHON`。依赖在项目根目录运行 `uv sync --locked` 安装。独立的 `train/train.py`、`onnx/export.py`、`run.py` 不会自动启动这一步。

转换输出为 `<工作区>/esp32p4-dl/<训练时间戳>/{models,output,work}`，两种 `.espdl` 位于 `output/w8a8/`、`output/w16a16/`，另附量化参数、模型描述、真实音频对照报告、标签及 MFCC 参数。`<工作区>/esp32p4-dl/latest.json` 仅指向完整成功的一次转换，`model` 字段为相对于 `esp32p4-dl/` 的默认 INT16 模型路径。输入仍为 MFCC 与外部状态，不是原始音频。

界面点击停止后，后续导出或转换阶段不会启动；转换期间同样支持停止。失败、停止均不更新上一份成功结果。所有转换子进程继承界面的进程作业，退出界面时统一回收。

## 工作区 MFCC 预处理

每次开始训练后先自动运行 `data/preprocess.py --force`，无论全新训练、清空重训或接续训练，都会重新扫描当前 WAV 并重新执行全部 VAD、位置/增益增强和 MFCC 计算，将增强后的 1 秒样本直接处理为 float32 MFCC，保存在 `<工作区>/training/<本次训练日期时间>/preprocess/`，每份特征为 `40 × 98`。MFCC、字典、清单和 `report.md` 均随本次运行独立保存；接续训练保留旧结果并在新运行中重新生成，清空重训时随训练历史一起清理。显式指定 `--run-dir` 时，预处理目录为该运行目录下的 `preprocess/`。训练、ONNX 验证和 ESP32-P4 校准直接读取本次生成的结果。原始 WAV 不修改，也不保存大量增强 WAV。

- 关键词：使用 `src/silero-vad` 中固定版本的 Silero VAD v6.0 官方源码及 TorchScript 模型 `silero_vad.jit`，通过 `torch.cuda.is_available()` 自动选择设备：CUDA 可用时优先使用 CUDA，否则使用 CPU。模型与输入使用同一设备，日志打印 `SILERO_VAD_DEVICE=cuda/cpu`；此行为也适用于 ONNX 推理入口开启 VAD 检查时。每条关键词录音前后各附加 2 秒从 `silence/` 真实录音抽取的上下文，短于 2 秒的静音素材循环取样。模型在每条拼接录音开始前清空状态，按 512 点（32ms）逐帧检测，采用官方语音段后处理，阈值 0.5、最短语音 250ms、最短静音 100ms、段边界扩展 30ms。只保留与原关键词录音重叠的语音段，扣除前置 32000 个采样并裁定到 `[0, 原录音采样数]`，得到原始录音坐标下的 Tstart/Tend。只在附加素材中出现的语音会忽略。
- VAD 字典位于本次训练目录的 `preprocess/keywords/keywordn/vad.json`，记录原录音峰值、哈希、版本、原始坐标语音段 `segments`、拼接坐标语音段 `segments_padded`、前后填充素材的来源和采样起点 `silence_context`。概率覆盖整段拼接音频，`probability_time_offset_seconds=-2.0`，第 i 项对应原录音时间区间从 `i×512/16000−2` 秒开始。兼容现有 2 秒和 3 秒录音；无语音录音记录后跳过，某关键词没有可用训练样本时明确报错。
- 语音段导出：每条检出语音的关键词录音，按原始录音坐标 `[Tstart, Tend)` 裁剪一份 WAV，输出为 `preprocess/keywords/keywordn/原录音名.wav`，保持每个关键词独立文件夹；不从前后附加静音的拼接音频中截取。两端之外的静音去除，区间内的停顿保留；格式为 16 kHz、单声道 PCM16，保留原幅度，不进行位置/增益增强或补齐到 1 秒。`vad.json` 的 `speech_clip` 记录文件路径、起止采样位置及采样数，无语音时为 `null`。即使手动复用 VAD/MFCC 缓存，也会生成这些语音段；原始录音不修改。
- 设原始关键词录音时长为 S（至少 2 秒），VAD 最后语音段的终点为 Tend。Tend≤S−100 ms 时取 10 个固定种子生成的 a∈[0,1)，每个 1 秒窗口为 `[Tend+a×100 ms−1 s, Tend+a×100 ms)`；这同时覆盖 Tend<1 s 和 Tend=1 s。窗口起点为负时，从原录音末尾取相应长度，拼接原录音开头至窗口终点，不补零。Tend>S−100 ms 时令 W=S−Tend，生成 `fix(W/20 ms)+1` 个 a∈[0,1)，窗口为 `[Tend+a×W−1 s, Tend+a×W)`；Tend=S 时仍生成一份 `[S−1 s,S)`。全部位置仍按原录音坐标裁剪。
- 每个位置样本保留原增益，再增加 +3/+6/+9/+12/+15 dB 及 −3/−6/−9 dB 版本。向上增益使用原录音整体峰值 dBFS 判断，超过 0 dBFS 的版本不生成。
- silence：按文件真实长度随机截取总共 1000 份 1 秒样本，其中 500 份应用均匀随机 [−8,+10] dB 增益，其余保持原增益。支持短素材补零及多个文件按可截取长度加权；增益后越过 PCM 幅度范围的部分限幅。
- unknown：每个原文件按 1 秒窗口、20 ms 步进完整扫描。若末尾不能落在步进网格上，追加最后一个完整窗口，短于 1 秒则补零；另生成 100 份峰值归一化高斯白噪声，并应用 [−40,−10] dB 随机增益。

`features/` 保存 `.npy` 数组及对应增强参数字典；`training.json`、`validation.json`、`testing.json` 是缓存样本清单；`metadata.json` 汇总样本数及构建/复用数量。`report.md` 为数据预处理报告，按 silence、unknown、各关键词目录及文本列出 MFCC 总数、训练/验证/测试数量，并说明 unknown 中录音切片与生成白噪声的数量；每次预处理完成后更新，复用缓存时也会生成。报告只统计本次有效样本，不累计未引用的历史缓存。每次训练在自己的 `training/<时间戳>/manifests/` 保存本次使用的清单。工作区先完成位置和增益增强，再按每个类别的 MFCC 样本数，以固定随机种子分配约 80% 训练集、20% 验证集；不能整除时按最接近的整数分配，每类至少留一份训练样本。工作区不另建测试集，训练后的测试流程使用验证集。由于同一录音的增强切片可能进入两个集合，验证结果可能高估对新录音的泛化能力。

训练入口始终传入 `--force`，跳过旧 VAD 和 MFCC 缓存的复用，即使 WAV 未变化也重新计算；新增、修改和删除的 WAV 都反映到本次清单、关键词字典和报告中。清空重训仍只清空 `training/`、`esp32p4-dl/`，预处理随后重新生成当前结果；未引用的历史特征文件保留，供旧训练记录追溯。预处理期间点击停止会跳过后续训练、导出与转换，下次开始训练仍完整重新处理。

单独手动运行 `data/preprocess.py --config <config.yaml>` 而不加 `--force` 时，仍可按原录音 SHA-256、处理参数、随机种子、MFCC 配置/实现/库版本及 Silero VAD 模型、源码和上下文录音哈希校验并复用缓存；静音素材更新会使关键词 VAD 缓存失效。这不影响训练入口强制重算的行为。

Silero 源码来源、固定提交及本地适配说明见 [src/silero-vad/KWS_INTEGRATION.md](../silero-vad/KWS_INTEGRATION.md)。旧 TEN-VAD 已从预处理调用链移除；磁盘中的旧目录尚未清理，历史训练产物保留供追溯。
