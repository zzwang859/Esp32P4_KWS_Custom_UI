# KWS 工作区中的 Silero VAD v6

源码来源：[snakers4/silero-vad 官方 v6.0](https://github.com/snakers4/silero-vad/tree/v6.0)，固定提交 `fba061dc5559f696e62171e9a0741782b0fdc23c`。本目录为该提交的 GitHub 源码快照，包含 MIT LICENSE、官方模型和示例。归档来源及 SHA-256 保存在 `UPSTREAM.json`。

KWS 通过 `../ds-tc-resnet/data/silero.py` 加载本目录的 `init_jit_model`、`get_speech_timestamps` 和 `src/silero_vad/data/silero_vad.jit`，运行期间无需下载。通过 `torch.cuda.is_available()` 判断 CUDA 是否可用，可用时将模型与输入放到 CUDA，否则使用 CPU；日志打印 `SILERO_VAD_DEVICE=cuda` 或 `SILERO_VAD_DEVICE=cpu`。训练预处理和 ONNX 推理入口开启 VAD 检查时共用此适配器，每条录音独立重置状态。CUDA 推理需要支持 CUDA 的 PyTorch 和兼容驱动，CPU 版 PyTorch 使用 CPU。唯一上游源码改动是将 `utils_vad.py` 中推理异常时启动交互式 ipdb 的行为改为抛出异常，避免桌面训练任务进入调试器；详细记录见 `UPSTREAM.json`。

工作区适配行为：

- 从 `silence/` 真实录音中按固定种子为关键词前后各抽取 32000 个采样（16kHz 下 2 秒），静音不足 2 秒时循环取样。抽取信息保存到关键词的 `vad.json`。
- 拼接音频完整参与检测，使用官方默认语音段参数：阈值 0.5、最短语音 250ms、最短静音 100ms、边界扩展 30ms。
- 语音段与原关键词范围求交，再减去前置 32000 个采样，得到相对于原录音的坐标。Tstart/Tend 分别为有效段最早开始和最晚结束，始终在原录音范围内。
- 导出 WAV 与后续 1 秒位置增强都使用原录音及其坐标。附加的两段静音素材不会进入关键词导出片段。
- 概率序列覆盖拼接后的全部音频，512 点一帧；第 i 项的原录音时间起点为 `i×512/16000−2` 秒。`segments` 为原始采样坐标，`segments_padded` 为拼接采样坐标。
- 取消请求通过官方进度回调检查，缓存校验涵盖模型、适配代码、上游工具代码、参数及静音录音哈希。
