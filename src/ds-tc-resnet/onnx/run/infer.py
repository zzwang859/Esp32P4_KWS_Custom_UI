#!/usr/bin/env python3
"""Infer the complete WAV, log chunk logits, and report debounced keywords."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from windows_runtime import preload_msvc
preload_msvc()

import numpy as np
import onnxruntime as ort
import torch

from config import load_config, resolve_path  # noqa: E402
from data.audio import MfccExtractor, load_wav  # noqa: E402
from device_contract import PROFILE
from kws_debounce import Debouncer, load_filter_config, softmax_probabilities

RESULT_PATH = PROJECT_ROOT.parents[1] / "workspace/test/infer_result.md"


class StreamingInference:
    """Feed consecutive 16 kHz mono samples into the same frontend and ONNX model as WAV inference."""

    def __init__(self, config: dict, model_path: Path, filter_config: dict):
        self.runtime = OnnxKWS(model_path)
        if self.runtime.mode != "stream_external" or int(self.runtime.metadata["chunk_frames"]) != 1:
            raise ValueError("实时仿真需要 chunk_frames=1 的 stream_external ONNX 模型")
        self.labels = json.loads(self.runtime.metadata["labels"])
        if self.labels[:2] != ["silence", "unknown"]:
            raise ValueError("模型类别必须以 silence、unknown 开头")
        self.frontend = MfccExtractor(config["data"], config["features"])
        self.debounce = Debouncer(filter_config)
        self.history = np.zeros(320, dtype=np.float32)
        self.pending = np.empty(0, dtype=np.float32)
        self.vad_pending = np.empty(0, dtype=np.float32)
        self.vad_high = False
        self.vad = None
        self.frame_index = 0
        self.set_filter(filter_config)

    def set_filter(self, config: dict) -> None:
        self.debounce.configure(config)
        if not self.debounce.need_vad_checked:
            self.vad = None
            self.vad_pending = np.empty(0, dtype=np.float32)
            self.vad_high = False
        elif self.vad is None:
            from data.silero import SileroVad
            self.vad = SileroVad().model
            self.vad.reset_states()

    @torch.inference_mode()
    def feed(self, samples: np.ndarray):
        """Yield (frame, logits, probabilities, detected_index) for each complete 10 ms frame."""
        self.pending = np.concatenate((self.pending, np.asarray(samples, dtype=np.float32)))
        while len(self.pending) >= 160:
            frame, self.pending = self.pending[:160], self.pending[160:]
            self.vad_pending = np.concatenate((self.vad_pending, frame))
            if self.vad is not None:
                while len(self.vad_pending) >= 512:
                    chunk, self.vad_pending = self.vad_pending[:512], self.vad_pending[512:]
                    self.vad_high = bool(self.vad(torch.from_numpy(chunk.copy()), 16000).item() >= 0.5)
            else:
                self.vad_pending = np.empty(0, dtype=np.float32)
            audio = torch.from_numpy(np.concatenate((self.history, frame)))
            features = self.frontend(audio).numpy()
            self.history = audio.numpy()[160:].copy()
            logits = self.runtime.predict(features)[0]
            if logits.shape != (len(self.labels),):
                raise ValueError("模型输出类别数与标签不匹配")
            probabilities = softmax_probabilities(logits)
            self.frame_index += 1
            now_us = self.frame_index * 10_000
            detected = self.debounce.update(int(np.argmax(logits)), logits, now_us,
                                             vad_high=self.vad_high)
            if detected >= 0:
                self.debounce.acknowledge(True, now_us)
            yield self.frame_index, logits, probabilities, detected


def audio_vad_flags(audio) -> np.ndarray:
    """Offline VAD gate using only completed 32-ms Silero chunks at each 10-ms step."""
    from data.silero import SileroVad, HOP, RATE
    model = SileroVad().model
    model.reset_states()
    probabilities = []
    for offset in range(0, len(audio) - HOP + 1, HOP):
        probabilities.append(float(model(audio[offset:offset + HOP], RATE).item()))
    chunk_count = (len(audio) + 159) // 160
    complete_frames = np.minimum(np.arange(1, chunk_count + 1) * 160, len(audio)) // HOP - 1
    flags = np.zeros(chunk_count, dtype=bool)
    if probabilities:
        available = complete_frames >= 0
        flags[available] = np.asarray(probabilities)[complete_frames[available]] >= 0.5
    return flags


class OnnxKWS:
    def __init__(self, model_path: Path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        self.metadata = self.session.get_modelmeta().custom_metadata_map
        if self.metadata.get("frontend_profile") != PROFILE:
            raise ValueError("ONNX 使用旧版模型/前端契约，请重新训练并导出")
        self.feature_name = self.metadata["feature_input_name"]
        self.logits_name = self.metadata["logits_output_name"]
        self.output_indices = {item.name: index for index, item in enumerate(self.outputs)}
        self.mode = self.metadata.get("mode", "non_stream")
        self.state_pairs = json.loads(self.metadata.get("external_state_pairs", "[]"))
        self.reset()

    def reset(self) -> None:
        self.states = {}
        for input_name, _ in self.state_pairs:
            info = next(item for item in self.inputs if item.name == input_name)
            shape = tuple(1 if not isinstance(value, int) else value for value in info.shape)
            self.states[input_name] = np.zeros(shape, dtype=np.float32)

    def predict_chunks(self, features: np.ndarray):
        """Yield each chunk's raw logits while retaining external state."""
        features = np.asarray(features, dtype=np.float32)
        if self.mode == "non_stream":
            yield self.session.run(None, {self.inputs[0].name: features})[0]
            return
        chunk_frames = int(self.metadata["chunk_frames"])
        for start in range(0, features.shape[2], chunk_frames):
            chunk = features[:, :, start:start + chunk_frames]
            if chunk.shape[2] < chunk_frames:
                chunk = np.pad(chunk, ((0, 0), (0, 0), (0, chunk_frames - chunk.shape[2])))
            feed = chunk.transpose(0, 2, 1) if self.metadata.get("features_layout") == "NTC" else chunk
            values = self.session.run(None, {self.feature_name: feed, **self.states})
            logits = values[self.output_indices[self.logits_name]]
            self.states = {
                input_name: values[self.output_indices[output_name]]
                for input_name, output_name in self.state_pairs
            }
            yield logits

    def predict(self, features: np.ndarray) -> np.ndarray:
        logits = None
        for logits in self.predict_chunks(features):
            pass
        if logits is None:
            raise ValueError("No feature frames to infer")
        return logits


def infer(config: dict, wav_path: Path, model_value: str | Path | None = None, *,
          result_path: Path | None = None, workspace: Path | None = None) -> np.ndarray:
    # Explicit --model paths are used directly, relative only to the shell CWD.
    model_path = (Path(model_value).expanduser().resolve() if model_value is not None
                  else resolve_path(config, config["onnx"]["output"]))
    workspace = Path(workspace) if workspace is not None else RESULT_PATH.parent
    debounce = Debouncer(load_filter_config(workspace / "workspace.json"))
    runtime = OnnxKWS(model_path)
    if runtime.mode != "stream_external" or int(runtime.metadata["chunk_frames"]) != 1:
        raise ValueError("整段音频消抖推理需要 chunk_frames=1 的 stream_external 模型")
    labels = json.loads(runtime.metadata["labels"])
    if labels[:2] != ["silence", "unknown"]:
        raise ValueError("消抖推理要求 index 0=silence，index 1=unknown")
    sample_rate = int(config["data"]["sample_rate"])
    audio = load_wav(wav_path, sample_rate)
    if audio.numel() == 0:
        raise ValueError("音频文件没有采样数据")
    features = MfccExtractor(config["data"], config["features"]).streaming(audio).numpy()
    vad_flags = audio_vad_flags(audio) if debounce.need_vad_checked else None
    recognized = []
    result_path = Path(result_path) if result_path is not None else workspace / "infer_result.md"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"model: {model_path}")
    print(f"mode: {runtime.mode}, duration: {audio.numel() / sample_rate:.3f}s, chunks: {features.shape[2]}")
    print(f"filter config: {workspace / 'workspace.json'}")
    print(f"debounce: warmup={debounce.warmup_chunks} chunks, stable={debounce.stable_chunks} chunks, "
          f"probability>={debounce.config['keyword_threshold_percent']}%, cooldown={debounce.config['cooldown_ms']}ms, "
          f"need_vad_checked={debounce.need_vad_checked}")
    if debounce.need_vad_checked:
        print("VAD gate: local Silero VAD v6; device VAD source is AFE")
    with result_path.open("w", encoding="utf-8", newline="") as output:
        headers = ["chunk", "time_s", *[f"{index}:{label}" for index, label in enumerate(labels)]]
        headers = [html.escape(value).replace("|", "&#124;").replace("\r", "").replace("\n", "<br>")
                   for value in headers]
        output.write("| " + " | ".join(headers) + " |\n")
        output.write("| " + " | ".join(["---:"] * len(headers)) + " |\n")
        for chunk_index, values in enumerate(runtime.predict_chunks(features), 1):
            logits = values[0]
            if logits.shape != (len(labels),):
                raise ValueError("模型输出类别数量与标签不匹配")
            probabilities = softmax_probabilities(logits)
            now_us = chunk_index * 10_000
            index = int(np.argmax(logits))
            cells = [f"{value:.2f}[{probability * 100:.2f}%]"
                     for value, probability in zip(logits, probabilities)]
            cells[index] = f'<span style="color: red;">{cells[index]}</span>'
            output.write("| " + " | ".join([str(chunk_index), f"{now_us / 1_000_000:.2f}", *cells]) + " |\n")
            detected = debounce.update(index, logits, now_us,
                                       vad_high=bool(vad_flags[chunk_index - 1]) if vad_flags is not None else False)
            if detected >= 0:
                if detected >= 2:
                    probability = probabilities[detected]
                    recognized.append(labels[detected])
                    print(f"[{now_us / 1_000_000:.2f}s] 识别结果: {labels[detected]} "
                          f"(index={detected}, probability={probability * 100:.2f}%)", flush=True)
                # Background results also rearm repeated-keyword suppression.
                debounce.acknowledge(True, now_us)
    print(f"logits: {result_path}")
    print("消抖识别结果: " + ("、".join(recognized) if recognized else "未检测到通过消抖的关键词"))
    return probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--model", type=Path, help="ONNX 文件完整路径；相对路径按当前工作目录解释，不拼接项目路径")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--workspace", type=Path, default=RESULT_PATH.parent)
    args = parser.parse_args()
    infer(load_config(args.config), args.wav, args.model, workspace=args.workspace)


if __name__ == "__main__":
    main()

