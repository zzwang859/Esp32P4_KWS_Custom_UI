"""Persistent, source-addressed 1-second augmentation and MFCC cache."""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import wave

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from windows_runtime import preload_msvc
preload_msvc()

import numpy as np
import torch
import torchaudio

from config import load_config
from data.audio import MfccExtractor, load_wav
from data.silero import MODEL, UTILS, PARAMETERS, PADDING_SAMPLES, SileroVad
from device_contract import frontend_sha256

VERSION = 3
RATE = 16000
SPLITS = ("training", "validation", "testing")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def crop(audio, start, size=RATE, *, wrap_negative=False):
    """Extract a window; keyword starts before zero continue from the source tail."""
    result = np.zeros(size, np.float32)
    if wrap_negative and start < 0:
        missing = min(-start, size)
        result[:missing] = audio[-missing:]
    left, right = max(0, start), min(len(audio), start + size)
    if right > left:
        result[left - start:right - start] = audio[left:right]
    return result


def export_speech(path, audio, info, check_stop):
    """Export the original-amplitude Tstart:Tend region, without augmentation."""
    start = max(0, min(len(audio), round(info["Tstart"] * RATE)))
    end = max(0, min(len(audio), round(info["Tend"] * RATE)))
    if end <= start:
        raise ValueError(f"Invalid Silero VAD speech interval: {path}")
    check_stop()
    pcm = np.clip(np.rint(audio[start:end] * 32768), -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with wave.open(str(temporary), "wb") as wav:
            wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            wav.writeframes(pcm.tobytes())
        check_stop()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(path=str(path), start_sample=start, end_sample=end,
                samples=end - start, sample_rate=RATE)


def keyword_windows(end, duration_seconds, rng):
    duration_samples = round(duration_seconds * RATE)
    end_sample = round(end * RATE)
    if duration_samples < 2 * RATE or not 0 < end_sample <= duration_samples:
        raise ValueError("关键词录音须至少 2 秒，且 VAD 结束位置须在原录音内")
    tail_samples = duration_samples - end_sample
    if tail_samples >= RATE // 10:
        count, spread_samples = 10, RATE // 10
    else:
        spread_samples = tail_samples
        count = tail_samples // (RATE // 50) + 1  # fix((S - Tend) / 20 ms) + 1
    return [dict(start_sample=round(end_sample + a * spread_samples) - RATE, position_a=float(a))
            for a in rng.random(count)]


def silence_context(sources, audios, rng):
    """Draw exactly 2 seconds from recorded silence, wrapping short recordings."""
    lengths = np.asarray([len(audio) for audio in audios], dtype=np.float64)
    index = int(rng.choice(len(audios), p=lengths / lengths.sum()))
    audio, source = audios[index], sources[index]
    start = int(rng.integers(max(1, len(audio) - PADDING_SAMPLES + 1))) if len(audio) >= PADDING_SAMPLES else int(rng.integers(len(audio)))
    context = audio[(start + np.arange(PADDING_SAMPLES)) % len(audio)].copy()
    return context, dict(path=source["path"], source_sha256=source["sha256"], start_sample=start,
                         samples=PADDING_SAMPLES, wrapped=start + PADDING_SAMPLES > len(audio))


def keyword_gains(peak):
    db = 20 * math.log10(peak) if peak > 0 else -math.inf
    return [0, *[gain for gain in (3, 6, 9, 12, 15) if db + gain <= 1e-7], -3, -6, -9]


def unknown_starts(length):
    if length <= RATE:
        return [0]
    starts = list(range(0, length - RATE + 1, 320))
    if starts[-1] != length - RATE:
        starts.append(length - RATE)
    return starts


def load_feature(entry):
    """Load exactly the augmented sample used in training/quantization."""
    values = np.load(entry["feature_path"], mmap_mode="r", allow_pickle=False)
    return np.array(values[int(entry["feature_index"])], dtype=np.float32, copy=True)


def write_report(root, classes, grouped):
    """Count active MFCC samples, excluding unreferenced historical cache files."""
    counts = {split: Counter(entry["label_index"] for entry in grouped[split]) for split in SPLITS}
    lines = ["# 数据预处理报告", "", f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}", "",
             "数量按本次有效清单中的 MFCC 样本统计（含位置、增益增强），不是 .npy 文件数；不计未引用的历史缓存。", "",
             "| 类别 index | 素材目录 | 类别文本 | MFCC 总数 | 训练集 | 验证集 | 测试集 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for item in classes:
        index = item["index"]
        directory = Path(item["directory"]).name
        if index >= 2:
            directory = "keywords/" + directory
        label = item["label"].replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")
        values = [counts[split][index] for split in SPLITS]
        lines.append(f"| {index} | {directory} | {label} | {sum(values)} | {values[0]} | {values[1]} | {values[2]} |")
    totals = [len(grouped[split]) for split in SPLITS]
    white = sum(entry["label_index"] == 1 and entry["path"] is None for entries in grouped.values() for entry in entries)
    unknown = sum(counts[split][1] for split in SPLITS)
    lines.extend([f"| 合计 | — | — | {sum(totals)} | {totals[0]} | {totals[1]} | {totals[2]} |", "",
                  f"unknown 包含：录音切片 {unknown - white} 条，生成白噪声：{white} 条。", "",
                  "关键词 Silero VAD v6 语音段保存在 `keywords/keywordn/原录音名.wav`，"
                  "按 Tstart 到 Tend 裁剪，保留原幅度，格式为 16 kHz 单声道 PCM16；无语音录音不导出。"
                  "检测时前后各附加 2 秒 silence 录音，Tstart/Tend 及裁剪均使用原始关键词录音坐标。"
                  "`vad.json` 的 `speech_clip` 记录导出路径和采样边界。", ""])
    temporary = root / "report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(root / "report.md")


def preprocess(config, *, force=False):
    data = config["data"]
    if int(data["sample_rate"]) != RATE or int(data["clip_duration_ms"]) != 1000:
        raise ValueError("Workspace preprocessing requires 16 kHz and 1000 ms clips")
    workspace = Path(data["root"]).resolve()
    root = Path(data["preprocess_dir"])
    run_dir = Path(config["train"]["run_dir"]).expanduser().resolve()
    if root.is_symlink() or root.resolve() != run_dir / "preprocess" or root.parent.resolve() != run_dir:
        raise ValueError("Preprocess cache must be training run_dir/preprocess without external links")
    root.mkdir(parents=True, exist_ok=True)
    stop_file = Path(config["train"]["stop_file"]) if config["train"].get("stop_file") else None

    def check_stop():
        if stop_file is not None and stop_file.exists():
            raise InterruptedError("Preprocessing stopped by user")

    check_stop()
    print("PREPROCESS: rebuilding all WAV data" if force else "PREPROCESS: checking reusable data", flush=True)
    torch.set_num_threads(4)
    extractor = MfccExtractor(data, config["features"]).eval()
    seed = int(config["project"]["seed"])
    recipe = dict(version=VERSION, seed=seed, sample_rate=RATE, samples=RATE, features=config["features"],
                  torch=torch.__version__, torchaudio=torchaudio.__version__,
                  mfcc_implementation=digest(Path(__file__).with_name("audio.py")),
                  frontend_asset_sha256=frontend_sha256(), keyword_end_jitter_ms=100,
                  keyword_tail_step_ms=20, keyword_positions=10, keyword_negative_start_wrap=True,
                  up_steps=5, down_steps=3, gain_step_db=3,
                  silence_count=1000, silence_gain_count=500, silence_gain_range=[-8, 10],
                  unknown_stride=320, white_count=100, white_gain_range=[-40, -10])
    vad_signature = key(dict(backend="silero-vad", version="6.0", seed=seed, model=digest(MODEL),
                             adapter=digest(Path(__file__).with_name("silero.py")),
                             upstream_utils=digest(UTILS), parameters=PARAMETERS,
                             padding_samples=PADDING_SAMPLES, context_recipe_version=1))
    # Refresh raw sources even on a standalone repeated invocation: augmented
    # manifests omit recordings rejected by VAD and contain synthetic entries.
    from data.prepare import prepare
    prepare(config)
    manifests = Path(data["manifests_dir"])
    sources = []
    seen_sources = set()
    for split in SPLITS:
        for line in (manifests / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["path"] is None or entry["path"] in seen_sources:
                continue
            seen_sources.add(entry["path"])
            path = Path(entry["path"]).resolve()
            if not path.is_relative_to(workspace) or not path.is_file():
                raise ValueError(f"Unsafe or missing source WAV: {path}")
            sources.append(dict(path=str(path), split=split, label=entry["label"], label_index=entry["label_index"],
                                sha256=digest(path), relative=path.relative_to(workspace).as_posix()))
    sources.sort(key=lambda source: (source["label_index"], source["relative"]))
    # Source split assignments are inherited. All positions and gain variants of
    # one recording/speaker remain together, including overlapping unknown clips.
    grouped = {split: [] for split in SPLITS}
    vad = None
    vad_dictionaries = {}
    generated = reused = skipped = 0
    source_records = []

    def rng_for(identity):
        return np.random.default_rng(int(key([seed, identity])[:16], 16))

    def build(source, plans, audio=None, *, kind, extra=None):
        nonlocal generated, reused
        check_stop()
        identity = source["relative"]
        signature = key(dict(recipe=recipe, source=source["sha256"], relative=identity,
                             kind=kind, extra=extra, plans=plans))
        folder = root / "features" / Path(identity).parent
        folder.mkdir(parents=True, exist_ok=True)
        stem = Path(identity).stem + "-" + signature[:20]
        feature_path, meta_path = folder / (stem + ".npy"), folder / (stem + ".json")
        cached = False
        if not force:
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                cached = metadata["signature"] == signature and metadata["samples"] == len(plans) and digest(feature_path) == metadata["mfcc_sha256"]
            except (OSError, ValueError, KeyError):
                pass
        if cached:
            reused += 1
        else:
            if audio is None and kind != "white_noise":
                audio = load_wav(source["path"], RATE).numpy()
            batches = []
            noise_rng = rng_for(signature)
            for offset in range(0, len(plans), 64):
                check_stop()
                samples = []
                for plan in plans[offset:offset + 64]:
                    if kind == "white_noise":
                        sample = noise_rng.standard_normal(RATE).astype(np.float32)
                        sample /= max(float(np.abs(sample).max()), 1e-20)
                    else:
                        sample = crop(audio, plan["start_sample"], wrap_negative=kind == "keyword")
                    sample = sample * np.float32(10 ** (plan["gain_db"] / 20))
                    samples.append(np.clip(sample, -1, 1))
                # Preserve the batch/channel axes for independent MFCC samples.
                with torch.no_grad():
                    features = extractor(torch.from_numpy(np.stack(samples)[:, None, :])).squeeze(1).numpy()
                if not np.isfinite(features).all():
                    raise ValueError(f"Non-finite MFCC: {identity}")
                batches.append(features)
            values = np.concatenate(batches).astype(np.float32)
            temporary = feature_path.with_suffix(".npy.tmp")
            with temporary.open("wb") as output:
                np.save(output, values, allow_pickle=False)
            check_stop()
            temporary.replace(feature_path)
            atomic_json(meta_path, dict(signature=signature, source=source["path"], source_sha256=source["sha256"],
                                       samples=len(plans), shape=list(values.shape), mfcc_sha256=digest(feature_path), plans=plans))
            generated += 1
        source_records.append(dict(source=source["path"], kind=kind, feature_path=str(feature_path),
                                   samples=len(plans), reused=cached, source_sha256=source["sha256"]))
        for index, plan in enumerate(plans):
            split = plan.get("split", source["split"])
            grouped[split].append(dict(path=source["path"], label=source["label"], label_index=source["label_index"],
                                       feature_path=str(feature_path), feature_index=index, augmentation=plan))
        print(f"PREPROCESS {'reuse' if cached else 'build'}: {identity} -> {len(plans)} MFCC samples", flush=True)

    # Exactly 1000 background samples globally; source selection is weighted by
    # actual recording duration. Exactly 500 get independently drawn gain values.
    silence = [source for source in sources if source["label_index"] == 0]
    silence_audio = [load_wav(source["path"], RATE).numpy() for source in silence]
    if not silence or any(len(audio) == 0 for audio in silence_audio):
        raise ValueError("Silence recordings must not be empty")
    vad_signature = key(dict(backend=vad_signature,
                             silence_sources=[(source["relative"], source["sha256"]) for source in silence]))
    silence_rng = rng_for(["silence", [item["sha256"] for item in silence]])
    weights = np.asarray([max(1, len(audio) - RATE + 1) for audio in silence_audio], np.float64)
    chosen = silence_rng.choice(len(silence), size=1000, p=weights / weights.sum())
    silence_plans = defaultdict(list)
    for index, source_index in enumerate(chosen):
        check_stop()
        start = int(silence_rng.integers(max(1, len(silence_audio[source_index]) - RATE + 1)))
        gain = float(silence_rng.uniform(-8, 10)) if index < 500 else 0.0
        silence_plans[int(source_index)].append(dict(start_sample=start, gain_db=gain, random_gain=index < 500))
    for index, plans in silence_plans.items():
        build(silence[index], plans, silence_audio[index], kind="silence")

    for source in sources:
        check_stop()
        label_index = source["label_index"]
        if label_index == 0:
            continue
        audio = load_wav(source["path"], RATE).numpy()
        if not len(audio):
            raise ValueError(f"Empty recording: {source['path']}")
        if label_index == 1:
            plans = [dict(start_sample=start, gain_db=0) for start in unknown_starts(len(audio))]
            build(source, plans, audio, kind="unknown")
            continue
        directory = Path(source["relative"]).parent.as_posix()
        if directory not in vad_dictionaries:
            path = root / directory / "vad.json"
            vad_dictionaries[directory] = {}
            if not force:
                try:
                    vad_dictionaries[directory] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
        dictionary = vad_dictionaries[directory]
        name = Path(source["path"]).name
        info = dictionary.get(name, {})
        if force or info.get("source_sha256") != source["sha256"] or info.get("vad_signature") != vad_signature:
            if vad is None:
                vad = SileroVad()
            context_rng = rng_for(["vad_context", source["relative"], source["sha256"], vad_signature])
            prefix, prefix_info = silence_context(silence, silence_audio, context_rng)
            suffix, suffix_info = silence_context(silence, silence_audio, context_rng)
            info = vad.analyze(audio, check_stop, prefix=prefix, suffix=suffix)
            peak = float(np.abs(audio).max())
            info.update(source_sha256=source["sha256"], vad_signature=vad_signature, duration_seconds=len(audio) / RATE,
                        silence_context=dict(prefix=prefix_info, suffix=suffix_info),
                        max_amplitude=peak, max_pcm=peak * 32768, peak_dbfs=20 * math.log10(peak) if peak > 0 else None)
        clip_path = root / directory / name
        if clip_path.is_symlink() or not clip_path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe speech output path: {clip_path}")
        info["speech_clip"] = (None if info["Tstart"] is None else
                               export_speech(clip_path, audio, info, check_stop))
        if info["speech_clip"] is None:
            clip_path.unlink(missing_ok=True)
        dictionary[name] = info
        atomic_json(root / directory / "vad.json", dictionary)
        if info["Tstart"] is None:
            skipped += 1
            print(f"PREPROCESS skip (no speech): {source['relative']}", flush=True)
            continue
        rng = rng_for([source["sha256"], source["relative"]])
        windows = keyword_windows(info["Tend"], len(audio) / RATE, rng)
        plans = [{**window, "gain_db": gain} for window in windows for gain in keyword_gains(info["max_amplitude"])]
        build(source, plans, audio, kind="keyword", extra=vad_signature)

    noise_rng = rng_for("white_noise")
    from data.prepare import split_for
    white = dict(path=None, sha256=key(["white_noise", seed]), relative="unknown/generated_white_noise",
                 label=data["classes"][1]["label"], label_index=1, split="training")
    plans = [dict(gain_db=float(noise_rng.uniform(-40, -10)),
                  split=split_for(Path(f"white_noise_{index}.wav"), data["validation_percentage"], data["testing_percentage"]))
             for index in range(100)]
    build(white, plans, kind="white_noise")
    for item in data["classes"]:
        if not any(entry["label_index"] == item["index"] for entry in grouped["training"]):
            raise ValueError(f"No usable training samples for class {item['index']} ({item['label']}); check VAD metadata")
    check_stop()
    counts = {split: len(entries) for split, entries in grouped.items()}
    for split, entries in grouped.items():
        temporary = manifests / f"{split}.jsonl.tmp"
        temporary.write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries), encoding="utf-8")
        temporary.replace(manifests / f"{split}.jsonl")
        atomic_json(root / f"{split}.json", entries)
    result = dict(recipe=recipe, vad_signature=vad_signature, counts=counts, force_rebuild=bool(force),
                  built_sources=generated, reused_sources=reused, skipped_no_speech=skipped, sources=source_records)
    atomic_json(root / "metadata.json", result)
    write_report(root, data["classes"], grouped)
    print(f"PREPROCESS completed: {counts}, built={generated}, reused={reused}, no_speech={skipped}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Rebuild all VAD and MFCC results from current WAV files")
    args = parser.parse_args()
    preprocess(load_config(args.config), force=args.force)
