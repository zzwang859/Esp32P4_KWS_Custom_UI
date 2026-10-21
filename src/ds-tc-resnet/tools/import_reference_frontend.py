"""Reproduce the pinned frontend asset from the original reference ONNX."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from windows_runtime import preload_msvc
preload_msvc()
import numpy as np
import onnx
from onnx import numpy_helper

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
args = parser.parse_args()
source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
if source_hash != "7ff161caded03c8beef17d3da0933bbe7143f731f5cc76421cdc25664a9f1a56":
    raise ValueError("Source differs from the pinned reference model; update the contract explicitly first")
constants = {value.name: numpy_helper.to_array(value) for value in onnx.load(args.source).graph.initializer}
prefix = "streaming/speech_features/"
arrays = dict(window=constants[prefix + "windowing_1/mul/y:0"],
              mel_filters=constants[prefix + "magnitude_rdf_tmel_1/Const:0"],
              dct=constants[prefix + "dct_1/MatMul/b:0"],
              normalization=constants["ConstantFolding/streaming/speech_features/normalizer_1/truediv_recip:0"])
for name, shape in dict(window=(480,), mel_filters=(224, 80), dct=(80, 40), normalization=(40,)).items():
    if arrays[name].shape != shape:
        raise ValueError(f"Unexpected {name} shape")
destination = ROOT / "data/esp32p4_frontend.npz"
np.savez_compressed(destination, **arrays)
destination.with_suffix(".json").write_text(json.dumps(dict(frontend_profile="mfcc40_16k_int16_state6_v1",
    source_model=args.source.name, source_sha256=source_hash,
    asset_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
    shapes={name: list(value.shape) for name, value in arrays.items()},
    fft_size=512, window_samples=480, hop_samples=160, magnitude_bins=224, floor=1e-12), indent=2), encoding="utf-8")
print(destination)
