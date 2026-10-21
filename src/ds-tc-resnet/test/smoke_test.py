#!/usr/bin/env python3
"""Fast structural and streaming-equivalence tests."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from windows_runtime import preload_msvc
preload_msvc()
import torch

from config import load_config  # noqa: E402
from model import DSTCResNet  # noqa: E402
from runtime import predict_features  # noqa: E402


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = load_config(PROJECT_ROOT / "config.yaml")
        self.model = DSTCResNet(self.config, label_count=8).eval()

    def test_reference_parameter_count(self) -> None:
        trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        self.assertEqual(trainable, 21472)

    def test_stream_modes_match_causal_non_stream(self) -> None:
        features = torch.randn(2, 40, 17)
        expected = predict_features(self.model, features, "non_stream")
        for mode in ("stream_internal", "stream_external"):
            actual = predict_features(self.model, features, mode, chunk_frames=1)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

