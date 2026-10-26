from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare-initialization.py"
SPEC = importlib.util.spec_from_file_location("prepare_cnn_initialization", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class CNNInitializationTests(unittest.TestCase):
    def test_seed_reproduces_every_file_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            PREPARE.generate(first, seed=42)
            PREPARE.generate(second, seed=42)
            first_manifest = json.loads((first / "manifest.json").read_text())
            second_manifest = json.loads((second / "manifest.json").read_text())
            self.assertEqual(first_manifest, second_manifest)
            for filename in first_manifest["files"]:
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())

    def test_expected_kernel_shapes_and_zero_biases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            PREPARE.generate(output, seed=7)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["files"]["conv1-weight-f32le.bin"]["shape"], [32, 1, 5, 5])
            self.assertEqual(manifest["files"]["conv3-weight-f32le.bin"]["shape"], [64, 32, 5, 5])
            self.assertEqual(manifest["files"]["fc1-weight-f32le.bin"]["shape"], [256, 576])
            for name, shape, _, _ in PREPARE.LAYERS:
                bias = (output / f"{name}-bias-f32le.bin").read_bytes()
                self.assertEqual(bias, bytes(shape[0] * 4))


if __name__ == "__main__":
    unittest.main()
