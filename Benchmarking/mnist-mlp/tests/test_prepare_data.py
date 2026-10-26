from __future__ import annotations

import importlib.util
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare-data.py"
SPEC = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREPARE_DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE_DATA)

INIT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare-initialization.py"
INIT_SPEC = importlib.util.spec_from_file_location("prepare_initialization", INIT_SCRIPT)
assert INIT_SPEC is not None and INIT_SPEC.loader is not None
PREPARE_INITIALIZATION = importlib.util.module_from_spec(INIT_SPEC)
INIT_SPEC.loader.exec_module(PREPARE_INITIALIZATION)


def npy_uint8(shape: tuple[int, ...], payload: bytes) -> bytes:
    header = repr(
        {"descr": "|u1", "fortran_order": False, "shape": shape}
    )
    # NPY v1 header ends in a newline and aligns the data to a 16-byte boundary.
    prefix_size = 10
    header_bytes = header.encode("latin1")
    padding = (-(prefix_size + len(header_bytes) + 1)) % 16
    header_bytes += b" " * padding + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes + payload


class ReadNpyTests(unittest.TestCase):
    def test_reads_uint8_shape_and_payload(self) -> None:
        raw = npy_uint8((2, 2), bytes([0, 1, 254, 255]))
        shape, payload = PREPARE_DATA.read_npy_uint8(io.BytesIO(raw))
        self.assertEqual(shape, (2, 2))
        self.assertEqual(payload, bytes([0, 1, 254, 255]))

    def test_rejects_wrong_dtype(self) -> None:
        raw = npy_uint8((1,), b"\x07").replace(b"'|u1'", b"'<f4'", 1)
        with self.assertRaises(ValueError):
            PREPARE_DATA.read_npy_uint8(io.BytesIO(raw))

    def test_rejects_truncated_payload(self) -> None:
        raw = npy_uint8((2,), b"\x01")
        with self.assertRaisesRegex(ValueError, "payload size mismatch"):
            PREPARE_DATA.read_npy_uint8(io.BytesIO(raw))


class SharedInitializationTests(unittest.TestCase):
    def test_same_seed_produces_identical_weight_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            PREPARE_INITIALIZATION.generate(first, seed=42)
            PREPARE_INITIALIZATION.generate(second, seed=42)
            first_manifest = json.loads((first / "manifest.json").read_text())
            second_manifest = json.loads((second / "manifest.json").read_text())
            self.assertEqual(first_manifest, second_manifest)
            for filename in first_manifest["files"]:
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())

    def test_biases_are_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            PREPARE_INITIALIZATION.generate(output, seed=7)
            biases = (output / "dense3-bias-f32le.bin").read_bytes()
            self.assertEqual(biases, bytes(len(biases)))

if __name__ == "__main__":
    unittest.main()
