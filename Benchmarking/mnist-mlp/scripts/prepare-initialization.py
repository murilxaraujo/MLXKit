#!/usr/bin/env python3
"""Create identical seeded Glorot-uniform weights for both implementations."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path


LAYERS = [
    ("dense1", 784, 256),
    ("dense2", 256, 256),
    ("dense3", 256, 10),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(output_dir: Path, seed: int) -> Path:
    if array.array("f").itemsize != 4 or sys.byteorder != "little":
        raise RuntimeError("Shared initialization requires 32-bit little-endian floats")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    files: dict[str, dict[str, int | str | list[int]]] = {}

    for name, fan_in, fan_out in LAYERS:
        # Store weights in MLX Linear's [out, in] order. The Keras runner
        # transposes these same values to its [in, out] Dense-kernel layout.
        count = fan_in * fan_out
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        values = array.array("f", (rng.uniform(-bound, bound) for _ in range(count)))
        weight_name = f"{name}-weight-f32le.bin"
        bias_name = f"{name}-bias-f32le.bin"
        write_atomic(output_dir / weight_name, values.tobytes())
        write_atomic(output_dir / bias_name, array.array("f", [0.0] * fan_out).tobytes())
        files[weight_name] = {
            "shape": [fan_out, fan_in],
            "bytes": count * 4,
            "sha256": sha256_file(output_dir / weight_name),
        }
        files[bias_name] = {
            "shape": [fan_out],
            "bytes": fan_out * 4,
            "sha256": sha256_file(output_dir / bias_name),
        }

    manifest = {
        "format_version": 1,
        "seed": seed,
        "algorithm": "Glorot/Xavier uniform [-sqrt(6/(fan_in+fan_out)), +sqrt(...)]",
        "bias_initializer": "zeros",
        "weight_layout": "little-endian float32 [out, in] (MLX Linear layout)",
        "generator": "Python random.Random; identical binary values consumed by both implementations",
        "python_version": sys.version.split()[0],
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    write_atomic(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    print(f"Prepared shared model initialization at {output_dir}")
    return manifest_path


def write_atomic(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="init-", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(contents)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed cannot be negative")
    generate(args.output_dir.expanduser().resolve(), args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
