#!/usr/bin/env python3
"""Generate shared PyTorch-layout parameters for the two-layer tanh RNN."""

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


EMBEDDING_SIZE = 100
HIDDEN_SIZE = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="init-", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(contents)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generate(output_dir: Path, data_dir: Path, seed: int) -> Path:
    if array.array("f").itemsize != 4 or sys.byteorder != "little":
        raise RuntimeError("Shared initialization requires 32-bit little-endian floats")
    dataset = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    vocabulary_size = int(dataset["vocabulary_size"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    files: dict[str, dict[str, int | str | list[int]]] = {}

    def save(name: str, shape: list[int], bound: float) -> None:
        count = math.prod(shape)
        values = array.array("f", (rng.uniform(-bound, bound) for _ in range(count)))
        filename = f"{name}-f32le.bin"
        write_atomic(output_dir / filename, values.tobytes())
        files[filename] = {"shape": shape, "bytes": count * 4, "sha256": sha256_file(output_dir / filename)}

    save("embedding-weight", [vocabulary_size, EMBEDDING_SIZE], 0.1)
    recurrent_bound = 1 / math.sqrt(HIDDEN_SIZE)
    for layer in range(2):
        input_size = EMBEDDING_SIZE if layer == 0 else HIDDEN_SIZE
        prefix = f"rnn-layer{layer}"
        save(f"{prefix}-weight-ih", [HIDDEN_SIZE, input_size], recurrent_bound)
        save(f"{prefix}-weight-hh", [HIDDEN_SIZE, HIDDEN_SIZE], recurrent_bound)
        save(f"{prefix}-bias-ih", [HIDDEN_SIZE], recurrent_bound)
        save(f"{prefix}-bias-hh", [HIDDEN_SIZE], recurrent_bound)
    save("output-weight", [vocabulary_size, HIDDEN_SIZE], recurrent_bound)
    save("output-bias", [vocabulary_size], recurrent_bound)

    manifest = {
        "format_version": 1,
        "architecture": "wikitext2-vanilla-rnn-v1",
        "seed": seed,
        "vocabulary_size": vocabulary_size,
        "dataset_vocabulary_sha256": dataset["vocabulary_sha256"],
        "algorithm": "Python random.Random uniform; embedding [-0.1,0.1], recurrent and output params [-1/sqrt(hidden_size),+1/sqrt(hidden_size)]",
        "weight_layout": "PyTorch [out,in] for linear/recurrent weights; float32 little-endian",
        "files": files,
    }
    path = output_dir / "manifest.json"
    write_atomic(path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    print(f"Prepared shared RNN initialization at {output_dir} (vocab={vocabulary_size:,})")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed cannot be negative")
    generate(args.output_dir.expanduser().resolve(), args.data_dir.expanduser().resolve(), args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
