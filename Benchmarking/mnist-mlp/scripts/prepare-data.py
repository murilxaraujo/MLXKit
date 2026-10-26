#!/usr/bin/env python3
"""Download Keras MNIST once and expose portable raw uint8 arrays."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import struct
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import BinaryIO


SOURCE_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
ARCHIVE_NAME = "mnist.npz"
ARRAYS = {
    "x_train.npy": ("train-images-uint8.bin", (60_000, 28, 28)),
    "y_train.npy": ("train-labels-uint8.bin", (60_000,)),
    "x_test.npy": ("test-images-uint8.bin", (10_000, 28, 28)),
    "y_test.npy": ("test-labels-uint8.bin", (10_000,)),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_npy_uint8(stream: BinaryIO) -> tuple[tuple[int, ...], bytes]:
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("MNIST archive member is not an NPY array")
    version = stream.read(2)
    if len(version) != 2:
        raise ValueError("Truncated NPY version")
    if version[0] == 1:
        header_length_bytes = stream.read(2)
        if len(header_length_bytes) != 2:
            raise ValueError("Truncated NPY v1 header length")
        header_length = struct.unpack("<H", header_length_bytes)[0]
    elif version[0] in (2, 3):
        header_length_bytes = stream.read(4)
        if len(header_length_bytes) != 4:
            raise ValueError("Truncated NPY v2/v3 header length")
        header_length = struct.unpack("<I", header_length_bytes)[0]
    else:
        raise ValueError(f"Unsupported NPY version {version[0]}.{version[1]}")

    header_bytes = stream.read(header_length)
    if len(header_bytes) != header_length:
        raise ValueError("Truncated NPY header")
    header = ast.literal_eval(header_bytes.decode("latin1").strip())
    if header.get("descr") != "|u1":
        raise ValueError(f"Expected uint8 NPY data, received {header.get('descr')!r}")
    if header.get("fortran_order"):
        raise ValueError("Fortran-ordered MNIST arrays are not supported")

    shape = tuple(int(dimension) for dimension in header["shape"])
    payload = stream.read()
    expected_bytes = 1
    for dimension in shape:
        expected_bytes *= dimension
    if len(payload) != expected_bytes:
        raise ValueError(
            f"NPY payload size mismatch: expected {expected_bytes}, got {len(payload)}"
        )
    return shape, payload


def download_archive(archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists() and archive_path.stat().st_size > 0:
        return

    print(f"Downloading MNIST from {SOURCE_URL}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="mnist-", suffix=".npz", dir=archive_path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
        os.replace(temporary_path, archive_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def cache_is_valid(data_dir: Path) -> bool:
    manifest_path = data_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_url") != SOURCE_URL:
            return False
        if not isinstance(manifest.get("preparation_python_version"), str):
            return False
        for filename, metadata in manifest["files"].items():
            path = data_dir / filename
            if not path.is_file() or path.stat().st_size != metadata["bytes"]:
                return False
            if sha256_file(path) != metadata["sha256"]:
                return False
        return len(manifest["files"]) == len(ARRAYS)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def prepare(data_dir: Path, force: bool = False) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    if not force and cache_is_valid(data_dir):
        print(f"Using prepared MNIST cache at {data_dir}")
        return data_dir / "manifest.json"

    archive_path = data_dir / "source" / ARCHIVE_NAME
    download_archive(archive_path)
    written_files: dict[str, dict[str, int | str]] = {}

    with zipfile.ZipFile(archive_path) as archive:
        for member_name, (output_name, expected_shape) in ARRAYS.items():
            with archive.open(member_name) as member:
                shape, payload = read_npy_uint8(member)
            if shape != expected_shape:
                raise ValueError(
                    f"Unexpected shape for {member_name}: {shape}; expected {expected_shape}"
                )
            output_path = data_dir / output_name
            temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
            temporary_path.write_bytes(payload)
            os.replace(temporary_path, output_path)
            written_files[output_name] = {
                "shape": list(shape),
                "bytes": len(payload),
                "sha256": sha256_file(output_path),
            }

    manifest = {
        "format_version": 1,
        "source_url": SOURCE_URL,
        "source_archive": ARCHIVE_NAME,
        "source_archive_sha256": sha256_file(archive_path),
        "image_dtype": "uint8",
        "image_layout": "N,H,W row-major bytes; flatten to 784 and divide by 255",
        "label_dtype": "uint8",
        "preparation_python_version": platform.python_version(),
        "files": written_files,
    }
    manifest_path = data_dir / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    print(f"Prepared MNIST cache at {data_dir}")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Recreate prepared files")
    arguments = parser.parse_args()
    try:
        prepare(arguments.data_dir.expanduser().resolve(), force=arguments.force)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"MNIST data preparation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
