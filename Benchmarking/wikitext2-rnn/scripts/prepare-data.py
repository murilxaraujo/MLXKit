#!/usr/bin/env python3
"""Prepare pinned WikiText-2 text as shared train/validation/test token windows."""

from __future__ import annotations

import argparse
import array
import collections
import hashlib
import json
import os
import random
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable


UPSTREAM = "https://github.com/liux2/RNN-on-wikitext2"
REVISION = "b2726e6bf08ddca5580e1fced26c57c1fd8e3657"
SPLITS = {
    "train": ("wiki.train.txt", "2d84b35af79e411d73ea7cae83c73964c6ac83b4"),
    "validation": ("wiki.valid.txt", "eec944eb2d9e46061cd34a6913528797db6842fc"),
    "test": ("wiki.test.txt", "6f3fc88e4cd88456ea892971f9f12243e8428624"),
}
SEQUENCE_LENGTH = 30
WINDOW_SIZE = SEQUENCE_LENGTH + 1
FORMAT_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(contents: bytes) -> str:
    header = f"blob {len(contents)}\0".encode("ascii")
    return hashlib.sha1(header + contents).hexdigest()


def write_atomic(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="wikitext-", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(contents)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def download_split(data_dir: Path, filename: str, expected_git_sha: str) -> Path:
    source_dir = data_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / filename
    if path.is_file():
        existing = path.read_bytes()
        if git_blob_sha1(existing) == expected_git_sha:
            return path
    url = f"https://raw.githubusercontent.com/liux2/RNN-on-wikitext2/{REVISION}/data/{filename}"
    request = urllib.request.Request(url, headers={"User-Agent": "MLXKit-wikitext2-benchmark/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        contents = response.read()
    actual = git_blob_sha1(contents)
    if actual != expected_git_sha:
        raise RuntimeError(f"Pinned source checksum mismatch for {filename}: {actual}")
    write_atomic(path, contents)
    return path


def tokenize_text(text: str, rng: random.Random, nltk_module=None) -> list[str]:
    if nltk_module is None:
        import nltk as nltk_module

    sentences = nltk_module.sent_tokenize(text)
    tokenized = [nltk_module.word_tokenize(sentence) for sentence in sentences]
    rng.shuffle(tokenized)
    tokens: list[str] = []
    for sentence in tokenized:
        sentence = [token.lower() for token in sentence]
        if not sentence:
            continue
        if not sentence[-1].isalnum():
            sentence = sentence[:-1]
        tokens.extend(["<s>", *sentence, "</s>"])
    return tokens


def sequence_pairs(tokens: Iterable[str], token_to_id: dict[str, int]) -> tuple[list[int], list[int], int]:
    encoded = [token_to_id.get(token, token_to_id["<unk>"]) for token in tokens]
    usable = len(encoded) // WINDOW_SIZE * WINDOW_SIZE
    inputs: list[int] = []
    targets: list[int] = []
    for offset in range(0, usable, WINDOW_SIZE):
        window = encoded[offset : offset + WINDOW_SIZE]
        inputs.extend(window[:-1])
        targets.extend(window[1:])
    return inputs, targets, usable // WINDOW_SIZE


def int32_bytes(values: list[int]) -> bytes:
    if array.array("i").itemsize != 4 or sys.byteorder != "little":
        raise RuntimeError("Prepared token IDs require 32-bit little-endian integers")
    packed = array.array("i", values)
    return packed.tobytes()


def prepare(data_dir: Path, seed: int, force: bool = False) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    if not force and manifest_path.is_file():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (current.get("format_version") == FORMAT_VERSION and current.get("seed") == seed
                and current.get("upstream_revision") == REVISION
                and all((data_dir / name).is_file() and sha256_file(data_dir / name) == metadata["sha256"]
                        for name, metadata in current.get("files", {}).items())):
            print(f"Using prepared WikiText-2 cache at {data_dir}")
            return current

    import nltk

    for resource in ("punkt", "punkt_tab"):
        if not nltk.download(resource, quiet=True, raise_on_error=True):
            raise RuntimeError(f"Could not download NLTK tokenizer resource {resource}")

    source_paths = {split: download_split(data_dir, *spec) for split, spec in SPLITS.items()}
    rng = random.Random(seed)
    tokenized = {
        split: tokenize_text(path.read_text(encoding="utf-8"), rng, nltk)
        for split, path in source_paths.items()
    }

    # The upstream script adds the held-out splits to the vocabulary. Keep the
    # model benchmark leakage-free: derive IDs from training tokens only.
    counts = collections.Counter(tokenized["train"])
    vocab = ["<unk>"] + sorted(
        (token for token in counts if token != "<unk>"),
        key=lambda token: (-counts[token], token),
    )
    token_to_id = {token: index for index, token in enumerate(vocab)}
    output_files: dict[str, dict[str, int | str]] = {}
    sequence_counts: dict[str, int] = {}
    target_tokens: dict[str, int] = {}
    for split, tokens in tokenized.items():
        inputs, targets, count = sequence_pairs(tokens, token_to_id)
        for suffix, values in (("inputs", inputs), ("targets", targets)):
            name = f"{split}-{suffix}-i32le.bin"
            payload = int32_bytes(values)
            write_atomic(data_dir / name, payload)
            output_files[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        sequence_counts[split] = count
        target_tokens[split] = count * SEQUENCE_LENGTH

    vocab_payload = (json.dumps(vocab, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    write_atomic(data_dir / "vocabulary.json", vocab_payload)
    output_files["vocabulary.json"] = {
        "bytes": len(vocab_payload),
        "sha256": hashlib.sha256(vocab_payload).hexdigest(),
    }
    source_hashes = {path.name: sha256_file(path) for path in source_paths.values()}
    manifest = {
        "format_version": FORMAT_VERSION,
        "dataset": "WikiText-2 as stored by liux2/RNN-on-wikitext2",
        "source_url": UPSTREAM,
        "upstream_revision": REVISION,
        "upstream_git_blob_sha1": {filename: spec[1] for filename, spec in SPLITS.items()},
        "source_file_sha256": source_hashes,
        "seed": seed,
        "tokenizer": "NLTK sent_tokenize + word_tokenize; lowercase; shuffled sentences; <s>/</s> boundaries; strip terminal non-alphanumeric token",
        "nltk_version": nltk.__version__,
        "vocabulary_policy": "all token types from train split only; held-out unknown tokens map to <unk>; sorted by descending training frequency then token",
        "vocabulary_size": len(vocab),
        "vocabulary_sha256": output_files["vocabulary.json"]["sha256"],
        "sequence_length": SEQUENCE_LENGTH,
        "window_size": WINDOW_SIZE,
        "windowing": "non-overlapping chunks of 31 tokens; first 30 are input and last 30 are one-step-shifted targets; incomplete final chunk dropped per split",
        "sequence_counts": sequence_counts,
        "target_token_counts": target_tokens,
        "files": output_files,
    }
    write_atomic(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    print(f"Prepared WikiText-2 at {data_dir}: vocab={len(vocab):,}, sequences={sequence_counts}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed cannot be negative")
    prepare(args.data_dir.expanduser().resolve(), args.seed, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
