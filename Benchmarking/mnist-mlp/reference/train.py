#!/usr/bin/env python3
"""Train the TensorFlow/Keras MNIST MLP and emit benchmark records."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import tensorflow as tf


INPUTS = 784
HIDDEN = 256
CLASSES = 10
TRAIN_COUNT = 48_000
VALIDATION_COUNT = 12_000
TEST_COUNT = 10_000
LEARNING_RATE = 1e-3
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-7
WARMUP_STEPS_DEFAULT = 10


class BenchmarkTiming(tf.keras.callbacks.Callback):
    def __init__(self, warmup_steps: int) -> None:
        super().__init__()
        self.warmup_steps = warmup_steps
        self.batch_durations: list[float] = []
        self.epoch_durations: list[float] = []
        self._batch_started = 0.0
        self._epoch_started = 0.0

    def on_train_batch_begin(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        self._batch_started = time.perf_counter()

    def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        self.batch_durations.append(time.perf_counter() - self._batch_started)

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self._epoch_started = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self.epoch_durations.append(time.perf_counter() - self._epoch_started)


def load_images(path: Path, sample_count: int) -> np.ndarray:
    expected_bytes = sample_count * 28 * 28
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{path} has an unexpected size; rerun prepare-data.py")
    return np.fromfile(path, dtype=np.uint8).reshape(sample_count, 28 * 28).astype(np.float32) / 255.0


def load_labels(path: Path, sample_count: int) -> np.ndarray:
    if path.stat().st_size != sample_count:
        raise ValueError(f"{path} has an unexpected size; rerun prepare-data.py")
    return np.fromfile(path, dtype=np.uint8).astype(np.int32)


def load_shared_initialization(directory: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights: list[np.ndarray] = []
    for name, fan_in, fan_out in [("dense1", 784, 256), ("dense2", 256, 256), ("dense3", 256, 10)]:
        weight_path = directory / f"{name}-weight-f32le.bin"
        bias_path = directory / f"{name}-bias-f32le.bin"
        weight = np.fromfile(weight_path, dtype="<f4")
        bias = np.fromfile(bias_path, dtype="<f4")
        if weight.size != fan_in * fan_out or bias.size != fan_out:
            raise ValueError(f"Invalid shared initialization files for {name}")
        # The prepared binary layout is [out, in]; Keras Dense stores [in, out].
        weights.extend([weight.reshape(fan_out, fan_in).T.copy(), bias])
    return weights, {**manifest, "manifest_path": str(manifest_path)}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def peak_resident_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports ru_maxrss in bytes; most other Unix platforms report KiB.
    return int(usage if sys.platform == "darwin" else usage * 1024)


def system_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def sysctl_value(name: str) -> str | None:
    try:
        value = subprocess.check_output(
            ["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return value or None
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_epoch_csv(path: Path, epochs: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "elapsed_seconds"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(epochs)


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--initialization-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON report")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS_DEFAULT)
    arguments = parser.parse_args()
    if arguments.epochs <= 0 or arguments.batch_size <= 0 or arguments.warmup_steps < 0:
        parser.error("epochs and batch-size must be positive; warmup-steps cannot be negative")
    return arguments


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No dataset manifest at {manifest_path}; run prepare-data.py first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    initialization_weights, initialization_manifest = load_shared_initialization(
        args.initialization_dir.expanduser().resolve()
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    train_images = load_images(data_dir / "train-images-uint8.bin", 60_000)
    train_labels = load_labels(data_dir / "train-labels-uint8.bin", 60_000)
    test_images = load_images(data_dir / "test-images-uint8.bin", TEST_COUNT)
    test_labels = load_labels(data_dir / "test-labels-uint8.bin", TEST_COUNT)
    if train_labels.shape[0] != train_images.shape[0] or test_labels.shape[0] != test_images.shape[0]:
        raise ValueError("Image and label sample counts do not match")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(INPUTS,)),
            tf.keras.layers.Dense(HIDDEN, activation="relu"),
            tf.keras.layers.Dense(HIDDEN, activation="relu"),
            tf.keras.layers.Dense(CLASSES, activation="softmax"),
        ],
        name="mnist_mlp",
    )
    model.set_weights(initialization_weights)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            beta_1=BETA_1,
            beta_2=BETA_2,
            epsilon=EPSILON,
            amsgrad=False,
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    physical_gpus = tf.config.list_physical_devices("GPU")
    logical_gpus = tf.config.list_logical_devices("GPU")
    metal_version = package_version("tensorflow-metal")
    if logical_gpus and metal_version:
        backend = "tensorflow-metal"
    elif logical_gpus:
        backend = "tensorflow-gpu-unverified-plugin"
    else:
        backend = "tensorflow-cpu"

    callback = BenchmarkTiming(args.warmup_steps)
    initial_validation = model.evaluate(
        train_images[TRAIN_COUNT:],
        train_labels[TRAIN_COUNT:],
        batch_size=args.batch_size,
        verbose=0,
        return_dict=True,
    )
    started = time.perf_counter()
    history = model.fit(
        train_images[:TRAIN_COUNT],
        train_labels[:TRAIN_COUNT],
        validation_data=(train_images[TRAIN_COUNT:], train_labels[TRAIN_COUNT:]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=[callback],
        verbose=2,
    )
    fit_wall_seconds = time.perf_counter() - started

    test_values = model.evaluate(
        test_images,
        test_labels,
        batch_size=args.batch_size,
        verbose=0,
        return_dict=True,
    )

    epoch_rows: list[dict[str, float | int]] = []
    for index in range(args.epochs):
        epoch_rows.append(
            {
                "epoch": index + 1,
                "train_loss": float(history.history["loss"][index]),
                "train_accuracy": float(history.history["accuracy"][index]),
                "validation_loss": float(history.history["val_loss"][index]),
                "validation_accuracy": float(history.history["val_accuracy"][index]),
                "elapsed_seconds": float(callback.epoch_durations[index]),
            }
        )

    steady_steps = callback.batch_durations[args.warmup_steps :]
    all_batch_seconds = sum(callback.batch_durations)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "implementation": "tensorflow-keras",
        "created_at_utc": now,
        "git_revision": os.environ.get("MLXKIT_GIT_REVISION", "unknown"),
        "git_worktree_dirty": os.environ.get("MLXKIT_GIT_DIRTY", "false") == "true",
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hardware_model": sysctl_value("hw.model"),
            "processor": sysctl_value("machdep.cpu.brand_string") or platform.processor() or None,
            "macos_version": platform.mac_ver()[0] or None,
            "physical_memory_bytes": system_memory_bytes(),
            "peak_process_resident_bytes": peak_resident_bytes(),
        },
        "software": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": package_version("keras"),
            "numpy": np.__version__,
            "tensorflow_metal": metal_version,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "backend": {
            "name": backend,
            "physical_gpus": [device.name for device in physical_gpus],
            "logical_gpus": [device.name for device in logical_gpus],
        },
        "dataset": {
            "source_url": manifest["source_url"],
            "source_archive_sha256": manifest["source_archive_sha256"],
            "manifest": str(manifest_path),
            "preparation_python_version": manifest.get("preparation_python_version", "unknown"),
            "prepared_file_sha256": {
                filename: metadata["sha256"] for filename, metadata in manifest["files"].items()
            },
            "train_examples": TRAIN_COUNT,
            "validation_examples": VALIDATION_COUNT,
            "test_examples": TEST_COUNT,
            "input_shape": [INPUTS],
            "preprocessing": "uint8 pixels flattened and cast to float32, divided by 255",
            "validation_split": "last 12000 of the original 60000 training examples",
        },
        "initialization": initialization_manifest,
        "configuration": {
            "architecture": [784, 256, 256, 10],
            "hidden_activation": "relu",
            "output": "10-way softmax probabilities",
            "parameters": model.count_params(),
            "kernel_initializer": "shared seeded Glorot uniform",
            "bias_initializer": "zeros",
            "loss": "sparse categorical cross-entropy",
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "beta_1": BETA_1,
            "beta_2": BETA_2,
            "epsilon": EPSILON,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "shuffle": True,
        },
        "measurements": {
            "initial_validation_loss": float(initial_validation["loss"]),
            "initial_validation_accuracy": float(initial_validation["accuracy"]),
            "fit_wall_seconds_including_validation": fit_wall_seconds,
            "training_batch_seconds": all_batch_seconds,
            "training_examples_per_second_batch_time": (
                TRAIN_COUNT * args.epochs / all_batch_seconds if all_batch_seconds > 0 else None
            ),
            "training_step_latency_median_seconds_excluding_warmup": (
                median(steady_steps) if steady_steps else None
            ),
            "training_step_latency_p90_seconds_excluding_warmup": percentile(steady_steps, 0.90),
            "warmup_steps_excluded_from_latency": min(args.warmup_steps, len(callback.batch_durations)),
            "test_loss": float(test_values["loss"]),
            "test_accuracy": float(test_values["accuracy"]),
            "peak_process_resident_memory_method": "resource.getrusage(RUSAGE_SELF).ru_maxrss; macOS bytes",
        },
        "epochs": epoch_rows,
    }

    write_json(output_path, report)
    write_epoch_csv(output_path.with_name(output_path.stem + "-epochs.csv"), epoch_rows)
    print(f"Test loss: {float(test_values['loss']):.6f}")
    print(f"Test accuracy: {float(test_values['accuracy']):.4%}")
    print(f"Training throughput: {report['measurements']['training_examples_per_second_batch_time']:.1f} examples/s")
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"MNIST TensorFlow benchmark failed: {error}", file=sys.stderr)
        raise
