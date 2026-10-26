#!/usr/bin/env python3
"""Train the PyTorch MNIST CNN and write a paired benchmark report."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


TRAIN_COUNT = 48_000
VALIDATION_COUNT = 12_000
TEST_COUNT = 10_000
BATCH_SIZE_DEFAULT = 32
EPOCHS_DEFAULT = 5
WARMUP_STEPS_DEFAULT = 10


class MNISTCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = F.relu(self.conv1(images))
        images = F.dropout(F.relu(F.max_pool2d(self.conv2(images), 2)), p=0.5, training=self.training)
        images = F.dropout(F.relu(F.max_pool2d(self.conv3(images), 2)), p=0.5, training=self.training)
        images = images.reshape(images.shape[0], 3 * 3 * 64)
        images = F.dropout(F.relu(self.fc1(images)), p=0.5, training=self.training)
        return F.log_softmax(self.fc2(images), dim=1)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def load_images(path: Path, sample_count: int, device: torch.device) -> torch.Tensor:
    expected_bytes = sample_count * 28 * 28
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{path} has an unexpected size; rerun the shared MNIST preparation script")
    values = np.fromfile(path, dtype=np.uint8).reshape(sample_count, 1, 28, 28).copy()
    return torch.from_numpy(values).to(device=device, dtype=torch.float32).div_(255.0)


def load_labels(path: Path, sample_count: int, device: torch.device) -> torch.Tensor:
    if path.stat().st_size != sample_count:
        raise ValueError(f"{path} has an unexpected size; rerun the shared MNIST preparation script")
    return torch.from_numpy(np.fromfile(path, dtype=np.uint8).astype(np.int64)).to(device)


def load_initialization(directory: Path, model: MNISTCNN) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with torch.no_grad():
        for name, layer in (("conv1", model.conv1), ("conv2", model.conv2), ("conv3", model.conv3),
                            ("fc1", model.fc1), ("fc2", model.fc2)):
            weights_path = directory / f"{name}-weight-f32le.bin"
            bias_path = directory / f"{name}-bias-f32le.bin"
            expected = layer.weight.numel()
            weight = np.fromfile(weights_path, dtype="<f4")
            bias = np.fromfile(bias_path, dtype="<f4")
            if weight.size != expected or bias.size != layer.bias.numel():
                raise ValueError(f"Invalid shared initialization files for {name}")
            layer.weight.copy_(torch.from_numpy(weight.copy()).reshape_as(layer.weight))
            layer.bias.copy_(torch.from_numpy(bias.copy()).reshape_as(layer.bias))
    return {**manifest, "manifest_path": str(manifest_path)}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def peak_resident_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if sys.platform == "darwin" else usage * 1024)


def system_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def sysctl_value(name: str) -> str | None:
    try:
        value = subprocess.check_output(["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL).strip()
        return value or None
    except (OSError, subprocess.CalledProcessError):
        return None


def evaluate(model: MNISTCNN, images: torch.Tensor, labels: torch.Tensor, batch_size: int,
             criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    weighted_loss = 0.0
    correct = 0
    with torch.inference_mode():
        for start in range(0, labels.shape[0], batch_size):
            end = min(start + batch_size, labels.shape[0])
            logits = model(images[start:end])
            loss = criterion(logits, labels[start:end])
            weighted_loss += float(loss.item()) * (end - start)
            correct += int((logits.argmax(dim=1) == labels[start:end]).sum().item())
    synchronize(device)
    return weighted_loss / labels.shape[0], correct / labels.shape[0]


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    with path.with_name(path.stem + "-epochs.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "elapsed_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["epochs"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--initialization-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS_DEFAULT)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.seed < 0 or args.warmup_steps < 0:
        parser.error("epochs and batch-size must be positive; seed and warmup-steps cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    data_manifest_path = data_dir / "manifest.json"
    if not data_manifest_path.is_file():
        raise FileNotFoundError(f"No dataset manifest at {data_manifest_path}")
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))

    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = MNISTCNN().to(device)
    initialization = load_initialization(args.initialization_dir.expanduser().resolve(), model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 228_010:
        raise AssertionError(f"Unexpected model parameter count: {parameter_count}")

    train_images = load_images(data_dir / "train-images-uint8.bin", 60_000, device)
    train_labels = load_labels(data_dir / "train-labels-uint8.bin", 60_000, device)
    test_images = load_images(data_dir / "test-images-uint8.bin", TEST_COUNT, device)
    test_labels = load_labels(data_dir / "test-labels-uint8.bin", TEST_COUNT, device)
    criterion = nn.CrossEntropyLoss()
    initial_val_loss, initial_val_accuracy = evaluate(
        model, train_images[TRAIN_COUNT:], train_labels[TRAIN_COUNT:], args.batch_size, criterion, device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    batch_durations: list[float] = []
    epochs: list[dict[str, float | int]] = []

    backend = "pytorch-mps" if device.type == "mps" else "pytorch-cpu"
    print(f"Training {args.epochs} epochs on {backend}...")
    sys.stdout.flush()
    fit_started = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        for start in range(0, TRAIN_COUNT, args.batch_size):
            end = min(start + args.batch_size, TRAIN_COUNT)
            batch_images = train_images[start:end]
            batch_labels = train_labels[start:end]
            synchronize(device)
            batch_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            log_probabilities = model(batch_images)
            loss = criterion(log_probabilities, batch_labels)
            loss.backward()
            optimizer.step()
            synchronize(device)
            epoch_loss += float(loss.item()) * (end - start)
            epoch_correct += int((log_probabilities.argmax(dim=1) == batch_labels).sum().item())
            batch_durations.append(time.perf_counter() - batch_started)

        train_loss = epoch_loss / TRAIN_COUNT
        train_accuracy = epoch_correct / TRAIN_COUNT
        validation_loss, validation_accuracy = evaluate(
            model, train_images[TRAIN_COUNT:], train_labels[TRAIN_COUNT:], args.batch_size, criterion, device
        )
        record: dict[str, float | int] = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        epochs.append(record)
        print(f"Epoch {epoch + 1}/{args.epochs} — loss {train_loss:.5f}, accuracy {train_accuracy:.3f}, "
              f"val_loss {validation_loss:.5f}, val_accuracy {validation_accuracy:.3f}, "
              f"{record['elapsed_seconds']:.2f}s")
        sys.stdout.flush()
    fit_wall_seconds = time.perf_counter() - fit_started
    test_loss, test_accuracy = evaluate(model, test_images, test_labels, args.batch_size, criterion, device)

    steady_steps = batch_durations[min(args.warmup_steps, len(batch_durations)):]
    batch_seconds = sum(batch_durations)
    dataset = {
        "source_url": data_manifest["source_url"],
        "source_archive_sha256": data_manifest["source_archive_sha256"],
        "manifest": str(data_manifest_path),
        "preparation_python_version": data_manifest.get("preparation_python_version", "unknown"),
        "prepared_file_sha256": {name: meta["sha256"] for name, meta in data_manifest["files"].items()},
        "train_examples": TRAIN_COUNT,
        "validation_examples": VALIDATION_COUNT,
        "test_examples": TEST_COUNT,
        "input_shape": [1, 28, 28],
        "preprocessing": "uint8 pixels cast to float32, divided by 255; NCHW for PyTorch",
        "validation_split": "last 12000 of original 60000 training examples",
    }
    revision = os.environ.get("MLXKIT_GIT_REVISION", "unknown")
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "implementation": "pytorch",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_revision": revision,
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
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "backend": {
            "name": backend,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "device": str(device),
            "mps_current_allocated_memory_bytes": (
                torch.mps.current_allocated_memory() if device.type == "mps" else None
            ),
            "mps_driver_allocated_memory_bytes": (
                torch.mps.driver_allocated_memory() if device.type == "mps" else None
            ),
        },
        "dataset": dataset,
        "initialization": initialization,
        "configuration": {
            "architecture": ["Conv2d(1,32,5)", "Conv2d(32,32,5)", "Conv2d(32,64,5)", "Linear(576,256)", "Linear(256,10)"],
            "convolution_padding": "valid",
            "pooling": "2x2 max pooling after conv2 and conv3",
            "hidden_activation": "relu",
            "dropout_probability": 0.5,
            "dropout_locations": ["after second max-pool", "after third max-pool", "after fc1 ReLU"],
            "output": "10-way log-softmax",
            "parameters": parameter_count,
            "kernel_initializer": "shared seeded Glorot uniform",
            "bias_initializer": "zeros",
            "loss": "CrossEntropyLoss applied to log-softmax output (as in source notebook)",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-8,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "shuffle": False,
        },
        "measurements": {
            "initial_validation_loss": initial_val_loss,
            "initial_validation_accuracy": initial_val_accuracy,
            "fit_wall_seconds_including_validation": fit_wall_seconds,
            "training_batch_seconds": batch_seconds,
            "training_examples_per_second_batch_time": (
                TRAIN_COUNT * args.epochs / batch_seconds if batch_seconds > 0 else None
            ),
            "training_step_latency_median_seconds_excluding_warmup": median(steady_steps) if steady_steps else None,
            "training_step_latency_p90_seconds_excluding_warmup": percentile(steady_steps, 0.90),
            "warmup_steps_excluded_from_latency": min(args.warmup_steps, len(batch_durations)),
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "peak_process_resident_memory_method": "resource.getrusage(RUSAGE_SELF).ru_maxrss; macOS bytes",
            "device_synchronization": "torch.mps.synchronize before and after each timed MPS batch",
        },
        "epochs": epochs,
    }
    write_report(output_path, report)
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test accuracy: {test_accuracy:.4%}")
    print(f"Training throughput: {report['measurements']['training_examples_per_second_batch_time']:.1f} examples/s")
    print(f"PyTorch backend: {backend}")
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"MNIST PyTorch CNN benchmark failed: {error}", file=sys.stderr)
        raise
