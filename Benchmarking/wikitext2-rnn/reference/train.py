#!/usr/bin/env python3
"""Train the PyTorch WikiText-2 vanilla RNN and emit a benchmark report."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
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
from torch.utils.data import DataLoader, TensorDataset

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from seeded_shuffle import splitmix64_shuffle_indices


EMBEDDING_SIZE = 100
HIDDEN_SIZE = 100
NUM_LAYERS = 2
SEQUENCE_LENGTH = 30
BATCH_SIZE_DEFAULT = 20
EPOCHS_DEFAULT = 20
LEARNING_RATE = 0.5
DROPOUT = 0.25
WARMUP_STEPS_DEFAULT = 10


class WikiTextRNN(nn.Module):
    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, EMBEDDING_SIZE)
        self.rnn = nn.RNN(
            input_size=EMBEDDING_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            nonlinearity="tanh",
            batch_first=True,
            bias=True,
        )
        self.output = nn.Linear(HIDDEN_SIZE, vocabulary_size)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(tokens))
        # Match the upstream implementation: reset state for each batch. Hidden
        # state is not carried across shuffled sequence windows.
        hidden = torch.zeros(NUM_LAYERS, tokens.shape[0], HIDDEN_SIZE, device=tokens.device)
        output, _ = self.rnn(embedded, hidden)
        return self.output(output).reshape(-1, self.output.out_features)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def load_split(data_dir: Path, split: str, sequence_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    size = sequence_count * SEQUENCE_LENGTH
    inputs_path = data_dir / f"{split}-inputs-i32le.bin"
    targets_path = data_dir / f"{split}-targets-i32le.bin"
    inputs = np.fromfile(inputs_path, dtype="<i4")
    targets = np.fromfile(targets_path, dtype="<i4")
    if inputs.size != size or targets.size != size:
        raise ValueError(f"Unexpected token count in {split} data; rerun prepare-data.py")
    return (
        torch.from_numpy(inputs.copy()).reshape(sequence_count, SEQUENCE_LENGTH).long(),
        torch.from_numpy(targets.copy()).reshape(sequence_count, SEQUENCE_LENGTH).long(),
    )


def load_initialization(directory: Path, model: WikiTextRNN) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with torch.no_grad():
        model.embedding.weight.copy_(torch.from_numpy(np.fromfile(directory / "embedding-weight-f32le.bin", dtype="<f4").copy()).reshape_as(model.embedding.weight))
        for layer in range(NUM_LAYERS):
            prefix = f"rnn-layer{layer}"
            for parameter_name, filename in (
                (f"weight_ih_l{layer}", f"{prefix}-weight-ih-f32le.bin"),
                (f"weight_hh_l{layer}", f"{prefix}-weight-hh-f32le.bin"),
                (f"bias_ih_l{layer}", f"{prefix}-bias-ih-f32le.bin"),
                (f"bias_hh_l{layer}", f"{prefix}-bias-hh-f32le.bin"),
            ):
                parameter = getattr(model.rnn, parameter_name)
                values = np.fromfile(directory / filename, dtype="<f4")
                if values.size != parameter.numel():
                    raise ValueError(f"Invalid shared initialization file: {filename}")
                parameter.copy_(torch.from_numpy(values.copy()).reshape_as(parameter))
        for parameter, filename in (
            (model.output.weight, "output-weight-f32le.bin"),
            (model.output.bias, "output-bias-f32le.bin"),
        ):
            values = np.fromfile(directory / filename, dtype="<f4")
            if values.size != parameter.numel():
                raise ValueError(f"Invalid shared initialization file: {filename}")
            parameter.copy_(torch.from_numpy(values.copy()).reshape_as(parameter))
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def evaluate(model: WikiTextRNN, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, int, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            token_count = targets.numel()
            total_loss += float(criterion(logits, targets.reshape(-1)).item()) * token_count
            correct_tokens += int((logits.argmax(dim=-1) == targets.reshape(-1)).sum().item())
            total_tokens += token_count
    synchronize(device)
    if total_tokens == 0:
        raise ValueError("Evaluation split produced no complete batches")
    mean_loss = total_loss / total_tokens
    return mean_loss, math.exp(min(mean_loss, 80)), total_tokens, correct_tokens / total_tokens


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def peak_resident_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


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


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    with path.with_name(path.stem + "-epochs.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["epoch", "train_loss", "train_perplexity", "train_token_accuracy", "validation_loss", "validation_perplexity", "validation_token_accuracy", "tokens", "elapsed_seconds"]
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
    parser.add_argument("--max-train-batches", type=int, default=0, help="optional bounded smoke-test limit; 0 uses the full split")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.seed < 0 or args.warmup_steps < 0 or args.max_train_batches < 0:
        parser.error("epochs and batch-size must be positive; seed, warmup-steps, and max-train-batches cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    init_dir = args.initialization_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    data_manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if data_manifest["vocabulary_sha256"] != json.loads((init_dir / "manifest.json").read_text(encoding="utf-8"))["dataset_vocabulary_sha256"]:
        raise ValueError("Dataset and initialization vocabulary hashes differ")
    counts = data_manifest["sequence_counts"]
    train_x, train_y = load_split(data_dir, "train", counts["train"])
    valid_x, valid_y = load_split(data_dir, "validation", counts["validation"])
    test_x, test_y = load_split(data_dir, "test", counts["test"])
    if args.max_train_batches:
        train_x = train_x[: args.max_train_batches * args.batch_size]
        train_y = train_y[: args.max_train_batches * args.batch_size]

    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    vocabulary_size = int(data_manifest["vocabulary_size"])
    model = WikiTextRNN(vocabulary_size).to(device)
    initialization = load_initialization(init_dir, model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_parameters = vocabulary_size * EMBEDDING_SIZE + 2 * (
        HIDDEN_SIZE * EMBEDDING_SIZE + HIDDEN_SIZE * HIDDEN_SIZE + 2 * HIDDEN_SIZE
    ) + vocabulary_size * HIDDEN_SIZE + vocabulary_size
    if parameter_count != expected_parameters:
        raise AssertionError(f"Unexpected model parameter count: {parameter_count} != {expected_parameters}")

    train_data = TensorDataset(train_x, train_y)
    valid_data = TensorDataset(valid_x, valid_y)
    test_data = TensorDataset(test_x, test_y)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    initial_validation_loss, initial_validation_perplexity, initial_validation_tokens, initial_validation_accuracy = evaluate(
        model, valid_loader, criterion, device
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    batch_durations: list[float] = []
    epochs: list[dict[str, float | int]] = []
    total_fit_tokens = 0
    backend = "pytorch-mps" if device.type == "mps" else "pytorch-cpu"
    print(f"Training {args.epochs} epochs on {backend} (vocab={vocabulary_size:,}, sequences={len(train_data):,})...")
    sys.stdout.flush()
    fit_started = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_order = splitmix64_shuffle_indices(len(train_data), args.seed + epoch)
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            sampler=epoch_order,
            shuffle=False,
            drop_last=True,
            num_workers=0,
        )
        model.train()
        epoch_started = time.perf_counter()
        weighted_loss = 0.0
        token_count = 0
        correct_tokens = 0
        for batch_index, (inputs, targets) in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            synchronize(device)
            batch_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets.reshape(-1))
            loss.backward()
            optimizer.step()
            synchronize(device)
            batch_tokens = targets.numel()
            weighted_loss += float(loss.item()) * batch_tokens
            correct_tokens += int((logits.argmax(dim=-1) == targets.reshape(-1)).sum().item())
            token_count += batch_tokens
            batch_durations.append(time.perf_counter() - batch_started)

        train_loss = weighted_loss / token_count
        train_perplexity = math.exp(min(train_loss, 80))
        train_accuracy = correct_tokens / token_count
        validation_loss, validation_perplexity, validation_tokens, validation_accuracy = evaluate(model, valid_loader, criterion, device)
        epoch_seconds = time.perf_counter() - epoch_started
        total_fit_tokens += token_count
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_perplexity": train_perplexity,
            "train_token_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_perplexity": validation_perplexity,
            "validation_token_accuracy": validation_accuracy,
            "tokens": token_count,
            "elapsed_seconds": epoch_seconds,
        }
        epochs.append(row)
        print(f"Epoch {epoch + 1}/{args.epochs} — loss {train_loss:.5f}, ppl {train_perplexity:.2f}, "
              f"val_loss {validation_loss:.5f}, val_ppl {validation_perplexity:.2f}, "
              f"{token_count:,} tokens, {epoch_seconds:.2f}s")
        sys.stdout.flush()

    fit_wall_seconds = time.perf_counter() - fit_started
    test_loss, test_perplexity, test_tokens, test_accuracy = evaluate(model, test_loader, criterion, device)
    steady_steps = batch_durations[min(args.warmup_steps, len(batch_durations)):]
    training_batch_seconds = sum(batch_durations)
    revision = os.environ.get("MLXKIT_GIT_REVISION", "unknown")
    files = data_manifest["files"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "implementation": "pytorch",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_revision": revision,
        "git_worktree_dirty": os.environ.get("MLXKIT_GIT_DIRTY", "false") == "true",
        "upstream": {
            "repository": "https://github.com/liux2/RNN-on-wikitext2",
            "revision": data_manifest["upstream_revision"],
            "source_file_sha256": data_manifest["source_file_sha256"],
        },
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
            "nltk": data_manifest["nltk_version"],
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "backend": {
            "name": backend,
            "device": str(device),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "mps_current_allocated_memory_bytes": torch.mps.current_allocated_memory() if device.type == "mps" else None,
            "mps_driver_allocated_memory_bytes": torch.mps.driver_allocated_memory() if device.type == "mps" else None,
        },
        "dataset": {
            "source_url": data_manifest["source_url"],
            "upstream_revision": data_manifest["upstream_revision"],
            "source_file_sha256": data_manifest["source_file_sha256"],
            "manifest": str(data_dir / "manifest.json"),
            "prepared_file_sha256": {name: metadata["sha256"] for name, metadata in files.items()},
            "vocabulary_size": vocabulary_size,
            "vocabulary_sha256": data_manifest["vocabulary_sha256"],
            "sequence_length": SEQUENCE_LENGTH,
            "train_sequences": len(train_data),
            "validation_sequences": counts["validation"],
            "test_sequences": counts["test"],
            "initial_validation_tokens": initial_validation_tokens,
            "test_tokens": test_tokens,
            "preprocessing": data_manifest["tokenizer"],
            "vocabulary_policy": data_manifest["vocabulary_policy"],
            "windowing": data_manifest["windowing"],
        },
        "initialization": initialization,
        "configuration": {
            "architecture": "Embedding -> Dropout -> 2-layer tanh RNN (batch_first) -> Linear vocabulary projection",
            "embedding_size": EMBEDDING_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "recurrent_layers": NUM_LAYERS,
            "parameters": parameter_count,
            "embedding_initializer": "shared uniform[-0.1,0.1]",
            "rnn_and_output_initializer": "shared uniform[-1/sqrt(hidden_size),+1/sqrt(hidden_size)]",
            "dropout_probability": DROPOUT,
            "dropout_location": "embedding output only",
            "initial_hidden_state": "zeros per batch; no state carried between windows",
            "loss": "mean cross entropy over next-token targets",
            "optimizer": "SGD",
            "learning_rate": LEARNING_RATE,
            "momentum": 0.0,
            "gradient_clipping": None,
            "batch_size": args.batch_size,
            "sequence_length": SEQUENCE_LENGTH,
            "epochs": args.epochs,
            "seed": args.seed,
            "train_shuffle": "shared SplitMix64 Fisher-Yates permutation per epoch; same window order in both implementations",
            "drop_last": True,
            "max_train_batches": args.max_train_batches,
        },
        "measurements": {
            "initial_validation_loss": initial_validation_loss,
            "initial_validation_perplexity": initial_validation_perplexity,
            "initial_validation_token_accuracy": initial_validation_accuracy,
            "fit_wall_seconds_including_validation": fit_wall_seconds,
            "training_batch_seconds": training_batch_seconds,
            "training_tokens": total_fit_tokens,
            "training_tokens_per_second_batch_time": total_fit_tokens / training_batch_seconds if training_batch_seconds else None,
            "training_tokens_per_second_fit_wall": total_fit_tokens / fit_wall_seconds if fit_wall_seconds else None,
            "training_step_latency_median_seconds_excluding_warmup": median(steady_steps) if steady_steps else None,
            "training_step_latency_p90_seconds_excluding_warmup": percentile(steady_steps, 0.90),
            "warmup_steps_excluded_from_latency": min(args.warmup_steps, len(batch_durations)),
            "validation_loss": epochs[-1]["validation_loss"],
            "validation_perplexity": epochs[-1]["validation_perplexity"],
            "validation_token_accuracy": epochs[-1]["validation_token_accuracy"],
            "test_loss": test_loss,
            "test_perplexity": test_perplexity,
            "test_token_accuracy": test_accuracy,
            "peak_process_resident_memory_method": "resource.getrusage(RUSAGE_SELF).ru_maxrss; macOS bytes",
            "device_synchronization": "torch.mps.synchronize before and after each timed MPS batch",
        },
        "epochs": epochs,
    }
    write_report(output_path, report)
    print(f"Validation perplexity: {report['measurements']['validation_perplexity']:.3f}")
    print(f"Test perplexity: {test_perplexity:.3f}")
    print(f"Training throughput: {report['measurements']['training_tokens_per_second_batch_time']:.1f} tokens/s")
    print(f"PyTorch backend: {backend}")
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"WikiText-2 PyTorch RNN benchmark failed: {error}", file=sys.stderr)
        raise
