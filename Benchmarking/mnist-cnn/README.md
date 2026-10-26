# MNIST — CNN benchmark

This workload compares the PyTorch and MLXKit CNN training stacks on shared
MNIST bytes, a shared seeded initialization, and the same train/validation/test
split. The model follows the supplied `cnn-with-pytorch-for-mnist.ipynb`. It
compares complete framework stacks; it does not isolate MLXKit overhead or
establish a universal framework ranking.

## Model and protocol

- Input: 28×28 grayscale images, converted to `float32` in `[0, 1]`. PyTorch
  uses NCHW; MLX uses NHWC.
- Architecture: valid `Conv2d(1,32,5) → ReLU → Conv2d(32,32,5) → 2×2
  MaxPool → ReLU → Dropout(0.5) → Conv2d(32,64,5) → 2×2 MaxPool → ReLU →
  Dropout(0.5) → Flatten(576) → Linear(256) → ReLU → Dropout(0.5) → Linear(10)`.
  The first convolution is not pooled, matching the notebook. There are
  228,010 trainable parameters.
- Output/loss: the PyTorch reference retains the notebook's log-softmax output
  followed by `CrossEntropyLoss`; MLXKit uses logits with sparse
  cross-entropy-from-logits.
- Optimizer: Adam with learning rate `0.001`, betas `(0.9, 0.999)`, and
  epsilon `1e-8` (PyTorch's default epsilon).
- Batch size: 32. Epochs: 5. Sample order is preserved (`shuffle=False`), as in
  the notebook. Both frameworks use the same full batches and final partial
  batch.
- Split: first 48,000 of the canonical 60,000 MNIST training images for
  training, last 12,000 for validation, and the canonical 10,000 test images
  for final evaluation.
- Seed: 42 by default. One script creates identical seeded Glorot-uniform
  kernels and zero biases for both implementations. Dropout random streams are
  framework-specific.

### Adaptations from the supplied notebook

The notebook expects a Kaggle `train.csv` file, makes a random 85/15 split, does
not normalize the pixel values, and has no validation set. This benchmark uses
the same canonical MNIST archive as the MLP workload, normalizes pixels by 255,
and uses a fixed validation/test split so runs are repeatable and comparable.
It retains the CNN layer sizes, valid convolutions, pooling/dropout placement,
five-epoch budget, batch size, optimizer defaults, and no-shuffle behavior. The
notebook's implicit PyTorch parameter initialization is replaced with shared
Glorot weights and zero biases to start both implementations from identical
parameters. Its test-accuracy denominator also counts padded batch slots; this
benchmark counts the actual 10,000 test examples.

The shared fully connected weights are stored in PyTorch's `[out, channel,
height, width]` flattened convention. MLX transposes the `fc1` columns to match
its NHWC flatten order; this keeps the initial outputs aligned across layouts.

## Setup and run

Requirements: Apple-silicon Mac, macOS 27 or later, Xcode/Swift 6.4 or later,
and Conda for the PyTorch reference. Install Miniforge with
`brew install --cask miniforge` if Conda is not installed, then from the
repository root run:

```sh
conda env create -f Benchmarking/mnist-cnn/reference/environment.yml
```

The PyTorch runner will use MPS when PyTorch reports it available; otherwise,
it records and uses CPU. The MLX runner records MLX's default device.

Run each implementation separately with a shared run ID to pair reports:

```sh
RUN_ID=mnist-cnn-pair-01 Benchmarking/mnist-cnn/scripts/run-reference.sh
RUN_ID=mnist-cnn-pair-01 Benchmarking/mnist-cnn/scripts/run-mlxkit.sh
```

Both runners accept `--help` and the `EPOCHS`, `SEED`, `BATCH_SIZE`, and
`WARMUP_STEPS` environment overrides. Use `EPOCHS=1` only for a quick execution
check. The shared MNIST cache is prepared under
`Benchmarking/mnist-mlp/data/`; CNN-specific weights and run reports are under
`Benchmarking/mnist-cnn/results/<run-id>/`.

Reports include framework/device versions, dataset and initialization hashes,
configuration, per-epoch train/validation metrics, final test loss/accuracy,
training time, throughput, median/P90 steady-state step latency, and peak
process RSS. PyTorch MPS allocation counters and MLX active-memory counters are
reported separately and are not directly interchangeable. Timed MPS batches
are synchronized before and after each update; dataset loading, device staging,
and model construction are outside the training timer.

```sh
Benchmarking/mnist-cnn/scripts/cleanup.sh
```

Cleanup removes only this workload's results and Swift build directory. Add
`--data` to remove the shared MNIST cache as well; this also affects the MLP
workload. The Conda environment is never removed.

## Implementations

- [`reference/`](reference/README.md): pinned PyTorch reference.
- [`mlxkit/`](mlxkit/README.md): MLXKit executable target.
- `scripts/`: shared-data preparation entry point, CNN initialization, runners,
  and cleanup.
