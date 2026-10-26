# MNIST — MLP benchmark

This workload compares the full TensorFlow/Keras and MLXKit training stacks on
the same MNIST data and model. It follows the supplied `MLP with MNIST.ipynb`,
with explicit settings for reproducibility. It does not measure isolated
MLXKit overhead or claim a universal framework ranking.

## Workload

- MNIST images: 28×28 grayscale, flattened to 784 `float32` inputs and scaled
  to `[0, 1]` by dividing by 255.
- Model: `784 → Dense(256, ReLU) → Dense(256, ReLU) → Dense(10)`.
  Biases are enabled; the model has 269,322 trainable parameters.
- Loss: sparse categorical cross-entropy from logits.
- Optimizer: Adam, learning rate `0.001`, betas `(0.9, 0.999)`, epsilon
  `1e-7`, bias correction enabled in MLX.
- Batch size: 32. The notebook's prose mentions a batch of 1000, but its call
  to `model.fit` omits `batch_size`, so Keras actually uses its default of 32.
- Epochs: 10. Validation uses the last 12,000 samples of the original 60,000
  training set, matching Keras `validation_split=0.2`; the first 48,000 are
  used for training. The standard 10,000-image test set is held out.
- Seed: 42 by default. A shared preparation script creates identical seeded
  Glorot-uniform kernel bytes and zero biases for both implementations. The
  framework-specific per-epoch shuffle orders can still differ.

The TensorFlow implementation retains the notebook's softmax output and
standard sparse categorical cross-entropy. MLXKit trains on logits with
cross-entropy-from-logits; these formulations are mathematically equivalent and
have the same layer dimensions, parameters, and predicted classes.

## Requirements

- Apple-silicon Mac, macOS 27 or later, Xcode/Swift 6.4 or later.
- Conda (Miniforge or Miniconda) for the Python reference.
- Python environment defined in [`reference/environment.yml`](reference/environment.yml).
- Network access on first run to download the Keras MNIST archive and resolve
  Conda/Swift dependencies.

TensorFlow Metal availability depends on the compatible TensorFlow/plugin
combination. The reference reports discovered devices and whether the Metal
plugin is installed; do not assume it used the GPU. The MLX implementation uses
the MLX backend available on the Mac.

## Prepare the reference environment

From the repository root:

```sh
brew install --cask miniforge # if Conda is not already installed
conda env create -f Benchmarking/mnist-mlp/reference/environment.yml
```

After installing Miniforge, run `conda init zsh` for a zsh shell and open a new
terminal if the `conda` command is not yet on `PATH`.

The environment is named `mlxkit-mnist-mlp`. Override the name with
`MNIST_CONDA_ENV` if required. `run-reference.sh` does not create or delete
Conda environments automatically.

## Run

Run either implementation independently from the repository root:

```sh
Benchmarking/mnist-mlp/scripts/run-reference.sh
Benchmarking/mnist-mlp/scripts/run-mlxkit.sh
```

To pair results, set the same run ID for both commands:

```sh
RUN_ID=mnist-pair-01 Benchmarking/mnist-mlp/scripts/run-reference.sh
RUN_ID=mnist-pair-01 Benchmarking/mnist-mlp/scripts/run-mlxkit.sh
```

Useful overrides are `EPOCHS`, `SEED`, `BATCH_SIZE`, `WARMUP_STEPS`,
`MNIST_DATA_DIR`, and `MNIST_RESULTS_DIR`. For example, set `EPOCHS=1` for a
quick execution check; do not report that run as a benchmark result. Run scripts
accept `--help` for details.

On the first run, either implementation prepares the shared dataset cache. The
preparation step downloads Keras' `mnist.npz` and writes raw `uint8` image and
label files so both languages consume exactly the same bytes. Data download and
preparation are outside the training timer.

The MLX runner builds the dedicated `MNISTMLPBenchmark` SwiftPM target, whose
source lives in `mlxkit/`; it does not reuse the synthetic-data
`Examples/SupervisedClassifier` executable. Its build scratch directory is
isolated under this benchmark so cleanup does not remove the repository's
shared Swift build.

## Output and measurements

Each run writes a JSON report and per-epoch CSV under
`Benchmarking/mnist-mlp/results/<run-id>/`, plus a captured console log. Reports
include the configuration, dataset manifest/checksums, machine and software
versions, discovered backend/devices, per-epoch train/validation loss and
accuracy, final test loss and accuracy, training wall time, batch-time-based
throughput, median/p90 steady-state batch latency, and peak process resident
memory.

Batch latency excludes the first `WARMUP_STEPS` training batches (10 by
default). Throughput uses total training examples divided by the sum of all
training-batch intervals. Training wall time includes epoch validation and
training-engine overhead; data download/preparation and model construction are
excluded. Peak memory is the process maximum resident set size as reported by
the operating system; it includes runtime/framework allocations and is not a
dedicated GPU-memory counter. Compare raw values only with these definitions in
mind.

The notebook's recorded 97.56% test accuracy is historical output, not a
benchmark acceptance threshold. Record raw paired values and known backend or
configuration differences before drawing conclusions.

The TensorFlow/Keras versions are pinned to TensorFlow 2.16.2, Keras 3.3.3, and
TensorFlow Metal 1.1.0. The TensorFlow and MLX Adam implementations use the
configured learning rate, betas, epsilon, and bias correction. Their numerical
implementations are not guaranteed to be bit-identical; report the framework
versions and treat tiny optimizer-level differences as part of the full-stack
comparison.

## Cleanup

```sh
Benchmarking/mnist-mlp/scripts/cleanup.sh
```

By default, cleanup removes only generated result files and the local Swift
build directory. Add `--data` to also remove the downloaded/prepared MNIST
cache. The Conda environment is never removed by this script.

## Implementation notes

- `reference/` contains the TensorFlow/Keras implementation and pinned Conda
  environment.
- `mlxkit/` contains the Swift executable and its SwiftPM manifest.
- `scripts/` contains the shared dataset preparation, independent runners, and
  cleanup command.
- The prepared data and benchmark results are ignored by Git.
