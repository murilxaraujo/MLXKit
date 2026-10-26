# MNIST MLP — MLXKit implementation

The executable is [`Sources/MNISTMLPBenchmark.swift`](Sources/MNISTMLPBenchmark.swift).
It is the `MNISTMLPBenchmark` target in the repository's root `Package.swift`
and uses the pinned MLX Swift 0.32.3 dependency. It uses MLXKit's
`TrainingEngine`, `DataLoader`, sparse `CrossEntropyLoss`, and accuracy metric.

From the repository root:

```sh
Benchmarking/mnist-mlp/scripts/run-mlxkit.sh
```

The runner reads the same prepared `uint8` dataset files as the TensorFlow
reference and the same seeded Glorot kernels and zero biases. It reports the
MLX default device, MLX active-memory peak, process resident-memory peak,
per-epoch quality, batch latency, throughput, and test metrics. Its SwiftPM
scratch directory is isolated at `Benchmarking/mnist-mlp/.build`.
