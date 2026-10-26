# MNIST CNN — MLXKit implementation

The executable is [`Sources/MNISTCNNBenchmark.swift`](Sources/MNISTCNNBenchmark.swift),
registered as `MNISTCNNBenchmark` in the root `Package.swift`. It uses MLXKit's
`TrainingEngine`, `DataLoader`, sparse `CrossEntropyLoss`, and accuracy metric.

From the repository root:

```sh
Benchmarking/mnist-cnn/scripts/run-mlxkit.sh
```

The model uses MLX's NHWC convolution convention. Shared PyTorch OIHW
initialization files are transposed to MLX OHWI layout on load. It consumes the
same prepared dataset bytes and seeded weights as the reference. The runner
records MLX's default device, MLX active-memory peak, process RSS, epoch metrics,
step latency, throughput, and test metrics.
