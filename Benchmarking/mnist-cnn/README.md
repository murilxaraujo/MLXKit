# MNIST — CNN

**Status:** scaffold only. Dataset setup, exact architecture, reference
framework/version, run commands, and results are pending.

## Workload

- **Model:** small convolutional neural network for 10-class digit
  classification. Fix layer widths and pooling after selecting the reference.
- **Dataset:** MNIST; source and preprocessing pending.
- **Quality:** test accuracy, with validation loss as an optional companion.
- **Performance:** training images/second, median step latency, and peak
  memory, measured as specified in [`../README.md`](../README.md).

Keep image layout and normalization, initialization, optimizer, batch size, and
training budget aligned between implementations.

## Implementations

- [Reference implementation](reference/README.md)
- [MLXKit implementation](mlxkit/README.md)
