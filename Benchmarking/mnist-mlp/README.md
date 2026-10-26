# MNIST — MLP

**Status:** scaffold only. Dataset setup, exact architecture, reference
framework/version, run commands, and results are pending.

## Workload

- **Model:** fully connected multilayer perceptron for 10-class digit
  classification. Fix hidden widths and activation after selecting the
  reference implementation.
- **Dataset:** MNIST; source and preprocessing pending.
- **Quality:** test accuracy, with validation loss as an optional companion.
- **Performance:** training examples/second, median step latency, and peak
  memory, measured as specified in [`../README.md`](../README.md).

Keep preprocessing, initialization, optimizer, batch size, and training budget
aligned between the reference and MLXKit implementations.

## Implementations

- [Reference implementation](reference/README.md)
- [MLXKit implementation](mlxkit/README.md)
