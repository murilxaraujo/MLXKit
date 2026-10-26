# Cora — GCN

**Status:** scaffold only. Dataset source/split, exact graph representation,
reference framework/version, run commands, and results are pending.

## Workload

- **Model:** graph convolutional network for node classification. Fix layer
  count, hidden width, feature normalization, and adjacency handling with the
  reference.
- **Dataset:** Cora; choose the exact commonly used split and preprocessing.
- **Quality:** test node-classification accuracy.
- **Performance:** training nodes/second or graph steps/second, median step
  latency, and peak memory, measured as specified in [`../README.md`](../README.md).

Document whether adjacency and features are sparse or dense in each
implementation. The graph is small; the workload is intended to demonstrate
graph-model implementation, not large-graph throughput.

## Implementations

- [Reference implementation](reference/README.md)
- [MLXKit implementation](mlxkit/README.md)
