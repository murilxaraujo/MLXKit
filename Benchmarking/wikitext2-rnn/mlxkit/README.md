# WikiText-2 vanilla RNN with MLXKit

[`Sources/WikiText2RNNBenchmark.swift`](Sources/WikiText2RNNBenchmark.swift)
uses MLXKit's `TrainingEngine`, `DataLoader`, and cross-entropy objective. The
two-layer tanh RNN is unrolled with MLX operations. Its separate input and
recurrent projections preserve PyTorch's `weight_ih`, `weight_hh`, `bias_ih`,
and `bias_hh` parameters.

From the repository root:

```sh
Benchmarking/wikitext2-rnn/scripts/run-mlxkit.sh
```

It reads the token IDs and shared float32 initialization produced by the
workload scripts, uses MLX's default device, and writes JSON, CSV, and log
records to the requested run directory.
