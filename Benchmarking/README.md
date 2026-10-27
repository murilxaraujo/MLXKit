# MLXKit benchmarks

This directory contains paired implementations of standard neural-network
workloads: a reference implementation in PyTorch or TensorFlow/Keras and the
same model trained with MLXKit. The goal is to demonstrate how these
architectures are expressed with MLXKit and to report comparable quality and
performance measurements.

The [MNIST MLP](mnist-mlp/README.md), [MNIST CNN](mnist-cnn/README.md), and
[WikiText-2 RNN](wikitext2-rnn/README.md) have independent reference-framework
and MLXKit runners. Other workloads remain scaffolding while their dataset
sources, exact configurations, and reference-framework versions are selected.
No benchmark result or performance claim is implied by this plan.

## Workload order

The initial order favors small data and simpler implementations. Approximate
dataset memory depends on preprocessing, dtype, and whether inputs are fully
materialized; model activations and optimizer state require additional memory.

| Order | Workload | Approximate dataset footprint | Implementation complexity | Quality metric |
|---:|---|---|---|---|
| 1 | [MNIST MLP](mnist-mlp/README.md) | ~47 MB as `uint8`, ~188 MB as `float32` for 60,000 training images | Low | Test accuracy |
| 2 | [MNIST CNN](mnist-cnn/README.md) | Same MNIST data as above | Low–medium | Test accuracy |
| 3 | [WikiText-2 vanilla RNN](wikitext2-rnn/README.md) | A few to tens of MB after tokenization | Medium | Validation perplexity |
| 4 | [WikiText-2 LSTM](wikitext2-lstm/README.md) | Same WikiText-2 corpus as above | Medium | Validation perplexity |
| 5 | [CIFAR-10 ResNet-18](cifar10-resnet18/README.md) | ~154 MB as `uint8`, ~614 MB as `float32` for 50,000 training images | Medium | Test accuracy |
| 6 | [WikiText-2 Transformer LM](wikitext2-transformer-lm/README.md) | Same WikiText-2 corpus as above | Medium–high | Validation perplexity |
| 7 | [Cora GCN](cora-gcn/README.md) | Small: typically tens of MB or less | High if sparse graph operations need custom handling | Test accuracy |
| 8 | [IWSLT14 Transformer](iwslt14-transformer/README.md) | Usually tens of MB after tokenization; preprocessing-dependent | High | BLEU (and validation loss) |

The estimates are planning guides, not measured memory requirements. In
particular, Cora is tiny as a dataset but may be complex to implement because
of graph and sparse-operation requirements. A small Transformer configuration
demonstrates the architecture; it is not a large-scale language-model claim.

## Layout

Each workload directory contains:

- `reference/` — the selected PyTorch or TensorFlow/Keras implementation.
- `mlxkit/` — the equivalent MLXKit implementation.
- `README.md` — workload definition, setup, execution, metrics, and result notes.

The README files inside the implementation directories record the exact source,
versions, commands, and any deviations once those choices are made. Keep the
two implementations separate so their dependencies and setup instructions stay
clear.

## Comparison protocol

Before publishing a comparison, record and align the dataset split and
preprocessing, architecture and parameter count, initialization, optimizer,
batching, number of training examples/tokens, and evaluation procedure. Record
any unavoidable differences. Run both implementations on the same machine when
possible, and identify each framework's actual device/backend. A PyTorch or
TensorFlow comparison measures the complete stacks; it does not isolate MLXKit
overhead or establish a universal framework ranking.

For each run, report:

| Measurement | Suggested definition |
|---|---|
| Quality | Workload metric on the documented held-out split: accuracy, perplexity, or BLEU as applicable. Include validation loss where useful. |
| Throughput | Examples/images per second, tokens per second, or another workload-appropriate unit. State whether data loading is included. |
| Step latency | Median steady-state training-step time in milliseconds; state batch/token count and what the timer includes. |
| Peak memory | Peak process or framework memory in bytes/GB, with the measurement method. Compare only like-for-like measurements. |

Use the same warm-up and repeated-measurement protocol for both implementations.
Report medians and spread, not just the fastest run. Ensure lazy/eager ML work is
completed before stopping timers. Keep setup and first-use costs separate from
steady-state measurements. Do not compare memory counters from different
frameworks as if they were equivalent without documenting their definitions.

For metrics where lower is better (loss, perplexity, latency, memory), report
`(MLXKit / reference - 1) × 100%`; for metrics where higher is better
(accuracy, BLEU, throughput), report `(MLXKit / reference - 1) × 100%`. Also
include the raw values and the absolute quality-metric difference so the
percentage does not hide the scale of the quality change.

## Run status

The MNIST MLP, CNN, and WikiText-2 RNN commands and setup are documented in
their workload READMEs. The other workloads do not have runnable benchmark
commands yet. Performance runs are intended for a local Apple-silicon machine,
not routine CI. Keep bounded correctness smoke checks separate from full timed
benchmark results.

## Results record template

Copy this record for each implementation and run; publish paired records only
after confirming that the configurations are comparable.

```text
Workload / configuration:
Dataset, split, preprocessing, and source:
Reference framework / version / source revision:
MLXKit revision / MLX Swift version:
Machine / macOS / memory / device backend:
Model, dtype, batch or token count, optimizer, seed:
Training/evaluation protocol:
Quality metric and value:
Throughput and unit:
Median step latency and spread:
Peak-memory value and measurement method:
Warm-up, repeats, timed region, and data-loading policy:
Known deviations / notes:
```
