# WikiText-2 — Transformer language model

**Status:** scaffold only. Tokenizer, dataset setup, exact small-model
configuration, reference framework/version, run commands, and results are
pending.

## Workload

- **Model:** small decoder-only Transformer for next-token prediction. Record
  width, attention heads, layer count, positional encoding, and parameter count.
- **Dataset:** WikiText-2; exact source, split, and tokenization pending.
- **Quality:** validation perplexity (and validation loss).
- **Performance:** training tokens/second, median step latency, and peak
  memory, measured as specified in [`../README.md`](../README.md).

Align tokenizer/vocabulary, sequence length, causal masking, initialization,
optimizer, and training token budget. This is a small-model architecture
benchmark, not a large-scale language-model result.

## Implementations

- [Reference implementation](reference/README.md)
- [MLXKit implementation](mlxkit/README.md)
