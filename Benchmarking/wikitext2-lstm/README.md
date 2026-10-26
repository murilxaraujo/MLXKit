# WikiText-2 — LSTM language model

**Status:** scaffold only. Tokenizer, dataset setup, exact model configuration,
reference framework/version, run commands, and results are pending.

## Workload

- **Model:** recurrent next-token language model using an LSTM. Fix embedding
  size, hidden size, layer count, and tying policy with the reference.
- **Dataset:** WikiText-2; exact source, split, and tokenization pending.
- **Quality:** validation perplexity (and validation loss).
- **Performance:** training tokens/second, median step latency, and peak
  memory, measured as specified in [`../README.md`](../README.md).

Align tokenizer/vocabulary, sequence length, padding or batching policy,
initialization, optimizer, and training token budget.

## Implementations

- [Reference implementation](reference/README.md)
- [MLXKit implementation](mlxkit/README.md)
