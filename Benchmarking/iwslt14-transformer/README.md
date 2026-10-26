# IWSLT14 — Transformer translation

**Status:** scaffold only. Dataset variant, language direction, tokenization,
exact model configuration, reference framework/version, run commands, and
results are pending.

## Workload

- **Model:** small encoder-decoder Transformer for machine translation.
- **Dataset:** IWSLT14; select and record the exact language pair, dataset
  preprocessing, and train/validation/test split.
- **Quality:** BLEU on the documented test set; validation loss may also be
  reported.
- **Performance:** training tokens/second, median step latency, and peak
  memory, measured as specified in [`../README.md`](../README.md).

Align tokenization, vocabulary, sequence lengths, padding masks, decoding and
BLEU settings, initialization, optimizer, and training token budget.

## Implementations

- [Reference implementation](reference/README.md)
- [MLXKit implementation](mlxkit/README.md)
