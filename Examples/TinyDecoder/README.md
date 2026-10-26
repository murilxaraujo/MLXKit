# Tiny causal decoder example

This Tuist command-line executable trains a small decoder-only language model on a
fixed local token fixture, evaluates held-out rows, greedily generates a short
continuation, and round-trips an inference artifact. It is a runnable
implementation study, not a claim of general language capability or a faithful
reproduction of the original Transformer experiments.

## Requirements and commands

- Supported baseline: Apple silicon, macOS 27+, and the dependencies resolved
  by the root package and Tuist project.
- Training and inference use generated local integer-token rows. No corpus,
  tokenizer files, credentials, or dataset downloads are required.
- Resolve/fetch dependencies with `make examples-generate`, then run from the
  repository root:

```sh
tuist run --path Examples TinyDecoder
tuist run --path Examples TinyDecoder --smoke
```

The default configuration is vocabulary 256, width 128, four attention heads,
two decoder blocks, sequence length 64, batch size four, and 12 epochs. The
input batch therefore has shape `[4, 64]`, as in the reference workload.
`--smoke` runs the same checks with vocabulary 32, width 16, two heads, two
blocks, sequence length 16, 16 training rows, four held-out rows, and 12 epochs.
Both modes use Adam and fixed local rows. Flags `--vocab`, `--width`, `--heads`,
`--sequence-length`, and `--epochs` configure the default-sized run; `--help`
lists them.

## Model and paper component

The model has learned token embeddings, learned absolute position embeddings,
two pre-normalized decoder blocks, and a vocabulary projection tied to the
token-embedding weights. Each block has a residual multi-head self-attention
layer and a residual four-times-width GELU feed-forward network. Attention is
implemented in `TinyDecoder/Core/TinyDecoder.swift` with public MLX
operations:

```text
Q, K, V = split_heads(Linear(LayerNorm(hidden)))
scores = Q · transpose(K) / sqrt(head_width)
weights = softmax(scores + causal_mask)
attention = merge_heads(weights · V)
```

The causal mask adds zero to key positions at or before each query and `-1e9`
to future positions. The attention projection matrices are bias-free. The
embedding and linear layers use MLXNN's default initializers; `seed(42)` fixes
the example's model initialization. The input is an integer ID tensor of
shape `[batch, sequence]`; model output is `[batch, sequence, vocabulary]`.

Paper reference: A. Vaswani et al., “Attention Is All You Need,” NeurIPS 2017,
arXiv:1706.03762, <https://arxiv.org/abs/1706.03762>. This example implements a
paper-inspired masked scaled dot-product self-attention component from scratch
using MLX array operations and public MLXKit training APIs. It does not copy
paper code or claim a paper reproduction.

The example code is covered by this repository's MIT license. It reuses no
paper implementation code, external assets, or dataset.

## Fixture, objective, and evaluation

The local fixture has no tokenizer or text corpus. Each row starts at
`(rowIndex * 73 + splitOffset) mod vocabularySize` and advances the integer
token by one modulo the vocabulary at every position. Each fixture row has
`sequenceLength + 1` tokens. The batcher uses the first `sequenceLength` tokens
as input and the one-position-shifted sequence as labels. Training and held-out
rows use distinct start offsets and are iterated in a fixed order.

`NextTokenObjective` applies mean cross-entropy to each position's vocabulary
logits and shifted integer target. Training updates model parameters through
`TrainingEngine`. The full run uses 32 training rows, eight held-out rows, batch
size four, Adam learning rate `0.003`, 12 epochs, and seed `42`. Smoke uses 16
training rows and four held-out rows, batch size four, Adam learning rate
`0.01`, 12 epochs, and the same seed. There is no row shuffle. The smoke run
checks finite objective/gradients, a nonzero gradient, changed parameters,
expected logits shape, and lower final training and held-out losses than their
initial values. It also performs greedy
autoregressive decoding, preserving the prompt and constraining generated IDs
to the vocabulary. Each next token is selected from the final position's logits
and the model recomputes the full prefix; there is no KV cache.

The held-out loss and deterministic next-token pattern verify only that this
small fixture exercises the training and inference path. They do not establish
language understanding, natural-text quality, benchmark performance, or
generalization to a corpus.

On the documented local macOS 27 / Apple-silicon setup, the full fixture
reported training loss `6.0772 → 0.0337` and held-out loss `6.1132 → 0.2704`.
The smoke fixture reported training loss `4.1507 → 0.0019` and held-out loss
`4.1219 → 0.0044`. These are deterministic fixture observations, not
cross-device bitwise guarantees or paper-level results.

## Inference artifact

After training, the example writes an M6 inference artifact in a temporary
directory. Caller-owned metadata records the architecture version, dimensions,
integer token mapping, learned positional-embedding choice, fixture, seed, and
paper reference. The example explicitly constructs a new model, loads the
weights, verifies metadata equality, and compares logits within `rtol=1e-5,
atol=1e-5`. Model code and token mapping remain in the caller; the artifact does
not serialize executable architecture or a tokenizer. The temporary artifact
is removed after the check.

This is an inference artifact, not a resumable training checkpoint. The example
uses MLXOptimizers' Adam but does not save or restore its optimizer state.

## Deviations and limitations

- The original Transformer encoder-decoder used six layers per stack, width
  512, eight heads, sinusoidal positions, post-normalization, dropout, and
  translation data/objectives. This example uses only a decoder, two blocks,
  configurable smaller dimensions (128 width/four heads by default), learned
  absolute positions, pre-normalization, no dropout, tied token/output weights,
  and synthetic integer cycles.
- The model processes the complete training sequence with a causal mask. Greedy
  inference recomputes the full prefix and has no incremental cache.
- No real tokenizer, text data, sampling strategy, optimizer-state recovery,
  performance measurement, or broad language-model capability is included.

See the [paper implementation template](../PaperImplementation/README.md) for a
repeatable project scaffold and reproduction checklist. See the [examples
index](../README.md) for the other runnable MLXKit examples and smoke commands.
