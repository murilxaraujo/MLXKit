# Tiny Decoder Example

The `TinyDecoderExample` executable demonstrates one custom decoder-style
model built with public MLX and MLXKit APIs. It uses a fixed local token pattern,
sets the MLX random seed, trains with a custom next-token objective, and
evaluates on held-out rows from the same deterministic fixture. It does not
download data.

Run it with the default configuration:

```sh
swift run TinyDecoderExample
```

The vocabulary size, hidden width, attention head count, maximum sequence
length, and epoch count can be changed from the command line. The width must be
divisible by the head count.

```sh
swift run TinyDecoderExample --vocab 24 --width 48 --heads 4 --sequence-length 10 --epochs 3
```

The model combines token and position embeddings, a causal multi-head
self-attention module, residual connections, a feed-forward MLP, and a tied
token-embedding output projection. Its custom objective computes mean
cross-entropy over next-token logits. The example uses the typed model contract
with `MLXArray` token inputs and logits, builds typed `Batch<MLXArray, MLXArray>`
values, and passes a re-iterable asynchronous sequence to `TrainingEngine`.

The example also supplies a `TrainingStep` that adds a small scalar logit
penalty to the next-token loss. This shows that `TrainingStep` can customize
loss computation for both `fit` and `evaluate`; MLXKit still performs
autodiff and optimizer updates, and MLXOptimizers still owns the optimizer
algorithm. Use direct MLX and MLXOptimizers APIs for a custom update algorithm.

The decoder implementation lives in the example's internal `TinyDecoderCore`
target. Its `TinyDecoder` conforms to `TrainableModel`, and
`NextTokenObjective` conforms to `Objective<MLXArray, MLXArray>`. The batch
sequence pairs each token row with the same row shifted left by one position.
`TrainingEngine` supplies each model output and its corresponding target to the
objective, then owns gradient calculation and optimizer updates. The GPU test
suite checks causal masking, finite gradients, parameter updates, and parity
with a direct MLX update. The parity assertion uses an absolute per-parameter
tolerance of `1e-5` on the fixed test fixture.

Training processes each complete sequence with a causal attention mask. It does
not share mutable inference-cache state across differentiable forward passes.
The example does not add a cache API; its fixed fixture and model are local to
the executable, consistent with <doc:CustomModelState>.

This small workload validates an implementation path and public API composition.
It does not demonstrate large-scale language-model capability, production
training, or performance.
