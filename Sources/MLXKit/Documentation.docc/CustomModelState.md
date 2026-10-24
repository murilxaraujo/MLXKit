# Custom Model State and Inference Caches

This guide defines the state boundary for custom models built with MLX and
trained through MLXKit. It does not add a general-purpose state or cache API.

## Three kinds of state

### Trainable parameters

Store learned values as parameters of an MLX `Module`. MLX discovers these
parameters, differentiates them, and applies optimizer updates. MLXKit does not
maintain a second parameter registry. The parameters included by MLXKit's
weight-saving APIs are the module's trainable parameters.

### Temporary forward values

Activations, attention scores, masks, and other values computed for a forward
pass are temporary computation values. Keep them local to that computation
unless the model has a clear reason to retain them. They are not trainable
parameters and are not part of MLXKit's weight checkpoints.

### Inference cache

An autoregressive model may keep derived values, such as keys and values from
earlier tokens, to avoid repeating work during generation. This cache is
inference-only state. It is not a trainable parameter and is not included in
MLXKit weight or training checkpoints.

The caller owns the cache and its lifetime. Scope it to one generation request
and one model configuration. Discard or reset it before a new request and after
loading or changing model weights. Also reset it if another change makes its
contents incompatible with the model or request, such as changing the model
configuration, restarting from a different input prefix, or switching
tokenizers. Appending generated tokens during the same request is the normal
cache-use path.

## Training uses the full sequence

For causal language-model training, pass the complete input sequence through
the model and apply a causal attention mask. Each position can attend only to
itself and earlier positions. The model computes the sequence outputs in one
differentiable forward pass; it does not reuse a mutable inference cache across
training examples or batches.

This keeps the gradient path explicit and avoids carrying request-specific
mutable state between optimizer updates. With M2, the caller supplies an
asynchronous sequence of typed `Batch` values to `TrainingEngine`; each batch
contains the sequence data needed for its full-sequence training step. See
<doc:M2TrainingSemantics> for the engine's objective, mode, and batch behavior.

## Checkpoint boundary

MLXKit's current weight-saving API writes module trainable parameters. Its full
training checkpoint also records epoch and step metadata. Neither API captures
an application-owned inference cache or serializes arbitrary temporary forward
state. After loading weights, start generation with an empty cache.

Do not treat cache contents as model weights or as resumable training state.
This guide does not define cache serialization, a cache ownership protocol, or
future checkpoint formats; checkpoint and resume behavior beyond the current
API remains a separate milestone.

## Scope

Custom architectures, objectives, and inference loops can use public MLX and
Swift APIs. MLXKit does not provide a generic cache abstraction, cache reset
mechanism, or cache serialization API. Any inference cache stays local to the
application or example that creates it.
