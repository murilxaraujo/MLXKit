# Contrastive embedding example

This standalone example trains an encoder on paired views of deterministic local
vectors. It demonstrates a custom contrastive scalar objective with MLXKit's
`TrainingEngine`, paired retrieval evaluation, and embedding generation for new
vectors. It does not download a dataset.

## Requirements

- A supported Apple-silicon Mac running macOS 27 or later.
- The dependencies already resolved by this repository's `Package.resolved`.
- No dataset download, credentials, or network access at run time.

## Run

From the repository root, after generating the examples project:

```sh
make examples-generate
tuist run --path Examples ContrastiveEmbeddingExample
```

For a smaller bounded run suitable for CI smoke checks or quick local checks:

```sh
tuist run --path Examples ContrastiveEmbeddingExample --smoke
```

Use `tuist run --path Examples ContrastiveEmbeddingExample --help` to print the options. The
smoke mode changes fixture size and epoch count, not the model, objective, or
correctness checks.

## Fixture and encoder

- Each input example has two views with shape `[2, 128]`; a batch has shape
  `[batch, 2, 128]`.
- The first 64 input coordinates contain a per-example identity signal shared
  by both views, with small independent perturbations. The remaining 64
  coordinates are independent nuisance values for each view.
- Training uses fixture seed `2026`, validation uses seed `90210`, and the
  separate new-input vector demo uses seed `424242`. Model initialization and
  training shuffle use seed `42`. Training and validation identities are
  disjoint.
- The shared encoder is `128 → 128 → 64` with ReLU, followed by L2
  normalization. It returns one unit-length 64-dimensional vector per input
  view. `encode(_:)` also accepts a batch of unpaired `[batch, 128]` inputs.
- Batch size is 32. The default uses 256 training pairs, 128 validation pairs,
  and 40 epochs. `--smoke` uses 96 training pairs, 64 validation pairs, and 30
  epochs.

## Objective and metric

The example uses symmetric in-batch NT-Xent. For each of the `2B` normalized
view embeddings, the other view of the same pair is the positive. Every other
view in that batch is a negative. The anchor itself is excluded from the
denominator. With cosine similarity `s` and temperature `τ = 0.1`, each anchor
contributes:

```text
-log( exp(s(anchor, positive) / τ) /
      sum_{candidate != anchor} exp(s(anchor, candidate) / τ) )
```

The objective averages this value across all `2B` anchors. The typed MLXKit
`TrainingStep` computes only this scalar objective; autodiff, gradient checks,
optimizer updates, batching, and lifecycle remain in MLXKit.

Validation reports paired Recall@1: for each view-0 embedding, retrieve the
most similar view-1 embedding in its batch and count whether it has the same
pair identity. The example checks finite loss and gradients, verifies that
training changes model parameters, checks that validation and generated vectors
are unit-normalized, and requires both validation loss and paired Recall@1 to
improve over the untrained model. Final Recall@1 must also exceed the
random-retrieval baseline `1/32 = 0.03125` for a full batch. These are checks
for this fixed synthetic task, not claims about general embedding quality.

## Reproducibility and limits

The fixture, split, shuffle, model seed, shapes, and training configuration are
fixed. Repeated runs should produce the same results on the same supported MLX
and device setup. Bitwise equality across hardware, OS, or MLX versions is not
promised. This small synthetic retrieval task does not demonstrate a general
purpose embedding model, external-dataset performance, or production deployment.
