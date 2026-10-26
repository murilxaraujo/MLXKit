# Supervised classifier example

This example demonstrates a complete local classification workflow with MLXKit:
define a model, build fixed train/validation splits, batch with `DataLoader`,
train with cross-entropy and accuracy, evaluate, predict, then save and reload
an inference artifact.

## Requirements

- A supported Apple-silicon Mac running macOS 27 or later.
- The dependencies already resolved by this repository's `Package.resolved`.
- No dataset download, credentials, or network access at run time.

## Run

From the repository root, after generating the examples project:

```sh
make examples-generate
tuist run --path Examples MNISTExample
```

For a smaller bounded run suitable for a CI smoke check or quick local check:

```sh
tuist run --path Examples MNISTExample --smoke
```

Both commands use the same model, data-generation method, validation checks,
and artifact round-trip verification. The smoke option reduces the number of
examples and epochs. `tuist run --path Examples MNISTExample --help` prints the
available options.

## Fixture and model

- Inputs are synthetic flattened vectors with shape `[batch, 784]`; they are not
  MNIST images and do not come from an external dataset.
- There are 10 balanced classes. Each class has eight strong class-specific
  feature coordinates and four small deterministic distractor values.
- Training fixture seed: `2026`. Validation fixture seed: `90210`. The full
  shuffle seed and model-initialization seed are both `42`.
- The model is `784 → 32 → 10`, with ReLU between the two dense layers.
- The default run uses 40 training examples and 12 validation examples per
  class for 12 epochs. `--smoke` uses 24 training examples and 8 validation
  examples per class for 8 epochs. Batch size is 32.
- The inference artifact stores model weights and caller-owned metadata for the
  architecture, input dimensions, class count, label mapping, and fixture.
  Model construction and preprocessing remain in the example code.

## Expected result

The executable prints per-epoch training and validation metrics. It exits with
an error if any reported metric or prediction is non-finite, if final validation
loss does not improve over the untrained model, if validation accuracy does not
exceed 50% on this deliberately learnable fixture, or if reloaded predictions
differ from the original predictions beyond `rtol=1e-5, atol=1e-5`.

These checks are fixture-specific correctness signals, not general accuracy
targets. Fixed data, seeds, and model settings make runs repeatable on the same
supported MLX/device setup; bitwise-identical training across macOS, MLX, or
hardware versions is not promised.

## Limits

This is a small synthetic supervised-learning example. It does not demonstrate
MNIST ingestion, image preprocessing, model selection, production deployment,
benchmark performance, or generalization to real-world data.
