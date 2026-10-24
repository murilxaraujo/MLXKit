# Implementing a paper component with MLXKit

This guide and [`PROJECT_TEMPLATE.md`](PROJECT_TEMPLATE.md) provide a reviewable
structure for a paper-inspired MLXKit example. Copy the template into a new
directory under `Examples/`, then adapt it to one specific component or result.
The intent is a runnable, scoped implementation study—not an automated
reproduction platform and not a guarantee that a paper's claims have been
reproduced.

## Start from the root package

Keep examples in the existing Swift package unless a real dependency or release
boundary requires another package. Add an executable target/product to the root
`Package.swift` only when the example needs a command-line executable. Use the
existing MLXKit and MLX package products. Do not add a nested package graph just
to make the example look independent.

Implement model-specific layers and differentiable operations with public MLX
APIs. Use MLXKit's public `TrainableModel`, `Objective`, typed `Batch`,
`TrainingConfiguration`, and `TrainingEngine` contracts where applicable.
Keep architecture, preprocessing, vocabularies, and artifact reconstruction in
the example. Do not edit MLXKit internals to make a single paper example work.

## Required reproduction record

The project template asks the author to record:

1. The paper citation, version, source URL, license/copyright notes, and the
exact component or result in scope.
2. Every input, intermediate, and output tensor shape, including batch/time/
channel axes and dtype where it affects behavior.
3. Architecture, positional/attention/masking choices, layer ordering, and
initialization. Distinguish paper settings from MLX defaults.
4. The objective formula and reduction, target alignment, optimizer, learning
rate, batch size, update count, seed, and other training controls.
5. Fixture provenance and generation, split construction, evaluation method,
deterministic correctness checks, and any expected local result.
6. Artifact contents and caller-owned metadata, reconstruction steps, and
whether optimizer state is or is not recoverable.
7. Reproduction commands, supported hardware/software, observed results,
deviations, limitations, and unverified claims.

Prefer deterministic local fixtures for correctness and CI. A downloaded
dataset may be an optional extension, but it must not be required for build,
smoke, or correctness paths. Avoid universal accuracy, quality, or speed claims
from tiny fixtures. State what each check demonstrates and what it does not.

## Suggested verification sequence

From the repository root, replace the placeholders with the new product and
test names:

```sh
swift build --product <ExampleProduct>
swift run <ExampleProduct> --smoke
swift test --filter <RelevantTestName>
```

If the test requires Apple silicon/Metal, mark it explicitly and include the
relevant command in the hardware-gated validation record, such as `make test-gpu`.
The smoke mode should be deterministic, bounded, and use local data.
It should check more than process startup when the workload has meaningful
correctness invariants.

## Validation against the M7 decoder

The [Tiny causal decoder](../TinyDecoder/README.md) applies this structure to
masked scaled dot-product self-attention from *Attention Is All You Need*. It
records tensor shapes, the causal-mask formula, seed and MLXNN initialization
boundary, next-token objective, token fixture, held-out loss, inference artifact
metadata, run commands, deviations, and scale limits. Its generated integer
cycle is deliberately synthetic; its learning signal validates the small
training/inference path and does not validate paper-level translation or
language claims.

The decoder is not a claim that the original paper was reproduced. It uses a
smaller two-block decoder-only architecture and differs in positional encoding,
normalization placement, training data, and other settings. The complete list
is recorded in its README. The example and this template are written as
Markdown, so they are linked from the examples index and are not standalone
DocC articles.
