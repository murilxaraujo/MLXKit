# MLXKit examples

These projects are runnable examples built as executable products of the root
Swift package. They use public MLXKit and MLX APIs; they do not add a nested
package or a second dependency graph. Resolve the dependencies in
`Package.resolved`, then run commands from the repository root.

All three examples support the documented macOS 27+ Apple-silicon baseline.
Their correctness and smoke paths use fixed local or generated fixtures. They
do not download datasets. The first SwiftPM build may need network access to
resolve or fetch package dependencies; example execution does not need network
access.

## Core examples

| Project | Purpose and capability boundary | Full run | Bounded smoke / quick run | Data and expected result |
|---|---|---|---|---|
| [Supervised classifier](SupervisedClassifier/README.md) (`MNISTExample`) | End-to-end dense classification: train, evaluate, predict, and save/reload an inference artifact. Synthetic 784-value vectors are not MNIST; this is not an image pipeline or a generalization benchmark. | `swift run MNISTExample` | `swift run MNISTExample --smoke` | Deterministic local 10-class fixture. Checks finite metrics/predictions, lower validation loss, a fixture-specific accuracy floor, and prediction parity after artifact reload. |
| [Contrastive embedding](ContrastiveEmbedding/README.md) (`ContrastiveEmbeddingExample`) | Paired-view NT-Xent training, paired Recall@1, normalized embeddings, and generation for new vectors. The small synthetic retrieval fixture is not a general-purpose embedding evaluation. | `swift run ContrastiveEmbeddingExample` | `swift run ContrastiveEmbeddingExample --smoke` | Deterministic local paired feature vectors. Checks finite/nonzero gradients, parameter updates, normalized output, and improved validation loss and Recall@1. |
| [Tiny causal decoder](TinyDecoder/README.md) (`TinyDecoderExample`) | Two-block decoder with from-scratch causal attention, next-token training, held-out evaluation, greedy generation, and inference-artifact verification. It is not an LLM or language benchmark. | `swift run TinyDecoderExample` | `swift run TinyDecoderExample --smoke` | Fixed local integer-token rows. Checks finite/nonzero gradients, parameter update, logits shape, train/held-out loss improvement, generation, and artifact reload. |

To run all bounded M7 examples locally:

```sh
make smoke-examples
```

The macOS 27 Apple-silicon CI job runs the same three commands after unit tests.
See [CI](../.github/workflows/ci.yml) for the complete build, test, smoke, and
GPU-test sequence.

## Paper implementation template

Copy the [paper implementation project template](PaperImplementation/PROJECT_TEMPLATE.md)
and follow its [reproduction guide](PaperImplementation/README.md). The
[TinyDecoder project](TinyDecoder/README.md) demonstrates how to fill the
template for a paper-inspired masked-attention component and records its
citation, shapes, initialization boundary, deviations, and fixture limits.

## Validation record

Validation was run on macOS 27.0.1, Apple silicon (`arm64`), Xcode 27.0, and Swift 6.4. All checks below passed locally; no hardware-gated checks were skipped. The hosted GitHub Actions job has not been run from this workspace. No dataset downloads are used.

| Check | Command | Coverage |
|---|---|---|
| Package build | `swift build` — **passed** | All library and executable products. |
| Unit tests | `swift test` — **passed, 92 tests** | Platform-independent test suite. |
| Example smoke runs | `make smoke-examples` — **passed** | Classifier, embedding, and decoder; fixed local data, bounded workloads; included in macOS 27 Apple-silicon CI. |
| GPU tests | `make test-gpu` — **passed, 189 tests** | Includes causal masking, decoder output shape, finite gradients, and optimizer update checks. |
| DocC generation | `make docs` — **passed** | MLXKit API documentation; example/template guides are Markdown, not DocC articles. |
| Full example runs | `swift run MNISTExample`; `swift run ContrastiveEmbeddingExample`; `swift run TinyDecoderExample` — **passed** | Synthetic classifier validation accuracy 100%; embedding paired Recall@1 0.9141; decoder training loss 0.0337 and held-out loss 0.2704. These are fixture checks, not general quality claims. |
