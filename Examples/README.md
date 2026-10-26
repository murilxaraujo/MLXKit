# MLXKit examples

These command-line examples are a standalone Tuist project. They import MLXKit
from the root Swift package and use [Swift Argument Parser](https://github.com/apple/swift-argument-parser)
for their command-line interfaces. The root package contains the MLXKit
abstractions and their tests; example implementations and dependencies live
here. Run commands from the repository root.

Install and generate the examples project once:

```sh
make examples-generate
```

The examples support the documented macOS 27+ Apple-silicon baseline. Their
correctness and smoke paths use fixed local or generated fixtures. They do not
download datasets. The first Tuist install may need network access to resolve or
fetch package dependencies; example execution does not need network access.

## Core examples

| Project | Purpose and capability boundary | Full run | Bounded smoke / quick run | Data and expected result |
|---|---|---|---|---|
| [Supervised classifier](SupervisedClassifier/README.md) (`MNISTExample`) | End-to-end dense classification: train, evaluate, predict, and save/reload an inference artifact. Synthetic 784-value vectors are not MNIST; this is not an image pipeline or a generalization benchmark. | `tuist run --path Examples MNISTExample` | `tuist run --path Examples MNISTExample --smoke` | Deterministic local 10-class fixture. Checks finite metrics/predictions, lower validation loss, a fixture-specific accuracy floor, and prediction parity after artifact reload. |
| [Contrastive embedding](ContrastiveEmbedding/README.md) (`ContrastiveEmbeddingExample`) | Paired-view NT-Xent training, paired Recall@1, normalized embeddings, and generation for new vectors. The small synthetic retrieval fixture is not a general-purpose embedding evaluation. | `tuist run --path Examples ContrastiveEmbeddingExample` | `tuist run --path Examples ContrastiveEmbeddingExample --smoke` | Deterministic local paired feature vectors. Checks finite/nonzero gradients, parameter updates, normalized output, and improved validation loss and Recall@1. |
| [Tiny causal decoder](TinyDecoder/README.md) (`TinyDecoderExample`) | Two-block decoder with from-scratch causal attention, next-token training, held-out evaluation, greedy generation, and inference-artifact verification. It is not an LLM or language benchmark. | `tuist run --path Examples TinyDecoder` | `tuist run --path Examples TinyDecoder --smoke` | Fixed local integer-token rows. Checks finite/nonzero gradients, parameter update, logits shape, train/held-out loss improvement, generation, and artifact reload. |

## Data pipeline example

`DataPipelineExample` demonstrates indexed in-memory data and a generated
unknown-length stream with bounded shuffle and prefetch. Run it with
`make run-data-pipeline`; this command resolves dependencies and generates the
Tuist project before launch.

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
| Package build | `swift build` — **passed** | The MLXKit library product. |
| Unit tests | `swift test` — **passed, 92 tests** | Platform-independent test suite. |
| Example smoke runs | `make smoke-examples` — **passed** | Classifier, embedding, and decoder; fixed local data, bounded workloads; included in macOS 27 Apple-silicon CI. |
| Library GPU tests | `make test-gpu` — **passed, 186 tests** | MLXKit GPU-dependent unit tests. |
| Example GPU tests | `make test-examples-gpu` — **passed, 3 tests** | Decoder causal masking, output shape, gradients, and optimizer update parity. |
| DocC generation | `make docs` — **passed** | MLXKit API documentation; example/template guides are Markdown, not DocC articles. |
| Full example runs | Prior pre-extraction full-run record — **passed** | Synthetic classifier validation accuracy 100%; embedding paired Recall@1 0.9141; decoder training loss 0.0337 and held-out loss 0.2704. These are fixture checks, not general quality claims. |
