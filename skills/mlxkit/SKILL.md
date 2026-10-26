---
name: mlxkit
description: Use MLXKit, the typed Swift training workflow built on MLX Swift, in an application or package. Helps agents integrate models, data pipelines, training, evaluation, callbacks, schedules, and checkpoints.
triggers:
  - MLXKit
  - mlxkit
  - MLXKit training
  - TrainingEngine
  - MLXKit DataLoader
  - TrainableModel
---

# MLXKit Integration

Use this skill when building or changing a Swift project that consumes the
MLXKit package. MLXKit coordinates typed training and evaluation workflows on
Apple silicon. MLX and MLXOptimizers remain responsible for arrays, model
parameters, autodiff, device execution, and optimizer algorithms. Do not treat
MLXKit as a replacement for those APIs.

## Before coding

1. Inspect the consumer project's `Package.swift`, Swift version, deployment
   target, and resolved MLXKit version. Check that version's README and API docs
   before copying an example; signatures and platform requirements can change.
2. Add the MLXKit package and its `MLXKit` product to the target. If source files
   directly import `MLX`, `MLXNN`, or `MLXOptimizers`, declare those products in
   the target as well. See the project's package manager conventions.
3. Use public APIs only. If an API or behavior is unclear, inspect version-matched
   DocC, runnable examples, or public declarations instead of guessing.
4. For MLX array operations, model layers, autodiff, or optimizer-specific
   behavior, use the MLX Swift APIs directly; this skill covers MLXKit's
   orchestration layer.

The current repository README lists macOS 27+, Swift 6.4 / Xcode 27, and Apple
silicon as its supported baseline. Verify the requirements for the version in
the consumer project rather than assuming this baseline applies to every
release.

## Core workflow

Define a model as an `MLXNN.Module` that conforms to `TrainableModel`. Choose an
objective and metrics whose output and target types match the model's output and
batch target. Build a `TrainingConfiguration`, create typed batches, then use
`TrainingEngine`:

```swift
import MLX
import MLXNN
import MLXOptimizers
import MLXKit

final class Classifier: Module, TrainableModel {
    let hidden = Linear(inputDimensions: 784, outputDimensions: 128)
    let output = Linear(inputDimensions: 128, outputDimensions: 10)

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        output(maximum(hidden(input), 0))
    }
}

let model = Classifier()
let configuration = TrainingConfiguration(
    optimizer: Adam(learningRate: 1e-3),
    objective: CrossEntropyLoss(),
    metrics: [Accuracy()]
)

let trainingData = try DataLoader(
    source: InMemoryDataset(features: trainFeatures, labels: trainLabels),
    batchSize: 32,
    shuffle: .full(seed: 42)
)
let validationData = try DataLoader(
    source: InMemoryDataset(features: validationFeatures, labels: validationLabels),
    batchSize: 32
)

let engine = TrainingEngine()
let history = try await engine.fit(
    model,
    configuration: configuration,
    data: trainingData,
    epochs: 10,
    validationData: validationData,
    callbacks: [ProgressReporter(), EarlyStopping(monitor: "val_loss", patience: 3)]
)
let testMetrics = try await engine.evaluate(
    model, configuration: configuration, data: testData
)
let prediction = engine.predict(model, input: newInput)
```

The arrays and `testData` above are application-owned values. Keep their shapes,
types, preprocessing, and label encoding consistent with the model and objective.

## Choosing the data path

- Use `InMemoryDataset` for indexed feature/label arrays.
- Use `DataLoader` with typed `Sample<Input, Target>` values when the data is
  structured or requires custom collation.
- Use an `AsyncSequence<Sample<Input, Target>>` for generated or streaming data;
  provide a `collate` closure that returns `Batch(input:target:sampleCount:)`.
- Select shuffle intentionally: `.full(seed:)` requires an indexed source;
  `.buffered(windowSize:seed:)` supports streams with bounded memory; `.none`
  preserves source order.
- Add `.prefetched(bufferSize:)` only when bounded producer/consumer overlap helps.

`fit` and `evaluate` consume complete `Batch` values through async sequences.
Sequences used for multiple epochs must be re-iterable and create fresh
iterators. A one-shot stream is not replayed for later epochs. The collator must
set `sampleCount` to the number of examples in that batch. Partial final batches
are emitted unless `dropLast` is enabled.

## Contracts that prevent common bugs

- Every configured objective or `trainingStep` must return an `MLXArray` scalar
  with shape `[]`. A one-element vector with shape `[1]` is not a scalar. Use a
  mean-reduced loss or reduce explicitly.
- `TrainingStep` customizes loss computation only. It does not replace
  autodiff, gradient processing, optimizer updates, callbacks, or engine
  lifecycle.
- Evaluation computes objective and metrics but does not train, update
  optimizer state, or advance the learning-rate schedule.
- Metrics reset for each epoch/evaluation. Metric names must be unique and may
  not be `loss` or start with `val_`.
- Gradient transforms run in configured order, then optional global-norm
  clipping runs before the optimizer update. Accumulation combines consecutive
  batches within an epoch; it changes optimizer-update and schedule frequency.
- The default `CrossEntropyLoss.weightedMean` supports logits shaped
  `[batch, classes]` and integer targets shaped `[batch]`. Do not assume it
  handles token-level, structured, or probability targets; implement those
  objectives explicitly.

## Persistence and recovery

- `Checkpointing.saveWeights` / `loadWeights` store and restore model parameters
  only. They do not restore optimizer state or constitute a training resume.
- Inference artifacts add caller-provided string metadata. The application must
  still package/recreate model code, labels, preprocessing, and other assets.
- Faithful epoch-boundary training recovery requires the checkpoint APIs and an
  optimizer conforming to `CheckpointableOptimizer`. Built-in optimizers do not
  automatically satisfy this contract. Verify that the actual optimizer can
  serialize and restore all required state before promising resume support.
- Resumed `fit` uses `epochs` as the total target epoch count and requires
  recreated data sources for the next epoch.

## Where to look next

Prefer docs and examples that match the dependency version in the project.
In this repository, useful starting points are:

- [README / installation and quick start](https://github.com/murilxaraujo/MLXKit/blob/main/README.md)
- [Getting Started](https://github.com/murilxaraujo/MLXKit/blob/main/Sources/MLXKit/Documentation.docc/GettingStarted.md)
- [Data pipeline](https://github.com/murilxaraujo/MLXKit/blob/main/Sources/MLXKit/Documentation.docc/DataPipeline.md)
- [Training semantics](https://github.com/murilxaraujo/MLXKit/blob/main/Sources/MLXKit/Documentation.docc/M2TrainingSemantics.md)
- [Training controls](https://github.com/murilxaraujo/MLXKit/blob/main/Sources/MLXKit/Documentation.docc/M5TrainingControlSemantics.md)
- [Persistence semantics](https://github.com/murilxaraujo/MLXKit/blob/main/Sources/MLXKit/Documentation.docc/M6PersistenceSemantics.md)
- [Runnable examples](https://github.com/murilxaraujo/MLXKit/tree/main/Examples)
