# Migrating from MLXKit 1.x to 2.0

This guide compares the released `1.0.0` API with the MLXKit 2.0 API. The
2.0.0 release is not available until its release gate passes. During
pre-release development, use the repository revision or candidate tag supplied
by the project rather than assuming `2.0.0` can already be resolved.

MLXKit permits source-breaking changes before 2.0.0. Starting with 2.0.0,
public source API is stable within the 2.x major version. MLXKit does not
promise binary/ABI compatibility. See <doc:CompatibilityPolicy> and
<doc:PlatformSupport> for the complete limits.

## Update the package requirements

The 1.0.0 package declared Swift tools 6.2, macOS 14 and iOS 17 deployment
targets, and MLX Swift from 0.31.3. The 2.0 candidate uses Swift tools 6.4 and
the documented macOS 27 Apple-silicon/Metal baseline, with MLX Swift 0.32.3 as
the validated dependency resolution. The 2.0 support contract does not include
iOS, Intel Macs, Linux, Windows, or a non-Apple compute backend.

After 2.0.0 is published, update the dependency requirement:

```swift
dependencies: [
    .package(url: "https://github.com/murilxaraujo/MLXKit", from: "2.0.0")
]
```

The package's lower-bound version range is not evidence that every permitted
MLX Swift version or platform has been tested. Follow the support matrix for
validated combinations.

## Model and training workflow

In 1.0.0, a model implemented `forward(_:)`, and `compile` returned one
non-generic configuration:

```swift
final class Classifier: Module, TrainableModel {
    let hidden = Linear(784, 128)
    let logits = Linear(128, 10)

    func forward(_ input: MLXArray) -> MLXArray {
        logits(maximum(hidden(input), 0))
    }
}

let engine = TrainingEngine()
let configuration = engine.compile(
    model: model,
    optimizer: Adam(learningRate: 1e-3),
    loss: CrossEntropyLoss(),
    metrics: [Accuracy()]
)
let history = try await engine.fit(
    model, config: configuration, data: loader, epochs: 10
)
```

In 2.0, implement `callAsFunction(_:)`; declare the model's input and output
types through its implementation. Configure the workflow as a typed value and
pass it to each engine operation:

```swift
final class Classifier: Module, TrainableModel {
    let hidden = Linear(784, 128)
    let logits = Linear(128, 10)

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        logits(maximum(hidden(input), 0))
    }
}

let engine = TrainingEngine()
let configuration = TrainingConfiguration(
    optimizer: Adam(learningRate: 1e-3),
    objective: CrossEntropyLoss(),
    metrics: [Accuracy()]
)
let history = try await engine.fit(
    model, configuration: configuration, data: loader, epochs: 10
)
```

There is no `compile` step or model-owned compiled state. Model input and output
types can be structured Swift values; batch and sequence dimensions remain
dynamic `MLXArray` dimensions. `fit`, `evaluate`, and `predict` operate on the
model and typed configuration directly.

## Objectives, reductions, and metrics

- `TrainingConfiguration` is generic over model output and target. Its
  `objective:` replaces the old `loss:` parameter and conforms to
  `Objective<Output, Target>`.
- A differentiable training objective must return a scalar. Reduce an
  unreduced loss explicitly; `fit` now reports a typed error for non-scalar or
  non-finite objective values instead of extracting an arbitrary scalar.
- `Metric` is generic over full model output and target. Update custom metrics
  to implement `update(output:target:)`, `compute()`, and `reset()` for the
  required types.
- Training loss is owned and aggregated by the engine. `RunningLoss` remains a
  public standalone accumulator, but it is not a `Metric` and must not be
  passed in the metric list.
- Non-finite objectives and gradients fail before an optimizer update. The
  former `TerminateOnNaN` callback is not the 2.0 mechanism; handle typed
  `TrainingError` failures instead.

For workflows that need batch-level objective logic, pass a custom
`TrainingStep` to `fit` or `evaluate`. It replaces objective computation only;
MLXKit continues to own autodiff, optimizer updates, callbacks, and lifecycle.

## Data loading

The 1.0.0 loader accepted an indexed `Dataset` and an unseeded Boolean shuffle:

```swift
let loader = DataLoader(dataset: dataset, batchSize: 32, shuffle: true)
for await batch in loader {
    let output = model(batch.features)
}
```

The 2.0 path uses typed `Sample<Input, Target>` and
`Batch<Input, Target>` values. For in-memory array data:

```swift
let dataset = InMemoryDataset(features: features, labels: labels)
let loader = try DataLoader(
    source: dataset,
    batchSize: 32,
    shuffle: .full(seed: 42),
    dropLast: false
)
for try await batch in loader {
    let output = model(batch.input)
    let sampleCount = batch.sampleCount
}
```

Array-specialized `features`, `labels`, and `size` conveniences remain
available on `Sample` and `Batch`, but new code should use `input`, `target`,
and `sampleCount`. The `DataLoader(dataset:batchSize:shuffle:dropLast:)`
initializer does not remain: use `source:` and an explicit
`ShuffleStrategy`. `.full(seed:)` provides reproducible indexed shuffling;
`.buffered(windowSize:seed:)` supports bounded shuffling of streams. Structured
samples use a throwing collator. Unknown-length async sources do not need to be
materialized.

## Training controls, callbacks, and errors

Configure learning-rate scheduling through `learningRateSchedule:`. MLXKit
updates only optimizers that support `LearningRateControllable`; unsupported
schedule/optimizer combinations now fail clearly rather than being silently
ignored. The old configuration property `lrSchedule` is replaced by
`learningRateSchedule`.

Callback lifecycle methods are synchronous and throwing. Add `throws` to
custom callback implementations and handle propagated errors. `fit` and
`evaluate` are `async throws`; cancellation surfaces as `CancellationError`.
Training errors now distinguish invalid configuration, empty input, invalid
objectives, non-finite values, unsupported schedules, invalid gradient
transforms, and resume-progress errors. See <doc:M2TrainingSemantics> and
<doc:M5TrainingControlSemantics> for exact behavior.

Gradient transforms, global-norm clipping, sample-aware accumulation, and
weighted cross-entropy are explicit M5 configuration/objective features. They
do not change the boundary that custom `TrainingStep` supplies a scalar loss
while MLXKit owns the optimizer update.

## Persistence and recovery

`saveWeights` and `loadWeights` remain the weight-only path. The 1.0.0
`save(model:epoch:step:to:)` and `load(into:from:)` calls are deprecated
compatibility wrappers: they store model parameters plus progress counters,
not optimizer state, and do not provide faithful training resume.

Choose the 2.0 operation by guarantee:

- `saveInferenceArtifact` / `loadInferenceArtifact` store weights plus
  caller-owned string metadata. Reconstruct the model and application assets
  explicitly.
- `saveTrainingProgress` / `loadTrainingProgress` store weights and progress
  counters, but do not restore optimizer state.
- `saveTrainingCheckpoint` / `loadTrainingCheckpoint` support epoch-boundary
  recovery only for an optimizer that explicitly conforms to
  `CheckpointableOptimizer`. Recreate the data source at the next epoch.

Do not assume that a 1.x artifact is interchangeable with a 2.0 artifact or
that a weights-only file resumes training. Format versions and legacy readers
are documented in <doc:M6PersistenceSemantics>.

## Migration checklist

- [ ] Update supported OS, Swift/Xcode, and MLX Swift versions; remove any
  unsupported iOS or non-Apple deployment assumption.
- [ ] Replace model `forward(_:)` with `callAsFunction(_:)` and represent
  non-array inputs or outputs with explicit Swift types where needed.
- [ ] Replace `compile` and `config:` with `TrainingConfiguration` and
  `configuration:`; rename `loss:` to `objective:` and `lrSchedule:` to
  `learningRateSchedule:`.
- [ ] Make training objectives scalar and update custom objective/metric
  conformances.
- [ ] Replace Boolean-shuffle `DataLoader(dataset:...)` calls with
  `DataLoader(source:..., shuffle:...)`; use explicit seeds and collators.
- [ ] Update callbacks to throwing lifecycle methods and handle async
  cancellation/errors.
- [ ] Select the persistence API by the state that must be restored; do not
  treat progress-only files as resumable checkpoints.
- [ ] Rebuild and run tests and examples on a supported configuration. Review
  every warning or behavioral change rather than assuming a compatibility
  shim exists.
