# M1 Training Workflow — Working Draft

**Status:** pair-approved direction for issue #8: the standard workflow is
typed by model output and target; a custom training step replaces only
per-batch scalar-objective computation; MLXKit retains differentiation,
optimizer updates, and lifecycle; optimizer schedules use a capability bridge
instead of concrete-type inspection; callbacks may throw synchronously. The
integrated names and signatures are in <doc:API2Design>, pending pair approval.

## Standard workflow

The common workflow builds a typed `TrainingConfiguration` value from an MLX
optimizer, objective, optional metrics, and optional schedule. Training and
prediction still receive the model explicitly; the configuration stores the
training choices, not a hidden model instance.

```swift
let model = Classifier() // Module, TrainableModel; Input/Output are MLXArray
let engine = TrainingEngine()

let objective = ClosureObjective<MLXArray, MLXArray> { logits, labels in
    CrossEntropyLoss(reduction: .mean)(
        predictions: logits,
        targets: labels
    )
}

let configuration = TrainingConfiguration(
    optimizer: Adam(learningRate: 1e-3),
    objective: objective,
    metrics: [Accuracy()]
)

let history = try await engine.fit(
    model,
    configuration: configuration,
    data: trainingData,
    epochs: 10,
    validationData: validationData,
    callbacks: [ProgressReporter()]
)

let evaluation = try await engine.evaluate(
    model,
    configuration: configuration,
    data: testData
)
let logits = engine.predict(model, input: images)
```

The actual data and configuration types must align with `Model.Input`,
`Model.Output`, and the objective target type. Issue #11 owns the data
abstraction names and batching contract. `predict` returns the model's complete
`Output` type, as decided in issue #10.

These are proposed public signatures, not compilable examples against MLXKit
1.x. The classifier and dataset declarations are abbreviated; the model
contract example is in <doc:M1ModelContract>.

The example preserves a discoverable configure/fit workflow without requiring
the model to store its configuration or passing the model to an API that only
builds configuration.

## Custom training-step boundary

The typed `Objective<Output, Target>` is the preferred extension for a custom
loss that depends only on model output and target. A `TrainingStep` is the
lower-level extension for workflows that need the model and full batch, such as
paired views, masks, or auxiliary batch fields:

```swift
struct ContrastiveInput {
    let anchor: MLXArray
    let positive: MLXArray
    let negative: MLXArray
}

struct ContrastiveBatch {
    let input: ContrastiveInput
}

struct ContrastiveOutput {
    let query: MLXArray
    let positive: MLXArray
    let negative: MLXArray
}

// Encoder is a Module, TrainableModel with these Input/Output types.
let encoder: Encoder = ...
typealias TrainingStep<Model: TrainableModel, Batch> = (Model, Batch) -> MLXArray

let contrastiveStep: TrainingStep<Encoder, ContrastiveBatch> = { model, batch in
    let output = model(batch.input)
    let positive = output.query * output.positive
    let negative = output.query * output.negative
    let positiveScore = positive.sum(axis: -1)
    let negativeScore = negative.sum(axis: -1)
    return MLX.maximum(1 - positiveScore + negativeScore, 0).mean()
}

// Configured for Encoder.Output and the data target type. Its objective is
// used by default; trainingStep overrides objective computation for this run.
let contrastiveConfiguration: TrainingConfiguration<ContrastiveOutput, MLXArray> = ...

let history = try await engine.fit(
    encoder,
    configuration: contrastiveConfiguration,
    data: contrastiveData,
    epochs: 5,
    trainingStep: contrastiveStep
)

let evaluation = try await engine.evaluate(
    encoder,
    configuration: contrastiveConfiguration,
    data: contrastiveValidationData,
    trainingStep: contrastiveStep
)
```

`TrainingStep` computes one differentiable scalar objective from a model and a
batch. MLXKit still computes gradients with MLX, applies the configured
MLXOptimizers optimizer, updates metrics/logs, and dispatches lifecycle events.
The step does not own the optimizer or mutate trainable parameters directly.
M2 defines how a custom step's output participates in metrics, how often the
model is evaluated, and how errors are reported. When supplied, the custom
step replaces the configured objective for that operation; passing the same
step to evaluation keeps its reported loss consistent with training.

Users who need a different gradient algorithm, multiple optimizer updates per
batch, or full control over parameter mutation can use MLX and MLXOptimizers
directly. The high-level engine is optional; this is the documented escape
hatch, not an unsupported way to bypass private MLXKit internals.

## Optimizers and learning-rate schedules

MLXOptimizers owns optimizer algorithms and the `Optimizer.update` contract.
That protocol does not expose a common learning-rate property, although many
concrete optimizers do. MLXKit must not use reflection or a switch over known
optimizer classes to set rates.

The design adds a narrow optional capability, provisionally
`LearningRateControllable`, for an optimizer that can accept a rate update:

```swift
protocol LearningRateControllable: Optimizer {
    func setLearningRate(_ value: Float) throws
}
```

MLXKit can provide conformances/adapters for supported MLXOptimizers types;
custom optimizers can adopt the capability when they support it. The existing
MLXKit schedule abstraction (or an equivalent step-to-rate closure) computes
the value. MLXKit does not reimplement optimizer behavior. Configuring a
schedule with an optimizer that does not support rate updates must fail clearly
before training rather than silently ignore the schedule. M2 starts at schedule
step zero and advances once per successful optimizer update; evaluation does
not read or mutate the schedule.

## Lifecycle, cancellation, and errors

- `fit`, `evaluate`, and data-backed prediction are asynchronous throwing
  operations because data may be asynchronous and failures must reach callers.
- Callback hooks are synchronous and throwing. A callback failure propagates
  from `fit`; it is not converted to a log message or ignored.
- Swift task cancellation is surfaced as `CancellationError`. The engine
  checks cancellation before batch work, after callbacks, and after sequence
  completion.
- A callback can request an orderly early stop from `onBatchEnd` or
  `onEpochEnd`; the engine completes the current batch/epoch boundary.
- Objective/gradient, data-source, configuration, schedule, and callback
  failures are thrown. Non-finite objectives and gradients fail before the
  optimizer update.
- Once train-begin dispatch starts, train-end dispatch runs exactly once on
  completion, early stop, cancellation, or failure. The original run error
  wins if train-end also throws.
- `fit`, `evaluate`, and `predict` restore the model's prior train/eval mode.

The lifecycle remains observable through callbacks and per-epoch history. The
M2 implementation and tests define exact hook order and failure cleanup.

## Alternatives and rationale

- **One generic objective only:** concise and sufficient for most custom loss
  functions, but cannot express custom access to the model/batch boundary when
  an objective is not just `(Output, Target) -> scalar`.
- **Custom step owns gradients and optimizer:** maximally flexible, but
  duplicates MLXKit's update/lifecycle contract and makes callback, metric, and
  schedule behavior ambiguous. Direct MLX is the cleaner escape hatch for this
  level of control.
- **Concrete optimizer inspection:** rejected. It duplicates knowledge of
  upstream optimizer types and silently skips unsupported types.
- **A parallel optimizer abstraction:** rejected. MLXOptimizers remains the
  source of optimizer algorithms; the optional rate-control capability only
  describes one optional property of an optimizer.

## Deferred to later milestones

- Issue #11 defines the batch type consumed by `TrainingStep`.
- M5 owns gradient controls and training controls beyond one MLX gradient and
  one configured optimizer update per batch.

The integrated names and signatures are in <doc:API2Design>.
