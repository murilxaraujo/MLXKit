# M1 Objective and Metric Contract — Working Draft

**Status:** pair-approved direction for issue #9: typed objectives support both
reusable conforming types and a closure-backed form; metrics are typed over
model output and target; the trainer reports loss separately from metrics; an
unreduced `.none` loss requires an explicit reduction before training. Final
names and composition are in <doc:API2Design>, pending pair approval.

Code snippets are API sketches. They assume `import MLX`, `import MLXNN`, and
`import MLXKit`; they do not compile against the current 1.x API.

## Constraints

- The model contract allows a user-defined `Output` type. A training objective
  and metric must be able to select the tensors or fields they use without
  requiring the model to flatten its output to one `MLXArray`.
- Objectives use MLX operations and autodiff. MLXKit must not duplicate those
  operations.
- Gradient-based training requires one scalar `MLXArray` objective per update.
  Swift's type system cannot prove an `MLXArray` is scalar, so M2 must validate
  this at the training boundary and report a useful error.
- Shape and dtype compatibility remain runtime properties. The API can expose
  clear boundaries for validation, but must not claim static shape safety.

## Objective alternatives

The current `LossFunction` only accepts `MLXArray` predictions and targets.
That is convenient for classifiers but does not compose directly with the
approved structured model output. Two objective styles should be supported:

### Conforming objective type

An objective type can carry configuration and be reused by name:

```swift
protocol Objective<Output, Target> {
    associatedtype Output
    associatedtype Target

    func callAsFunction(output: Output, target: Target) -> MLXArray
}

struct CompositeOutput {
    let prediction: MLXArray
    let embedding: MLXArray
}

struct CompositeObjective: Objective {
    typealias Output = CompositeOutput
    typealias Target = MLXArray

    let embeddingPenalty: Float

    func callAsFunction(output: CompositeOutput, target: MLXArray) -> MLXArray {
        let residual = output.prediction - target
        let dataLoss = (residual * residual).mean()
        let penalty = (output.embedding * output.embedding).mean()
        return dataLoss + penalty * embeddingPenalty
    }
}
```

This example consumes a named prediction and an auxiliary embedding from one
model output and uses MLX operations to return one differentiable scalar.

### Closure-backed objective

A closure adapter avoids conformance boilerplate for local or inline
objectives:

```swift
struct ClosureObjective<ModelOutput, TargetValue>: Objective {
    typealias Output = ModelOutput
    typealias Target = TargetValue

    private let body: (ModelOutput, TargetValue) -> MLXArray

    init(_ body: @escaping (ModelOutput, TargetValue) -> MLXArray) {
        self.body = body
    }

    func callAsFunction(output: ModelOutput, target: TargetValue) -> MLXArray {
        body(output, target)
    }
}

struct ClassifierOutput {
    let logits: MLXArray
}

let objective = ClosureObjective<ClassifierOutput, MLXArray> { output, labels in
    CrossEntropyLoss()(
        predictions: output.logits,
        targets: labels
    )
}
```

The closure-backed form and conforming form must share the same typed contract;
they must not create separate training behaviors. Exact public names and
initializers remain subject to the integrated M1 review.

The protocol and adapter above are illustrative signatures; see
<doc:API2Design> for the selected names and generic relationships.

## Objective contract and reductions

The objective supplied to gradient-based `fit` returns a scalar MLX array. It
receives the model's full `Output` value and the typed target value. This lets a
custom objective use one output field, combine multiple fields, or use other
MLX operations without changing the model protocol.

Built-in tensor losses remain convenient when the model output and target are
`MLXArray`. Their configured reduction is part of the objective behavior:

- Scalar reductions such as `.mean` or `.sum` can be used for training.
- An unreduced result such as `.none` remains useful for direct loss
  calculation, but it is not a valid training objective unless the caller
  explicitly reduces it to one scalar.
- The training engine must not silently choose a reduction for a custom
  objective or unreduced built-in loss.

M2 defines exact accepted shapes, reduction/weighting semantics, uneven-batch
aggregation, and the validation error for a non-scalar objective. If a built-in
loss value with `.none` is passed to `fit`, the request must fail clearly or be
made scalar explicitly; it must not silently proceed to a scalar extraction
that traps. An explicit closure can reduce the unreduced array before returning
it, but the engine does not select a reduction on the caller's behalf.

## Typed metric contract

Metrics observe the same model output and target types as the training
configuration. Built-in metrics can specialize to `MLXArray`; a custom metric
can inspect a structured output directly.

```swift
protocol Metric<Output, Target>: AnyObject {
    associatedtype Output
    associatedtype Target

    var name: String { get }
    func update(output: Output, target: Target)
    func compute() -> Float
    func reset()
}
```

The lifecycle is:

1. Reset before an independent training epoch or evaluation run.
2. Update once for each observed batch.
3. Compute one named scalar value at the documented reporting boundary.

Metric names must be stable and unique within one training configuration. The
training log reserves `loss` for the objective and uses a documented validation
prefix. M2 owns exact collision validation, empty-input behavior, and batch
aggregation semantics.

`RunningLoss` should not conform to `Metric`: it does not consume model output
and target, and its current adapter treats its arguments as fake predictions
and targets. Loss reporting and aggregation belong to the trainer. A metric
measures a separate property of model output/target pairs.

For example, a custom metric can inspect one field of a structured output:

```swift
struct EmbeddingOutput {
    let prediction: MLXArray
    let embedding: MLXArray
}

final class MeanEmbeddingEnergy: Metric {
    typealias Output = EmbeddingOutput
    typealias Target = MLXArray

    let name = "mean_embedding_energy"
    private var total: Float = 0
    private var count = 0

    func update(output: EmbeddingOutput, target: MLXArray) {
        let perExample = (output.embedding * output.embedding).sum(axis: -1)
        total += perExample.sum().item(Float.self)
        count += perExample.size
    }

    func compute() -> Float {
        count == 0 ? 0 : total / Float(count)
    }

    func reset() {
        total = 0
        count = 0
    }
}
```

The metric receives the target because that is the shared protocol contract;
this example does not need it. The value is a scalar for logging, while M2
defines empty-input and aggregation semantics for the production API.

## Examples and compatibility

- A classification example uses an output containing logits, a cross-entropy
  objective specialized to `MLXArray`, and an accuracy metric specialized to
  `MLXArray`.
- A research example uses a structured embedding output, a conforming
  contrastive objective or closure-backed equivalent, and a custom metric that
  reads the relevant output fields.
- The objective and metric receive the complete output value; projections are
  explicit in the objective/metric and do not require a second model protocol.
- Existing `LossFunction` and `Metric` signatures may change before 2.0 under
  the M0 compatibility policy. The integrated API design must record these
  source changes and migration guidance.

## Decisions deferred to M2 and later milestones

- M2 tests scalar objective validation, reduction and sample-weighting
  semantics, metric aggregation for uneven batches, and failure behavior.
- M1 issue #8 aligns objective and metric generic types with model and target
  types in the training configuration.
- <doc:API2Design> selects the integrated names, type-erasure/storage direction,
  concurrency constraints, and migration map; its pair review is still open.
- Metrics remain scalar in this initial design. Non-scalar metric results would
  require a separately reviewed logging/result model rather than an implicit
  conversion to `Float`.
