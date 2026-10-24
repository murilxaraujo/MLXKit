# MLXKit 2.0 Public API Design

**Status:** integrated M1 design draft. The contract directions in issues #8–#11
were approved in pair review. This page resolves their names and composition
into one proposed API. It is a design record, not the current 1.x API reference
or a final 2.0 compatibility promise until this review is approved.

Detailed contract notes are in <doc:M1ModelContract>,
<doc:M1ObjectiveMetricContract>, <doc:M1TrainingWorkflow>, and
<doc:M1DataContract>.

## Design principles

1. Keep MLX `Module`, `MLXArray`, MLX autodiff, and MLXOptimizers as the source
   of model parameters, tensor operations, differentiation, and optimizer
   algorithms.
2. Make the common supervised workflow concise with typed configuration,
   training, evaluation, and prediction APIs.
3. Make custom models, objectives, metrics, steps, and data sources possible
   through public Swift/MLX composition. Do not require users to modify MLXKit
   or implement a parallel tensor/autodiff stack.
4. Use Swift types for model inputs, outputs, targets, and structured samples;
   retain dynamic batch and sequence dimensions in `MLXArray` values.
5. Make invalid or unsupported requests visible. Do not silently select loss
   reductions, ignore unsupported optimizer schedules, swallow callback errors,
   or report failed training as successful.
6. Treat Keras as a usability reference, not an API-parity target. Use Swift and
   MLX idioms, and preserve direct MLX as the escape hatch.

## Model contract

Models remain MLX modules. MLX owns parameter discovery, train/eval state, and
gradient application. MLXKit adds typed input and output contracts:

```swift
public protocol TrainableModel: Module {
    associatedtype Input
    associatedtype Output

    func callAsFunction(_ input: Input) -> Output
}
```

Single-array models use `Input == MLXArray` and `Output == MLXArray`. Structured
models use ordinary user-defined Swift types. No MLXKit tensor-tree protocol is
required. One `Output` type is used for both training and inference; an
objective or metric selects the fields it needs. Model I/O types are not
globally constrained to `Sendable`; concurrency requirements apply at the
specific API boundary that needs them.

## Objectives and metrics

An objective is typed by full model output and target and returns one scalar
`MLXArray` when used by gradient-based training:

```swift
public protocol Objective<Output, Target> {
    associatedtype Output
    associatedtype Target

    func callAsFunction(output: Output, target: Target) -> MLXArray
}

public struct ClosureObjective<ModelOutput, TargetValue>: Objective {
    public typealias Output = ModelOutput
    public typealias Target = TargetValue

    private let body: (ModelOutput, TargetValue) -> MLXArray

    public init(_ body: @escaping (ModelOutput, TargetValue) -> MLXArray) {
        self.body = body
    }

    public func callAsFunction(output: ModelOutput, target: TargetValue) -> MLXArray {
        body(output, target)
    }
}
```

Reusable objective types conform to `Objective`; local objectives use
`ClosureObjective`. Built-in losses specialize to array predictions and
targets and adapt to `Objective<MLXArray, MLXArray>`. Their direct
`predictions:targets:` call remains available for unreduced loss calculation.
An unreduced loss such as `.none` is not accepted by `fit` unless the caller
explicitly reduces it. M2 checks the result shape; MLXKit does not silently
choose a reduction.

Metrics are stateful, typed by the same output/target pair, and return named
scalar values:

```swift
public protocol Metric<Output, Target>: AnyObject {
    associatedtype Output
    associatedtype Target

    var name: String { get }
    func update(output: Output, target: Target)
    func compute() -> Float
    func reset()
}
```

Neither `Objective` nor `Metric` has a global `Sendable` requirement. They run
in the training task; an API that moves one across an actor/task boundary adds
the constraint at that boundary.

Built-in classification/regression metrics specialize to `MLXArray`. Custom
metrics can inspect structured outputs. The trainer owns objective loss
aggregation and reserves the `loss` log key; `RunningLoss` is not a metric.
Metric names are stable and unique within a configuration. M2 defines exact
aggregation and collision validation.

## Configuration and training workflow

Configuration is a typed value, independent of a model instance. There is no
`TrainingEngine.compile` primary API: it receives a model only to infer types
and currently evaluates the model as a side effect. The model is passed to
`fit`, `evaluate`, or `predict` where it is used.

```swift
public struct TrainingConfiguration<Output, Target> {
    public let optimizer: any Optimizer
    public let objective: any Objective<Output, Target>
    public let metrics: [any Metric<Output, Target>]
    public let learningRateSchedule: (any LRSchedule)?

    public init<ObjectiveType: Objective<Output, Target>>(
        optimizer: any Optimizer,
        objective: ObjectiveType,
        metrics: [any Metric<Output, Target>] = [],
        learningRateSchedule: (any LRSchedule)? = nil
    ) {
        self.optimizer = optimizer
        self.objective = objective
        self.metrics = metrics
        self.learningRateSchedule = learningRateSchedule
    }
}
```

An initializer accepts any conforming objective and typed metrics, then stores
them behind their associated-type-constrained existentials. The optimizer stays
the MLXOptimizers existential; the configuration does not implement optimizer
algorithms.

`TrainingStep` is an optional lower-level loss-computation hook. It receives
the concrete model and full typed batch and returns a scalar objective. MLXKit
still owns autodiff, one optimizer update per batch, callbacks, and lifecycle.
For a custom gradient/update algorithm, users use MLX and MLXOptimizers
directly.

```swift
public typealias TrainingStep<Model: TrainableModel, Target> =
    (Model, Batch<Model.Input, Target>) -> MLXArray
```

Supplying a step overrides objective computation for that `fit` or `evaluate`
call. Pass the same step to evaluation when its loss should match training.

`fit` and `evaluate` are `async throws`. Callback hooks are synchronous and
throwing; callback errors propagate. Swift task cancellation surfaces as
`CancellationError`. M2 defines exact event order, cancellation checkpoints,
cleanup, error types, early-stop timing, and metric/output reuse.

```swift
public func fit<Model: TrainableModel, Target>(
    _ model: Model,
    configuration: TrainingConfiguration<Model.Output, Target>,
    data: DataLoader<Model.Input, Target>,
    epochs: Int,
    validationData: DataLoader<Model.Input, Target>? = nil,
    trainingStep: TrainingStep<Model, Target>? = nil,
    callbacks: [any Callback] = []
) async throws -> History

public func evaluate<Model: TrainableModel, Target>(
    _ model: Model,
    configuration: TrainingConfiguration<Model.Output, Target>,
    data: DataLoader<Model.Input, Target>,
    trainingStep: TrainingStep<Model, Target>? = nil
) async throws -> [String: Float]

public func predict<Model: TrainableModel>(
    _ model: Model,
    input: Model.Input
) -> Model.Output
```

An overload accepts an arbitrary `AsyncSequence` of matching `Batch` values.
The direct MLX sequence overload has separate forms with and without
validation data so its iterator type remains concrete and errors remain
throwing.

## Optimizers and schedules

MLXKit uses `Optimizer.update` and does not inspect concrete optimizer types.
An optional `LearningRateControllable: Optimizer` capability exposes
`setLearningRate(_:)`; MLXKit adapts supported MLXOptimizers types, and custom
optimizers may conform. A schedule paired with an optimizer that lacks this
capability fails clearly before training. MLXKit supplies schedules, not a
parallel optimizer implementation.

The M2 adapters cover SGD, RMSprop, AdaGrad, AdaDelta, Adam/AdamW, Adamax,
Lion, Adafactor, and Muon. `MultiOptimizer` is not adaptable as a single rate
because its children can use different optimizers; users can schedule its
children directly or provide a custom optimizer that implements the capability.
For Adafactor, configuring a schedule switches off its default relative-rate
mode so the supplied rate is effective.

```swift
public protocol LearningRateControllable: Optimizer {
    func setLearningRate(_ value: Float) throws
}
```

## Data API

Samples and batches carry model input and target types:

```swift
public struct Sample<Input, Target> {
    public let input: Input
    public let target: Target
}

public struct Batch<Input, Target> {
    public let input: Input
    public let target: Target
    public let sampleCount: Int
}
```

The source contract is an `AsyncSequence<Sample<Input, Target>, Failure>` with
`Failure: Error`. A source need not know its length or provide random access.
The typed `DataLoader<Input, Target>` is the primary sample-to-batch pipeline
and conforms to `AsyncSequence<Batch<Input, Target>, any Error>`. It accepts
any async sample source, performs explicit batching, and hides the source's
concrete iterator type. In-memory `MLXArray` data has a built-in collator;
structured samples use a throwing collate closure.

`fit`'s primary data input is `DataLoader<Model.Input, Target>`. A convenience
overload accepts any asynchronous sequence of `Batch<Model.Input, Target>`.
This keeps the common path discoverable while allowing prebuilt/custom batch
sequences. The generic overloads accept a sequence with `Failure: Error`; a
separate overload handles calls with and without validation data. Random access
is an optional capability.

The `DataLoader` initializer accepts any async sample source, performs explicit
batching, and hides the source's concrete iterator type:

```swift
public struct DataLoader<Input, Target>: AsyncSequence {
    public typealias Element = Batch<Input, Target>
    public typealias Failure = any Error

    public init<Source: AsyncSequence>(
        source: Source,
        batchSize: Int,
        shuffle: ShuffleStrategy = .none,
        dropLast: Bool = false,
        collate: @escaping ([Sample<Input, Target>]) throws -> Batch<Input, Target>
    ) where Source.Element == Sample<Input, Target>
}

extension DataLoader where Input == MLXArray, Target == MLXArray {
    public init(
        source: InMemoryDataset,
        batchSize: Int,
        shuffle: ShuffleStrategy = .none,
        dropLast: Bool = false
    )
}
```

An in-memory `MLXArray`/`MLXArray` initializer supplies the built-in collator;
structured samples use a throwing collate closure. The loader may erase the
concrete source iterator internally but retains the `Input` and `Target` types.

Shuffle strategies are explicit: `.full(seed:)` for indexed sources and
`.buffered(windowSize:seed:)` for streams. Streaming shuffle never
materializes the complete source. Prefetch is bounded and applies backpressure;
upstream, transform, and collate failures propagate. M4 defines exact ordering,
reproducibility, buffering, cancellation, and cleanup semantics.

```swift
public enum ShuffleStrategy {
    case none
    case full(seed: UInt64)
    case buffered(windowSize: Int, seed: UInt64)
}
```

## Integrated examples

### Conventional classifier

```swift
import MLX
import MLXNN
import MLXOptimizers
import MLXKit

final class Classifier: Module, TrainableModel {
    typealias Input = MLXArray
    typealias Output = MLXArray

    let hidden = Linear(inputDimensions: 784, outputDimensions: 128)
    let logits = Linear(inputDimensions: 128, outputDimensions: 10)

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        logits(MLX.maximum(hidden(input), 0))
    }
}

let model = Classifier()
let configuration = TrainingConfiguration(
    optimizer: Adam(learningRate: 1e-3),
    objective: CrossEntropyLoss(),
    metrics: [Accuracy()],
    learningRateSchedule: CosineDecay(initialLR: 1e-3, totalSteps: 10_000)
)

let training = DataLoader(
    source: InMemoryDataset(inputs: trainImages, targets: trainLabels),
    batchSize: 32,
    shuffle: .full(seed: 42),
    dropLast: false
)
let validation = DataLoader(
    source: InMemoryDataset(inputs: validationImages, targets: validationLabels),
    batchSize: 32
)

let engine = TrainingEngine()
let history = try await engine.fit(
    model,
    configuration: configuration,
    data: training,
    epochs: 10,
    validationData: validation,
    callbacks: [ProgressReporter()]
)
let metrics = try await engine.evaluate(model, configuration: configuration, data: validation)
let predictions = engine.predict(model, input: testImages)
```

`InMemoryDataset` and `DataLoader` names are retained as conveniences, but the
source, sample, and batch contracts are generic. A real project supplies the
named image/label arrays and may use the generic batch-sequence overload.

### Custom parameterized, multi-input model and objective

```swift
struct EncoderInput {
    let tokenEmbeddings: MLXArray      // [batch, sequence, 128]
    let positionEmbeddings: MLXArray   // [batch, sequence, 128]
}

struct EncoderOutput {
    let embeddings: MLXArray           // [batch, sequence, 128]
    let logits: MLXArray               // [batch, sequence, vocabulary]
}

final class LearnableScale: Module {
    let scale: MLXArray

    init(width: Int) {
        scale = MLXArray.ones([width])
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        input * scale
    }
}

final class TokenEncoder: Module, TrainableModel {
    typealias Input = EncoderInput
    typealias Output = EncoderOutput

    let scale = LearnableScale(width: 128)
    let projection = Linear(inputDimensions: 128, outputDimensions: 256)

    func callAsFunction(_ input: EncoderInput) -> EncoderOutput {
        let hidden = scale(input.tokenEmbeddings + input.positionEmbeddings)
        return EncoderOutput(embeddings: hidden, logits: projection(hidden))
    }
}

struct TokenTargets {
    let tokenIDs: MLXArray // [batch, sequence]
}

struct TokenObjective: Objective {
    typealias Output = EncoderOutput
    typealias Target = TokenTargets

    let embeddingPenalty: Float

    func callAsFunction(output: EncoderOutput, target: TokenTargets) -> MLXArray {
        let tokenLoss = CrossEntropyLoss()(
            predictions: output.logits,
            targets: target.tokenIDs
        )
        let penalty = (output.embeddings * output.embeddings).mean()
        return tokenLoss + penalty * embeddingPenalty
    }
}
```

The batch and sequence axes are dynamic; only the feature width and output
vocabulary are fixed by this example. The model composes a custom parameterized
MLX module and exposes multiple input fields and structured outputs. The
objective combines token loss with an auxiliary embedding term using public
MLXKit/MLX APIs. A user-defined throwing collator can pad variable-length
samples and create `Batch<EncoderInput, TokenTargets>`; no MLXKit internals are
required.

An optional `TrainingStep<TokenEncoder, TokenTargets>` can receive that batch
and compute the scalar objective when a workflow needs model/batch-level logic
beyond the standard objective interface. Its optimizer update and lifecycle
remain owned by MLXKit.

## 1.x to 2.0 migration map

No compatibility shims are proposed in this design. Pre-2.0 source breaks are
allowed by <doc:CompatibilityPolicy>; the 2.0 release must provide a fuller
before-and-after migration guide. Do not assume checkpoint files are portable
across versions; M6 owns the artifact policy.

| 1.x surface | 2.0 direction | Migration action / behavior change |
|---|---|---|
| `TrainableModel.forward(_:) -> MLXArray` | `TrainableModel` with associated `Input`/`Output` and `callAsFunction(_:)` | Rename `forward` and add type aliases or structured Swift types. The 2.0 model is still an MLX `Module`. |
| `TrainingEngine.compile(model:optimizer:loss:metrics:lrSchedule:)` | Direct `TrainingConfiguration<Output, Target>` value | Move optimizer, objective, metrics, and schedule into the initializer. Remove the model argument and its eager `eval` side effect. |
| `LossFunction` over array predictions/targets | Typed `Objective<Output, Target>` protocol and `ClosureObjective` | Adapt custom loss types or closures; built-ins remain convenient for array I/O. The objective passed to training must be scalar. |
| `Reduction.none` passed to `fit` | Direct unreduced evaluation plus explicit scalar reduction for training | Add a reduction in user objective code or choose `.mean`/`.sum`; `fit` rejects non-scalar output instead of attempting scalar extraction. |
| `Metric` over two `MLXArray`s | `Metric<Output, Target>` with `update`/`compute`/`reset` | Update conformances and choose a stable unique scalar result name. |
| `RunningLoss` as a `Metric` | Trainer-owned `loss` reporting | Remove `RunningLoss` from metric arrays; `History["loss"]` remains a trainer log. |
| `Dataset: RandomAccessCollection` with `features`/`labels` | Async sample source; optional random-access capability | Custom sources yield `Sample<Input, Target>` and may have unknown length. |
| `Sample`/`Batch` with fixed `features`/`labels: MLXArray` | Generic `Sample<Input, Target>` and `Batch<Input, Target>` | Move model inputs/targets into typed values; provide a collate closure for custom structured samples. |
| `DataLoader(dataset:batchSize:shuffle:dropLast:)` with boolean shuffle | Typed `DataLoader<Input, Target>` over sample sources | Replace the boolean with an explicit seeded strategy; add collate for structured data. Streams can use bounded shuffle and prefetch. |
| `Metric.update(predictions:targets:)` | `update(output:target:)` with full typed model output | Select output fields inside a metric rather than narrowing the model output. |
| `predict(input: MLXArray) -> MLXArray` | `predict(input: Model.Input) -> Model.Output` | Update call sites and handle the full structured output type where applicable. |
| Concrete optimizer casts for schedules | Optional `LearningRateControllable` capability | Custom optimizer integrations conform explicitly; unsupported schedule/optimizer combinations fail clearly. |
| Nonthrowing callback hooks | Synchronous throwing hooks | Add `throws` to callback conformances and handle propagated callback errors. Async training cancellation surfaces as `CancellationError`. |
| `Checkpointing` weight/full-checkpoint calls | No M1 mapping selected | M6 will distinguish weights, inference artifacts, and resumable checkpoints. Do not assume cross-version artifact compatibility. |

## Alternatives rejected

- A model protocol fixed to `MLXArray -> MLXArray` or an MLXKit-owned model
  base class/parameter registry.
- A universal tensor-tree or `Batchable` protocol for arbitrary user values.
- A random-access requirement for every dataset or implicit full materializing
  shuffle for streams.
- Losses/metrics restricted to the model's single array output.
- `compile` as the primary API despite configuration being an independent
  value; the model is passed only where it is used.
- Reflection or concrete-type switches for optimizer schedule updates.
- A training-step hook that owns gradient and optimizer mutation while also
  claiming MLXKit owns training lifecycle semantics.

## M0 design review checklist

The review checks <doc:ProductScope>, <doc:MLXKitAndMLX>,
<doc:PlatformSupport>, <doc:CompatibilityPolicy>, and
<doc:ReferenceWorkloads>.

- [x] **Product scope:** concise conventional workflow plus custom model,
  objective, metric, step, and data extension points.
- [x] **Responsibility boundary:** use MLX modules, arrays, autodiff, and
  optimizers; no MLXKit internals or duplicate tensor stack.
- [x] **Platform contract:** no new platform claims; build/test target remains
  the M0 macOS 27.x / Apple silicon / Metal baseline.
- [x] **Compatibility:** pre-2.0 breaking source changes are allowed; no
  unapproved compatibility shims or binary-stability claims.
- [x] **Reference workloads:** classifier, embedding/contrastive, and small
  decoder examples remain representable; dynamic sequence and batch sizes are
  not fixed by the API.
- [x] **Performance:** structured autodiff, collate/prefetch, and optimizer
  schedule costs are measured in later milestones rather than asserted here.

## Deferred decisions

- **M2:** runtime validation, scalar/reduction semantics, metric aggregation,
  training lifecycle ordering, cancellation/error cleanup, optimizer step
  accounting, and deterministic integration tests.
- **M3:** custom architecture guides, state/cache boundaries, and gradient
  correctness examples.
- **M4:** concrete data transforms, collate/padding, shuffle reproducibility,
  bounded prefetch/backpressure, and stream cancellation tests.
- **M5:** gradient transforms, clipping, accumulation, sample weighting, and
  update controls beyond one MLX optimizer update per batch.
- **M6:** weight/export/resume artifact formats and checkpoint compatibility.
- **M7–M8:** runnable example projects and measured local benchmarks.
- **M9:** final API freeze, complete migration guide, and 2.0 release review.

The M1 design is ready for implementation only after the integrated example
and migration map receive pair approval. M2 and later remain drafts until their
preceding milestone is complete.
