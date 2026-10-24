# M1 Model Contract — Working Draft

**Status:** approved model-contract direction for issue #10. The typed I/O
direction, `callAsFunction`, and boundary-only `Sendable` constraints have been
selected through pair review. User-defined Swift I/O types are allowed without
a required tensor-tree protocol, and one `Output` type is shared by training
and inference. The integrated design is in <doc:API2Design> and remains subject
to final pair approval; this is not yet a final 2.0 API promise.

This draft compares an array-only model contract with a typed input/output
contract. The working direction is the typed contract: it keeps the ordinary
single-array case concise and lets a model state its structured interface
without adding a second parameter or tensor system to MLXKit.

## Constraints from M0 and MLX

- MLXKit owns high-level workflow contracts. MLX owns `MLXArray`, module
  parameters, differentiation, and numerical operations.
- Custom components must work through public MLX APIs and must not require
  changes to MLXKit internals.
- Batch and sequence dimensions remain dynamic MLX array dimensions.
- `MLXNN.Module` already provides parameter traversal, train/eval mode, and
  trainable-parameter selection. MLXKit should not create a parallel model base
  class or parameter registry.
- MLX Swift's `valueAndGrad(model:_:)` has an overload for arbitrary argument
  types. It can carry a structured input through the model, but the upstream
  documentation says this overload rebuilds the gradient transform on each
  invocation. The `[MLXArray]` overloads are preferred when their argument
  shape is sufficient. M2 must measure the effect before the training API
  commits to an implementation strategy.

## Alternative A: array-only contract

```swift
protocol TrainableModel: Module {
    func callAsFunction(_ input: MLXArray) -> MLXArray
}
```

This is simple for a conventional classifier. It does not directly express
independent input modalities or multiple named outputs. A second protocol,
packing convention, or custom training path would be needed for those cases.
Packing unrelated tensors into one array is not a general or type-safe answer.

## Alternative B: typed input/output contract (working direction)

```swift
protocol TrainableModel: Module {
    associatedtype Input
    associatedtype Output

    func callAsFunction(_ input: Input) -> Output
}
```

The array case remains direct:

```swift
import MLX
import MLXNN
import MLXKit

final class Classifier: Module, TrainableModel {
    typealias Input = MLXArray
    typealias Output = MLXArray

    let hiddenLayer = Linear(inputDimensions: 784, outputDimensions: 128)
    let outputLayer = Linear(inputDimensions: 128, outputDimensions: 10)

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        outputLayer(MLX.maximum(hiddenLayer(input), 0))
    }
}

let model = Classifier()
let images: MLXArray = ...
let logits = model(images)
```

The same contract leaves custom parameterized components in ordinary MLXNN
modules. MLX discovers this module's `MLXArray` parameter; MLXKit does not need
to register it:

```swift
final class LearnableScale: Module {
    let scale: MLXArray

    init(width: Int) {
        self.scale = MLXArray.ones([width])
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        input * scale
    }
}
```

A structured model can use ordinary named Swift values whose tensor leaves are
`MLXArray` values:

```swift
struct EncoderInput {
    let tokenIDs: MLXArray       // [batch, sequence]
    let attentionMask: MLXArray  // [batch, sequence]
}

struct EncoderOutput {
    let logits: MLXArray         // [batch, sequence, vocabulary]
    let pooled: MLXArray         // [batch, width]
}

final class Encoder: Module, TrainableModel {
    typealias Input = EncoderInput
    typealias Output = EncoderOutput

    func callAsFunction(_ input: EncoderInput) -> EncoderOutput {
        // Compose MLXNN modules and MLX operations here.
        fatalError("illustrative API sketch")
    }
}
```

Named structs make fields discoverable and avoid positional meaning being
hidden in a tuple. They are ordinary user/model values, not a new MLXKit tensor
container. The core contract should not prescribe how every such value is
batched; that is the data contract in issue #11.

## Autodiff and training interaction

The typed input can be passed as an arbitrary argument to MLX's module-aware
gradient transform. A training example can carry both structured input and
target while MLX differentiates the model's trainable parameters:

```swift
struct Example {
    let input: EncoderInput
    let target: MLXArray
}

let lossAndGradient = valueAndGrad(model: model) {
    (model: Encoder, example: Example) -> [MLXArray] in
    let output = model(example.input)
    return [objective(output.logits, example.target)]
}
```

This is an integration sketch, not a final training API. The objective contract,
target typing, efficient argument representation, metric storage, and data
batching are coordinated with issues #8, #9, and #11. M2 must validate that the
chosen path does not impose unacceptable overhead and can retain an efficient
specialized path for the common single-array case.

## Approved model-contract direction

1. Keep MLX `Module` as the source of trainable parameter state and train/eval
   behavior. Do not add an MLXKit parameter protocol or registry.
2. Use associated `Input` and `Output` types on the high-level model contract.
   A model with one tensor input/output uses `MLXArray` for both.
3. Require a callable model method (`callAsFunction`) to match MLXNN composition
   and Swift call-site ergonomics; record the rename from the 1.x `forward`
   method in the migration map.
4. Permit arbitrary user-defined Swift value types for model I/O. The
   objective, data, and concurrency APIs state their own requirements at the
   points where they consume those types. Do not require users to implement a
   generic tensor-tree protocol.
5. Do not require associated `Input` and `Output` types to be `Sendable` at the
   model protocol boundary. Require `Sendable` only where a specific API crosses
   a task or actor-isolation boundary.
6. Keep training and inference APIs generic over the concrete model type where
   practical. Do not erase `Input`/`Output` to `Any` or store a model existential
   merely to support `compile`-style configuration.
7. Do not promise static tensor shape checking. Shapes remain runtime properties
   of `MLXArray`; diagnostics belong at documented validation boundaries.
8. Use one `Output` type for both training and inference. A structured output
   can include logits, embeddings, or other values; objectives and metrics
   select what they consume. Do not add a separate training-output protocol.

## Deferred decisions and validation

- The M2 acceptance fixture covers a deterministic `MLXArray` regression path.
  A structured-input performance baseline remains future work; no universal
  performance threshold is promised.
- M4 owns generalized sample-source batching and collation for user-defined
  structured `Input` values.
- The integrated API uses `callAsFunction`; see <doc:API2Design> for the
  source migration direction from `forward`.

The callable method and boundary-only concurrency constraints above are
implemented, as is allowing user-defined I/O types without a required tree
protocol and sharing one `Output` type between training and inference.

The final design must be consistent with <doc:MLXKitAndMLX> and the wider M1
design record. Decisions about batching and training-loop behavior remain with
issues #11 and #8/#9 and the later M2/M4 milestones.
