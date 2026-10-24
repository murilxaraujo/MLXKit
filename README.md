# MLXKit

A high-level Swift framework for machine learning on Apple silicon, built on [MLX Swift](https://github.com/ml-explore/mlx-swift).

MLXKit provides a typed training workflow: define a model, create a typed training configuration, and call `fit()`. It coordinates gradient computation, metric tracking, callbacks, learning-rate scheduling, and checkpointing while MLX and MLXOptimizers provide the numerical core.

## Requirements

- macOS 27.x
- Swift 6.4 with Xcode 27.0
- Apple Silicon (`arm64`, M1 or later)

## Installation

Add MLXKit to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/murilxaraujo/MLXKit", from: "1.0.0"),
]
```

Then add it to your target:

```swift
.target(name: "MyApp", dependencies: ["MLXKit"])
```

## Quick Start

```swift
import MLX
import MLXNN
import MLXOptimizers
import MLXKit

// 1. Define a model
final class Classifier: Module, TrainableModel {
    let linear1 = Linear(inputDimensions: 784, outputDimensions: 128)
    let linear2 = Linear(inputDimensions: 128, outputDimensions: 10)

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = linear1(input)
        x = maximum(x, 0)
        return linear2(x)
    }
}

// 2. Configure
let model = Classifier()
let engine = TrainingEngine()
let config = TrainingConfiguration(
    optimizer: Adam(learningRate: 1e-3),
    objective: CrossEntropyLoss(),
    metrics: [Accuracy()],
    learningRateSchedule: CosineDecay(initialLR: 1e-3, totalSteps: 10000)
)

// 3. Train
let loader = try DataLoader(
    source: trainData,
    batchSize: 32,
    shuffle: .full(seed: 42)
)
let history = try await engine.fit(
    model, configuration: config, data: loader,
    epochs: 10,
    validationData: valLoader,
    callbacks: [ProgressReporter(), EarlyStopping(monitor: "val_loss", patience: 3)]
)

// 4. Evaluate & predict
let metrics = try await engine.evaluate(model, configuration: config, data: testLoader)
let output = engine.predict(model, input: newData)

// 5. Save & load
try Checkpointing.saveWeights(model, to: weightsURL)
try Checkpointing.loadWeights(into: restoredModel, from: weightsURL)
```

## Features

| Module | Description |
|---|---|
| **Training Engine** | Typed `fit()` / `evaluate()` / `predict()` over async batch sequences |
| **M5 Training Controls** | Ordered gradient transforms, global-norm clipping, sample-aware accumulation, and explicit sample/class-weighted cross entropy |
| **Losses** | CrossEntropy, BCE, MSE, L1, Huber, KLDiv with configurable reduction |
| **Metrics** | Accuracy, TopK, MSE, and MAE with batch accumulation; `RunningLoss` is a standalone helper |
| **Callbacks** | EarlyStopping, ModelCheckpoint, ProgressReporter, Lambda |
| **Data** | Typed indexed/async sample sources, composable transforms, seeded shuffle, batching, and bounded prefetch |
| **LR Schedules** | CosineDecay, StepDecay, ExponentialDecay, composable Warmup |
| **Serialization** | Versioned weight and inference artifacts; epoch-boundary recovery when an optimizer implements `CheckpointableOptimizer` |
| **Utilities** | `model.printSummary()` with tabular param counts and memory estimates |

## Running Tests

```bash
# Unit tests (no GPU required)
make test

# All tests including GPU (requires Apple Silicon + Metal)
make test-all

# Build only
make build
```

See the [Makefile](Makefile) for all available commands.

## Documentation

Full API documentation is available at **[murilxaraujo.github.io/MLXKit](https://murilxaraujo.github.io/MLXKit/documentation/mlxkit)**.
See the [data pipeline guide](Sources/MLXKit/Documentation.docc/DataPipeline.md) for typed in-memory and streaming batches, shuffle/prefetch semantics, and migration guidance. Run `swift run DataPipelineExample` for a local example.
See the [M2 training semantics](Sources/MLXKit/Documentation.docc/M2TrainingSemantics.md) for scalar objectives, aggregation, schedules, callback lifecycle, cancellation, and failure behavior.
See the [M5 training-control semantics](Sources/MLXKit/Documentation.docc/M5TrainingControlSemantics.md) for ordered gradient transforms, clipping, sample-aware accumulation, weighted cross entropy, and the `TrainingStep` boundary. `MNISTExample` demonstrates standard-path controls; `TinyDecoderExample` demonstrates a custom loss-computation step. Both use local data.
See the [M6 persistence semantics](Sources/MLXKit/Documentation.docc/M6PersistenceSemantics.md) for weight-only artifacts, inference metadata, and the explicit optimizer-state requirements for epoch-boundary recovery.
Read the [MLXKit 2.0 product scope](Sources/MLXKit/Documentation.docc/ProductScope.md) for its target users, product goals, non-goals, and roadmap success criteria.
See the [integrated MLXKit 2.0 API design](Sources/MLXKit/Documentation.docc/API2Design.md) for the full API direction and the data pipeline's public contracts.
See the [platform support matrix](Sources/MLXKit/Documentation.docc/PlatformSupport.md), [compatibility policy](Sources/MLXKit/Documentation.docc/CompatibilityPolicy.md), [MLX responsibility boundary](Sources/MLXKit/Documentation.docc/MLXKitAndMLX.md), and [reference workload plan](Sources/MLXKit/Documentation.docc/ReferenceWorkloads.md) for the M0 product contract.

To build locally:

```bash
make docs
```

## Example

A standalone MNIST-style training example is included:

```bash
swift run MNISTExample
swift run DataPipelineExample
```

`DataPipelineExample` runs both an indexed in-memory workflow and a generated
unknown-length stream with bounded shuffle and prefetch. Both examples use
local synthetic data and do not require network access.

## License

MIT
