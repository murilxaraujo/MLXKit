# MLXKit

A high-level Swift framework for machine learning on Apple silicon, built on [MLX Swift](https://github.com/ml-explore/mlx-swift).

MLXKit provides a Keras-style training API: define a model, compile it with an optimizer and loss, call `fit()`. It handles gradient computation, metric tracking, callbacks, learning rate scheduling, and checkpointing.

## Requirements

- macOS 14+ / iOS 17+
- Swift 6.2+
- Apple Silicon (M1 or later)

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

    func forward(_ input: MLXArray) -> MLXArray {
        var x = linear1(input)
        x = maximum(x, 0)
        return linear2(x)
    }
}

// 2. Compile
let model = Classifier()
let engine = TrainingEngine()
let config = engine.compile(
    model: model,
    optimizer: Adam(learningRate: 1e-3),
    loss: CrossEntropyLoss(),
    metrics: [Accuracy()],
    lrSchedule: CosineDecay(initialLR: 1e-3, totalSteps: 10000)
)

// 3. Train
let loader = DataLoader(dataset: trainData, batchSize: 32, shuffle: true)
let history = try await engine.fit(
    model, config: config, data: loader,
    epochs: 10,
    validationData: valLoader,
    callbacks: [ProgressReporter(), EarlyStopping(monitor: "val_loss", patience: 3)]
)

// 4. Evaluate & predict
let metrics = try await engine.evaluate(model, config: config, data: testLoader)
let output = engine.predict(model, input: newData)

// 5. Save & load
try Checkpointing.save(model: model, epoch: 10, to: checkpointURL)
let info = try Checkpointing.load(into: restoredModel, from: checkpointURL)
```

## Features

| Module | Description |
|---|---|
| **Training Engine** | Keras-style `compile()` / `fit()` / `evaluate()` / `predict()` |
| **Losses** | CrossEntropy, BCE, MSE, L1, Huber, KLDiv with configurable reduction |
| **Metrics** | Accuracy, TopK, MSE, MAE, RunningLoss with batch accumulation |
| **Callbacks** | EarlyStopping, ModelCheckpoint, ProgressReporter, TerminateOnNaN, Lambda |
| **Data** | `Dataset` protocol, `InMemoryDataset`, `AsyncSequence`-based `DataLoader` |
| **LR Schedules** | CosineDecay, StepDecay, ExponentialDecay, composable Warmup |
| **Serialization** | Weight save/load and full checkpoint (model + epoch + metadata) via safetensors |
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

Documentation is available as a [DocC catalog](Sources/MLXKit/Documentation.docc/).

To build locally:

```bash
make docs
```

## Example

A standalone MNIST-style training example is included:

```bash
swift run MNISTExample
```

## License

MIT
