# Getting Started with MLXKit

Set up your project and train your first model in minutes.

## Overview

MLXKit is a Swift package that provides a high-level training API on top of
MLX Swift. This guide walks you through adding MLXKit to your project and
training a simple model.

### Adding MLXKit to Your Project

Add MLXKit as a dependency in your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/murilxaraujo/MLXKit", from: "0.1.0"),
]
```

Then add it to your target's dependencies:

```swift
.target(
    name: "MyApp",
    dependencies: ["MLXKit"]
),
```

### Defining a Model

Create a `Module` subclass that conforms to ``TrainableModel``:

```swift
import MLX
import MLXNN
import MLXKit

final class Classifier: Module, TrainableModel {
    let linear1 = Linear(inputDimensions: 784, outputDimensions: 128)
    let linear2 = Linear(inputDimensions: 128, outputDimensions: 10)

    func forward(_ input: MLXArray) -> MLXArray {
        var x = linear1(input)
        x = maximum(x, 0) // ReLU
        return linear2(x)
    }
}
```

### Compiling and Training

Use ``TrainingEngine`` to compile and train:

```swift
let model = Classifier()
let engine = TrainingEngine()

// Compile: bundle optimizer, loss, and metrics
let config = engine.compile(
    model: model,
    optimizer: Adam(learningRate: 1e-3),
    loss: CrossEntropyLoss(),
    metrics: [Accuracy()]
)

// Create a data loader
let dataset = InMemoryDataset(features: trainFeatures, labels: trainLabels)
let loader = DataLoader(dataset: dataset, batchSize: 32, shuffle: true)

// Train
let history = try await engine.fit(
    model, config: config, data: loader,
    epochs: 10,
    callbacks: [ProgressReporter()]
)
```

### Evaluating and Predicting

```swift
// Evaluate on test data
let metrics = try await engine.evaluate(model, config: config, data: testLoader)
print("Test accuracy: \(metrics["accuracy"]!)")

// Predict on new inputs
let output = engine.predict(model, input: newData)
```

### Saving and Loading

```swift
// Save a checkpoint
try Checkpointing.save(model: model, epoch: 10, to: checkpointURL)

// Load into a new model
let restored = Classifier()
let info = try Checkpointing.load(into: restored, from: checkpointURL)
print("Resumed from epoch \(info.epoch)")
```

### Inspecting Your Model

```swift
model.printSummary()
// ──────────────────────────────────────
//  Layer    │ Type   │ Params
// ──────────────────────────────────────
//  linear1  │ Linear │ 100,480
//  linear2  │ Linear │ 1,290
// ──────────────────────────────────────
//  Total params: 101,770
//  Estimated size: 397.5 KB
// ──────────────────────────────────────
```
