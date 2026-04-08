# ``MLXKit``

A high-level Swift framework for machine learning on Apple silicon, built on MLX.

## Overview

MLXKit provides a Keras-style training API on top of
[MLX Swift](https://github.com/ml-explore/mlx-swift). Define a model,
compile it with an optimizer and loss, then call `fit()` — MLXKit handles
gradient computation, metric tracking, callbacks, learning rate scheduling,
and checkpointing.

```swift
let model = MyModel()
let engine = TrainingEngine()
let config = engine.compile(
    model: model,
    optimizer: Adam(learningRate: 1e-3),
    loss: CrossEntropyLoss(),
    metrics: [Accuracy()]
)

let history = try await engine.fit(
    model, config: config, data: trainLoader,
    epochs: 10, callbacks: [.progressReporter, .earlyStopping]
)
```

## Topics

### Essentials

- <doc:GettingStarted>
- <doc:Architecture>

### Core Training

- ``TrainableModel``
- ``TrainingEngine``
- ``TrainingConfiguration``
- ``TrainingError``

### Losses

- ``LossFunction``
- ``Reduction``
- ``CrossEntropyLoss``
- ``BinaryCrossEntropyLoss``
- ``MSELoss``
- ``L1Loss``
- ``HuberLoss``
- ``KLDivLoss``

### Metrics

- ``Metric``
- ``Accuracy``
- ``TopKAccuracy``
- ``MeanSquaredError``
- ``MeanAbsoluteError``
- ``RunningLoss``

### Data Loading

- ``Dataset``
- ``Sample``
- ``Batch``
- ``InMemoryDataset``
- ``DataLoader``

### Callbacks

- ``Callback``
- ``CallbackList``
- ``TrainingState``
- ``History``
- ``EarlyStopping``
- ``ModelCheckpoint``
- ``ProgressReporter``
- ``TerminateOnNaN``
- ``LambdaCallback``

### Learning Rate Schedules

- ``LRSchedule``
- ``CosineDecay``
- ``StepDecay``
- ``ExponentialDecay``
- ``WarmupSchedule``

### Serialization

- ``Checkpointing``
- ``CheckpointError``

### Utilities

- ``ModelSummary``
- ``SummaryRow``
