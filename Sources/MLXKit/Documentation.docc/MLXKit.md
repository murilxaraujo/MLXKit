# ``MLXKit``

A high-level Swift framework for machine learning on Apple silicon, built on MLX.

## Overview

MLXKit provides a typed training workflow on top of
[MLX Swift](https://github.com/ml-explore/mlx-swift). Define a model and a
typed ``TrainingConfiguration``, then call `fit()` — MLXKit coordinates
gradient computation, metric tracking, callbacks, learning-rate scheduling,
and checkpointing while MLX and MLXOptimizers provide the numerical core.

```swift
let model = MyModel()
let engine = TrainingEngine()
let config = TrainingConfiguration(
    optimizer: Adam(learningRate: 1e-3),
    objective: CrossEntropyLoss(),
    metrics: [Accuracy()]
)

let history = try await engine.fit(
    model, configuration: config, data: trainLoader,
    epochs: 10, callbacks: [.progressReporter, .earlyStopping]
)
```

## Topics

### Essentials

- <doc:ProductScope>
- <doc:API2Design>
- <doc:MLXKitAndMLX>
- <doc:PlatformSupport>
- <doc:CompatibilityPolicy>
- <doc:ReferenceWorkloads>
- <doc:M2TrainingSemantics>
- <doc:M5TrainingControlSemantics>
- <doc:CustomModelState>
- <doc:TinyDecoderExample>
- <doc:M1ModelContract>
- <doc:M1ObjectiveMetricContract>
- <doc:M1TrainingWorkflow>
- <doc:M1DataContract>
- <doc:DataPipeline>
- <doc:GettingStarted>
- <doc:Architecture>

### Core Training

- ``TrainableModel``
- ``TrainingEngine``
- ``TrainingConfiguration``
- ``TrainingError``
- ``GradientProcessingConfiguration``
- ``GradientTransform``

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
- ``DataLoaderError``
- ``ShuffleStrategy``
- ``PrefetchedSequence``
- ``PrefetchError``

### Callbacks

- ``Callback``
- ``CallbackList``
- ``TrainingState``
- ``History``
- ``EarlyStopping``
- ``ModelCheckpoint``
- ``ProgressReporter``
- ``LambdaCallback``

### Learning Rate Schedules

- ``LRSchedule``
- ``LearningRateControllable``
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
