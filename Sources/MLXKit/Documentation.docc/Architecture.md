# Architecture

An overview of MLXKit's design and how the components fit together.

## Overview

MLXKit follows a modular, protocol-oriented design inspired by Keras.
Each concern — losses, metrics, callbacks, data loading, scheduling,
and serialization — is encapsulated behind a protocol with concrete
implementations.

### Component Stack

The training engine sits at the center, orchestrating all components:

```
┌─────────────────────────────────────────────┐
│              TrainingEngine                  │
│                                             │
│  ┌───────────┐  ┌──────────┐  ┌──────────┐ │
│  │  Losses   │  │ Metrics  │  │Callbacks │ │
│  └───────────┘  └──────────┘  └──────────┘ │
│  ┌───────────┐  ┌──────────┐  ┌──────────┐ │
│  │DataLoader │  │LRSchedule│  │Checkpoint│ │
│  └───────────┘  └──────────┘  └──────────┘ │
└─────────────────────────────────────────────┘
```

### Training Flow

1. **Define** a ``TrainableModel`` — any `Module` with a `forward()` method.
2. **Compile** with ``TrainingEngine/compile(model:optimizer:loss:metrics:lrSchedule:)``
   to bundle the optimizer, loss function, metrics, and LR schedule into a
   ``TrainingConfiguration``.
3. **Fit** with ``TrainingEngine/fit(_:config:data:epochs:validationData:callbacks:)``
   which runs the training loop:
   - Callback lifecycle: `onTrainBegin` → epoch → batch → `onTrainEnd`
   - Gradient computation via `valueAndGrad` (JAX pattern)
   - Optimizer parameter updates
   - Metric accumulation and LR scheduling
   - Optional validation at epoch end
4. **Evaluate** with ``TrainingEngine/evaluate(_:config:data:)``
   for inference-mode metrics.
5. **Predict** with ``TrainingEngine/predict(_:input:)`` for single-input inference.

### Protocol Contracts

| Protocol | Purpose | Key Method |
|---|---|---|
| ``TrainableModel`` | Model with forward pass | `forward(_:) -> MLXArray` |
| ``LossFunction`` | Differentiable loss | `callAsFunction(predictions:targets:)` |
| ``Metric`` | Stateful accumulator | `update()` / `compute()` / `reset()` |
| ``Callback`` | Lifecycle hooks | `onEpochEnd(state:)`, etc. |
| ``Dataset`` | Indexed data source | `RandomAccessCollection` conformance |
| ``LRSchedule`` | Step-based LR | `learningRate(atStep:)` |

### Gradient Computation

MLXKit uses MLX Swift's `valueAndGrad(model:_:)` to compute gradients
in a functional style. The training step:

1. Forward pass through the model
2. Compute scalar loss
3. Automatic differentiation produces parameter gradients
4. Optimizer applies gradients to update model parameters
5. `eval()` materializes lazy computation

### Data Pipeline

``DataLoader`` is an `AsyncSequence` that yields ``Batch`` values from
any ``Dataset``. It supports batching, shuffling, and `dropLast` for
incomplete final batches. ``InMemoryDataset`` provides a simple
array-backed implementation.

### Serialization

``Checkpointing`` uses MLX's safetensors format for weight storage.
Full checkpoints include training metadata (epoch, step) in the
safetensors metadata dictionary, enabling seamless training resumption.
