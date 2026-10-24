# Architecture

MLXKit is a small coordination layer over MLX Swift and MLXOptimizers. It
provides typed model I/O, scalar objectives, metric aggregation, callbacks,
learning-rate schedules, and checkpoint helpers. MLX remains responsible for
tensor operations, autodiff, model parameters, and device execution.

## Training flow

1. Define a ``TrainableModel`` by declaring `Input` and `Output` and
   implementing `callAsFunction(_:)`.
2. Create a ``TrainingConfiguration`` with an MLXOptimizers optimizer, an
   objective, optional typed metrics, and an optional learning-rate schedule.
3. Call ``TrainingEngine/fit(_:configuration:data:epochs:trainingStep:callbacks:)``
   with an asynchronous sequence of typed ``Batch`` values. The engine
   computes MLX gradients and applies one optimizer update per valid batch.
4. Optionally provide a validation sequence. The engine logs `val_loss` and
   prefixes validation metrics with `val_` after each epoch.
5. Call ``TrainingEngine/evaluate(_:configuration:data:trainingStep:)`` for
   inference-mode loss and metrics, or ``TrainingEngine/predict(_:input:)``
   for typed inference.

## Component contracts

| Component | Responsibility |
|---|---|
| ``TrainableModel`` | Typed model input and complete output, backed by an MLX `Module`. |
| ``Objective`` / ``LossFunction`` | Compute a scalar differentiable objective. |
| ``Metric`` | Accumulate a named value across batches and reset at run boundaries. |
| ``Callback`` | Observe training lifecycle synchronously; callback errors propagate. |
| ``LRSchedule`` | Map successful optimizer-update steps to learning rates. |
| ``Dataset`` / ``DataLoader`` | Existing array-backed sample batching for MLX arrays. |
| ``Checkpointing`` | Persist and restore MLX model weights. |

## M2 sequence boundary and deferred data pipeline

M2 training accepts an `AsyncSequence` whose elements are already typed
``Batch`` values. ``InMemoryDataset`` and ``DataLoader`` remain the current
array-backed convenience path. The generalized typed sample-source loader,
collation, transforms, bounded shuffle, prefetch, and backpressure are deferred
to M4; see <doc:M1DataContract> for that design boundary.

For scalar reduction, epoch aggregation, metric timing, schedules, lifecycle,
failure behavior, and model-mode restoration, see <doc:M2TrainingSemantics>.
