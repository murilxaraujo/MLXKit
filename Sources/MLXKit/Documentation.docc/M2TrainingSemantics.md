# M2 Training Semantics

This page records behavior implemented by the typed M2 training engine. MLXKit
coordinates the workflow; MLX and MLXOptimizers remain responsible for tensor
semantics, autodiff, and optimizer algorithms.

## Objective and epoch loss

Each training objective must be an `MLXArray` scalar with shape `[]`. A shape
such as `[1]` is not a scalar. The engine does not choose a reduction. Use a
mean-reduced built-in loss or reduce explicitly in a custom objective or
``TrainingStep``. MLXKit reports `TrainingError.nonScalarObjective` for a
non-scalar result. MLX operator errors caused by incompatible shape or dtype
remain MLX runtime errors.

At epoch end, `loss` is the `sampleCount`-weighted mean of the scalar objective
reported for each successfully updated batch:

```text
epochLoss = sum(batchObjective * batch.sampleCount) / sum(batch.sampleCount)
```

This is the dataset mean when the objective is a per-batch mean. If an objective
uses a different scale, the epoch log preserves that objective's scale and
weights the batch values by sample count.

## Metrics and logs

Metrics reset before every training epoch and evaluation run. They update once
per batch and compute their final value at the reporting boundary. Training
metrics receive the model output and target from before that batch's optimizer
update. Evaluation metrics receive outputs from inference mode.

The engine reserves `loss` and `val_` log keys. Metric names must be non-empty,
unique, and must not equal `loss` or start with `val_`. Invalid names fail
before training starts. Validation logs use `val_loss` and `val_<metricName>`.

## Learning-rate schedules

The first scheduled optimizer update uses step `0`. The step advances once
after each successful optimizer update and is shared across epochs. Evaluation
does not read the schedule or mutate the optimizer's rate. An optimizer paired
with a schedule must conform to ``LearningRateControllable`` or the engine
throws `TrainingError.unsupportedLearningRateSchedule` before training.

## Callbacks, cancellation, and failures

Callbacks run synchronously and serially in this order on successful training:

```text
train begin
  epoch begin
    batch begin → update → batch end (for each batch)
  epoch end
train end
```

The engine places its `History` callback before user callbacks. A callback can
request early stop at `onBatchEnd` or `onEpochEnd`. A batch-end stop finishes
that batch, validation, and epoch-end dispatch; history contains that completed
epoch and no later epochs.

Callback errors propagate to the caller. Once train-begin dispatch starts,
train-end dispatch runs exactly once on success, early stop, cancellation, a
sequence error, or a callback error. If train-end also throws while another
error is active, the original error takes precedence.

Task cancellation propagates as `CancellationError`. The engine checks
cancellation before batch work, after callbacks, and after sequence completion.
It does not start another optimizer update after observing cancellation. An
upstream sequence's own cancellation and cleanup behavior is outside MLXKit's
control.

Non-finite training objectives and gradients throw
`TrainingError.nonFiniteObjective` and `TrainingError.nonFiniteGradient` before
an optimizer update. The former `TerminateOnNaN` callback is removed; stopping
after an invalid update is not treated as successful training.

## Empty input and model mode

`fit` requires a positive epoch count, positive `sampleCount` for each batch,
and at least one training batch per epoch. `evaluate` requires positive batch
sample counts and at least one batch. These failures use `TrainingError` cases
such as `invalidEpochCount`, `invalidSampleCount`, `emptyTrainingData`, and
`emptyEvaluationData`. Errors thrown by the input `AsyncSequence` propagate
unchanged. Data-backed prediction returns an empty array for an empty sequence.

`fit` runs epochs in training mode and restores the model's prior mode on
success or failure. `evaluate` and `predict` run in inference mode and restore
the prior mode before returning or throwing.

## M2 boundary

M2 consumes typed asynchronous sequences of already-built batches. It does not
implement M4 sample sources, transforms, collation, streaming shuffle,
prefetch, or backpressure. M5 gradient controls and multi-update training,
and M6 distributed execution, are also deferred. MLXKit validates the scalar
objective contract and its own configuration; MLX continues to define runtime
array shape, dtype, and device behavior.

The prospective M5 control ordering, accumulation, weighting, and failure
contracts are recorded in <doc:M5TrainingControlSemantics>. They do not change
the implemented M2 behavior until M5 implementation is complete.
