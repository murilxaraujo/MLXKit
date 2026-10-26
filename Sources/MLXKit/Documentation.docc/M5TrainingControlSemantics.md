# M5 Training-Control Semantics

**Status:** Gradient transforms, global-norm clipping, sample-aware
accumulation, and weighted cross entropy are implemented. This page extends the
implemented behavior in <doc:M2TrainingSemantics>.

MLXKit continues to coordinate training. MLX owns tensor operations and
autodiff, and MLXOptimizers owns optimizer algorithms and state. The engine
continues to consume an asynchronous sequence of complete, typed ``Batch``
values.

## Training-step and control boundaries

``TrainingStep`` remains a loss-computation hook. It receives the model and a
complete batch and returns one scalar objective. It may replace the configured
objective for a `fit` or `evaluate` call, but it does not replace autodiff,
gradient processing, the optimizer update, callbacks, or the engine lifecycle.
Custom optimizer/update algorithms remain a direct-MLX/MLXOptimizers use case.

M5 controls apply only during `fit`:

- Gradient transforms and clipping process the gradient tree immediately before
  an optimizer update.
- Accumulation groups consecutive input batches into one optimizer update.
- Weighted cross entropy is an objective-level operation. The engine does not
  inspect or infer the layout of `Batch.target`.

`evaluate` computes the configured objective (including weighting explicitly
implemented by that objective) and metrics. It does not differentiate, apply
gradient transforms, clip, accumulate, read the learning-rate schedule, or
update optimizer state. `predict` is unchanged.

## Gradient and update order

For every input batch, the engine computes and validates the scalar objective,
then obtains its gradient tree with MLX autodiff. Raw objective and gradient
values must be finite. With accumulation disabled, that batch's gradient is
the update gradient. With accumulation enabled, its gradient contributes to
the current window as defined below. At an update boundary, the engine:

1. Divides the accumulated gradient sum by the window's total sample count.
2. Applies configured user gradient transforms in declaration order.
3. Applies optional global-norm clipping to the transformed tree.
4. Validates the final tree and its finite values.
5. Calls `Optimizer.update` once and materializes the model and optimizer
   updates.

Every transform receives and returns the complete gradient tree. It may throw.
After each transform, MLXKit requires the same parameter-tree structure and
leaf shapes as the raw gradients, and finite values. A thrown transform,
changed structure or shape, or non-finite value fails the run before the
affected optimizer update. Clipping uses the L2 norm over all gradient leaves:

```text
globalNorm = sqrt(sum_over_all_leaves(sum(gradient * gradient)))
scale = min(1, maxNorm / globalNorm)
clippedGradient = gradient * scale
```

`maxNorm` must be finite and greater than zero. A gradient below the threshold
is unchanged; a zero-norm tree is also unchanged. A non-finite computed norm
fails before the update. Invalid configuration and invalid gradients fail
clearly; a configured control is never silently ignored. Earlier successful
optimizer updates are not rolled back when a later operation fails.

Gradient transforms and clipping are configured through
``GradientProcessingConfiguration``:

```swift
TrainingConfiguration(
    optimizer: optimizer,
    objective: objective,
    gradientProcessing: GradientProcessingConfiguration(
        transforms: [transformA, transformB],
        globalNormClip: 1.0
    )
)
```

Transforms can throw and are declared as ``GradientTransform``. The default
configuration has no transforms or clipping and uses one batch per update.
Set ``TrainingConfiguration/gradientAccumulationSteps`` to a value greater
than one to accumulate consecutive batches:

```swift
TrainingConfiguration(
    optimizer: optimizer,
    objective: objective,
    gradientAccumulationSteps: 4,
    gradientProcessing: GradientProcessingConfiguration(
        transforms: [transformA],
        globalNormClip: 1.0
    )
)
```

## Sample-aware accumulation

An accumulation window contains the configured number of consecutive input
batches in one epoch. It never spans epochs. Let `g_j` be the gradient of the
scalar objective for batch `j`, and `n_j` its positive `sampleCount`. The
window gradient before user transforms is:

```text
windowGradient = sum(n_j * g_j) / sum(n_j)
```

This matches the gradient of a combined-batch mean for uneven microbatches when
each batch objective is a per-example mean and examples contribute
independently. This is the objective-scaling precondition for sample-aware
equivalence. Objectives that use a sum or another scale must normalize their
own result if they require this equivalence. Accumulation does not promise the
same result as sequential optimizer updates: the optimizer updates once per
window, so its state and learning-rate schedule also advance once. Batch-coupled
operations or stochastic/stateful model behavior can also make separate
microbatches differ from a physically combined batch.

The final incomplete window is flushed on normal epoch exhaustion. If an
`onBatchEnd` callback requests early stop, the current window is also flushed
before validation and `onEpochEnd`; this commits the gradients for every batch
the run accepted before stopping. A complete window is updated as soon as its
last microbatch has been processed. Empty windows do not cause updates.

Pending gradients are discarded, without an extra optimizer update, if the run
is interrupted by task cancellation, a source/transform/validation error, or a
throwing callback before the window is flushed. In particular, a throwing
`onBatchEnd` callback discards an incomplete window; it cannot undo an update
already made for a complete window. A failure from `onEpochEnd` occurs after
the epoch's final window has already been flushed. Successful earlier updates
remain applied in all cases. This is not a rollback or resumable-checkpoint
contract.

## Callbacks, logs, and schedules

Callbacks remain per input batch, not per optimizer update. `state.batch` and
`onBatchBegin`/`onBatchEnd` count input batches. There is no new optimizer-step
callback in M5. Batch metrics continue to observe each microbatch's
pre-update model output. The batch `loss` is that microbatch's scalar objective.

When a batch completes a full window, its optimizer update occurs before that
batch's `onBatchEnd`. For an incomplete final window, `onBatchEnd` runs when
the microbatch has been accumulated; the flush then occurs at normal epoch
exhaustion or after an early-stop request, before validation and
`onEpochEnd`. Thus callback boundaries describe consumed batches, not a promise
that each callback corresponds to an optimizer update.

Epoch loss remains the sample-count-weighted mean of the scalar objectives for
the input batches accepted in a successfully completed or early-stopped epoch:

```text
epochLoss = sum(batchObjective * batch.sampleCount) / sum(batch.sampleCount)
```

For the built-in weighted-mean objective below, this yields the weighted
per-example loss divided by the number of examples across the epoch. Metrics
remain unweighted unless a metric explicitly implements weighting. Validation
loss and metrics retain their existing batch-wise evaluation semantics.

The first optimizer update uses schedule step `0`; the step advances once
after each successful optimizer update, including a flushed partial window.
When a window starts, the engine sets the learning rate from the current
optimizer-update step before the first microbatch's `onBatchBegin`. The same
rate applies to every microbatch in that window, and the schedule is not read
again until the next window starts. With accumulation disabled, this preserves
the current one-schedule-value-per-batch behavior. Evaluation does not read or
advance the schedule. A window that is discarded or fails before the update
does not advance the schedule step, although the configured rate may already
have been set for that attempted window.

## Weighted cross entropy

M5's built-in weighting scope is integer-target cross entropy for ordinary
classification batches: logits have shape `[batch, classes]`, targets have
shape `[batch]`, per-sample weights (when supplied) have shape `[batch]`, and
class weights (when supplied) have shape `[classes]`. Token-level, structured,
probability-target, or otherwise non-matching layouts are not inferred or
silently reshaped by the engine. A caller can express those objectives with an
explicit custom objective or ``TrainingStep``.

`CrossEntropyLoss.weightedMean(predictions:targets:sampleWeights:classWeights:)`
implements the explicit weighted operation. For per-example cross entropy
`loss_i`, sample weight `s_i` (default `1`), and class weight `c_yi` for target
class `y_i` (default `1`), the weighted mean is:

```text
batchObjective = sum_i(s_i * c_yi * loss_i) / sampleCount
```

The denominator is the actual number of examples, not the sum of weights. The
same rule applies to uneven batches, so the existing epoch aggregation gives:

```text
epochLoss = sum_all_examples(s_i * c_yi * loss_i) / totalSampleCount
```

Weights must be finite and nonnegative. Zero weights are valid and contribute
zero to the numerator; an all-zero weight batch has zero loss and gradient.
Class weights multiply the per-example loss after the configured
label-smoothing calculation. The weighted built-in path is a scalar mean;
weighted `.sum` and `.none` modes are not part of this contract. Without
weights, existing `CrossEntropyLoss` reductions remain unchanged.

The operation validates the supported ranks and dimensions before the loss
calculation. Invalid weight shapes or values produce a non-finite objective;
`fit` and `evaluate` reject that objective, and `fit` does not apply an
optimizer update for it. The operation requires logits shaped `[batch,
classes]` and integer targets shaped `[batch]`; token-level, structured,
probability-target, and other layouts require an explicit custom operation.

The loss operation accepts weights explicitly. For example, if per-sample
weights are carried alongside labels in a user-defined batch target, a
`TrainingStep` can pass them to `weightedMean`. Use the same `TrainingStep` with
`fit` and `evaluate` to preserve objective semantics in both paths. The engine
does not inspect arbitrary target values to discover weights.
Arbitrary `Objective` and `TrainingStep` implementations remain responsible
for their own weighting, and metrics remain unweighted unless their own API
states otherwise.

## Rejected alternatives and rationale

- **Clip each microbatch before accumulation:** rejected because it changes
  the accumulated gradient direction and magnitude. Accumulation and sample
  normalization happen first; transforms then run once per optimizer update.
- **Clip before user transforms:** rejected because a transform could increase
  the norm after clipping. User transforms run in declared order, followed by
  clipping.
- **Flush pending gradients on cancellation or arbitrary failure:** rejected
  because it would apply an optimizer update after the run was interrupted or
  after an invalid operation. Only normal exhaustion and an explicit early
  stop flush a partial window.
- **Divide weighted loss by the sum of weights:** rejected for this API because
  it changes the specified objective and interacts differently with
  sample-count-based accumulation and epoch reporting. The denominator is the
  actual `sampleCount`.
- **Infer sample/class weights from generic targets or metrics:** rejected
  because targets may be arbitrary Swift values and such inference would be
  ambiguous. Weighting is explicit at the objective boundary.
- **Add a custom optimizer/update hook or parallel optimizer abstraction:**
  rejected because `TrainingStep` already supports custom scalar loss
  computation, while MLXOptimizers owns optimizer algorithms. Use MLX and
  MLXOptimizers directly for custom update algorithms.

## Implementation boundary

This contract does not add distributed training, persistent accumulated
gradients, mixed-precision loss scaling, optimizer algorithms, a generic tensor
tree API, or a new callback lifecycle. The M5 implementation issues add these
controls without changing the M4 source-to-batch boundary or the M2 ownership
split.

Invalid global-norm limits and accumulation-window sizes fail configuration
validation before data iteration with ``TrainingError/invalidGlobalNormClip(_:)``
and ``TrainingError/invalidGradientAccumulationSteps(_:)``. Invalid weighted
cross-entropy shapes or weights produce a non-finite scalar; `fit` and
`evaluate` report ``TrainingError/nonFiniteObjective``. Gradient transform
errors, tree-key changes, shape changes, and non-finite transformed values fail
before the affected update. Earlier successful updates are not rolled back.

`MNISTExample` demonstrates the standard objective path with a gradient
transform, global-norm clipping, and two-batch accumulation. `TinyDecoderExample`
demonstrates a ``TrainingStep`` that adds a scalar penalty to the configured
objective. In both cases, MLXKit retains autodiff and update ownership, and
MLXOptimizers retains optimizer-algorithm ownership. Use direct MLX and
MLXOptimizers APIs when an algorithm needs a custom optimizer update.

Run the deterministic local examples and validation commands with:

```sh
make run-example
make run-tiny-decoder
make run-data-pipeline
make test
make test-gpu
make docs
```

The examples use generated local data and do not download datasets.

Validation record for this integration (2026-10-25): `swift test` passed with
92 tests, `make test-gpu` passed with 178 tests, and `make docs` completed. All
three listed example commands ran successfully on the local Apple Silicon/Metal
environment.
