# M6 Persistence and Recovery Semantics

MLXKit uses versioned safetensors artifacts. The artifact kind defines what is
stored and what a successful load guarantees. A successful weight load is not a
training resume.

## Artifact kinds

### Weights

``Checkpointing/saveWeights(_:to:)`` stores only an MLX module's trainable
parameters. ``Checkpointing/loadWeights(into:from:)`` checks that parameter
keys, shapes, and dtypes match before it updates the destination model. It does
not restore optimizer state, training progress, a learning-rate schedule, or
application state.

Legacy unversioned weight files remain readable. New artifacts carry the
`mlxkit.artifact` format identifier and a format version. Unknown identifiers or
versions fail rather than being interpreted as weights.

### Inference artifacts

An inference artifact is a versioned weight artifact with caller-provided string
metadata:

```swift
try Checkpointing.saveInferenceArtifact(
    model,
    metadata: ["architecture": "Classifier-v1", "labels": "cat,dog"],
    to: artifactURL
)

let artifact = try Checkpointing.loadInferenceArtifact(
    into: restoredModel,
    from: artifactURL
)
print(artifact.metadata["architecture"] ?? "unknown")
```

The metadata does not contain executable model code or guarantee that the model
can be reconstructed. The application owns model construction, tokenizer and
label files, preprocessing, and other deployment assets. Inference caches are
not persisted.

### Training progress metadata

``Checkpointing/saveTrainingProgress(model:progress:metadata:to:)`` stores model
parameters and completed-epoch/update counters. It is useful for reporting or
application-managed restarts, but does **not** restore an optimizer and must not
be treated as a faithful training-resume checkpoint. The older `save(model:epoch:step:to:)`
and `load(into:from:)` methods remain as deprecated compatibility wrappers with
this same limited guarantee.

## Resumable training checkpoints

``Checkpointing/saveTrainingCheckpoint(model:optimizer:progress:metadata:to:)-(_,CheckpointableOptimizer,_,_,_)``
and ``Checkpointing/loadTrainingCheckpoint(into:optimizer:from:)-(_,CheckpointableOptimizer,_)`` store model
parameters, optimizer-owned state, and ``TrainingProgress``. The optimizer must
conform to ``CheckpointableOptimizer``. MLXOptimizers currently exposes
optimizer state for evaluation but does not expose a general restore API, so its
built-in optimizers do not conform automatically. A custom optimizer can opt in
when it can serialize and validate all state needed by its update algorithm.

The resume boundary is the end of a fully consumed epoch. Construct progress
with the count of completed epochs and successful optimizer updates. Pending
gradient-accumulation windows are flushed at epoch exhaustion by
``TrainingEngine``; a callback-requested stop during a batch is not a completed
epoch. ``TrainingState/epochCompleted`` lets callbacks distinguish these cases.

For example, after loading a checkpoint, pass the restored progress to `fit`:

```swift
let restored = try Checkpointing.loadTrainingCheckpoint(
    into: model,
    optimizer: optimizer,
    from: checkpointURL
)

let history = try await TrainingEngine().fit(
    model,
    configuration: configuration,
    data: makeTrainingBatches(), // Recreate the source at the next epoch.
    epochs: 20,                  // Total target epoch count, not extra epochs.
    resumeFrom: restored.progress
)
```

The caller must provide the same model/objective and compatible optimizer and
learning-rate schedule configuration. The schedule is evaluated from the
restored optimizer-step count. An optimizer conformer is responsible for
validating its identifier, hyperparameters, and state before changing itself.
`History` returned by a resumed `fit` contains only epochs run by that call.
MLXKit does not capture arbitrary `AsyncSequence` iterator position, callback
state, random-generator state, or application-owned state. It does not promise
mid-epoch or cross-device bitwise reproducibility.

## Validation and compatibility

New loads validate artifact kind/version and exact trainable parameter keys,
shapes, and dtypes before updating the model. Training loads also check the
optimizer identifier and progress metadata before restoring. A checkpointable
optimizer must make its own restore validation atomic: a thrown validation
error must not leave partially changed optimizer state. Files are written to a
temporary sibling and published only after serialization succeeds.

Format version 1 is the current format. This release does not promise
cross-version compatibility beyond the documented legacy weight/progress
readers. A future format change needs an explicit migration path and tests
before compatibility is promised.

## Scope boundary

MLXKit does not serialize executable model definitions, inference caches,
arbitrary application metadata beyond string key/value pairs, arbitrary
asynchronous source state, or distributed checkpoints. A caller-provided
metadata field can describe external assets; it does not package those assets.

## Validation record

On 2026-10-25, `swift test` passed (92 tests), `make test-gpu` passed (188
tests), and `make docs` completed without DocC warnings. The supervised
classifier example also completed its inference-artifact save/load path on
Apple Silicon/Metal.
