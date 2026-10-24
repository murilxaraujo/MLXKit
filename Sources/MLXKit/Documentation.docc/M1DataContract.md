# M1 Data Contract — Working Draft

**Status:** pair-approved direction for issue #11: sources yield generic
samples; batching is an explicit pipeline stage; structured samples use a
throwing collate closure rather than a required tensor-tree or `Batchable`
protocol; random access is optional; indexed and streaming sources use
different explicit seeded-shuffle strategies. Exact public names and
signatures remain part of the integrated M1 design.

## Data model

The model contract accepts user-defined input and output types. The data API
must therefore carry input and target types without imposing a single array
layout:

```swift
struct Sample<Input, Target> {
    let input: Input
    let target: Target
}

struct Batch<Input, Target> {
    let input: Input
    let target: Target
    let sampleCount: Int
}
```

`Sample` is one logical example. `Batch` contains the user/model inputs and
targets for one training or evaluation update, plus its actual example count.
`input` and `target` may be `MLXArray` or user-defined Swift values. This
contract does not require an MLXKit tensor-tree protocol. `sampleCount` is
explicit because the final batch may be smaller than the configured size and
training/metric aggregation must know its size.

## Source and batch pipeline

The base source contract is an asynchronous sequence of samples. A source does
not need a known count or random access:

```swift
let source: some AsyncSequence<Sample<EncoderInput, TokenTargets>, any Error> = ...
```

The pipeline applies transforms to samples and then groups them into batches.
The training engine consumes the resulting asynchronous sequence of typed
batches. Sources that fail can throw from iteration; their errors propagate to
the caller of the asynchronous throwing training/evaluation operation.

Indexed data is an optional source capability, not a requirement on every
source. An in-memory array-backed source can provide fast indexing, length, and
the convenient common-case collator. A stream, file reader, or generator can
yield samples with an unknown final length and must not materialize the full
source to participate in training.

## Batching and collation

Batching custom data uses a throwing collate closure:

```swift
let batches = source.batches(of: 32, dropLast: false) { samples in
    Batch(
        input: EncoderInput(
            tokenIDs: stack(samples.map { $0.input.tokenIDs }),
            attentionMask: stack(samples.map { $0.input.attentionMask })
        ),
        target: stack(samples.map(\.target)),
        sampleCount: samples.count
    )
}
```

The closure owns the conversion from a set of logical examples to the model's
batched `Input` and the objective's batched `Target`. It can stack arrays,
preserve nested structure, pad variable-length sequences, or reject malformed
examples by throwing. The common `MLXArray`/`MLXArray` in-memory path has a
built-in batching operation and does not require a custom closure.

A required `Batchable` conformance is rejected as the core contract: the same
sample type may be batched differently for padding, bucketing, multi-view
training, or evaluation. An explicit collate closure keeps that choice at the
pipeline call site and avoids another global structural protocol.

Batch size, incomplete-final-batch handling, ordering of mapping/filtering/
batching, and the exact collate-error diagnostics are configured at the
pipeline boundary. M4 owns their precise runtime semantics and tests.

## In-memory and streaming examples

The usual array-backed classifier remains concise:

```swift
let trainingBatches = InMemoryDataset(
    inputs: imageArrays,
    targets: classIDs
).batches(
    of: 32,
    shuffle: .full(seed: 42),
    dropLast: false
)
```

Structured in-memory samples supply their own collator. A streaming source can
use the same pipeline without declaring a length or loading all examples first:

```swift
let stream: AsyncThrowingStream<Sample<EncoderInput, TokenTargets>, Error> =
    tokenizedExampleStream()

let trainingBatches = stream
    .map(normalizeTokens)
    .filter(isUsable)
    .shuffle(.buffered(windowSize: 512, seed: 42))
    .batches(of: 32, dropLast: false, collate: collateTokenExamples)
    .prefetched(bufferSize: 2)
```

These signatures are API sketches; the source and transform wrapper names are
not final. They show that transformation order is explicit and that prefetch
has a visible finite bound.

## Shuffle and reproducibility

Random access is optional, so shuffle policy is explicit:

- Indexed sources can use a full seeded permutation without loading or
  reshuffling tensor data.
- Streaming sources use a bounded seeded shuffle buffer/window. This is an
  approximate streaming shuffle; it must not silently read the entire stream
  into memory.
- A seed is user-configurable. M4 documents the reproducibility scope,
  including source order, seed, implementation/version, and limits across
  devices or releases.

The pipeline makes shuffle position relative to mapping, filtering, and batching
visible. M4 defines ordering guarantees and verifies them with deterministic
tests.

## Cancellation, errors, and prefetch

- Source iteration is asynchronous and may throw. Upstream failures and
  collate/transform failures propagate through `fit`/`evaluate`; they are not
  treated as end-of-data or successful completion.
- Task cancellation is propagated as `CancellationError`. M4 tests that
  cancellation releases upstream iteration and does not leave prefetch work
  unbounded.
- Prefetch is opt-in or explicitly configured with a finite capacity. It must
  apply backpressure and stop requesting upstream work when cancelled.
- Unknown length is valid. APIs expose counts only when the source can provide
  them; training must not require `totalBatches` to be known.

M4 defines exact error types, cancellation checkpoints, buffer bounds,
backpressure behavior, and cleanup guarantees.

## Alternatives and rationale

- **Require random-access collections:** rejected. It excludes streams and
  forces full materialization.
- **Require every source to yield prebuilt batches:** rejected as the only
  abstraction. Sample-level transforms and a common in-memory convenience path
  need a sample source; custom prebatched pipelines may still adapt to the
  trainer's batch-sequence boundary.
- **Require a tensor-tree/batching protocol:** rejected. It narrows the custom
  input promise and forces one batching strategy per type.
- **Implicit full shuffle for all sources:** rejected. It can materialize an
  unbounded stream and gives misleading memory expectations.

## Deferred to M4 and the integrated M1 design

- M4 implements and tests transforms, collation, partial batches, indexed and
  bounded streaming shuffle, prefetch/backpressure, errors, and cancellation.
- Issue #8 aligns the batch sequence with the `TrainingStep<Model, Batch>`
  extension point.
- The integrated issue #12 selects final names and Swift generic/throwing
  signatures and adds migration guidance from the 1.x `Dataset` and
  `DataLoader` APIs.
