# Typed data pipeline

MLXKit converts typed samples into typed batches. `DataLoader` accepts an
indexed collection or an asynchronous sample sequence. It applies sample
transforms first, shuffle second, and batching/collation last. The resulting
`AsyncSequence<Batch<Input, Target>>` works with the existing
``TrainingEngine/fit(_:configuration:data:epochs:resumeFrom:trainingStep:callbacks:)`` and
``TrainingEngine/evaluate(_:configuration:data:trainingStep:)`` sequence
boundary.

For a runnable, network-free example that covers both indexed data and a
generated stream, run:

```sh
swift run DataPipelineExample
```

The existing in-memory classifier example is also local and synthetic:

```sh
swift run MNISTExample
```

## In-memory array data

Use ``InMemoryDataset`` for the concise `MLXArray`-to-`MLXArray` workflow. Full
shuffle creates a seeded permutation of indices, then the built-in collator
stacks each batch:

```swift
let dataset = InMemoryDataset(features: trainInputs, labels: trainTargets)
let batches = try DataLoader(
    source: dataset,
    batchSize: 32,
    shuffle: .full(seed: 42),
    dropLast: false
)

let history = try await engine.fit(
    model,
    configuration: configuration,
    data: batches,
    epochs: 5
)
```

Full shuffle is only valid for indexed sources. It retains the data in place
and shuffles indices; its index permutation uses memory proportional to the
number of samples.

## Unknown-length streams

An asynchronous source only needs to yield `Sample` values. It does not need a
count or random-access interface. Sample transforms are ordinary async-sequence
`map` and `filter` operations. The loader applies its shuffle strategy after
those transforms and before batching:

```swift
struct GeneratedSamples: AsyncSequence {
    typealias Element = Sample<MLXArray, MLXArray>
    typealias Failure = Never

    func makeAsyncIterator() -> Iterator { Iterator() }

    struct Iterator: AsyncIteratorProtocol {
        typealias Element = Sample<MLXArray, MLXArray>
        typealias Failure = Never
        var value: Int32 = 0

        mutating func next() async -> Element? {
            guard value < 100 else { return nil }
            defer { value += 1 }
            let input = MLXArray([Float32(value)])
            return Sample(input: input, target: input * 2)
        }
    }
}

let transformed = GeneratedSamples()
    .map { sample in Sample(input: sample.input, target: sample.target) }
    .filter { $0.input.item(Float.self) >= 0 }

let batches = try DataLoader(
    source: transformed,
    batchSize: 16,
    shuffle: .buffered(windowSize: 64, seed: 42),
    collate: { samples in
        Batch(
            input: stacked(samples.map(\.input)),
            target: stacked(samples.map(\.target)),
            sampleCount: samples.count
        )
    }
)
let prefetched = try batches.prefetched(bufferSize: 2)
let metrics = try await engine.evaluate(
    model,
    configuration: configuration,
    data: prefetched
)
```

The example source has no `count` property and generates each value when the
iterator requests it. The buffered shuffle holds at most its configured window
of samples. Prefetch holds at most its configured number of unconsumed batches,
including a source read in progress. Neither stage materializes the complete
stream. The collator receives a final partial batch unless `dropLast` is true,
and `sampleCount` must equal the number of samples the collator received.

A source used for multiple training epochs must create a fresh iterator with
fresh data from `makeAsyncIterator()`. A one-shot stream is not replayed
automatically. Use a reiterable source or wrap a source factory when each epoch
must start again.

## Ordering, errors, and cancellation

- The order is **source → transforms → shuffle → batching/collation → optional
  prefetch**. Transforms therefore run before the shuffle strategy.
- `.full(seed:)` uses a seeded index permutation and requires a random-access
  source. `.buffered(windowSize:seed:)` keeps a bounded replacement window and
  supports asynchronous sources. For the same source order, seed, and
  implementation version, the shuffle order is deterministic. Cross-version
  and cross-device shuffle identity is not guaranteed.
- Empty sources produce no batches. A partial final batch is emitted by
  default; `dropLast: true` discards it. The loader checks every collator's
  reported `sampleCount`.
- Source, transform, and collator errors propagate from iteration and from
  `fit`/`evaluate`; they are not converted to end-of-data.
- Prefetch preserves batch order. Its finite `bufferSize` is a bound on
  unconsumed batches; capacity is reserved before advancing upstream, so an
  in-flight read counts toward that bound.
- Cancelling the consumer or dropping its iterator cancels producer work and
  releases buffered values. Prompt shutdown of an in-flight source read depends
  on that source cooperating with task cancellation.

## Migration from the array-only API

The data API now uses generic `Sample<Input, Target>` and
`Batch<Input, Target>` values. The existing `Dataset` protocol is limited to
indexed `MLXArray` samples; use `InMemoryDataset` for the common array path, or
provide any `AsyncSequence<Sample<Input, Target>>` for generic and streaming
data.

Replace Boolean shuffle and dataset-labeled loader construction:

```swift
// Previous array-oriented spelling:
let oldLoader = DataLoader(dataset: dataset, batchSize: 32, shuffle: true)

// Current indexed array path:
let loader = try DataLoader(
    source: InMemoryDataset(features: features, labels: labels),
    batchSize: 32,
    shuffle: .full(seed: 42)
)
```

The replacement requires `try` because invalid batch sizes and shuffle
configurations fail during construction. Use `.none` for source order, a seeded
`.full(seed:)` for indexed data, or `.buffered(windowSize:seed:)` for streams.
There is no implicit Boolean-shuffle compatibility overload.

For structured data, replace fixed `features`/`labels` fields with typed samples
and provide a collator:

```swift
let records: [Sample<Record, Label>] = loadLocalRecords()
let batches = try DataLoader(source: records, batchSize: 16) { samples in
    let inputs = try pad(samples.map(\.input))
    let targets = stackLabels(samples.map(\.target))
    return Batch(input: inputs, target: targets, sampleCount: samples.count)
}
```

The collator owns padding and structure. No global tensor-tree or batching
protocol is required. For the complete model, objective, and training API
migration table, see <doc:API2Design>.
