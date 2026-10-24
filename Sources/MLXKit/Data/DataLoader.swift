import Foundation
import MLX

/// Selects a deterministic shuffle strategy for a sample source.
public enum ShuffleStrategy: Sendable, Equatable {
    /// Preserve source order.
    case none

    /// Shuffle all indices of an indexed source with the supplied seed.
    case full(seed: UInt64)

    /// Shuffle a stream with a bounded replacement window.
    case buffered(windowSize: Int, seed: UInt64)
}

/// Configuration errors raised while constructing a ``DataLoader``.
public enum DataLoaderError: Error, LocalizedError, Equatable {
    case invalidBatchSize(Int)
    case invalidShuffleWindowSize(Int)
    case fullShuffleRequiresIndexedSource
    case collatedSampleCountMismatch(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidBatchSize(let value):
            "Batch size must be greater than zero; received \(value)."
        case .invalidShuffleWindowSize(let value):
            "Shuffle window size must be greater than zero; received \(value)."
        case .fullShuffleRequiresIndexedSource:
            "Full shuffle requires an indexed source. Use buffered shuffle for an asynchronous stream."
        case .collatedSampleCountMismatch(let expected, let actual):
            "The collator reported sampleCount \(actual), but the batch contains \(expected) samples."
        }
    }
}

/// An asynchronous sequence that groups typed samples into typed batches.
///
/// Sources can be transformed with `AsyncSequence.map` and `filter` before
/// they are passed to the loader. The loader applies its shuffle strategy to
/// those samples, then batches and collates them. A collator must report the
/// actual number of samples in the returned batch; a mismatch fails iteration.
///
/// ```swift
/// let loader = try DataLoader(
///     source: dataset,
///     batchSize: 32,
///     shuffle: .full(seed: 42)
/// )
/// for try await batch in loader {
///     let output = model(batch.input)
/// }
/// ```
public struct DataLoader<Input, Target>: AsyncSequence {
    public typealias Element = Batch<Input, Target>
    public typealias Failure = any Error

    public let batchSize: Int
    public let shuffle: ShuffleStrategy
    public let dropLast: Bool

    private let iteratorFactory: () -> AnyAsyncIterator<Element>

    /// Creates a loader for an asynchronous sample source.
    ///
    /// Use `.buffered(windowSize:seed:)` or `.none` for asynchronous sources.
    /// `.full(seed:)` is available only when the source is indexed.
    ///
    /// - Parameters:
    ///   - source: A possibly throwing asynchronous sequence of samples.
    ///   - batchSize: The maximum number of samples in each batch.
    ///   - shuffle: The sample-level shuffle strategy, applied before batching.
    ///   - dropLast: Whether to discard an incomplete final batch.
    ///   - collate: A throwing conversion from samples to a batch. The loader
    ///     verifies that `sampleCount` equals the number of supplied samples.
    public init<Source: AsyncSequence>(
        source: Source,
        batchSize: Int,
        shuffle: ShuffleStrategy = .none,
        dropLast: Bool = false,
        collate: @escaping ([Sample<Input, Target>]) throws -> Batch<Input, Target>
    ) throws where Source.Element == Sample<Input, Target> {
        try Self.validate(batchSize: batchSize, shuffle: shuffle)

        let factory: () -> AnyAsyncIterator<Element>
        switch shuffle {
        case .none:
            let batches = BatchingSequence(
                source: source,
                batchSize: batchSize,
                dropLast: dropLast,
                collate: collate
            )
            factory = { AnyAsyncIterator(batches.makeAsyncIterator()) }
        case .buffered(let windowSize, let seed):
            let shuffled = BufferedShuffleSequence(
                source: source,
                windowSize: windowSize,
                seed: seed
            )
            let batches = BatchingSequence(
                source: shuffled,
                batchSize: batchSize,
                dropLast: dropLast,
                collate: collate
            )
            factory = { AnyAsyncIterator(batches.makeAsyncIterator()) }
        case .full:
            throw DataLoaderError.fullShuffleRequiresIndexedSource
        }

        self.batchSize = batchSize
        self.shuffle = shuffle
        self.dropLast = dropLast
        self.iteratorFactory = factory
    }

    /// Creates a loader for an indexed sample collection.
    ///
    /// Full shuffle stores and shuffles only the collection's indices. It does
    /// not copy or materialize the samples.
    public init<Source: RandomAccessCollection>(
        source: Source,
        batchSize: Int,
        shuffle: ShuffleStrategy = .none,
        dropLast: Bool = false,
        collate: @escaping ([Sample<Input, Target>]) throws -> Batch<Input, Target>
    ) throws where Source.Element == Sample<Input, Target> {
        try Self.validate(batchSize: batchSize, shuffle: shuffle)

        let factory: () -> AnyAsyncIterator<Element>
        switch shuffle {
        case .none, .full:
            let indexed = IndexedSampleSequence(
                source: source,
                seed: shuffle.fullSeed
            )
            let batches = BatchingSequence(
                source: indexed,
                batchSize: batchSize,
                dropLast: dropLast,
                collate: collate
            )
            factory = { AnyAsyncIterator(batches.makeAsyncIterator()) }
        case .buffered(let windowSize, let seed):
            let indexed = IndexedSampleSequence(
                source: source,
                seed: nil
            )
            let shuffled = BufferedShuffleSequence(
                source: indexed,
                windowSize: windowSize,
                seed: seed
            )
            let bufferedBatches = BatchingSequence(
                source: shuffled,
                batchSize: batchSize,
                dropLast: dropLast,
                collate: collate
            )
            self.batchSize = batchSize
            self.shuffle = shuffle
            self.dropLast = dropLast
            self.iteratorFactory = { AnyAsyncIterator(bufferedBatches.makeAsyncIterator()) }
            return
        }

        self.batchSize = batchSize
        self.shuffle = shuffle
        self.dropLast = dropLast
        self.iteratorFactory = factory
    }

    /// Creates a loader for an indexed `MLXArray` in-memory dataset.
    public init(
        source: InMemoryDataset,
        batchSize: Int = 32,
        shuffle: ShuffleStrategy = .none,
        dropLast: Bool = false
    ) throws where Input == MLXArray, Target == MLXArray {
        try self.init(
            source: source,
            batchSize: batchSize,
            shuffle: shuffle,
            dropLast: dropLast
        ) { samples in
            Batch(
                input: stacked(samples.map(\.input)),
                target: stacked(samples.map(\.target)),
                sampleCount: samples.count
            )
        }
    }

    public func makeAsyncIterator() -> Iterator {
        Iterator(iterator: iteratorFactory())
    }

    private static func validate(batchSize: Int, shuffle: ShuffleStrategy) throws {
        guard batchSize > 0 else {
            throw DataLoaderError.invalidBatchSize(batchSize)
        }
        if case .buffered(let windowSize, _) = shuffle, windowSize <= 0 {
            throw DataLoaderError.invalidShuffleWindowSize(windowSize)
        }
    }

    public struct Iterator: AsyncIteratorProtocol {
        public typealias Element = Batch<Input, Target>
        public typealias Failure = any Error

        private var iterator: AnyAsyncIterator<Element>

        fileprivate init(iterator: AnyAsyncIterator<Element>) {
            self.iterator = iterator
        }

        public mutating func next() async throws -> Element? {
            try await iterator.next()
        }
    }
}

private extension ShuffleStrategy {
    var fullSeed: UInt64? {
        guard case .full(let seed) = self else { return nil }
        return seed
    }
}

private struct IndexedSampleSequence<Source: RandomAccessCollection, Input, Target>: AsyncSequence
where Source.Element == Sample<Input, Target> {
    typealias Element = Sample<Input, Target>
    typealias Failure = Never

    let source: Source
    let indices: [Source.Index]?
    let count: Int

    init(source: Source, seed: UInt64?) {
        self.source = source
        if let seed {
            var order = Array(source.indices)
            order.seededShuffle(seed: seed)
            self.indices = order
        } else {
            self.indices = nil
        }
        self.count = source.count
    }

    func makeAsyncIterator() -> Iterator {
        Iterator(source: source, indices: indices, count: count)
    }

    struct Iterator: AsyncIteratorProtocol {
        typealias Element = Sample<Input, Target>
        typealias Failure = Never

        let source: Source
        let indices: [Source.Index]?
        let count: Int
        var offset = 0

        mutating func next() async -> Element? {
            guard offset < count else { return nil }
            defer { offset += 1 }
            let index = indices?[offset] ?? source.index(source.startIndex, offsetBy: offset)
            return source[index]
        }
    }
}

private struct BufferedShuffleSequence<Source: AsyncSequence>: AsyncSequence {
    typealias Element = Source.Element
    typealias Failure = any Error

    let source: Source
    let windowSize: Int
    let seed: UInt64

    func makeAsyncIterator() -> Iterator {
        Iterator(
            source: source,
            windowSize: windowSize,
            random: SplitMix64(seed: seed)
        )
    }

    struct Iterator: AsyncIteratorProtocol {
        typealias Element = Source.Element
        typealias Failure = any Error

        var source: Source.AsyncIterator
        let windowSize: Int
        var random: SplitMix64
        var window: [Element] = []
        var hasReachedEnd = false

        init(source: Source, windowSize: Int, random: SplitMix64) {
            self.source = source.makeAsyncIterator()
            self.windowSize = windowSize
            self.random = random
        }

        mutating func next() async throws -> Element? {
            while window.count < windowSize && !hasReachedEnd {
                if let element = try await source.next() {
                    window.append(element)
                } else {
                    hasReachedEnd = true
                }
            }

            guard !window.isEmpty else { return nil }

            let selectedIndex = Int(random.next() % UInt64(window.count))
            let selected = window.remove(at: selectedIndex)
            if !hasReachedEnd, let replacement = try await source.next() {
                window.append(replacement)
            } else {
                hasReachedEnd = true
            }
            return selected
        }
    }
}

private struct BatchingSequence<Source: AsyncSequence, Input, Target>: AsyncSequence
where Source.Element == Sample<Input, Target> {
    typealias Element = Batch<Input, Target>
    typealias Failure = any Error

    let source: Source
    let batchSize: Int
    let dropLast: Bool
    let collate: ([Sample<Input, Target>]) throws -> Batch<Input, Target>

    func makeAsyncIterator() -> Iterator {
        Iterator(
            source: source.makeAsyncIterator(),
            batchSize: batchSize,
            dropLast: dropLast,
            collate: collate
        )
    }

    struct Iterator: AsyncIteratorProtocol {
        typealias Element = Batch<Input, Target>
        typealias Failure = any Error

        var source: Source.AsyncIterator
        let batchSize: Int
        let dropLast: Bool
        let collate: ([Sample<Input, Target>]) throws -> Batch<Input, Target>

        mutating func next() async throws -> Element? {
            var samples: [Sample<Input, Target>] = []
            samples.reserveCapacity(batchSize)

            while samples.count < batchSize {
                guard let sample = try await source.next() else { break }
                samples.append(sample)
            }

            guard !samples.isEmpty else { return nil }
            if dropLast && samples.count < batchSize { return nil }

            let batch = try collate(samples)
            guard batch.sampleCount == samples.count else {
                throw DataLoaderError.collatedSampleCountMismatch(
                    expected: samples.count,
                    actual: batch.sampleCount
                )
            }
            return batch
        }
    }
}

fileprivate struct AnyAsyncIterator<Element>: AsyncIteratorProtocol {
    typealias Failure = any Error

    private let nextBody: () async throws -> Element?

    init<Iterator: AsyncIteratorProtocol>(_ iterator: Iterator) where Iterator.Element == Element {
        let box = AsyncIteratorBox(iterator)
        self.nextBody = { try await box.next() }
    }

    mutating func next() async throws -> Element? {
        try await nextBody()
    }
}

fileprivate final class AsyncIteratorBox<Iterator: AsyncIteratorProtocol> {
    private var iterator: Iterator

    init(_ iterator: Iterator) {
        self.iterator = iterator
    }

    func next() async throws -> Iterator.Element? {
        try await iterator.next()
    }
}

private struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        return value ^ (value >> 31)
    }
}

private extension Array {
    mutating func seededShuffle(seed: UInt64) {
        var random = SplitMix64(seed: seed)
        guard count > 1 else { return }
        for index in stride(from: count - 1, through: 1, by: -1) {
            let swapIndex = Int(random.next() % UInt64(index + 1))
            swapAt(index, swapIndex)
        }
    }
}
