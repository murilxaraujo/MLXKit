import MLX

/// An `AsyncSequence` that yields ``Batch`` values from a ``Dataset``.
///
/// Supports configurable batch size, optional shuffling, and `dropLast`
/// to discard incomplete final batches.
///
/// ```swift
/// let loader = DataLoader(dataset: trainData, batchSize: 32, shuffle: true)
/// for await batch in loader {
///     let output = model(batch.features)
/// }
/// ```
public struct DataLoader<D: Dataset>: AsyncSequence {
    public typealias Element = Batch

    public let dataset: D
    public let batchSize: Int
    public let shuffle: Bool
    public let dropLast: Bool

    public init(
        dataset: D,
        batchSize: Int = 32,
        shuffle: Bool = false,
        dropLast: Bool = false
    ) {
        self.dataset = dataset
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.dropLast = dropLast
    }

    public func makeAsyncIterator() -> Iterator {
        Iterator(loader: self)
    }

    public struct Iterator: AsyncIteratorProtocol {
        private let loader: DataLoader<D>
        private let indices: [Int]
        private var offset: Int = 0

        init(loader: DataLoader<D>) {
            self.loader = loader
            var idx = Array(0..<loader.dataset.count)
            if loader.shuffle {
                idx.shuffle()
            }
            self.indices = idx
        }

        public mutating func next() async -> Batch? {
            let remaining = indices.count - offset
            guard remaining > 0 else { return nil }

            if remaining < loader.batchSize && loader.dropLast {
                return nil
            }

            let currentBatchSize = Swift.min(loader.batchSize, remaining)
            let batchIndices = Array(indices[offset..<(offset + currentBatchSize)])
            offset += currentBatchSize

            let indexArray = MLXArray(batchIndices.map(Int32.init))
            let features = loader.dataset.features[indexArray]
            let labels = loader.dataset.labels[indexArray]

            return Batch(features: features, labels: labels, size: currentBatchSize)
        }
    }
}

