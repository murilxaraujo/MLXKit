import MLX
import MLXKit

/// Re-iterable, deterministic M2 batches built from a local token fixture.
struct FixedTokenBatches: AsyncSequence {
    typealias Element = Batch<MLXArray, MLXArray>
    typealias Failure = Never

    private let batches: [Element]

    init(tokenRows: [[Int32]], sequenceLength: Int, batchSize: Int) {
        precondition(sequenceLength > 0)
        precondition(batchSize > 0)
        precondition(tokenRows.allSatisfy { $0.count == sequenceLength + 1 })

        self.batches = stride(from: 0, to: tokenRows.count, by: batchSize).map { start in
            let end = Swift.min(start + batchSize, tokenRows.count)
            let rows = Array(tokenRows[start..<end])
            let inputRows = rows.map { Array($0.prefix(sequenceLength)) }
            let targetRows = rows.map { Array($0.dropFirst().prefix(sequenceLength)) }

            return Batch(
                input: MLXArray(inputRows),
                target: MLXArray(targetRows),
                sampleCount: rows.count
            )
        }
    }

    func makeAsyncIterator() -> Iterator {
        Iterator(batches: batches)
    }

    struct Iterator: AsyncIteratorProtocol {
        private let batches: [Element]
        private var index = 0

        fileprivate init(batches: [Element]) {
            self.batches = batches
        }

        mutating func next() async -> Element? {
            guard index < batches.count else { return nil }
            defer { index += 1 }
            return batches[index]
        }
    }
}

func makeTokenRows(count: Int, sequenceLength: Int, vocabularySize: Int) -> [[Int32]] {
    (0..<count).map { sample in
        (0...sequenceLength).map { position in
            Int32((sample + position) % vocabularySize)
        }
    }
}
