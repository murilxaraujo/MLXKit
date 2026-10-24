import Testing
import MLX
@testable import MLXKit

@Suite("DataLoader Configuration")
struct DataLoaderConfigTests {
    private let samples = (0..<5).map { Sample(input: $0, target: $0 * 10) }

    @Test func defaultConfiguration() throws {
        let loader = try DataLoader(
            source: samples,
            batchSize: 32,
            collate: collate
        )
        #expect(loader.batchSize == 32)
        #expect(loader.shuffle == .none)
        #expect(loader.dropLast == false)
    }

    @Test func customConfiguration() throws {
        let loader = try DataLoader(
            source: samples,
            batchSize: 64,
            shuffle: .full(seed: 42),
            dropLast: true,
            collate: collate
        )
        #expect(loader.batchSize == 64)
        #expect(loader.shuffle == .full(seed: 42))
        #expect(loader.dropLast == true)
    }

    @Test func invalidBatchSizeFailsAtConstruction() {
        #expect(throws: DataLoaderError.invalidBatchSize(0)) {
            try DataLoader(source: samples, batchSize: 0, collate: collate)
        }
    }

    @Test func invalidShuffleWindowFailsAtConstruction() {
        #expect(throws: DataLoaderError.invalidShuffleWindowSize(0)) {
            try DataLoader(
                source: samples,
                batchSize: 2,
                shuffle: .buffered(windowSize: 0, seed: 1),
                collate: collate
            )
        }
    }

    private func collate(_ samples: [Sample<Int, Int>]) -> Batch<Int, Int> {
        Batch(
            input: samples.map(\.input).reduce(0, +),
            target: samples.map(\.target).reduce(0, +),
            sampleCount: samples.count
        )
    }
}
