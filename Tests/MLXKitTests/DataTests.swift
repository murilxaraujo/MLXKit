import Testing
import MLX
@testable import MLXKit

@Suite("DataLoader Configuration")
struct DataLoaderConfigTests {
    @Test func defaultConfiguration() {
        let loader = DataLoader(dataset: StubDataset(), batchSize: 32)
        #expect(loader.batchSize == 32)
        #expect(loader.shuffle == false)
        #expect(loader.dropLast == false)
    }

    @Test func customConfiguration() {
        let loader = DataLoader(dataset: StubDataset(), batchSize: 64, shuffle: true, dropLast: true)
        #expect(loader.batchSize == 64)
        #expect(loader.shuffle == true)
        #expect(loader.dropLast == true)
    }
}

/// Stub that satisfies the protocol without creating MLXArrays (avoids Metal).
private struct StubDataset: Dataset {
    // These are never accessed in config-only tests.
    var features: MLXArray { fatalError("stub") }
    var labels: MLXArray { fatalError("stub") }
    var count: Int { 100 }
    subscript(index: Int) -> Sample { fatalError("stub") }
}
