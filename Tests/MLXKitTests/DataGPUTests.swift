// GPU-dependent tests — run via Xcode (xcodebuild test), not `swift test`.

#if MLX_GPU_TESTS
import Testing
import MLX
@testable import MLXKit

@Suite("InMemoryDataset (GPU)")
struct InMemoryDatasetGPUTests {
    @Test func countMatchesDim0() {
        let features = MLXArray.zeros([100, 10])
        let labels = MLXArray.zeros([100])
        let dataset = InMemoryDataset(features: features, labels: labels)
        #expect(dataset.count == 100)
    }

    @Test func subscriptReturnsCorrectSlices() {
        let features = MLXArray(0..<20, [4, 5]).asType(.float32)
        let labels = MLXArray([10, 20, 30, 40] as [Int32])
        let dataset = InMemoryDataset(features: features, labels: labels)

        let sample = dataset[2]
        #expect(sample.features.shape == [5])
        #expect(sample.labels.item(Int32.self) == 30)
    }

    @Test func randomAccessCollectionConformance() {
        let features = MLXArray.zeros([5, 3])
        let labels = MLXArray.zeros([5])
        let dataset = InMemoryDataset(features: features, labels: labels)

        #expect(dataset.startIndex == 0)
        #expect(dataset.endIndex == 5)

        var count = 0
        for _ in dataset { count += 1 }
        #expect(count == 5)
    }
}

@Suite("DataLoader (GPU)")
struct DataLoaderGPUTests {
    private func makeDataset(count: Int, featureDim: Int = 10) -> InMemoryDataset {
        InMemoryDataset(
            features: MLXArray.zeros([count, featureDim]),
            labels: MLXArray.zeros([count])
        )
    }

    @Test func correctBatchCount() async {
        let loader = DataLoader(dataset: makeDataset(count: 100), batchSize: 32)
        var batchCount = 0
        for await _ in loader { batchCount += 1 }
        // ceil(100/32) = 4
        #expect(batchCount == 4)
    }

    @Test func dropLastReducesBatchCount() async {
        let loader = DataLoader(dataset: makeDataset(count: 100), batchSize: 32, dropLast: true)
        var batchCount = 0
        for await _ in loader { batchCount += 1 }
        // floor(100/32) = 3
        #expect(batchCount == 3)
    }

    @Test func noShufflePreservesOrder() async {
        let features = MLXArray(0..<20, [10, 2]).asType(.float32)
        let labels = MLXArray(0..<10, [10]).asType(.int32)
        let dataset = InMemoryDataset(features: features, labels: labels)

        let loader = DataLoader(dataset: dataset, batchSize: 5, shuffle: false)
        var firstBatchLabel: Int32?
        for await batch in loader {
            firstBatchLabel = batch.labels[0].item(Int32.self)
            break
        }
        #expect(firstBatchLabel == 0)
    }

    @Test func batchShapes() async {
        let loader = DataLoader(dataset: makeDataset(count: 100, featureDim: 784), batchSize: 32)
        for await batch in loader {
            #expect(batch.features.dim(-1) == 784)
            #expect(batch.size <= 32)
            break
        }
    }

    @Test func lastBatchSmallerWhenNotDropped() async {
        let loader = DataLoader(dataset: makeDataset(count: 10), batchSize: 3, dropLast: false)
        var lastBatch: Batch?
        for await batch in loader { lastBatch = batch }
        // 10 % 3 = 1 remaining sample
        #expect(lastBatch?.size == 1)
    }
}
#endif
