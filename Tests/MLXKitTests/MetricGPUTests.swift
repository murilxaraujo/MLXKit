// GPU-dependent tests — run via Xcode (xcodebuild test), not `swift test`.
// SPM test runner doesn't bundle MLX Metal shaders, causing a fatal crash.

#if MLX_GPU_TESTS
import Testing
import MLX
@testable import MLXKit

@Suite("Accuracy (GPU)")
struct AccuracyGPUTests {
    @Test func knownPredictions() {
        let accuracy = Accuracy()
        let predictions = MLXArray([0, 1, 2] as [Int32])
        let targets = MLXArray([0, 1, 0] as [Int32])

        accuracy.update(predictions: predictions, targets: targets)
        let result = accuracy.compute()
        // 2 out of 3 correct
        #expect(abs(result - 2.0 / 3.0) < 1e-5)
    }

    @Test func multiBatchAccumulation() {
        let accuracy = Accuracy()

        // Batch 1: 2/3 correct
        accuracy.update(
            predictions: MLXArray([0, 1, 2] as [Int32]),
            targets: MLXArray([0, 1, 0] as [Int32])
        )
        // Batch 2: 1/2 correct
        accuracy.update(
            predictions: MLXArray([1, 0] as [Int32]),
            targets: MLXArray([1, 1] as [Int32])
        )

        // Total: 3/5
        let result = accuracy.compute()
        #expect(abs(result - 3.0 / 5.0) < 1e-5)
    }

    @Test func resetClearsState() {
        let accuracy = Accuracy()
        accuracy.update(
            predictions: MLXArray([0, 1] as [Int32]),
            targets: MLXArray([0, 1] as [Int32])
        )
        #expect(accuracy.compute() == 1.0)

        accuracy.reset()
        #expect(accuracy.compute() == 0)
    }

    @Test func logitPredictions() {
        let accuracy = Accuracy()
        // Logits: argmax([2.0, 0.5, -1.0]) = 0, argmax([0.1, 3.0, 1.0]) = 1
        let logits = MLXArray([2.0, 0.5, -1.0, 0.1, 3.0, 1.0] as [Float32]).reshaped([2, 3])
        let targets = MLXArray([0, 1] as [Int32])

        accuracy.update(predictions: logits, targets: targets)
        #expect(accuracy.compute() == 1.0)
    }
}

@Suite("TopKAccuracy (GPU)")
struct TopKAccuracyGPUTests {
    @Test func top2WithKnownLogits() {
        let topK = TopKAccuracy(k: 2)
        // Class 0 has highest logit, class 2 is second highest
        // Targets: [2, 0] — both in top-2
        let logits = MLXArray([
            3.0, 0.5, 2.0,
            0.1, -1.0, 1.0,
        ] as [Float32]).reshaped([2, 3])
        let targets = MLXArray([2, 0] as [Int32])

        topK.update(predictions: logits, targets: targets)
        #expect(topK.compute() == 1.0)
    }

    @Test func top1MissesSecondChoice() {
        let topK = TopKAccuracy(k: 1)
        // argmax = 0, but target is 2
        let logits = MLXArray([3.0, 0.5, 2.0] as [Float32]).reshaped([1, 3])
        let targets = MLXArray([2] as [Int32])

        topK.update(predictions: logits, targets: targets)
        #expect(topK.compute() == 0.0)
    }
}

@Suite("MeanSquaredError (GPU)")
struct MeanSquaredErrorGPUTests {
    @Test func manualComputation() {
        let mse = MeanSquaredError()
        let predictions = MLXArray([1.0, 2.0, 3.0] as [Float32])
        let targets = MLXArray([1.5, 2.5, 3.5] as [Float32])

        mse.update(predictions: predictions, targets: targets)
        // Each diff = 0.5, squared = 0.25, sum = 0.75, mean = 0.25
        let result = mse.compute()
        #expect(abs(result - 0.25) < 1e-5)
    }
}

@Suite("MeanAbsoluteError (GPU)")
struct MeanAbsoluteErrorGPUTests {
    @Test func manualComputation() {
        let mae = MeanAbsoluteError()
        let predictions = MLXArray([1.0, 2.0, 3.0] as [Float32])
        let targets = MLXArray([1.5, 2.5, 3.5] as [Float32])

        mae.update(predictions: predictions, targets: targets)
        // Each abs diff = 0.5, sum = 1.5, mean = 0.5
        let result = mae.compute()
        #expect(abs(result - 0.5) < 1e-5)
    }
}

@Suite("RunningLoss (GPU)")
struct RunningLossGPUTests {
    @Test func averageAcrossBatches() {
        let runningLoss = RunningLoss()

        // Batch 1: loss = 2.0, 10 samples
        runningLoss.update(loss: MLXArray(Float32(2.0)), count: 10)
        // Batch 2: loss = 4.0, 10 samples
        runningLoss.update(loss: MLXArray(Float32(4.0)), count: 10)

        // Total: 6.0 / 20 = 0.3
        let result = runningLoss.compute()
        #expect(abs(result - 0.3) < 1e-5)
    }
}
#endif
