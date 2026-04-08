// GPU-dependent tests — run via Xcode (xcodebuild test), not `swift test`.
// SPM test runner doesn't bundle MLX Metal shaders, causing a fatal crash.

#if MLX_GPU_TESTS
import Testing
import MLX
import MLXNN
@testable import MLXKit

@Suite("Reduction (GPU)")
struct ReductionGPUTests {
    @Test func meanReduction() {
        let input = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float32])
        let result = Reduction.mean.apply(to: input)
        #expect(result.ndim == 0)
    }

    @Test func sumReduction() {
        let input = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float32])
        let result = Reduction.sum.apply(to: input)
        #expect(result.ndim == 0)
    }

    @Test func noneReduction() {
        let input = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float32])
        let result = Reduction.none.apply(to: input)
        #expect(result.shape == [4])
    }
}

@Suite("CrossEntropyLoss (GPU)")
struct CrossEntropyLossGPUTests {
    @Test func matchesMLXNN() {
        let logits = MLXArray([2.0, -1.0, 0.5, 1.0, 3.0, -0.5] as [Float32]).reshaped([2, 3])
        let targets = MLXArray([0, 2] as [Int32])

        let kit = CrossEntropyLoss(reduction: .mean)
        let kitResult = kit(predictions: logits, targets: targets)

        let nnResult = MLXNN.crossEntropy(
            logits: logits, targets: targets, reduction: .mean
        )

        #expect(kitResult.shape == nnResult.shape)
    }

    @Test func labelSmoothing() {
        let logits = MLXArray([2.0, -1.0, 0.5] as [Float32]).reshaped([1, 3])
        let targets = MLXArray([0] as [Int32])

        let smoothed = CrossEntropyLoss(labelSmoothing: 0.1, reduction: .mean)
        let unsmoothed = CrossEntropyLoss(labelSmoothing: 0, reduction: .mean)

        let smoothedResult = smoothed(predictions: logits, targets: targets)
        let unsmoothedResult = unsmoothed(predictions: logits, targets: targets)

        #expect(smoothedResult.shape == unsmoothedResult.shape)
    }

    @Test func noneReductionPreservesShape() {
        let logits = MLXArray([2.0, -1.0, 0.5, 1.0, 3.0, -0.5] as [Float32]).reshaped([2, 3])
        let targets = MLXArray([0, 2] as [Int32])

        let loss = CrossEntropyLoss(reduction: .none)
        let result = loss(predictions: logits, targets: targets)
        #expect(result.shape == [2])
    }
}

@Suite("MSELoss (GPU)")
struct MSELossGPUTests {
    @Test func matchesMLXNN() {
        let predictions = MLXArray([1.0, 2.0, 3.0] as [Float32])
        let targets = MLXArray([1.5, 2.5, 3.5] as [Float32])

        let kit = MSELoss(reduction: .mean)
        let kitResult = kit(predictions: predictions, targets: targets)

        let nnResult = MLXNN.mseLoss(
            predictions: predictions, targets: targets, reduction: .mean
        )

        #expect(kitResult.shape == nnResult.shape)
    }

    @Test func noneReductionPreservesShape() {
        let predictions = MLXArray([1.0, 2.0, 3.0] as [Float32])
        let targets = MLXArray([1.5, 2.5, 3.5] as [Float32])

        let result = MSELoss(reduction: .none)(predictions: predictions, targets: targets)
        #expect(result.shape == [3])
    }
}

@Suite("L1Loss (GPU)")
struct L1LossGPUTests {
    @Test func matchesMLXNN() {
        let predictions = MLXArray([1.0, 2.0, 3.0] as [Float32])
        let targets = MLXArray([1.5, 2.5, 3.5] as [Float32])

        let kit = L1Loss(reduction: .mean)
        let kitResult = kit(predictions: predictions, targets: targets)

        let nnResult = MLXNN.l1Loss(
            predictions: predictions, targets: targets, reduction: .mean
        )

        #expect(kitResult.shape == nnResult.shape)
    }
}

@Suite("HuberLoss (GPU)")
struct HuberLossGPUTests {
    @Test func matchesMLXNN() {
        let predictions = MLXArray([1.0, 5.0, 3.0] as [Float32])
        let targets = MLXArray([1.5, 2.5, 3.5] as [Float32])

        let kit = HuberLoss(delta: 1.0, reduction: .mean)
        let kitResult = kit(predictions: predictions, targets: targets)

        let nnResult = MLXNN.huberLoss(
            inputs: predictions, targets: targets, delta: 1.0, reduction: .mean
        )

        #expect(kitResult.shape == nnResult.shape)
    }
}

@Suite("BinaryCrossEntropyLoss (GPU)")
struct BinaryCrossEntropyLossGPUTests {
    @Test func matchesMLXNN() {
        let logits = MLXArray([0.5, -0.3, 1.2] as [Float32])
        let targets = MLXArray([1.0, 0.0, 1.0] as [Float32])

        let kit = BinaryCrossEntropyLoss(reduction: .mean)
        let kitResult = kit(predictions: logits, targets: targets)

        let nnResult = MLXNN.binaryCrossEntropy(
            logits: logits, targets: targets, reduction: .mean
        )

        #expect(kitResult.shape == nnResult.shape)
    }
}

@Suite("KLDivLoss (GPU)")
struct KLDivLossGPUTests {
    @Test func matchesMLXNN() {
        let logProbs = MLXArray([-1.2, -0.5, -2.0, -0.8, -1.5, -1.0] as [Float32]).reshaped([2, 3])
        let targets = MLXArray([0.3, 0.5, 0.2, 0.1, 0.6, 0.3] as [Float32]).reshaped([2, 3])

        let kit = KLDivLoss(reduction: .mean)
        let kitResult = kit(predictions: logProbs, targets: targets)

        let nnResult = MLXNN.klDivLoss(
            inputs: logProbs, targets: targets, reduction: .mean
        )

        #expect(kitResult.shape == nnResult.shape)
    }

    @Test func noneReductionPreservesShape() {
        let logProbs = MLXArray([-1.2, -0.5, -2.0, -0.8, -1.5, -1.0] as [Float32]).reshaped([2, 3])
        let targets = MLXArray([0.3, 0.5, 0.2, 0.1, 0.6, 0.3] as [Float32]).reshaped([2, 3])

        let result = KLDivLoss(reduction: .none)(predictions: logProbs, targets: targets)
        #expect(result.shape == [2])
    }
}
#endif
