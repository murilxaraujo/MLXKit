import Testing
import MLX
@testable import MLXKit

// MARK: - Static lookup (no GPU needed)

@Suite("Static Lookup")
struct StaticLookupTests {
    @Test func crossEntropyLookup() {
        let loss: CrossEntropyLoss = .crossEntropy
        #expect(loss.labelSmoothing == 0)
        #expect(loss.reduction == .mean)
    }

    @Test func binaryCrossEntropyLookup() {
        let loss: BinaryCrossEntropyLoss = .binaryCrossEntropy
        #expect(loss.inputsAreLogits == true)
        #expect(loss.reduction == .mean)
    }

    @Test func mseLookup() {
        let loss: MSELoss = .mse
        #expect(loss.reduction == .mean)
    }

    @Test func l1Lookup() {
        let loss: L1Loss = .l1
        #expect(loss.reduction == .mean)
    }

    @Test func huberLookup() {
        let loss: HuberLoss = .huber
        #expect(loss.delta == 1.0)
        #expect(loss.reduction == .mean)
    }

    @Test func klDivLookup() {
        let loss: KLDivLoss = .klDiv
        #expect(loss.axis == -1)
        #expect(loss.reduction == .mean)
    }
}

// MARK: - Configuration tests (no GPU needed)

@Suite("Loss Configuration")
struct LossConfigurationTests {
    @Test func crossEntropyCustomConfig() {
        let loss = CrossEntropyLoss(labelSmoothing: 0.1, reduction: .sum)
        #expect(loss.labelSmoothing == 0.1)
        #expect(loss.reduction == .sum)
    }

    @Test func bceCustomConfig() {
        let loss = BinaryCrossEntropyLoss(inputsAreLogits: false, reduction: .none)
        #expect(loss.inputsAreLogits == false)
        #expect(loss.reduction == .none)
    }

    @Test func huberCustomDelta() {
        let loss = HuberLoss(delta: 2.5, reduction: .sum)
        #expect(loss.delta == 2.5)
        #expect(loss.reduction == .sum)
    }

    @Test func klDivCustomAxis() {
        let loss = KLDivLoss(axis: 0, reduction: .sum)
        #expect(loss.axis == 0)
        #expect(loss.reduction == .sum)
    }

    @Test func reductionEquality() {
        #expect(Reduction.mean == .mean)
        #expect(Reduction.sum == .sum)
        #expect(Reduction.none == .none)
        #expect(Reduction.mean != .sum)
    }
}
