import Testing
@testable import MLXKit

// MARK: - Static lookup (no GPU needed)

@Suite("Metric Static Lookup")
struct MetricStaticLookupTests {
    @Test func accuracyLookup() {
        let metric: Accuracy = .accuracy
        #expect(metric.name == "accuracy")
    }

    @Test func top5AccuracyLookup() {
        let metric: TopKAccuracy = .top5Accuracy
        #expect(metric.name == "top5_accuracy")
        #expect(metric.k == 5)
    }

    @Test func meanSquaredErrorLookup() {
        let metric: MeanSquaredError = .meanSquaredError
        #expect(metric.name == "mse")
    }

    @Test func meanAbsoluteErrorLookup() {
        let metric: MeanAbsoluteError = .meanAbsoluteError
        #expect(metric.name == "mae")
    }

    @Test func runningLossLookup() {
        let metric: RunningLoss = .runningLoss
        #expect(metric.name == "loss")
    }
}

// MARK: - Configuration tests (no GPU needed)

@Suite("Metric Configuration")
struct MetricConfigurationTests {
    @Test func topKCustomK() {
        let metric = TopKAccuracy(k: 3)
        #expect(metric.k == 3)
        #expect(metric.name == "top3_accuracy")
    }

    @Test func runningLossCustomName() {
        let metric = RunningLoss(name: "val_loss")
        #expect(metric.name == "val_loss")
    }

    @Test func computeReturnsZeroBeforeUpdate() {
        #expect(Accuracy().compute() == 0)
        #expect(TopKAccuracy(k: 5).compute() == 0)
        #expect(MeanSquaredError().compute() == 0)
        #expect(MeanAbsoluteError().compute() == 0)
        #expect(RunningLoss().compute() == 0)
    }
}
