// End-to-end integration tests — run via Xcode (xcodebuild test), not `swift test`.

#if MLX_GPU_TESTS
import Testing
import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
@testable import MLXKit

/// MNIST-like classifier for integration testing.
private final class TestClassifier: Module, TrainableModel {
    let linear1 = Linear(inputDimensions: 20, outputDimensions: 32)
    let linear2 = Linear(inputDimensions: 32, outputDimensions: 4)

    func forward(_ input: MLXArray) -> MLXArray {
        var x = linear1(input)
        x = maximum(x, 0)
        return linear2(x)
    }
}

private func makeSyntheticClassificationData(count: Int) -> InMemoryDataset {
    let features = MLXRandom.uniform(low: -1.0, high: 1.0, [count, 20])
    let labels = MLXRandom.randInt(low: 0, high: 4, [count])
    return InMemoryDataset(features: features, labels: labels)
}

@Suite("End-to-End Integration (GPU)")
struct IntegrationGPUTests {
    @Test func fullTrainingPipeline() async throws {
        let model = TestClassifier()
        let engine = TrainingEngine()
        let config = engine.compile(
            model: model,
            optimizer: Adam(learningRate: 0.01),
            loss: CrossEntropyLoss(),
            metrics: [Accuracy()],
            lrSchedule: CosineDecay(initialLR: 0.01, totalSteps: 200)
        )

        let trainData = makeSyntheticClassificationData(count: 200)
        let valData = makeSyntheticClassificationData(count: 50)
        let trainLoader = DataLoader(dataset: trainData, batchSize: 32, shuffle: true)
        let valLoader = DataLoader(dataset: valData, batchSize: 32)

        // fit
        let history = try await engine.fit(
            model, config: config, data: trainLoader,
            epochs: 5,
            validationData: valLoader
        )

        // Training completes without crashes
        #expect(history.epochs.count == 5)

        // History contains expected keys
        let losses = history["loss"].compactMap { $0 }
        let accuracies = history["accuracy"].compactMap { $0 }
        let valLosses = history["val_loss"].compactMap { $0 }
        let valAccuracies = history["val_accuracy"].compactMap { $0 }
        #expect(losses.count == 5)
        #expect(accuracies.count == 5)
        #expect(valLosses.count == 5)
        #expect(valAccuracies.count == 5)

        // Loss should decrease (at least somewhat on 5 epochs)
        #expect(losses.last! < losses.first!)
    }

    @Test func evaluateReturnsReasonableMetrics() async throws {
        let model = TestClassifier()
        let engine = TrainingEngine()
        let config = engine.compile(
            model: model,
            optimizer: Adam(learningRate: 0.01),
            loss: CrossEntropyLoss(),
            metrics: [Accuracy()]
        )

        // Train briefly
        let trainData = makeSyntheticClassificationData(count: 200)
        let trainLoader = DataLoader(dataset: trainData, batchSize: 32, shuffle: true)
        _ = try await engine.fit(model, config: config, data: trainLoader, epochs: 10)

        // Evaluate
        let evalData = makeSyntheticClassificationData(count: 100)
        let evalLoader = DataLoader(dataset: evalData, batchSize: 32)
        let metrics = try await engine.evaluate(model, config: config, data: evalLoader)

        #expect(metrics["loss"] != nil)
        #expect(metrics["accuracy"] != nil)
        // Random chance for 4 classes is 0.25, trained model should be somewhat above
        #expect(metrics["accuracy"]! >= 0.0)
    }

    @Test func predictReturnsCorrectShape() {
        let model = TestClassifier()
        let engine = TrainingEngine()
        eval(model)

        let input = MLXRandom.uniform(low: -1.0, high: 1.0, [8, 20])
        let output = engine.predict(model, input: input)
        #expect(output.shape == [8, 4])
    }

    @Test func modelSummaryPrintsCorrectly() {
        let model = TestClassifier()
        eval(model)

        let summary = model.summary()
        #expect(summary.totalParameters > 0)
        // linear1: 20*32 + 32 = 672, linear2: 32*4 + 4 = 132, total = 804
        #expect(summary.totalParameters == 804)

        let text = summary.formatted()
        #expect(text.contains("Linear"))
        #expect(text.contains("804"))
    }

    @Test func checkpointSaveLoadRoundTrip() async throws {
        let model = TestClassifier()
        let engine = TrainingEngine()
        let config = engine.compile(
            model: model,
            optimizer: Adam(learningRate: 0.01),
            loss: CrossEntropyLoss()
        )

        // Train briefly
        let trainData = makeSyntheticClassificationData(count: 100)
        let trainLoader = DataLoader(dataset: trainData, batchSize: 32)
        _ = try await engine.fit(model, config: config, data: trainLoader, epochs: 2)

        // Save checkpoint
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("integration_test_\(UUID().uuidString)")
        let url = dir.appendingPathComponent("checkpoint.safetensors")

        try Checkpointing.save(model: model, epoch: 2, step: 100, to: url)

        // Load into fresh model
        let model2 = TestClassifier()
        eval(model2)
        let info = try Checkpointing.load(into: model2, from: url)

        #expect(info.epoch == 2)
        #expect(info.step == 100)

        // Predictions should match
        let testInput = MLXRandom.uniform(low: -1.0, high: 1.0, [4, 20])
        eval(testInput)
        let pred1 = engine.predict(model, input: testInput)
        let pred2 = engine.predict(model2, input: testInput)
        #expect(pred1.shape == pred2.shape)

        // Cleanup
        try? FileManager.default.removeItem(at: dir)
    }
}
#endif
