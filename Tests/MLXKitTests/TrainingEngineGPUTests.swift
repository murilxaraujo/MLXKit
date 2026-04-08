// GPU-dependent tests — run via Xcode (xcodebuild test), not `swift test`.

#if MLX_GPU_TESTS
import Testing
import MLX
import MLXNN
import MLXOptimizers
@testable import MLXKit

/// Simple two-layer model for testing.
private final class TwoLayerModel: Module, TrainableModel {
    let linear1 = Linear(inputDimensions: 2, outputDimensions: 8)
    let linear2 = Linear(inputDimensions: 8, outputDimensions: 1)

    func forward(_ input: MLXArray) -> MLXArray {
        var x = linear1(input)
        x = maximum(x, 0) // ReLU
        return linear2(x)
    }
}

/// Simple classification model for XOR-like problems.
private final class ClassifierModel: Module, TrainableModel {
    let linear1 = Linear(inputDimensions: 2, outputDimensions: 16)
    let linear2 = Linear(inputDimensions: 16, outputDimensions: 2)

    func forward(_ input: MLXArray) -> MLXArray {
        var x = linear1(input)
        x = maximum(x, 0)
        return linear2(x)
    }
}

/// Creates a simple regression dataset: y = x0 + x1
private func makeRegressionDataset(count: Int) -> InMemoryDataset {
    let features = MLXRandom.uniform(low: -1.0, high: 1.0, [count, 2])
    let labels = features[0..., 0] + features[0..., 1]
    return InMemoryDataset(features: features, labels: labels.reshaped([-1, 1]))
}

@Suite("TrainingEngine fit() (GPU)", .serialized)
struct TrainingEngineFitGPUTests {
    @Test func lossDecreasesOverEpochs() async throws {
        let model = TwoLayerModel()
        let engine = TrainingEngine()
        let config = engine.compile(
            model: model,
            optimizer: Adam(learningRate: 0.01),
            loss: MSELoss()
        )

        let dataset = makeRegressionDataset(count: 200)
        let loader = DataLoader(dataset: dataset, batchSize: 32, shuffle: true)

        let history = try await engine.fit(
            model, config: config, data: loader, epochs: 20
        )

        let losses = history["loss"].compactMap { $0 }
        #expect(losses.count == 20)
        // Loss should decrease
        #expect(losses.last! < losses.first!)
    }

    @Test func historyContainsMetrics() async throws {
        let model = ClassifierModel()
        let engine = TrainingEngine()
        let config = engine.compile(
            model: model,
            optimizer: Adam(learningRate: 0.01),
            loss: CrossEntropyLoss(),
            metrics: [Accuracy()]
        )

        // Simple separable dataset
        let features = MLXArray([
            1.0, 1.0,
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
        ] as [Float32]).reshaped([4, 2])
        let labels = MLXArray([1, 1, 1, 0] as [Int32])
        let dataset = InMemoryDataset(features: features, labels: labels)
        let loader = DataLoader(dataset: dataset, batchSize: 4)

        let history = try await engine.fit(
            model, config: config, data: loader, epochs: 5
        )

        #expect(history.epochs.count == 5)
        // Should have both loss and accuracy
        #expect(history["loss"].compactMap { $0 }.count == 5)
        #expect(history["accuracy"].compactMap { $0 }.count == 5)
    }

    @Test func earlyStoppingStopsTraining() async throws {
        let model = TwoLayerModel()
        let engine = TrainingEngine()
        let config = engine.compile(
            model: model,
            optimizer: Adam(learningRate: 0.01),
            loss: MSELoss()
        )

        let dataset = makeRegressionDataset(count: 100)
        let loader = DataLoader(dataset: dataset, batchSize: 32)

        // Use a callback that stops after 3 epochs
        let stopper = LambdaCallback(onEpochEnd: { state in
            if state.epoch >= 2 {
                state.shouldStop = true
            }
        })

        let history = try await engine.fit(
            model, config: config, data: loader,
            epochs: 100, callbacks: [stopper]
        )

        // Should have stopped early (3 epochs: 0, 1, 2)
        #expect(history.epochs.count == 3)
    }

    @Test func validationMetricsAppearWithValPrefix() async throws {
        let model = TwoLayerModel()
        let engine = TrainingEngine()
        let config = engine.compile(
            model: model,
            optimizer: Adam(learningRate: 0.01),
            loss: MSELoss()
        )

        let trainData = makeRegressionDataset(count: 100)
        let valData = makeRegressionDataset(count: 50)
        let trainLoader = DataLoader(dataset: trainData, batchSize: 32)
        let valLoader = DataLoader(dataset: valData, batchSize: 32)

        let history = try await engine.fit(
            model, config: config, data: trainLoader,
            epochs: 3, validationData: valLoader
        )

        let valLosses = history["val_loss"].compactMap { $0 }
        #expect(valLosses.count == 3)
    }
}

@Suite("TrainingEngine evaluate() (GPU)", .serialized)
struct TrainingEngineEvaluateGPUTests {
    @Test func returnsMetrics() async throws {
        let model = TwoLayerModel()
        let engine = TrainingEngine()
        let config = engine.compile(
            model: model,
            optimizer: Adam(learningRate: 0.01),
            loss: MSELoss()
        )

        let dataset = makeRegressionDataset(count: 50)
        let loader = DataLoader(dataset: dataset, batchSize: 16)

        let metrics = try await engine.evaluate(model, config: config, data: loader)
        #expect(metrics["loss"] != nil)
    }
}

@Suite("TrainingEngine predict() (GPU)")
struct TrainingEnginePredictGPUTests {
    @Test func outputShape() {
        let model = TwoLayerModel()
        let engine = TrainingEngine()
        eval(model)

        let input = MLXArray.zeros([5, 2])
        let output = engine.predict(model, input: input)
        #expect(output.shape == [5, 1])
    }

    @Test func modelInEvalModeDuringPrediction() {
        let model = TwoLayerModel()
        let engine = TrainingEngine()
        eval(model)

        model.train(true)
        #expect(model.training == true)

        let input = MLXArray.zeros([1, 2])
        _ = engine.predict(model, input: input)

        // Should restore training mode after prediction
        #expect(model.training == true)
    }
}
#endif
