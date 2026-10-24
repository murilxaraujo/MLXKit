// GPU-dependent tests — run via Xcode (xcodebuild test), not `swift test`.

#if MLX_GPU_TESTS
import Testing
import MLX
import MLXNN
import MLXOptimizers
@testable import MLXKit

/// Simple two-layer model for testing.
private final class TwoLayerModel: Module, TrainableModel {
    typealias Input = MLXArray
    typealias Output = MLXArray

    let linear1 = Linear(inputDimensions: 2, outputDimensions: 8)
    let linear2 = Linear(inputDimensions: 8, outputDimensions: 1)

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = linear1(input)
        x = maximum(x, 0) // ReLU
        return linear2(x)
    }
}

/// Simple classification model for XOR-like problems.
private final class ClassifierModel: Module, TrainableModel {
    typealias Input = MLXArray
    typealias Output = MLXArray

    let linear1 = Linear(inputDimensions: 2, outputDimensions: 16)
    let linear2 = Linear(inputDimensions: 16, outputDimensions: 2)

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = linear1(input)
        x = maximum(x, 0)
        return linear2(x)
    }
}

private final class ScalarLinearModel: Module, TrainableModel {
    typealias Input = MLXArray
    typealias Output = MLXArray

    let weight = MLXArray(Float32(1))

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        input * weight
    }
}

private final class MeanOutputMetric: Metric {
    typealias Output = MLXArray
    typealias Target = MLXArray

    let name = "mean_output"
    private var sum: Float = 0
    private var count = 0

    func update(output: MLXArray, target: MLXArray) {
        sum += output.sum().item(Float.self)
        count += output.size
    }

    func compute() -> Float {
        count == 0 ? 0 : sum / Float(count)
    }

    func reset() {
        sum = 0
        count = 0
    }
}

private final class RecordingLearningRateSchedule: LRSchedule, @unchecked Sendable {
    private(set) var steps: [Int] = []

    func learningRate(atStep step: Int) -> Float {
        steps.append(step)
        return Float(step + 1) * 0.001
    }
}

private enum TrainingCallbackTestError: Error, Equatable {
    case first
    case terminal
}

private final class LockedInt: @unchecked Sendable {
    private(set) var value = 0
    func increment() { value += 1 }
}

private final class LifecycleRecorder: Callback, @unchecked Sendable {
    private(set) var events: [String] = []
    func onTrainBegin(state: TrainingState) { events.append("trainBegin") }
    func onEpochBegin(state: TrainingState) { events.append("epochBegin") }
    func onBatchBegin(state: TrainingState) { events.append("batchBegin") }
    func onBatchEnd(state: TrainingState) { events.append("batchEnd") }
    func onEpochEnd(state: TrainingState) { events.append("epochEnd") }
    func onTrainEnd(state: TrainingState) { events.append("trainEnd") }
}

private final class StopAtBatchEnd: Callback, @unchecked Sendable {
    func onBatchEnd(state: TrainingState) {
        state.shouldStop = true
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
    @Test func earlyStopAtBatchEndCompletesAndRecordsOneEpoch() async throws {
        let model = ScalarLinearModel()
        model.train(false)
        let inputs = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float32]).reshaped([4, 1])
        let targets = MLXArray.zeros([4, 1])
        let loader = DataLoader(
            dataset: InMemoryDataset(features: inputs, labels: targets),
            batchSize: 1
        )
        let recorder = LifecycleRecorder()
        let configuration = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.01),
            objective: MSELoss()
        )

        let history = try await TrainingEngine().fit(
            model,
            configuration: configuration,
            data: loader,
            epochs: 3,
            callbacks: [StopAtBatchEnd(), recorder]
        )

        #expect(history["loss"].count == 1)
        #expect(recorder.events == [
            "trainBegin", "epochBegin", "batchBegin", "batchEnd", "epochEnd", "trainEnd",
        ])
        #expect(!model.training)
    }

    @Test func callbackLifecycleOrderAndFitModeRestoration() async throws {
        let model = ScalarLinearModel()
        model.train(false)
        let inputs = MLXArray([1.0, 2.0] as [Float32]).reshaped([2, 1])
        let targets = MLXArray.zeros([2, 1])
        let loader = DataLoader(
            dataset: InMemoryDataset(features: inputs, labels: targets),
            batchSize: 2
        )
        let recorder = LifecycleRecorder()
        let configuration = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.01),
            objective: MSELoss()
        )

        _ = try await TrainingEngine().fit(
            model,
            configuration: configuration,
            data: loader,
            epochs: 1,
            callbacks: [recorder]
        )

        #expect(recorder.events == [
            "trainBegin", "epochBegin", "batchBegin", "batchEnd", "epochEnd", "trainEnd",
        ])
        #expect(!model.training)
    }

    @Test func nonFiniteObjectiveDoesNotUpdateParametersAndRestoresMode() async throws {
        let model = ScalarLinearModel()
        model.train(false)
        let inputs = MLXArray([1.0, 2.0] as [Float32]).reshaped([2, 1])
        let targets = MLXArray.zeros([2, 1])
        let loader = DataLoader(
            dataset: InMemoryDataset(features: inputs, labels: targets),
            batchSize: 2
        )
        let before = model.trainableParameters().flattened()[0].1.item(Float.self)
        let configuration = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.1),
            objective: ClosureObjective<MLXArray, MLXArray> { output, _ in
                output.mean() * 0 + MLXArray(Float32.nan)
            }
        )

        do {
            _ = try await TrainingEngine().fit(
                model,
                configuration: configuration,
                data: loader,
                epochs: 1
            )
            Issue.record("Expected a non-finite objective to fail")
        } catch let error as TrainingError {
            #expect(error == .nonFiniteObjective)
        }

        let after = model.trainableParameters().flattened()[0].1.item(Float.self)
        #expect(after == before)
        #expect(!model.training)
    }

    @Test func nonFiniteGradientDoesNotUpdateParameters() async throws {
        let model = ScalarLinearModel()
        let inputs = MLXArray.zeros([2, 1])
        let targets = MLXArray.zeros([2, 1])
        let loader = DataLoader(
            dataset: InMemoryDataset(features: inputs, labels: targets),
            batchSize: 2
        )
        let before = model.trainableParameters().flattened()[0].1.item(Float.self)
        let configuration = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.1),
            objective: ClosureObjective<MLXArray, MLXArray> { output, _ in
                sqrt(output.mean())
            }
        )

        do {
            _ = try await TrainingEngine().fit(
                model,
                configuration: configuration,
                data: loader,
                epochs: 1
            )
            Issue.record("Expected a non-finite gradient to fail")
        } catch let error as TrainingError {
            #expect(error == .nonFiniteGradient)
        }

        let after = model.trainableParameters().flattened()[0].1.item(Float.self)
        #expect(after == before)
    }

    @Test func terminalCallbackFailureIsReportedOnce() async throws {
        let model = ScalarLinearModel()
        let inputs = MLXArray([1.0] as [Float32]).reshaped([1, 1])
        let targets = MLXArray.zeros([1, 1])
        let loader = DataLoader(
            dataset: InMemoryDataset(features: inputs, labels: targets),
            batchSize: 1
        )
        let terminalCalls = LockedInt()
        let configuration = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.01),
            objective: MSELoss()
        )
        let callback = LambdaCallback(onTrainEnd: { _ in
            terminalCalls.increment()
            throw TrainingCallbackTestError.terminal
        })

        do {
            _ = try await TrainingEngine().fit(
                model,
                configuration: configuration,
                data: loader,
                epochs: 1,
                callbacks: [callback]
            )
            Issue.record("Expected the terminal callback error to propagate")
        } catch let error as TrainingCallbackTestError {
            #expect(error == .terminal)
        }

        #expect(terminalCalls.value == 1)
    }

    @Test func originalCallbackFailureWinsIfTerminalCallbackAlsoFails() async throws {
        let model = ScalarLinearModel()
        let inputs = MLXArray([1.0] as [Float32]).reshaped([1, 1])
        let targets = MLXArray.zeros([1, 1])
        let loader = DataLoader(
            dataset: InMemoryDataset(features: inputs, labels: targets),
            batchSize: 1
        )
        let callback = LambdaCallback(
            onTrainEnd: { _ in throw TrainingCallbackTestError.terminal },
            onBatchBegin: { _ in throw TrainingCallbackTestError.first }
        )
        let configuration = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.01),
            objective: MSELoss()
        )

        do {
            _ = try await TrainingEngine().fit(
                model,
                configuration: configuration,
                data: loader,
                epochs: 1,
                callbacks: [callback]
            )
            Issue.record("Expected the batch callback error to propagate")
        } catch let error as TrainingCallbackTestError {
            #expect(error == .first)
        }
    }

    @Test func scheduleStartsAtZeroAndAdvancesAfterSuccessfulUpdates() async throws {
        let model = ScalarLinearModel()
        let inputs = MLXArray([1.0, 2.0, 3.0] as [Float32]).reshaped([3, 1])
        let targets = MLXArray.zeros([3, 1])
        let loader = DataLoader(
            dataset: InMemoryDataset(features: inputs, labels: targets),
            batchSize: 2
        )
        let optimizer = SGD(learningRate: 0.5)
        let schedule = RecordingLearningRateSchedule()
        let configuration = TrainingConfiguration(
            optimizer: optimizer,
            objective: MSELoss(),
            learningRateSchedule: schedule
        )

        _ = try await TrainingEngine().fit(
            model,
            configuration: configuration,
            data: loader,
            epochs: 1
        )

        #expect(schedule.steps == [0, 1])
        #expect(optimizer.learningRate == 0.002)
    }

    @Test func evaluationDoesNotReadScheduleOrMutateLearningRate() async throws {
        let model = ScalarLinearModel()
        let inputs = MLXArray([1.0, 2.0] as [Float32]).reshaped([2, 1])
        let targets = MLXArray.zeros([2, 1])
        let loader = DataLoader(
            dataset: InMemoryDataset(features: inputs, labels: targets),
            batchSize: 2
        )
        let optimizer = SGD(learningRate: 0.5)
        let schedule = RecordingLearningRateSchedule()
        let configuration = TrainingConfiguration(
            optimizer: optimizer,
            objective: MSELoss(),
            learningRateSchedule: schedule
        )

        _ = try await TrainingEngine().evaluate(
            model,
            configuration: configuration,
            data: loader
        )

        #expect(schedule.steps.isEmpty)
        #expect(optimizer.learningRate == 0.5)
    }

    @Test func epochLossWeightsScalarBatchObjectivesBySampleCount() async throws {
        let model = ScalarLinearModel()
        let inputs = MLXArray([1.0, 1.0, 10.0] as [Float32]).reshaped([3, 1])
        let targets = MLXArray.zeros([3, 1])
        let loader = DataLoader(
            dataset: InMemoryDataset(features: inputs, labels: targets),
            batchSize: 2
        )
        let configuration = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.01),
            objective: MSELoss()
        )

        let history = try await TrainingEngine().fit(
            model,
            configuration: configuration,
            data: loader,
            epochs: 1,
            trainingStep: { model, batch in
                batch.input.mean() + model(batch.input).sum() * 0
            }
        )

        // Batch scalar objectives are 1 and 10, with sample counts 2 and 1.
        #expect(abs(history["loss"][0]! - 4) < 1e-5)
    }

    @Test func metricsObservePreUpdatePrediction() async throws {
        let model = ScalarLinearModel()
        let inputs = MLXArray([1.0, 2.0] as [Float32]).reshaped([2, 1])
        let targets = MLXArray.zeros([2, 1])
        let loader = DataLoader(
            dataset: InMemoryDataset(features: inputs, labels: targets),
            batchSize: 2
        )
        let before = TrainingEngine().predict(model, input: inputs).mean().item(Float.self)
        let configuration = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.1),
            objective: MSELoss(),
            metrics: [MeanOutputMetric()]
        )

        let history = try await TrainingEngine().fit(
            model,
            configuration: configuration,
            data: loader,
            epochs: 1
        )
        let after = TrainingEngine().predict(model, input: inputs).mean().item(Float.self)

        #expect(abs(history["mean_output"][0]! - before) < 1e-5)
        #expect(abs(after - before) > 1e-3)
    }

    @Test func nonScalarObjectiveIsRejectedBeforeScalarExtraction() async throws {
        let model = ScalarLinearModel()
        let inputs = MLXArray([1.0, 2.0] as [Float32]).reshaped([2, 1])
        let targets = MLXArray.zeros([2, 1])
        let loader = DataLoader(
            dataset: InMemoryDataset(features: inputs, labels: targets),
            batchSize: 2
        )
        let configuration = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.1),
            objective: MSELoss(reduction: .none)
        )

        do {
            _ = try await TrainingEngine().fit(
                model,
                configuration: configuration,
                data: loader,
                epochs: 1
            )
            Issue.record("Expected the unreduced objective to be rejected")
        } catch let error as TrainingError {
            #expect(error == .nonScalarObjective(shape: [2, 1]))
        }
    }

    @Test func shapeOneObjectiveIsNotTreatedAsScalar() async throws {
        let model = ScalarLinearModel()
        let inputs = MLXArray([1.0, 2.0] as [Float32]).reshaped([2, 1])
        let targets = MLXArray.zeros([2, 1])
        let loader = DataLoader(
            dataset: InMemoryDataset(features: inputs, labels: targets),
            batchSize: 2
        )
        let configuration = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.1),
            objective: ClosureObjective<MLXArray, MLXArray> { output, _ in
                output.mean().expandedDimensions(axis: 0)
            }
        )

        do {
            _ = try await TrainingEngine().fit(
                model,
                configuration: configuration,
                data: loader,
                epochs: 1
            )
            Issue.record("Expected shape [1] to be rejected as non-scalar")
        } catch let error as TrainingError {
            #expect(error == .nonScalarObjective(shape: [1]))
        }
    }

    @Test func oneBatchUpdateMatchesDirectMLX() async throws {
        let trainedModel = TwoLayerModel()
        let directModel = TwoLayerModel()
        directModel.update(parameters: trainedModel.trainableParameters())

        let inputs = MLXArray([
            0.25, -0.5,
            0.75, 0.1,
            -0.3, 0.8,
            0.2, 0.4,
        ] as [Float32]).reshaped([4, 2])
        let targets = MLXArray([0.1, -0.2, 0.5, 0.3] as [Float32]).reshaped([4, 1])
        let loader = DataLoader(
            dataset: InMemoryDataset(features: inputs, labels: targets),
            batchSize: 4
        )
        let engineOptimizer = SGD(learningRate: 0.02)
        let configuration = TrainingConfiguration(
            optimizer: engineOptimizer,
            objective: MSELoss()
        )

        _ = try await TrainingEngine().fit(
            trainedModel,
            configuration: configuration,
            data: loader,
            epochs: 1
        )

        let directOptimizer = SGD(learningRate: 0.02)
        let directLossAndGrad = valueAndGrad(model: directModel) {
            (model: TwoLayerModel, features: MLXArray, labels: MLXArray) -> MLXArray in
            MSELoss()(predictions: model(features), targets: labels)
        }
        let (_, gradients) = directLossAndGrad(directModel, inputs, targets)
        directOptimizer.update(model: directModel, gradients: gradients)
        eval(directModel, directOptimizer)

        let actual = trainedModel.trainableParameters().flattened().sorted { $0.0 < $1.0 }
        let expected = directModel.trainableParameters().flattened().sorted { $0.0 < $1.0 }
        #expect(actual.map(\.0) == expected.map(\.0))
        for (actualParameter, expectedParameter) in zip(actual, expected) {
            let difference = abs(actualParameter.1 - expectedParameter.1).max().item(Float.self)
            #expect(difference < 1e-5)
        }
    }

    @Test func lossDecreasesOverEpochs() async throws {
        let model = TwoLayerModel()
        let engine = TrainingEngine()
        let config = TrainingConfiguration(
            optimizer: Adam(learningRate: 0.01),
            objective: MSELoss()
        )

        let dataset = makeRegressionDataset(count: 200)
        let loader = DataLoader(dataset: dataset, batchSize: 32, shuffle: true)

        let history = try await engine.fit(
            model, configuration: config, data: loader, epochs: 20
        )

        let losses = history["loss"].compactMap { $0 }
        #expect(losses.count == 20)
        // Loss should decrease
        #expect(losses.last! < losses.first!)
    }

    @Test func historyContainsMetrics() async throws {
        let model = ClassifierModel()
        let engine = TrainingEngine()
        let config = TrainingConfiguration(
            optimizer: Adam(learningRate: 0.01),
            objective: CrossEntropyLoss(),
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
            model, configuration: config, data: loader, epochs: 5
        )

        #expect(history.epochs.count == 5)
        // Should have both loss and accuracy
        #expect(history["loss"].compactMap { $0 }.count == 5)
        #expect(history["accuracy"].compactMap { $0 }.count == 5)
    }

    @Test func earlyStoppingStopsTraining() async throws {
        let model = TwoLayerModel()
        let engine = TrainingEngine()
        let config = TrainingConfiguration(
            optimizer: Adam(learningRate: 0.01),
            objective: MSELoss()
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
            model, configuration: config, data: loader,
            epochs: 100, callbacks: [stopper]
        )

        // Should have stopped early (3 epochs: 0, 1, 2)
        #expect(history.epochs.count == 3)
    }

    @Test func validationMetricsAppearWithValPrefix() async throws {
        let model = TwoLayerModel()
        let engine = TrainingEngine()
        let config = TrainingConfiguration(
            optimizer: Adam(learningRate: 0.01),
            objective: MSELoss()
        )

        let trainData = makeRegressionDataset(count: 100)
        let valData = makeRegressionDataset(count: 50)
        let trainLoader = DataLoader(dataset: trainData, batchSize: 32)
        let valLoader = DataLoader(dataset: valData, batchSize: 32)

        let history = try await engine.fit(
            model, configuration: config, data: trainLoader,
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
        let config = TrainingConfiguration(
            optimizer: Adam(learningRate: 0.01),
            objective: MSELoss()
        )

        let dataset = makeRegressionDataset(count: 50)
        let loader = DataLoader(dataset: dataset, batchSize: 16)

        let metrics = try await engine.evaluate(model, configuration: config, data: loader)
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
