// GPU-dependent tests — run via Xcode (xcodebuild test), not `swift test`.

#if MLX_GPU_TESTS
import Testing
import MLX
import MLXNN
import MLXOptimizers
@testable import MLXKit

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

/// A fixed regression fixture with inputs `1...4` and target `2x`.
private func makeRegressionDataset(count: Int) -> InMemoryDataset {
    let values = (0..<count).map { Float32($0 % 4 + 1) }
    let features = MLXArray(values).reshaped([count, 1])
    let labels = MLXArray(values.map { $0 * 2 }).reshaped([count, 1])
    return InMemoryDataset(features: features, labels: labels)
}

private struct ReiterableMLXSampleSequence: AsyncSequence {
    typealias Element = Sample<MLXArray, MLXArray>
    typealias Failure = Never

    let samples: [Element]

    func makeAsyncIterator() -> Iterator {
        Iterator(samples: samples)
    }

    struct Iterator: AsyncIteratorProtocol {
        typealias Element = Sample<MLXArray, MLXArray>
        typealias Failure = Never

        let samples: [Element]
        var offset = 0

        mutating func next() async -> Element? {
            guard offset < samples.count else { return nil }
            defer { offset += 1 }
            return samples[offset]
        }
    }
}

@Suite("TrainingEngine fit() (GPU)", .serialized)
struct TrainingEngineFitGPUTests {
    @Test func earlyStopAtBatchEndCompletesAndRecordsOneEpoch() async throws {
        let model = ScalarLinearModel()
        model.train(false)
        let inputs = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float32]).reshaped([4, 1])
        let targets = MLXArray.zeros([4, 1])
        let loader = try DataLoader(
            source: InMemoryDataset(features: inputs, labels: targets),
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
        let loader = try DataLoader(
            source: InMemoryDataset(features: inputs, labels: targets),
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
        let loader = try DataLoader(
            source: InMemoryDataset(features: inputs, labels: targets),
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
        let loader = try DataLoader(
            source: InMemoryDataset(features: inputs, labels: targets),
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
        let loader = try DataLoader(
            source: InMemoryDataset(features: inputs, labels: targets),
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
        let loader = try DataLoader(
            source: InMemoryDataset(features: inputs, labels: targets),
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
        let loader = try DataLoader(
            source: InMemoryDataset(features: inputs, labels: targets),
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
        let loader = try DataLoader(
            source: InMemoryDataset(features: inputs, labels: targets),
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
        let loader = try DataLoader(
            source: InMemoryDataset(features: inputs, labels: targets),
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
        let loader = try DataLoader(
            source: InMemoryDataset(features: inputs, labels: targets),
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
        let loader = try DataLoader(
            source: InMemoryDataset(features: inputs, labels: targets),
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
        let loader = try DataLoader(
            source: InMemoryDataset(features: inputs, labels: targets),
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
        let trainedModel = ScalarLinearModel()
        let directModel = ScalarLinearModel()
        let inputs = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float32]).reshaped([4, 1])
        let targets = MLXArray([2.0, 4.0, 6.0, 8.0] as [Float32]).reshaped([4, 1])
        let loader = try DataLoader(
            source: InMemoryDataset(features: inputs, labels: targets),
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
            (model: ScalarLinearModel, features: MLXArray, labels: MLXArray) -> MLXArray in
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

    @Test func asynchronousTypedPipelineWorksWithFitAndEvaluate() async throws {
        let samples = (1...4).map { value in
            Sample(
                input: MLXArray([Float32(value)]),
                target: MLXArray([Float32(value * 2)])
            )
        }
        let transformedSource = ReiterableMLXSampleSequence(samples: samples)
            .filter { $0.input.size > 0 }
        let trainingBatches = try DataLoader(
            source: transformedSource,
            batchSize: 2,
            collate: { samples in
                Batch(
                    input: stacked(samples.map(\.input)),
                    target: stacked(samples.map(\.target)),
                    sampleCount: samples.count
                )
            }
        )
        let prefetchedBatches = try trainingBatches.prefetched(bufferSize: 2)
        let model = ScalarLinearModel()
        let configuration = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.01),
            objective: MSELoss()
        )

        let history = try await TrainingEngine().fit(
            model,
            configuration: configuration,
            data: prefetchedBatches,
            epochs: 1
        )
        #expect(history["loss"].first != nil)

        let evaluationBatches = try DataLoader(
            source: ReiterableMLXSampleSequence(samples: samples),
            batchSize: 3,
            collate: { samples in
                Batch(
                    input: stacked(samples.map(\.input)),
                    target: stacked(samples.map(\.target)),
                    sampleCount: samples.count
                )
            }
        )
        let metrics = try await TrainingEngine().evaluate(
            model,
            configuration: configuration,
            data: evaluationBatches
        )
        #expect(metrics["loss"]?.isFinite == true)
    }

    @Test func lossDecreasesOverEpochs() async throws {
        let model = ScalarLinearModel()
        let engine = TrainingEngine()
        let config = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.05),
            objective: MSELoss()
        )

        let dataset = makeRegressionDataset(count: 8)
        let loader = try DataLoader(source: dataset, batchSize: 4)

        let history = try await engine.fit(
            model, configuration: config, data: loader, epochs: 8
        )

        let losses = history["loss"].compactMap { $0 }
        #expect(losses.count == 8)
        // Loss should decrease
        #expect(losses.last! < losses.first!)
    }

    @Test func historyContainsMetrics() async throws {
        let model = ScalarLinearModel()
        let engine = TrainingEngine()
        let config = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.01),
            objective: MSELoss(),
            metrics: [MeanOutputMetric()]
        )

        let dataset = makeRegressionDataset(count: 4)
        let loader = try DataLoader(source: dataset, batchSize: 4)

        let history = try await engine.fit(
            model, configuration: config, data: loader, epochs: 5
        )

        #expect(history.epochs.count == 5)
        // Should have both loss and accuracy
        #expect(history["loss"].compactMap { $0 }.count == 5)
        #expect(history["mean_output"].compactMap { $0 }.count == 5)
    }

    @Test func earlyStoppingStopsTraining() async throws {
        let model = ScalarLinearModel()
        let engine = TrainingEngine()
        let config = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.01),
            objective: MSELoss()
        )

        let dataset = makeRegressionDataset(count: 8)
        let loader = try DataLoader(source: dataset, batchSize: 4)

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
        let model = ScalarLinearModel()
        let engine = TrainingEngine()
        let config = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.01),
            objective: MSELoss(),
            metrics: [MeanOutputMetric()]
        )

        let trainData = makeRegressionDataset(count: 8)
        let valData = makeRegressionDataset(count: 4)
        let trainLoader = try DataLoader(source: trainData, batchSize: 4)
        let valLoader = try DataLoader(source: valData, batchSize: 2)

        let history = try await engine.fit(
            model, configuration: config, data: trainLoader,
            epochs: 3, validationData: valLoader
        )

        let valLosses = history["val_loss"].compactMap { $0 }
        #expect(valLosses.count == 3)
        #expect(history["val_mean_output"].compactMap { $0 }.count == 3)

        let evaluation = try await engine.evaluate(
            model,
            configuration: config,
            data: valLoader
        )
        #expect(evaluation["loss"] != nil)
        #expect(evaluation["mean_output"] != nil)

        let predictions = try await engine.predict(model, data: valLoader)
        #expect(predictions.count == 2)
        #expect(predictions.allSatisfy { $0.shape == [2, 1] })
    }
}

@Suite("TrainingEngine evaluate() (GPU)", .serialized)
struct TrainingEngineEvaluateGPUTests {
    @Test func returnsMetrics() async throws {
        let model = ScalarLinearModel()
        let engine = TrainingEngine()
        let config = TrainingConfiguration(
            optimizer: Adam(learningRate: 0.01),
            objective: MSELoss()
        )

        let dataset = makeRegressionDataset(count: 8)
        let loader = try DataLoader(source: dataset, batchSize: 4)

        let metrics = try await engine.evaluate(model, configuration: config, data: loader)
        #expect(metrics["loss"] != nil)
    }
}

@Suite("TrainingEngine predict() (GPU)")
struct TrainingEnginePredictGPUTests {
    @Test func outputShape() {
        let model = ScalarLinearModel()
        let engine = TrainingEngine()
        eval(model)

        let input = MLXArray.zeros([5, 1])
        let output = engine.predict(model, input: input)
        #expect(output.shape == [5, 1])
    }

    @Test func modelInEvalModeDuringPrediction() {
        let model = ScalarLinearModel()
        let engine = TrainingEngine()
        eval(model)

        model.train(true)
        #expect(model.training == true)

        let input = MLXArray.zeros([1, 1])
        _ = engine.predict(model, input: input)

        // Should restore training mode after prediction
        #expect(model.training == true)
    }
}
#endif
