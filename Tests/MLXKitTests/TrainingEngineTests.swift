import Testing
import MLX
import MLXNN
import MLXOptimizers
@testable import MLXKit

private final class IntegerModel: Module, TrainableModel {
    typealias Input = Int
    typealias Output = Int

    func callAsFunction(_ input: Int) -> Int { input }
}

private final class IntegerMetric: Metric {
    typealias Output = Int
    typealias Target = Int

    let name: String
    init(name: String) { self.name = name }
    func update(output: Int, target: Int) {}
    func compute() -> Float { 0 }
    func reset() {}
}

private struct EmptyBatches<Element>: AsyncSequence {
    typealias Failure = Never

    func makeAsyncIterator() -> Iterator { Iterator() }

    struct Iterator: AsyncIteratorProtocol {
        mutating func next() async -> Element? { nil }
    }
}

private struct FixedBatches<Element>: AsyncSequence {
    typealias Failure = Never
    let batches: [Element]

    func makeAsyncIterator() -> Iterator {
        Iterator(batches: batches)
    }

    struct Iterator: AsyncIteratorProtocol {
        let batches: [Element]
        var index = 0

        mutating func next() async -> Element? {
            guard index < batches.count else { return nil }
            defer { index += 1 }
            return batches[index]
        }
    }
}

private struct FixedLearningRateSchedule: LRSchedule {
    func learningRate(atStep step: Int) -> Float { 0.01 }
}

@Suite("TrainingConfiguration")
struct TrainingConfigurationTests {
    @Test func storesAllComponents() {
        let config = TrainingConfiguration(
            optimizer: Adam(learningRate: 0.001),
            objective: MSELoss(),
            metrics: [Accuracy(), MeanSquaredError()],
            learningRateSchedule: CosineDecay(initialLR: 0.001, totalSteps: 1000)
        )
        #expect(config.metrics.count == 2)
        #expect(config.learningRateSchedule != nil)
    }

    @Test func configurationsAreIndependentValues() {
        let config1 = TrainingConfiguration(
            optimizer: Adam(learningRate: 0.001),
            objective: MSELoss()
        )
        let config2 = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.01),
            objective: CrossEntropyLoss()
        )
        // Both configs exist independently
        #expect(config1.metrics.isEmpty)
        #expect(config2.metrics.isEmpty)
    }
}

@Suite("TrainingError")
struct TrainingErrorTests {
    @Test func invalidEpochCountHasActionableMessage() {
        let error = TrainingError.invalidEpochCount(0)
        #expect(error.errorDescription?.contains("greater than zero") == true)
    }

    @Test func unsupportedScheduledOptimizerFailsBeforeDataIteration() async throws {
        let configuration = TrainingConfiguration(
            optimizer: MultiOptimizer(optimizers: [SGD(learningRate: 0.1)]),
            objective: ClosureObjective<Int, Int> { _, _ in MLXArray(0) },
            learningRateSchedule: FixedLearningRateSchedule()
        )

        do {
            _ = try await TrainingEngine().fit(
                IntegerModel(),
                configuration: configuration,
                data: EmptyBatches<Batch<Int, Int>>(),
                epochs: 1
            )
            Issue.record("Expected unsupported scheduled optimizer to be rejected")
        } catch let error as TrainingError {
            #expect(error == .unsupportedLearningRateSchedule)
        }
    }

    @Test func duplicateMetricNamesFailBeforeTraining() async throws {
        let metricA = IntegerMetric(name: "score")
        let metricB = IntegerMetric(name: "score")
        let configuration = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.1),
            objective: ClosureObjective<Int, Int> { _, _ in MLXArray(0) },
            metrics: [metricA, metricB]
        )

        do {
            _ = try await TrainingEngine().fit(
                IntegerModel(),
                configuration: configuration,
                data: EmptyBatches<Batch<Int, Int>>(),
                epochs: 1
            )
            Issue.record("Expected duplicate metric names to be rejected")
        } catch let error as TrainingError {
            #expect(error == .metricNameCollision("score"))
        }
    }

    @Test func emptyTrainingSequenceThrowsInsteadOfRecordingZeroLoss() async throws {
        let configuration = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.1),
            objective: ClosureObjective<Int, Int> { _, _ in MLXArray(0) }
        )

        do {
            _ = try await TrainingEngine().fit(
                IntegerModel(),
                configuration: configuration,
                data: EmptyBatches<Batch<Int, Int>>(),
                epochs: 1
            )
            Issue.record("Expected empty training data to be rejected")
        } catch let error as TrainingError {
            #expect(error == .emptyTrainingData)
        }
    }

    @Test func nonpositiveBatchSampleCountThrowsBeforeForwardPass() async throws {
        let configuration = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.1),
            objective: ClosureObjective<Int, Int> { _, _ in MLXArray(0) }
        )

        do {
            _ = try await TrainingEngine().fit(
                IntegerModel(),
                configuration: configuration,
                data: FixedBatches(batches: [Batch(input: 1, target: 1, sampleCount: 0)]),
                epochs: 1
            )
            Issue.record("Expected a nonpositive sample count to be rejected")
        } catch let error as TrainingError {
            #expect(error == .invalidSampleCount(0))
        }
    }

    @Test func emptyEvaluationThrowsAndRestoresModelMode() async throws {
        let model = IntegerModel()
        model.train(true)
        let configuration = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.1),
            objective: ClosureObjective<Int, Int> { _, _ in MLXArray(0) }
        )

        do {
            _ = try await TrainingEngine().evaluate(
                model,
                configuration: configuration,
                data: EmptyBatches<Batch<Int, Int>>()
            )
            Issue.record("Expected empty evaluation data to be rejected")
        } catch let error as TrainingError {
            #expect(error == .emptyEvaluationData)
        }

        #expect(model.training)
    }

    @Test(arguments: ["", "loss", "val_accuracy"])
    func reservedMetricNamesFailBeforeTraining(name: String) async throws {
        let configuration = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.1),
            objective: ClosureObjective<Int, Int> { _, _ in MLXArray(0) },
            metrics: [IntegerMetric(name: name)]
        )

        do {
            _ = try await TrainingEngine().fit(
                IntegerModel(),
                configuration: configuration,
                data: EmptyBatches<Batch<Int, Int>>(),
                epochs: 1
            )
            Issue.record("Expected reserved metric name to be rejected")
        } catch let error as TrainingError {
            #expect(error == .metricNameCollision(name))
        }
    }
}
