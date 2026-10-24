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

private final class TwoParameterScalarModel: Module, TrainableModel {
    typealias Input = MLXArray
    typealias Output = MLXArray

    let weight = MLXArray(Float32(1))
    let bias = MLXArray(Float32(1))

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        input * weight + bias
    }
}

private final class RecordingGradientOptimizer: LearningRateControllable {
    private(set) var receivedGradients: [ModuleParameters] = []
    private(set) var learningRates: [Float] = []

    func update(model: Module, gradients: ModuleParameters) {
        receivedGradients.append(gradients)
    }

    func setLearningRate(_ value: Float) throws {
        learningRates.append(value)
    }

    func innerState() -> [MLXArray] { [] }
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

private enum GradientTransformTestError: Error, Equatable {
    case expected
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

private struct BatchSequence: AsyncSequence {
    typealias Element = Batch<MLXArray, MLXArray>
    typealias Failure = Never

    let batches: [Element]

    func makeAsyncIterator() -> Iterator { Iterator(batches: batches) }

    struct Iterator: AsyncIteratorProtocol {
        typealias Element = Batch<MLXArray, MLXArray>
        typealias Failure = Never

        let batches: [Element]
        var offset = 0

        mutating func next() async -> Element? {
            guard offset < batches.count else { return nil }
            defer { offset += 1 }
            return batches[offset]
        }
    }
}

private struct WeightedClassificationTarget {
    let labels: MLXArray
    let sampleWeights: MLXArray
    let classWeights: MLXArray
}

private struct WeightedClassificationBatchSequence: AsyncSequence {
    typealias Element = Batch<MLXArray, WeightedClassificationTarget>
    typealias Failure = Never

    let batches: [Element]

    func makeAsyncIterator() -> Iterator { Iterator(batches: batches) }

    struct Iterator: AsyncIteratorProtocol {
        typealias Element = Batch<MLXArray, WeightedClassificationTarget>

        let batches: [Element]
        var offset = 0

        mutating func next() async -> Element? {
            guard offset < batches.count else { return nil }
            defer { offset += 1 }
            return batches[offset]
        }
    }
}

private final class LogitScaleModel: Module, TrainableModel {
    typealias Input = MLXArray
    typealias Output = MLXArray

    let scale = MLXArray(Float32(1))

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        input * scale
    }
}

private func regressionBatch(_ values: [Float32]) -> Batch<MLXArray, MLXArray> {
    let inputs = MLXArray(values).reshaped([values.count, 1])
    return Batch(input: inputs, target: MLXArray.zeros([values.count, 1]), sampleCount: values.count)
}

private final class CancelAtBatchEnd: Callback {
    func onBatchEnd(state: TrainingState) {
        withUnsafeCurrentTask { $0?.cancel() }
    }
}

@Suite("TrainingEngine fit() (GPU)", .serialized)
struct TrainingEngineFitGPUTests {
    @Test func invalidWeightedCrossEntropySkipsOptimizerUpdate() async throws {
        let loss = CrossEntropyLoss()
        let batch = Batch(
            input: MLXArray([2.0, -1.0, 0.5] as [Float32]).reshaped([1, 3]),
            target: WeightedClassificationTarget(
                labels: MLXArray([0] as [Int32]),
                sampleWeights: MLXArray([-1.0] as [Float32]),
                classWeights: MLXArray([1.0, 1.0, 1.0] as [Float32])
            ),
            sampleCount: 1
        )
        let optimizer = RecordingGradientOptimizer()
        let objective = ClosureObjective<MLXArray, WeightedClassificationTarget> { output, target in
            loss.weightedMean(
                predictions: output,
                targets: target.labels,
                sampleWeights: target.sampleWeights,
                classWeights: target.classWeights
            )
        }
        let configuration = TrainingConfiguration(optimizer: optimizer, objective: objective)
        let step: TrainingStep<LogitScaleModel, WeightedClassificationTarget> = { model, batch in
            loss.weightedMean(
                predictions: model(batch.input),
                targets: batch.target.labels,
                sampleWeights: batch.target.sampleWeights,
                classWeights: batch.target.classWeights
            )
        }

        do {
            _ = try await TrainingEngine().fit(
                LogitScaleModel(),
                configuration: configuration,
                data: WeightedClassificationBatchSequence(batches: [batch]),
                epochs: 1,
                trainingStep: step
            )
            Issue.record("Invalid weights must fail before an optimizer update")
        } catch let error as TrainingError {
            #expect(error == .nonFiniteObjective)
        }

        #expect(optimizer.receivedGradients.isEmpty)
    }

    @Test func weightedCrossEntropyFitAndEvaluateAgreeAcrossUnevenBatches() async throws {
        let loss = CrossEntropyLoss(labelSmoothing: 0.1)
        let classWeights = MLXArray([1.2, 0.4, 2.5] as [Float32])
        let batches = WeightedClassificationBatchSequence(batches: [
            Batch(
                input: MLXArray([2.0, -1.0, 0.5, 1.0, 3.0, -0.5] as [Float32]).reshaped([2, 3]),
                target: WeightedClassificationTarget(
                    labels: MLXArray([0, 2] as [Int32]),
                    sampleWeights: MLXArray([0.5, 2.0] as [Float32]),
                    classWeights: classWeights
                ),
                sampleCount: 2
            ),
            Batch(
                input: MLXArray([-0.2, 0.7, 1.4] as [Float32]).reshaped([1, 3]),
                target: WeightedClassificationTarget(
                    labels: MLXArray([1] as [Int32]),
                    sampleWeights: MLXArray([1.5] as [Float32]),
                    classWeights: classWeights
                ),
                sampleCount: 1
            ),
        ])
        let weightedStep: TrainingStep<LogitScaleModel, WeightedClassificationTarget> = { model, batch in
            loss.weightedMean(
                predictions: model(batch.input),
                targets: batch.target.labels,
                sampleWeights: batch.target.sampleWeights,
                classWeights: batch.target.classWeights
            )
        }
        let objective = ClosureObjective<MLXArray, WeightedClassificationTarget> { output, target in
            loss.weightedMean(
                predictions: output,
                targets: target.labels,
                sampleWeights: target.sampleWeights,
                classWeights: target.classWeights
            )
        }
        let configuration = TrainingConfiguration(
            optimizer: RecordingGradientOptimizer(),
            objective: objective
        )
        let model = LogitScaleModel()
        let engine = TrainingEngine()

        let history = try await engine.fit(
            model,
            configuration: configuration,
            data: batches,
            epochs: 1,
            trainingStep: weightedStep
        )
        let evaluation = try await engine.evaluate(
            model,
            configuration: configuration,
            data: batches,
            trainingStep: weightedStep
        )
        let expected = loss.weightedMean(
            predictions: MLXArray([
                2.0, -1.0, 0.5,
                1.0, 3.0, -0.5,
                -0.2, 0.7, 1.4,
            ] as [Float32]).reshaped([3, 3]),
            targets: MLXArray([0, 2, 1] as [Int32]),
            sampleWeights: MLXArray([0.5, 2.0, 1.5] as [Float32]),
            classWeights: classWeights
        ).item(Float.self)

        #expect(history.epochs.count == 1)
        #expect(abs((history.epochs[0]["loss"] ?? .nan) - expected) < 1e-5)
        #expect(abs((evaluation["loss"] ?? .nan) - expected) < 1e-5)
    }

    @Test func sampleAwareAccumulationMatchesCombinedBatchMeanForEqualAndUnevenMicrobatches() async throws {
        let microbatchExamples: [[[Float32]]] = [
            [[1], [2]],
            [[1], [2, 3]],
        ]

        for examples in microbatchExamples {
            let accumulatedOptimizer = RecordingGradientOptimizer()
            let accumulatedBatches = BatchSequence(batches: examples.map(regressionBatch))
            let accumulatedConfiguration = TrainingConfiguration(
                optimizer: accumulatedOptimizer,
                objective: MSELoss(),
                gradientAccumulationSteps: examples.count
            )
            _ = try await TrainingEngine().fit(
                ScalarLinearModel(),
                configuration: accumulatedConfiguration,
                data: accumulatedBatches,
                epochs: 1
            )

            let referenceOptimizer = RecordingGradientOptimizer()
            let combinedBatch = regressionBatch(examples.flatMap { $0 })
            let referenceConfiguration = TrainingConfiguration(
                optimizer: referenceOptimizer,
                objective: MSELoss()
            )
            _ = try await TrainingEngine().fit(
                ScalarLinearModel(),
                configuration: referenceConfiguration,
                data: BatchSequence(batches: [combinedBatch]),
                epochs: 1
            )

            #expect(accumulatedOptimizer.receivedGradients.count == 1)
            #expect(referenceOptimizer.receivedGradients.count == 1)
            let accumulated = accumulatedOptimizer.receivedGradients[0].flattened()[0].1
                .item(Float.self)
            let reference = referenceOptimizer.receivedGradients[0].flattened()[0].1
                .item(Float.self)
            #expect(abs(accumulated - reference) < 1e-5)

            let accumulatedModel = ScalarLinearModel()
            let accumulatedUpdateConfiguration = TrainingConfiguration(
                optimizer: SGD(learningRate: 0.01),
                objective: MSELoss(),
                gradientAccumulationSteps: examples.count
            )
            _ = try await TrainingEngine().fit(
                accumulatedModel,
                configuration: accumulatedUpdateConfiguration,
                data: accumulatedBatches,
                epochs: 1
            )

            let referenceModel = ScalarLinearModel()
            let referenceUpdateConfiguration = TrainingConfiguration(
                optimizer: SGD(learningRate: 0.01),
                objective: MSELoss()
            )
            _ = try await TrainingEngine().fit(
                referenceModel,
                configuration: referenceUpdateConfiguration,
                data: BatchSequence(batches: [combinedBatch]),
                epochs: 1
            )
            #expect(abs(
                accumulatedModel.weight.item(Float.self)
                    - referenceModel.weight.item(Float.self)
            ) < 1e-5)
        }
    }

    @Test func accumulatedTransformedAndClippedGradientMatchesDirectMLX() async throws {
        let batches = BatchSequence(batches: [
            regressionBatch([1]),
            regressionBatch([2, 3]),
        ])
        let recordedOptimizer = RecordingGradientOptimizer()
        let configuration = TrainingConfiguration(
            optimizer: recordedOptimizer,
            objective: MSELoss(),
            gradientProcessing: GradientProcessingConfiguration(
                transforms: [{ gradients in gradients.mapValues { $0 * 2 } }],
                globalNormClip: 0.5
            ),
            gradientAccumulationSteps: 2
        )

        _ = try await TrainingEngine().fit(
            ScalarLinearModel(),
            configuration: configuration,
            data: batches,
            epochs: 1
        )

        let directModel = ScalarLinearModel()
        let combinedInputs = MLXArray([1.0, 2.0, 3.0] as [Float32]).reshaped([3, 1])
        let combinedTargets = MLXArray.zeros([3, 1])
        let directLossAndGrad = valueAndGrad(model: directModel) {
            (model: ScalarLinearModel, features: MLXArray, targets: MLXArray) -> MLXArray in
            MSELoss()(predictions: model(features), targets: targets)
        }
        let (_, rawGradients) = directLossAndGrad(directModel, combinedInputs, combinedTargets)
        let transformedGradients = rawGradients.mapValues { $0 * 2 }
        let globalNorm = transformedGradients.flattened().reduce(Float.zero) {
            $0 + ($1.1 * $1.1).sum().item(Float.self)
        }.squareRoot()
        let clipScale = min(Float(1), Float(0.5) / globalNorm)
        let expectedGradients = transformedGradients.mapValues { $0 * clipScale }

        #expect(recordedOptimizer.receivedGradients.count == 1)
        let actualLeaves = Dictionary(uniqueKeysWithValues: recordedOptimizer.receivedGradients[0].flattened())
        let expectedLeaves = Dictionary(uniqueKeysWithValues: expectedGradients.flattened())
        #expect(Set(actualLeaves.keys) == Set(expectedLeaves.keys))
        for key in actualLeaves.keys {
            #expect(actualLeaves[key]!.allClose(expectedLeaves[key]!, rtol: 1e-5, atol: 1e-5).all().item(Bool.self))
        }

        let actualModel = ScalarLinearModel()
        _ = try await TrainingEngine().fit(
            actualModel,
            configuration: TrainingConfiguration(
                optimizer: SGD(learningRate: 0.01),
                objective: MSELoss(),
                gradientProcessing: GradientProcessingConfiguration(
                    transforms: [{ gradients in gradients.mapValues { $0 * 2 } }],
                    globalNormClip: 0.5
                ),
                gradientAccumulationSteps: 2
            ),
            data: batches,
            epochs: 1
        )
        SGD(learningRate: 0.01).update(model: directModel, gradients: expectedGradients)
        eval(directModel)

        #expect(abs(actualModel.weight.item(Float.self) - directModel.weight.item(Float.self)) < 1e-5)
    }

    @Test func fullAndPartialWindowsEachUpdateOnceAndUseGradientProcessing() async throws {
        for batchCount in [4, 5] {
            let optimizer = RecordingGradientOptimizer()
            let transformCalls = LockedInt()
            let configuration = TrainingConfiguration(
                optimizer: optimizer,
                objective: MSELoss(),
                gradientProcessing: GradientProcessingConfiguration(
                    transforms: [{ gradients in
                        transformCalls.increment()
                        return gradients.mapValues { $0 * 2 }
                    }],
                    globalNormClip: 1
                ),
                gradientAccumulationSteps: 2
            )
            let batches = BatchSequence(
                batches: (1...batchCount).map { regressionBatch([Float32($0)]) }
            )

            _ = try await TrainingEngine().fit(
                ScalarLinearModel(),
                configuration: configuration,
                data: batches,
                epochs: 1
            )

            let expectedUpdates = (batchCount + 1) / 2
            #expect(optimizer.receivedGradients.count == expectedUpdates)
            #expect(transformCalls.value == expectedUpdates)
            for gradients in optimizer.receivedGradients {
                let norm = gradients.flattened().reduce(Float.zero) {
                    $0 + $1.1.square().sum().item(Float.self)
                }.squareRoot()
                #expect(norm <= 1.0001)
            }
        }
    }

    @Test func earlyStopFlushesThePendingPartialWindow() async throws {
        let optimizer = RecordingGradientOptimizer()
        let schedule = RecordingLearningRateSchedule()
        let configuration = TrainingConfiguration(
            optimizer: optimizer,
            objective: MSELoss(),
            learningRateSchedule: schedule,
            gradientAccumulationSteps: 3
        )
        let history = try await TrainingEngine().fit(
            ScalarLinearModel(),
            configuration: configuration,
            data: BatchSequence(batches: [
                regressionBatch([2]), regressionBatch([4]), regressionBatch([6]),
            ]),
            epochs: 1,
            callbacks: [StopAtBatchEnd()]
        )

        #expect(optimizer.receivedGradients.count == 1)
        #expect(abs(optimizer.receivedGradients[0].flattened()[0].1.item(Float.self) - 8) < 1e-5)
        #expect(schedule.steps == [0])
        #expect(history.epochs.count == 1)
    }

    @Test func accumulationWindowsDoNotSpanEpochs() async throws {
        let optimizer = RecordingGradientOptimizer()
        let configuration = TrainingConfiguration(
            optimizer: optimizer,
            objective: MSELoss(),
            gradientAccumulationSteps: 4
        )
        let batches = BatchSequence(batches: [
            regressionBatch([1]), regressionBatch([2]), regressionBatch([3]),
        ])

        _ = try await TrainingEngine().fit(
            ScalarLinearModel(),
            configuration: configuration,
            data: batches,
            epochs: 2
        )

        // Each epoch flushes its own three-batch partial window.
        #expect(optimizer.receivedGradients.count == 2)
    }

    @Test func accumulationPreservesSampleWeightedLogsMetricsAndUpdateBasedSchedule() async throws {
        let optimizer = RecordingGradientOptimizer()
        let schedule = RecordingLearningRateSchedule()
        let configuration = TrainingConfiguration(
            optimizer: optimizer,
            objective: MSELoss(),
            metrics: [MeanOutputMetric()],
            learningRateSchedule: schedule,
            gradientAccumulationSteps: 2
        )
        let history = try await TrainingEngine().fit(
            ScalarLinearModel(),
            configuration: configuration,
            data: BatchSequence(batches: [
                regressionBatch([1]), regressionBatch([2, 3]), regressionBatch([4]),
            ]),
            epochs: 1
        )

        #expect(optimizer.receivedGradients.count == 2)
        #expect(schedule.steps == [0, 1])
        #expect(optimizer.learningRates == [0.001, 0.002])
        #expect(abs((history["loss"].compactMap { $0 }.first ?? 0) - 7.5) < 1e-5)
        #expect(abs((history["mean_output"].compactMap { $0 }.first ?? 0) - 2.5) < 1e-5)
    }

    @Test func invalidAccumulationWindowFailsBeforeTraining() async throws {
        for steps in [0, -1] {
            let optimizer = RecordingGradientOptimizer()
            let configuration = TrainingConfiguration(
                optimizer: optimizer,
                objective: MSELoss(),
                gradientAccumulationSteps: steps
            )
            do {
                _ = try await TrainingEngine().fit(
                    ScalarLinearModel(),
                    configuration: configuration,
                    data: BatchSequence(batches: [regressionBatch([1])]),
                    epochs: 1
                )
                Issue.record("Expected an invalid accumulation window to fail")
            } catch let error as TrainingError {
                #expect(error == .invalidGradientAccumulationSteps(steps))
            }
            #expect(optimizer.receivedGradients.isEmpty)
        }
    }

    @Test func callbackFailureDiscardsPendingAccumulatedGradients() async throws {
        let optimizer = RecordingGradientOptimizer()
        let configuration = TrainingConfiguration(
            optimizer: optimizer,
            objective: MSELoss(),
            gradientAccumulationSteps: 3
        )
        let callback = LambdaCallback(onBatchEnd: { _ in
            throw TrainingCallbackTestError.first
        })

        do {
            _ = try await TrainingEngine().fit(
                ScalarLinearModel(),
                configuration: configuration,
                data: BatchSequence(batches: [regressionBatch([1]), regressionBatch([2])]),
                epochs: 1,
                callbacks: [callback]
            )
            Issue.record("Expected callback failure")
        } catch let error as TrainingCallbackTestError {
            #expect(error == .first)
        }
        #expect(optimizer.receivedGradients.isEmpty)
    }

    @Test func sourceFailureDiscardsPendingAccumulatedGradients() async throws {
        let optimizer = RecordingGradientOptimizer()
        let configuration = TrainingConfiguration(
            optimizer: optimizer,
            objective: MSELoss(),
            gradientAccumulationSteps: 3
        )
        let source = AsyncThrowingStream<Batch<MLXArray, MLXArray>, Error> { continuation in
            continuation.yield(regressionBatch([1]))
            continuation.finish(throwing: TrainingCallbackTestError.first)
        }

        do {
            _ = try await TrainingEngine().fit(
                ScalarLinearModel(),
                configuration: configuration,
                data: source,
                epochs: 1
            )
            Issue.record("Expected source failure")
        } catch let error as TrainingCallbackTestError {
            #expect(error == .first)
        }
        #expect(optimizer.receivedGradients.isEmpty)
    }

    @Test func cancellationDiscardsPendingAccumulatedGradients() async throws {
        let optimizer = RecordingGradientOptimizer()
        let configuration = TrainingConfiguration(
            optimizer: optimizer,
            objective: MSELoss(),
            gradientAccumulationSteps: 3
        )

        do {
            _ = try await TrainingEngine().fit(
                ScalarLinearModel(),
                configuration: configuration,
                data: BatchSequence(batches: [regressionBatch([1]), regressionBatch([2])]),
                epochs: 1,
                callbacks: [CancelAtBatchEnd()]
            )
            Issue.record("Expected cancellation")
        } catch is CancellationError {
            // Expected.
        }
        #expect(optimizer.receivedGradients.isEmpty)
    }

    @Test func gradientTransformsRunInOrderAndOptimizerReceivesOneProcessedTreePerBatch() async throws {
        let model = ScalarLinearModel()
        let inputs = MLXArray([1.0, 2.0] as [Float32]).reshaped([2, 1])
        let targets = MLXArray.zeros([2, 1])
        let loader = try DataLoader(
            source: InMemoryDataset(features: inputs, labels: targets),
            batchSize: 1
        )
        let optimizer = RecordingGradientOptimizer()
        let schedule = RecordingLearningRateSchedule()
        var transformOrder: [Int] = []
        let configuration = TrainingConfiguration(
            optimizer: optimizer,
            objective: MSELoss(),
            learningRateSchedule: schedule,
            gradientProcessing: GradientProcessingConfiguration(transforms: [
                { gradients in
                    transformOrder.append(1)
                    return gradients.mapValues { $0 * 2 }
                },
                { gradients in
                    transformOrder.append(2)
                    return gradients.mapValues { $0 + 1 }
                },
            ], globalNormClip: 10)
        )

        _ = try await TrainingEngine().fit(
            model,
            configuration: configuration,
            data: loader,
            epochs: 1
        )

        #expect(transformOrder == [1, 2, 1, 2])
        #expect(optimizer.receivedGradients.count == 2)
        #expect(schedule.steps == [0, 1])
        #expect(optimizer.learningRates == [0.001, 0.002])
        for (index, gradients) in optimizer.receivedGradients.enumerated() {
            #expect(gradients.flattened().count == 1)
            let expected = index == 0 ? Float(5) : Float(10)
            #expect(abs(gradients.flattened()[0].1.item(Float.self) - expected) < 1e-5)
        }
    }

    @Test func globalNormClipScalesAcrossAllParameters() async throws {
        let model = TwoParameterScalarModel()
        let inputs = MLXArray([1.0] as [Float32])
        let targets = MLXArray.zeros([1])
        let loader = try DataLoader(
            source: InMemoryDataset(
                features: inputs.reshaped([1, 1]),
                labels: targets.reshaped([1, 1])
            ),
            batchSize: 1
        )
        let optimizer = RecordingGradientOptimizer()
        let configuration = TrainingConfiguration(
            optimizer: optimizer,
            objective: MSELoss(),
            gradientProcessing: GradientProcessingConfiguration(globalNormClip: 2)
        )

        _ = try await TrainingEngine().fit(
            model,
            configuration: configuration,
            data: loader,
            epochs: 1
        )

        #expect(optimizer.receivedGradients.count == 1)
        let leaves = Dictionary(
            uniqueKeysWithValues: optimizer.receivedGradients[0].flattened()
        )
        #expect(Set(leaves.keys) == Set(["weight", "bias"]))
        let weightGradient = leaves["weight"]!.item(Float.self)
        let biasGradient = leaves["bias"]!.item(Float.self)
        let expectedLeafGradient = Float(2).squareRoot()
        let actualGlobalNorm = (
            weightGradient * weightGradient + biasGradient * biasGradient
        ).squareRoot()
        #expect(abs(weightGradient - expectedLeafGradient) < 1e-4)
        #expect(abs(biasGradient - expectedLeafGradient) < 1e-4)
        #expect(abs(actualGlobalNorm - 2) < 1e-4)
    }

    @Test func globalNormClipLeavesGradientsBelowThresholdUnchanged() async throws {
        let model = TwoParameterScalarModel()
        let inputs = MLXArray([1.0] as [Float32])
        let targets = MLXArray.zeros([1])
        let loader = try DataLoader(
            source: InMemoryDataset(
                features: inputs.reshaped([1, 1]),
                labels: targets.reshaped([1, 1])
            ),
            batchSize: 1
        )
        let optimizer = RecordingGradientOptimizer()
        let configuration = TrainingConfiguration(
            optimizer: optimizer,
            objective: MSELoss(),
            gradientProcessing: GradientProcessingConfiguration(globalNormClip: 10)
        )

        _ = try await TrainingEngine().fit(
            model,
            configuration: configuration,
            data: loader,
            epochs: 1
        )

        let leaves = Dictionary(
            uniqueKeysWithValues: optimizer.receivedGradients[0].flattened()
        )
        #expect(abs(leaves["weight"]!.item(Float.self) - 4) < 1e-5)
        #expect(abs(leaves["bias"]!.item(Float.self) - 4) < 1e-5)
    }

    @Test func invalidGlobalNormClipFailsBeforeAnyUpdate() async throws {
        for maxNorm in [Float(0), -1, .nan, .infinity] {
            let model = ScalarLinearModel()
            let loader = try DataLoader(
                source: InMemoryDataset(
                    features: MLXArray([1.0] as [Float32]).reshaped([1, 1]),
                    labels: MLXArray.zeros([1, 1])
                ),
                batchSize: 1
            )
            let optimizer = RecordingGradientOptimizer()
            let configuration = TrainingConfiguration(
                optimizer: optimizer,
                objective: MSELoss(),
                gradientProcessing: GradientProcessingConfiguration(globalNormClip: maxNorm)
            )

            do {
                _ = try await TrainingEngine().fit(
                    model,
                    configuration: configuration,
                    data: loader,
                    epochs: 1
                )
                Issue.record("Expected invalid global-norm clip to fail")
            } catch let error as TrainingError {
                if case .invalidGlobalNormClip = error {
                    // Expected.
                } else {
                    Issue.record("Unexpected error: \(error)")
                }
            }

            #expect(optimizer.receivedGradients.isEmpty)
        }
    }

    @Test func nonFiniteTransformedGradientsFailBeforeAnyUpdate() async throws {
        let model = ScalarLinearModel()
        let loader = try DataLoader(
            source: InMemoryDataset(
                features: MLXArray([1.0] as [Float32]).reshaped([1, 1]),
                labels: MLXArray.zeros([1, 1])
            ),
            batchSize: 1
        )
        let optimizer = RecordingGradientOptimizer()
        let configuration = TrainingConfiguration(
            optimizer: optimizer,
            objective: MSELoss(),
            gradientProcessing: GradientProcessingConfiguration(transforms: [
                { gradients in gradients.mapValues { _ in MLXArray(Float32.nan) } },
            ])
        )

        do {
            _ = try await TrainingEngine().fit(
                model,
                configuration: configuration,
                data: loader,
                epochs: 1
            )
            Issue.record("Expected non-finite transformed gradients to fail")
        } catch let error as TrainingError {
            #expect(error == .nonFiniteGradient)
        }

        #expect(optimizer.receivedGradients.isEmpty)
    }

    @Test func transformedGradientKeysAndShapesMustMatch() async throws {
        let transforms: [GradientTransform] = [
            { _ in ModuleParameters.unflattened([("unexpected", MLXArray(Float32(1)))]) },
            { gradients in gradients.mapValues { _ in MLXArray.zeros([2]) } },
        ]

        for (index, transform) in transforms.enumerated() {
            let model = ScalarLinearModel()
            let loader = try DataLoader(
                source: InMemoryDataset(
                    features: MLXArray([1.0] as [Float32]).reshaped([1, 1]),
                    labels: MLXArray.zeros([1, 1])
                ),
                batchSize: 1
            )
            let optimizer = RecordingGradientOptimizer()
            let configuration = TrainingConfiguration(
                optimizer: optimizer,
                objective: MSELoss(),
                gradientProcessing: GradientProcessingConfiguration(transforms: [transform])
            )

            do {
                _ = try await TrainingEngine().fit(
                    model,
                    configuration: configuration,
                    data: loader,
                    epochs: 1
                )
                Issue.record("Expected incompatible transformed gradients to fail")
            } catch let error as TrainingError {
                if index == 0 {
                    if case .gradientTreeKeyMismatch = error {
                        // Expected.
                    } else {
                        Issue.record("Unexpected error: \(error)")
                    }
                } else if case .gradientShapeMismatch = error {
                    // Expected.
                } else {
                    Issue.record("Unexpected error: \(error)")
                }
            }

            #expect(optimizer.receivedGradients.isEmpty)
        }
    }

    @Test func throwingGradientTransformPropagatesWithoutUpdating() async throws {
        let model = ScalarLinearModel()
        let loader = try DataLoader(
            source: InMemoryDataset(
                features: MLXArray([1.0] as [Float32]).reshaped([1, 1]),
                labels: MLXArray.zeros([1, 1])
            ),
            batchSize: 1
        )
        let optimizer = RecordingGradientOptimizer()
        let configuration = TrainingConfiguration(
            optimizer: optimizer,
            objective: MSELoss(),
            gradientProcessing: GradientProcessingConfiguration(transforms: [
                { _ in throw GradientTransformTestError.expected },
            ])
        )

        do {
            _ = try await TrainingEngine().fit(
                model,
                configuration: configuration,
                data: loader,
                epochs: 1
            )
            Issue.record("Expected the gradient transform error to propagate")
        } catch let error as GradientTransformTestError {
            #expect(error == .expected)
        }

        #expect(optimizer.receivedGradients.isEmpty)
    }

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
