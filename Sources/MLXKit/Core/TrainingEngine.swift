import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Computes a scalar objective from a model and its complete batch. MLXKit
/// retains responsibility for autodiff, optimizer updates, and lifecycle.
public typealias TrainingStep<Model: TrainableModel, Target> =
    (Model, Batch<Model.Input, Target>) -> MLXArray

/// Coordinates training, evaluation, and prediction while leaving model
/// parameters, tensor operations, autodiff, and optimizer algorithms to MLX.
public final class TrainingEngine: @unchecked Sendable {
    public init() {}

    /// Trains a model from a typed asynchronous sequence of batches.
    public func fit<Model: TrainableModel, Target, Data: AsyncSequence>(
        _ model: Model,
        configuration: TrainingConfiguration<Model.Output, Target>,
        data: Data,
        epochs: Int,
        trainingStep: TrainingStep<Model, Target>? = nil,
        callbacks: [any Callback] = []
    ) async throws -> History
    where Data.Element == Batch<Model.Input, Target>, Data.Failure: Error {
        try await fitCore(
            model,
            configuration: configuration,
            data: data,
            epochs: epochs,
            validationData: EmptyBatchSequence<Batch<Model.Input, Target>>(),
            hasValidationData: false,
            trainingStep: trainingStep,
            callbacks: callbacks
        )
    }

    /// Trains a model and evaluates it against a validation sequence after
    /// each epoch.
    public func fit<
        Model: TrainableModel,
        Target,
        Data: AsyncSequence,
        ValidationData: AsyncSequence
    >(
        _ model: Model,
        configuration: TrainingConfiguration<Model.Output, Target>,
        data: Data,
        epochs: Int,
        validationData: ValidationData,
        trainingStep: TrainingStep<Model, Target>? = nil,
        callbacks: [any Callback] = []
    ) async throws -> History
    where
        Data.Element == Batch<Model.Input, Target>,
        Data.Failure: Error,
        ValidationData.Element == Batch<Model.Input, Target>,
        ValidationData.Failure: Error
    {
        try await fitCore(
            model,
            configuration: configuration,
            data: data,
            epochs: epochs,
            validationData: validationData,
            hasValidationData: true,
            trainingStep: trainingStep,
            callbacks: callbacks
        )
    }

    private func fitCore<
        Model: TrainableModel,
        Target,
        Data: AsyncSequence,
        ValidationData: AsyncSequence
    >(
        _ model: Model,
        configuration: TrainingConfiguration<Model.Output, Target>,
        data: Data,
        epochs: Int,
        validationData: ValidationData,
        hasValidationData: Bool,
        trainingStep: TrainingStep<Model, Target>?,
        callbacks: [any Callback]
    ) async throws -> History
    where
        Data.Element == Batch<Model.Input, Target>,
        Data.Failure: Error,
        ValidationData.Element == Batch<Model.Input, Target>,
        ValidationData.Failure: Error
    {
        guard epochs > 0 else {
            throw TrainingError.invalidEpochCount(epochs)
        }
        try validateMetricNames(configuration.metrics)
        try validateGradientProcessing(configuration.gradientProcessing)
        try Task.checkCancellation()

        let learningRateController: (any LearningRateControllable)?
        if configuration.learningRateSchedule != nil {
            guard let controller = configuration.optimizer as? any LearningRateControllable else {
                throw TrainingError.unsupportedLearningRateSchedule
            }
            learningRateController = controller
        } else {
            learningRateController = nil
        }

        let history = History()
        let callbackList = CallbackList([history] + callbacks)
        let state = TrainingState()
        state.totalEpochs = epochs
        let wasTraining = model.training
        defer { model.train(wasTraining) }

        var globalStep = 0
        let objectiveShape = ObjectiveShapeRecorder()
        let lossAndGrad = valueAndGrad(model: model) { (differentiableModel: Model, batch: Batch<Model.Input, Target>) in
            let output = differentiableModel(batch.input)
            let objective: MLXArray
            if let trainingStep {
                objective = trainingStep(differentiableModel, batch)
            } else {
                objective = configuration.objective(output: output, target: batch.target)
            }
            if objective.ndim != 0 {
                objectiveShape.nonScalarShape = objective.shape
            }
            // MLX's valueAndGrad requires scalar outputs. Sum preserves a true
            // scalar objective and lets us report vector objectives safely.
            return [objective.sum()]
        }

        var trainBeginStarted = false
        var trainEndStarted = false

        do {
            trainBeginStarted = true
            try callbackList.onTrainBegin(state: state)
            try Task.checkCancellation()

            for epoch in 0..<epochs {
                state.epoch = epoch
                state.batch = 0
                state.logs = [:]
                state.shouldStop = false

                model.train(true)
                for metric in configuration.metrics {
                    metric.reset()
                }
                try callbackList.onEpochBegin(state: state)
                try Task.checkCancellation()

                var weightedLoss: Float = 0
                var lossWeight: Int = 0
                var batchIndex = 0

                for try await batch in data {
                    try Task.checkCancellation()
                    state.batch = batchIndex
                    guard batch.sampleCount > 0 else {
                        throw TrainingError.invalidSampleCount(batch.sampleCount)
                    }

                    if let schedule = configuration.learningRateSchedule {
                        let learningRate = schedule.learningRate(atStep: globalStep)
                        try learningRateController?.setLearningRate(learningRate)
                    }

                    try callbackList.onBatchBegin(state: state)
                    try Task.checkCancellation()

                    objectiveShape.nonScalarShape = nil
                    let (values, gradients) = lossAndGrad(model, batch)
                    if let shape = objectiveShape.nonScalarShape {
                        throw TrainingError.nonScalarObjective(shape: shape)
                    }
                    guard let lossValue = values.first else {
                        throw TrainingError.nonScalarObjective(shape: [])
                    }
                    guard lossValue.ndim == 0 else {
                        throw TrainingError.nonScalarObjective(shape: lossValue.shape)
                    }
                    eval(lossValue, gradients)
                    let scalarLoss = lossValue.item(Float.self)
                    guard scalarLoss.isFinite else {
                        throw TrainingError.nonFiniteObjective
                    }

                    try validateFiniteGradients(gradients)

                    // Metrics observe predictions from the pre-update model.
                    let metricOutput = model(batch.input)
                    eval(metricOutput)
                    for metric in configuration.metrics {
                        metric.update(output: metricOutput, target: batch.target)
                    }

                    weightedLoss += scalarLoss * Float(batch.sampleCount)
                    lossWeight += batch.sampleCount

                    let processedGradients = try processGradients(
                        gradients,
                        using: configuration.gradientProcessing
                    )
                    configuration.optimizer.update(model: model, gradients: processedGradients)
                    eval(model, configuration.optimizer)
                    globalStep += 1

                    state.logs["loss"] = scalarLoss
                    try callbackList.onBatchEnd(state: state)
                    try Task.checkCancellation()

                    if state.shouldStop {
                        break
                    }
                    batchIndex += 1
                }

                try Task.checkCancellation()
                guard batchIndex > 0 || state.shouldStop else {
                    throw TrainingError.emptyTrainingData
                }

                state.logs["loss"] = lossWeight == 0 ? 0 : weightedLoss / Float(lossWeight)
                for metric in configuration.metrics {
                    state.logs[metric.name] = metric.compute()
                }

                if hasValidationData {
                    let validationValues = try await evaluate(
                        model,
                        configuration: configuration,
                        data: validationData,
                        trainingStep: trainingStep
                    )
                    for (name, value) in validationValues {
                        state.logs["val_\(name)"] = value
                    }
                }

                try callbackList.onEpochEnd(state: state)
                try Task.checkCancellation()
                if state.shouldStop {
                    break
                }
            }

            trainEndStarted = true
            try callbackList.onTrainEnd(state: state)
            try Task.checkCancellation()
            return history
        } catch {
            if trainBeginStarted && !trainEndStarted {
                trainEndStarted = true
                // Preserve the original failure if a terminal callback also fails.
                try? callbackList.onTrainEnd(state: state)
            }
            throw error
        }
    }

    /// Evaluates a model without applying gradients or optimizer updates.
    public func evaluate<Model: TrainableModel, Target, Data: AsyncSequence>(
        _ model: Model,
        configuration: TrainingConfiguration<Model.Output, Target>,
        data: Data,
        trainingStep: TrainingStep<Model, Target>? = nil
    ) async throws -> [String: Float]
    where Data.Element == Batch<Model.Input, Target>, Data.Failure: Error {
        let wasTraining = model.training
        try validateMetricNames(configuration.metrics)
        model.train(false)
        defer { model.train(wasTraining) }

        for metric in configuration.metrics {
            metric.reset()
        }

        var weightedLoss: Float = 0
        var sampleCount = 0
        var batchCount = 0

        for try await batch in data {
            try Task.checkCancellation()
            guard batch.sampleCount > 0 else {
                throw TrainingError.invalidSampleCount(batch.sampleCount)
            }

            let output = model(batch.input)
            eval(output)
            let loss = trainingStep?(model, batch)
                ?? configuration.objective(output: output, target: batch.target)
            guard loss.ndim == 0 else {
                throw TrainingError.nonScalarObjective(shape: loss.shape)
            }
            eval(loss)
            let scalarLoss = loss.item(Float.self)
            guard scalarLoss.isFinite else {
                throw TrainingError.nonFiniteObjective
            }
            weightedLoss += scalarLoss * Float(batch.sampleCount)
            sampleCount += batch.sampleCount

            for metric in configuration.metrics {
                metric.update(output: output, target: batch.target)
            }
            batchCount += 1
        }

        try Task.checkCancellation()

        guard batchCount > 0 else {
            throw TrainingError.emptyEvaluationData
        }

        var results: [String: Float] = ["loss": weightedLoss / Float(sampleCount)]
        for metric in configuration.metrics {
            results[metric.name] = metric.compute()
        }
        return results
    }

    private func validateMetricNames<Output, Target>(
        _ metrics: [any Metric<Output, Target>]
    ) throws {
        var names = Set<String>()
        for metric in metrics {
            let name = metric.name
            guard !name.isEmpty,
                  name != "loss",
                  !name.hasPrefix("val_"),
                  names.insert(name).inserted
            else {
                throw TrainingError.metricNameCollision(name)
            }
        }
    }

    private func validateGradientProcessing(
        _ configuration: GradientProcessingConfiguration
    ) throws {
        if let maxNorm = configuration.globalNormClip,
           !maxNorm.isFinite || maxNorm <= 0
        {
            throw TrainingError.invalidGlobalNormClip(maxNorm)
        }
    }

    private func processGradients(
        _ gradients: ModuleParameters,
        using configuration: GradientProcessingConfiguration
    ) throws -> ModuleParameters {
        var processed = gradients
        for transform in configuration.transforms {
            let transformed = try transform(processed)
            try validateGradientStructure(transformed, matches: gradients)
            try validateFiniteGradients(transformed)
            processed = transformed
        }

        if let maxNorm = configuration.globalNormClip {
            processed = try clipGlobalNorm(processed, maxNorm: maxNorm)
            try validateGradientStructure(processed, matches: gradients)
            try validateFiniteGradients(processed)
        }

        return processed
    }

    private func validateGradientStructure(
        _ candidate: ModuleParameters,
        matches expected: ModuleParameters
    ) throws {
        let expectedLeaves = expected.flattened().sorted { $0.0 < $1.0 }
        let actualLeaves = candidate.flattened().sorted { $0.0 < $1.0 }
        let expectedKeys = expectedLeaves.map(\.0)
        let actualKeys = actualLeaves.map(\.0)
        guard expectedKeys == actualKeys else {
            throw TrainingError.gradientTreeKeyMismatch(
                expected: expectedKeys,
                actual: actualKeys
            )
        }

        for (expectedLeaf, actualLeaf) in zip(expectedLeaves, actualLeaves) {
            guard expectedLeaf.1.shape == actualLeaf.1.shape else {
                throw TrainingError.gradientShapeMismatch(
                    parameter: expectedLeaf.0,
                    expected: expectedLeaf.1.shape,
                    actual: actualLeaf.1.shape
                )
            }
        }
    }

    private func validateFiniteGradients(_ gradients: ModuleParameters) throws {
        eval(gradients)
        for (_, gradient) in gradients.flattened() {
            let finiteCount = isFinite(gradient).sum().item(Int.self)
            guard finiteCount == gradient.size else {
                throw TrainingError.nonFiniteGradient
            }
        }
    }

    private func clipGlobalNorm(
        _ gradients: ModuleParameters,
        maxNorm: Float
    ) throws -> ModuleParameters {
        let squaredNorm = gradients.flattened().reduce(MLXArray(Float32(0))) { total, leaf in
            let values = leaf.1.asType(.float32)
            return total + (values * values).sum()
        }
        eval(squaredNorm)
        let norm = sqrt(squaredNorm).item(Float.self)
        guard norm.isFinite else {
            throw TrainingError.nonFiniteGradient
        }
        guard norm > maxNorm else {
            return gradients
        }

        let scale = maxNorm / norm
        return gradients.mapValues { $0 * scale }
    }

    /// Runs the model in inference mode for one typed input and restores its
    /// prior train/eval mode before returning.
    public func predict<Model: TrainableModel>(_ model: Model, input: Model.Input) -> Model.Output {
        let wasTraining = model.training
        model.train(false)
        defer { model.train(wasTraining) }

        let output = model(input)
        eval(output)
        return output
    }

    /// Predicts one output for each batch in an asynchronous sequence.
    public func predict<Model: TrainableModel, Target, Data: AsyncSequence>(
        _ model: Model,
        data: Data
    ) async throws -> [Model.Output]
    where Data.Element == Batch<Model.Input, Target>, Data.Failure: Error {
        let wasTraining = model.training
        model.train(false)
        defer { model.train(wasTraining) }

        var outputs: [Model.Output] = []
        for try await batch in data {
            try Task.checkCancellation()
            let output = model(batch.input)
            eval(output)
            outputs.append(output)
        }
        try Task.checkCancellation()
        return outputs
    }

}

private struct EmptyBatchSequence<Element>: AsyncSequence {
    typealias Failure = Never

    func makeAsyncIterator() -> Iterator { Iterator() }

    struct Iterator: AsyncIteratorProtocol {
        mutating func next() async -> Element? { nil }
    }
}

/// Captures objective shape during MLX's synchronous value-and-gradient call.
private final class ObjectiveShapeRecorder {
    var nonScalarShape: [Int]?
}
