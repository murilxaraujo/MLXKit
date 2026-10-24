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

        let history = History()
        let callbackList = CallbackList([history] + callbacks)
        let state = TrainingState()
        state.totalEpochs = epochs
        callbackList.onTrainBegin(state: state)

        var globalStep = 0
        let lossAndGrad = valueAndGrad(model: model) { (differentiableModel: Model, batch: Batch<Model.Input, Target>) in
            let output = differentiableModel(batch.input)
            let objective: MLXArray
            if let trainingStep {
                objective = trainingStep(differentiableModel, batch)
            } else {
                objective = configuration.objective(output: output, target: batch.target)
            }
            return [objective]
        }

        for epoch in 0..<epochs {
            state.epoch = epoch
            state.batch = 0
            state.logs = [:]
            state.shouldStop = false

            model.train(true)
            for metric in configuration.metrics {
                metric.reset()
            }
            callbackList.onEpochBegin(state: state)

            var weightedLoss: Float = 0
            var lossWeight: Int = 0
            var batchIndex = 0

            for try await batch in data {
                state.batch = batchIndex
                guard batch.sampleCount > 0 else {
                    throw TrainingError.invalidSampleCount(batch.sampleCount)
                }

                if let schedule = configuration.learningRateSchedule {
                    let learningRate = schedule.learningRate(atStep: globalStep)
                    setLearningRate(configuration.optimizer, learningRate)
                }

                callbackList.onBatchBegin(state: state)

                let (values, gradients) = lossAndGrad(model, batch)
                guard let lossValue = values.first else {
                    throw TrainingError.nonScalarObjective(shape: [])
                }
                guard lossValue.ndim == 0 else {
                    throw TrainingError.nonScalarObjective(shape: lossValue.shape)
                }
                eval(lossValue, gradients)

                // Metrics see the model output before this batch's update.
                // The objective forward pass remains inside MLX's gradient transform.
                let metricOutput = model(batch.input)
                eval(metricOutput)
                for metric in configuration.metrics {
                    metric.update(output: metricOutput, target: batch.target)
                }

                let scalarLoss = lossValue.item(Float.self)
                weightedLoss += scalarLoss * Float(batch.sampleCount)
                lossWeight += batch.sampleCount

                configuration.optimizer.update(model: model, gradients: gradients)
                eval(model, configuration.optimizer)

                state.logs["loss"] = scalarLoss
                callbackList.onBatchEnd(state: state)

                if state.shouldStop {
                    break
                }
                batchIndex += 1
                globalStep += 1
            }

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

            callbackList.onEpochEnd(state: state)
            if state.shouldStop {
                break
            }
        }

        callbackList.onTrainEnd(state: state)
        return history
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
            weightedLoss += scalarLoss * Float(batch.sampleCount)
            sampleCount += batch.sampleCount

            for metric in configuration.metrics {
                metric.update(output: output, target: batch.target)
            }
            batchCount += 1
        }

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
            let output = model(batch.input)
            eval(output)
            outputs.append(output)
        }
        return outputs
    }

    private func setLearningRate(_ optimizer: any Optimizer, _ learningRate: Float) {
        if let sgd = optimizer as? SGD {
            sgd.learningRate = learningRate
        } else if let adam = optimizer as? Adam {
            adam.learningRate = learningRate
        } else if let adamax = optimizer as? Adamax {
            adamax.learningRate = learningRate
        } else if let adaGrad = optimizer as? AdaGrad {
            adaGrad.learningRate = learningRate
        } else if let adaDelta = optimizer as? AdaDelta {
            adaDelta.learningRate = learningRate
        } else if let rmsProp = optimizer as? RMSprop {
            rmsProp.learningRate = learningRate
        } else if let lion = optimizer as? Lion {
            lion.learningRate = learningRate
        } else if let adafactor = optimizer as? Adafactor {
            adafactor.learningRate = learningRate
        }
    }
}

private struct EmptyBatchSequence<Element>: AsyncSequence {
    typealias Failure = Never

    func makeAsyncIterator() -> Iterator { Iterator() }

    struct Iterator: AsyncIteratorProtocol {
        mutating func next() async -> Element? { nil }
    }
}
