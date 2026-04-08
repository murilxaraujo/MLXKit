import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// The central training orchestrator that wires losses, metrics, callbacks,
/// data loading, LR schedules, and gradient computation into a Keras-style
/// training loop.
///
/// ```swift
/// let model = MyModel()
/// let engine = TrainingEngine()
/// let config = engine.compile(
///     model: model,
///     optimizer: Adam(learningRate: 1e-3),
///     loss: CrossEntropyLoss(),
///     metrics: [Accuracy()]
/// )
///
/// let trainLoader = DataLoader(dataset: trainData, batchSize: 32, shuffle: true)
/// let history = try await engine.fit(
///     model, config: config, data: trainLoader,
///     epochs: 10, callbacks: [.progressReporter, .earlyStopping]
/// )
/// ```
public final class TrainingEngine: @unchecked Sendable {

    public init() {}

    /// Configures training by bundling optimizer, loss, metrics, and LR schedule.
    ///
    /// - Returns: A ``TrainingConfiguration`` to pass to ``fit(_:config:data:epochs:validationData:callbacks:)``.
    public func compile(
        model: some TrainableModel,
        optimizer: any Optimizer,
        loss: any LossFunction,
        metrics: [any Metric] = [],
        lrSchedule: (any LRSchedule)? = nil
    ) -> TrainingConfiguration {
        eval(model)
        return TrainingConfiguration(
            optimizer: optimizer,
            loss: loss,
            metrics: metrics,
            lrSchedule: lrSchedule
        )
    }

    // MARK: - fit()

    /// Trains the model for the specified number of epochs.
    ///
    /// - Parameters:
    ///   - model: The model to train.
    ///   - config: The training configuration from ``compile(model:optimizer:loss:metrics:lrSchedule:)``.
    ///   - data: The training data loader.
    ///   - epochs: Number of epochs to train.
    ///   - validationData: Optional validation data loader.
    ///   - callbacks: Callbacks to invoke during training.
    /// - Returns: A ``History`` containing per-epoch metric logs.
    public func fit<D: Dataset>(
        _ model: some TrainableModel,
        config: TrainingConfiguration,
        data: DataLoader<D>,
        epochs: Int,
        validationData: DataLoader<D>? = nil,
        callbacks: [any Callback] = []
    ) async throws -> History {
        let history = History()
        let allCallbacks = [history] + callbacks
        let callbackList = CallbackList(allCallbacks)
        let state = TrainingState()
        state.totalEpochs = epochs

        let runningLoss = RunningLoss(name: "loss")

        // Build valueAndGrad closure
        let lossGrad = valueAndGrad(model: model) {
            (model: Module, x: MLXArray, y: MLXArray) -> MLXArray in
            let trainable = model as! any TrainableModel
            let yPred = trainable.forward(x)
            return config.loss(predictions: yPred, targets: y)
        }

        var globalStep = 0

        callbackList.onTrainBegin(state: state)

        for epoch in 0..<epochs {
            state.epoch = epoch
            state.batch = 0
            state.logs = [:]

            model.train(true)
            runningLoss.reset()
            for metric in config.metrics { metric.reset() }

            callbackList.onEpochBegin(state: state)

            var batchIndex = 0
            for await batch in data {
                state.batch = batchIndex

                // Apply LR schedule
                if let schedule = config.lrSchedule {
                    let lr = schedule.learningRate(atStep: globalStep)
                    setLearningRate(config.optimizer, lr)
                }

                callbackList.onBatchBegin(state: state)

                // Forward + backward + update
                let (lossValue, gradients) = lossGrad(model, batch.features, batch.labels)
                config.optimizer.update(model: model, gradients: gradients)
                eval(model, config.optimizer)

                // Update metrics
                let lossScalar = lossValue.item(Float.self)
                runningLoss.update(loss: lossValue, count: batch.size)

                let yPred = model.forward(batch.features)
                for metric in config.metrics {
                    metric.update(predictions: yPred, targets: batch.labels)
                }

                state.logs["loss"] = lossScalar
                callbackList.onBatchEnd(state: state)

                if state.shouldStop { break }
                batchIndex += 1
                globalStep += 1
            }

            // Epoch-level logs
            state.logs["loss"] = runningLoss.compute()
            for metric in config.metrics {
                state.logs[metric.name] = metric.compute()
            }

            // Validation
            if let validationData {
                let valMetrics = try await evaluate(
                    model, config: config, data: validationData
                )
                for (key, value) in valMetrics {
                    state.logs["val_\(key)"] = value
                }
            }

            callbackList.onEpochEnd(state: state)

            if state.shouldStop { break }
        }

        callbackList.onTrainEnd(state: state)

        return history
    }

    // MARK: - evaluate()

    /// Evaluates the model on the given data without computing gradients.
    ///
    /// - Parameters:
    ///   - model: The model to evaluate.
    ///   - config: The training configuration.
    ///   - data: The evaluation data loader.
    /// - Returns: A dictionary of metric names to their computed values.
    public func evaluate<D: Dataset>(
        _ model: some TrainableModel,
        config: TrainingConfiguration,
        data: DataLoader<D>
    ) async throws -> [String: Float] {
        let wasTraining = model.training
        model.train(false)
        defer { model.train(wasTraining) }

        let runningLoss = RunningLoss(name: "loss")
        for metric in config.metrics { metric.reset() }

        for await batch in data {
            let yPred = model.forward(batch.features)
            eval(yPred)

            let lossValue = config.loss(predictions: yPred, targets: batch.labels)
            runningLoss.update(loss: lossValue, count: batch.size)

            for metric in config.metrics {
                metric.update(predictions: yPred, targets: batch.labels)
            }
        }

        var results: [String: Float] = ["loss": runningLoss.compute()]
        for metric in config.metrics {
            results[metric.name] = metric.compute()
        }

        return results
    }

    // MARK: - predict()

    /// Runs the model in inference mode on a single input.
    ///
    /// - Parameters:
    ///   - model: The model to use for prediction.
    ///   - input: The input tensor.
    /// - Returns: The model's output.
    public func predict(_ model: some TrainableModel, input: MLXArray) -> MLXArray {
        let wasTraining = model.training
        model.train(false)
        defer { model.train(wasTraining) }

        let output = model.forward(input)
        eval(output)
        return output
    }

    /// Runs the model in inference mode on batched data from a data loader.
    ///
    /// - Parameters:
    ///   - model: The model to use for prediction.
    ///   - data: The data loader to iterate.
    /// - Returns: An array of output tensors, one per batch.
    public func predict<D: Dataset>(
        _ model: some TrainableModel,
        data: DataLoader<D>
    ) async -> [MLXArray] {
        let wasTraining = model.training
        model.train(false)
        defer { model.train(wasTraining) }

        var outputs: [MLXArray] = []
        for await batch in data {
            let output = model.forward(batch.features)
            eval(output)
            outputs.append(output)
        }
        return outputs
    }

    // MARK: - Helpers

    /// Sets the learning rate on an optimizer via reflection on known optimizer types.
    private func setLearningRate(_ optimizer: any Optimizer, _ lr: Float) {
        // MLX Swift optimizers expose `learningRate` as a stored property
        if let sgd = optimizer as? SGD {
            sgd.learningRate = lr
        } else if let adam = optimizer as? Adam {
            adam.learningRate = lr
        } else if let adamax = optimizer as? Adamax {
            adamax.learningRate = lr
        } else if let adaGrad = optimizer as? AdaGrad {
            adaGrad.learningRate = lr
        } else if let adaDelta = optimizer as? AdaDelta {
            adaDelta.learningRate = lr
        } else if let rmsProp = optimizer as? RMSprop {
            rmsProp.learningRate = lr
        } else if let lion = optimizer as? Lion {
            lion.learningRate = lr
        } else if let adafactor = optimizer as? Adafactor {
            adafactor.learningRate = lr
        }
    }
}
