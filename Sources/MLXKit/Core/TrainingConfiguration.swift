import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Stores the optimizer, loss, metrics, and LR schedule for training.
///
/// Created by ``TrainingEngine/compile(model:optimizer:loss:metrics:lrSchedule:)``
/// and consumed by ``TrainingEngine/fit(_:epochs:validationData:callbacks:)``.
public final class TrainingConfiguration: @unchecked Sendable {
    /// The optimizer used to update model parameters.
    public let optimizer: any Optimizer
    /// The loss function used to compute training loss.
    public let loss: any LossFunction
    /// Metrics to track during training.
    public let metrics: [any Metric]
    /// Optional learning rate schedule.
    public let lrSchedule: (any LRSchedule)?

    public init(
        optimizer: any Optimizer,
        loss: any LossFunction,
        metrics: [any Metric] = [],
        lrSchedule: (any LRSchedule)? = nil
    ) {
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.lrSchedule = lrSchedule
    }
}

/// Errors that occur during training.
public enum TrainingError: Error, LocalizedError {
    /// The model has not been compiled before calling fit/evaluate.
    case notCompiled

    public var errorDescription: String? {
        switch self {
        case .notCompiled:
            return "Model has not been compiled. Call TrainingEngine.compile() before fit() or evaluate()."
        }
    }
}
