import Foundation
import MLX
import MLXOptimizers

/// The optimizer, objective, metrics, and optional learning-rate schedule for
/// one typed model output and target pair.
public struct TrainingConfiguration<Output, Target> {
    public let optimizer: any Optimizer
    public let objective: any Objective<Output, Target>
    public let metrics: [any Metric<Output, Target>]
    public let learningRateSchedule: (any LRSchedule)?

    public init<ObjectiveType: Objective<Output, Target>>(
        optimizer: any Optimizer,
        objective: ObjectiveType,
        metrics: [any Metric<Output, Target>] = [],
        learningRateSchedule: (any LRSchedule)? = nil
    ) {
        self.optimizer = optimizer
        self.objective = objective
        self.metrics = metrics
        self.learningRateSchedule = learningRateSchedule
    }
}

/// Errors reported by the training and evaluation workflow.
public enum TrainingError: Error, LocalizedError, Equatable {
    case invalidEpochCount(Int)
    case emptyTrainingData
    case emptyEvaluationData
    case invalidSampleCount(Int)
    case nonScalarObjective(shape: [Int])
    case metricNameCollision(String)
    case unsupportedLearningRateSchedule
    case nonFiniteObjective
    case nonFiniteGradient

    public var errorDescription: String? {
        switch self {
        case .invalidEpochCount(let epochs):
            "Epoch count must be greater than zero; received \(epochs)."
        case .emptyTrainingData:
            "Training data produced no batches."
        case .emptyEvaluationData:
            "Evaluation data produced no batches."
        case .invalidSampleCount(let count):
            "Batch sampleCount must be greater than zero; received \(count)."
        case .nonScalarObjective(let shape):
            "Training objective must be scalar; received shape \(shape). Reduce the objective explicitly."
        case .metricNameCollision(let name):
            "Metric name '\(name)' collides with a reserved or duplicate training log key."
        case .unsupportedLearningRateSchedule:
            "The configured optimizer does not support learning-rate updates required by this schedule."
        case .nonFiniteObjective:
            "Training objective is not finite; optimizer update was not applied."
        case .nonFiniteGradient:
            "Training gradient contains a non-finite value; optimizer update was not applied."
        }
    }
}
