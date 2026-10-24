import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Transforms the complete gradient tree before an optimizer update.
///
/// Transforms run in the order configured by ``GradientProcessingConfiguration``.
/// Each transform may throw. It must return the same parameter keys and tensor
/// shapes it received, with finite values.
public typealias GradientTransform = (ModuleParameters) throws -> ModuleParameters

/// Gradient transforms and optional global-norm clipping for training.
public struct GradientProcessingConfiguration {
    /// User transforms, applied in array order before clipping.
    public let transforms: [GradientTransform]

    /// Optional maximum L2 norm across the complete gradient tree.
    public let globalNormClip: Float?

    public init(transforms: [GradientTransform] = [], globalNormClip: Float? = nil) {
        self.transforms = transforms
        self.globalNormClip = globalNormClip
    }
}

/// The optimizer, objective, metrics, schedule, and gradient controls for one
/// typed model output and target pair.
public struct TrainingConfiguration<Output, Target> {
    public let optimizer: any Optimizer
    public let objective: any Objective<Output, Target>
    public let metrics: [any Metric<Output, Target>]
    public let learningRateSchedule: (any LRSchedule)?
    /// Gradient transforms and clipping applied before each optimizer update.
    public let gradientProcessing: GradientProcessingConfiguration

    public init<ObjectiveType: Objective<Output, Target>>(
        optimizer: any Optimizer,
        objective: ObjectiveType,
        metrics: [any Metric<Output, Target>] = [],
        learningRateSchedule: (any LRSchedule)? = nil,
        gradientProcessing: GradientProcessingConfiguration = .init()
    ) {
        self.optimizer = optimizer
        self.objective = objective
        self.metrics = metrics
        self.learningRateSchedule = learningRateSchedule
        self.gradientProcessing = gradientProcessing
    }
}

/// Errors reported by the training and evaluation workflow.
public enum TrainingError: Error, LocalizedError, Equatable {
    case invalidEpochCount(Int)
    case emptyTrainingData
    case emptyEvaluationData
    case invalidSampleCount(Int)
    case invalidLearningRate(Float)
    case nonScalarObjective(shape: [Int])
    case metricNameCollision(String)
    case unsupportedLearningRateSchedule
    case nonFiniteObjective
    case nonFiniteGradient
    case invalidGlobalNormClip(Float)
    case gradientTreeKeyMismatch(expected: [String], actual: [String])
    case gradientShapeMismatch(parameter: String, expected: [Int], actual: [Int])

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
        case .invalidLearningRate(let value):
            "Learning rate must be finite and nonnegative; received \(value)."
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
        case .invalidGlobalNormClip(let value):
            "Global gradient-norm clip must be finite and greater than zero; received \(value)."
        case .gradientTreeKeyMismatch(let expected, let actual):
            "Gradient transform changed parameter keys; expected \(expected), received \(actual)."
        case .gradientShapeMismatch(let parameter, let expected, let actual):
            "Gradient transform changed the shape for '\(parameter)'; expected \(expected), received \(actual)."
        }
    }
}
