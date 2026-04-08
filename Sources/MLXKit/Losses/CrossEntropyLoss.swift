import MLX
import MLXNN

/// Computes the cross-entropy loss between logits and integer class targets.
///
/// Wraps ``MLXNN/crossEntropy(logits:targets:weights:axis:labelSmoothing:reduction:)``
/// with the ``LossFunction`` protocol.
///
/// ```swift
/// let loss = CrossEntropyLoss()
/// let value = loss(predictions: logits, targets: labels)
/// ```
///
/// - Parameters:
///   - labelSmoothing: Smoothing factor in `[0, 1)`. When greater than zero,
///     the target distribution is mixed with a uniform distribution.
///   - reduction: How to aggregate per-sample losses (default: `.mean`).
public struct CrossEntropyLoss: LossFunction {
    public let reduction: Reduction
    /// Label smoothing factor in `[0, 1)`.
    public let labelSmoothing: Float

    public init(labelSmoothing: Float = 0, reduction: Reduction = .mean) {
        self.labelSmoothing = labelSmoothing
        self.reduction = reduction
    }

    public func callAsFunction(predictions: MLXArray, targets: MLXArray) -> MLXArray {
        let unreduced = MLXNN.crossEntropy(
            logits: predictions,
            targets: targets,
            labelSmoothing: labelSmoothing,
            reduction: .none
        )
        return reduction.apply(to: unreduced)
    }
}
