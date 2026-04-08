import MLX
import MLXNN

/// Computes the Huber loss, a robust alternative to MSE that is less sensitive to outliers.
///
/// Wraps ``MLXNN/huberLoss(inputs:targets:delta:reduction:)``.
///
/// For residuals smaller than `delta`, Huber loss behaves like MSE; for larger
/// residuals it behaves like L1. This combines the stability of L1 with the
/// smoothness of MSE near zero.
///
/// ```swift
/// let loss = HuberLoss(delta: 1.0)
/// let value = loss(predictions: predicted, targets: actual)
/// ```
public struct HuberLoss: LossFunction {
    public let reduction: Reduction
    /// Threshold at which the loss transitions from quadratic to linear.
    public let delta: Float

    public init(delta: Float = 1.0, reduction: Reduction = .mean) {
        self.delta = delta
        self.reduction = reduction
    }

    public func callAsFunction(predictions: MLXArray, targets: MLXArray) -> MLXArray {
        let unreduced = MLXNN.huberLoss(
            inputs: predictions,
            targets: targets,
            delta: delta,
            reduction: .none
        )
        return reduction.apply(to: unreduced)
    }
}
