import MLX
import MLXNN

/// Computes the L1 (mean absolute error) loss between predictions and targets.
///
/// Wraps ``MLXNN/l1Loss(predictions:targets:reduction:)``.
///
/// The per-sample loss is `|predictions - targets|`.
///
/// ```swift
/// let loss = L1Loss()
/// let value = loss(predictions: predicted, targets: actual)
/// ```
public struct L1Loss: LossFunction {
    public let reduction: Reduction

    public init(reduction: Reduction = .mean) {
        self.reduction = reduction
    }

    public func callAsFunction(predictions: MLXArray, targets: MLXArray) -> MLXArray {
        let unreduced = MLXNN.l1Loss(
            predictions: predictions,
            targets: targets,
            reduction: .none
        )
        return reduction.apply(to: unreduced)
    }
}
