import MLX
import MLXNN

/// Computes the mean squared error loss between predictions and targets.
///
/// Wraps the MLXNN mean squared error operation.
///
/// The per-sample loss is `(predictions - targets)²`.
///
/// ```swift
/// let loss = MSELoss()
/// let value = loss(predictions: predicted, targets: actual)
/// ```
public struct MSELoss: LossFunction {
    public let reduction: Reduction

    public init(reduction: Reduction = .mean) {
        self.reduction = reduction
    }

    public func callAsFunction(predictions: MLXArray, targets: MLXArray) -> MLXArray {
        let unreduced = MLXNN.mseLoss(
            predictions: predictions,
            targets: targets,
            reduction: .none
        )
        return reduction.apply(to: unreduced)
    }
}
