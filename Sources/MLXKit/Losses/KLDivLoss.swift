import MLX
import MLXNN

/// Computes the Kullback-Leibler divergence loss.
///
/// Wraps ``MLXNN/klDivLoss(inputs:targets:axis:reduction:)``.
///
/// Measures how one probability distribution diverges from a target distribution.
/// Predictions should be log-probabilities and targets should be probabilities.
///
/// ```swift
/// let loss = KLDivLoss()
/// let value = loss(predictions: logProbs, targets: targetProbs)
/// ```
public struct KLDivLoss: LossFunction {
    public let reduction: Reduction
    /// The axis along which to compute the divergence.
    public let axis: Int

    public init(axis: Int = -1, reduction: Reduction = .mean) {
        self.axis = axis
        self.reduction = reduction
    }

    public func callAsFunction(predictions: MLXArray, targets: MLXArray) -> MLXArray {
        let unreduced = MLXNN.klDivLoss(
            inputs: predictions,
            targets: targets,
            axis: axis,
            reduction: .none
        )
        return reduction.apply(to: unreduced)
    }
}
