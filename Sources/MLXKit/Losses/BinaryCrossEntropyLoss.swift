import MLX
import MLXNN

/// Computes binary cross-entropy loss between logits and binary targets.
///
/// Wraps ``MLXNN/binaryCrossEntropy(logits:targets:weights:withLogits:reduction:)``.
///
/// ```swift
/// let loss = BinaryCrossEntropyLoss()
/// let value = loss(predictions: logits, targets: labels)
/// ```
///
/// - Parameters:
///   - inputsAreLogits: If `true` (default), predictions are treated as raw logits.
///     If `false`, predictions are treated as probabilities.
///   - reduction: How to aggregate per-sample losses (default: `.mean`).
public struct BinaryCrossEntropyLoss: LossFunction {
    public let reduction: Reduction
    /// Whether inputs are raw logits (`true`) or probabilities (`false`).
    public let inputsAreLogits: Bool

    public init(inputsAreLogits: Bool = true, reduction: Reduction = .mean) {
        self.inputsAreLogits = inputsAreLogits
        self.reduction = reduction
    }

    public func callAsFunction(predictions: MLXArray, targets: MLXArray) -> MLXArray {
        let unreduced = MLXNN.binaryCrossEntropy(
            logits: predictions,
            targets: targets,
            withLogits: inputsAreLogits,
            reduction: .none
        )
        return reduction.apply(to: unreduced)
    }
}
