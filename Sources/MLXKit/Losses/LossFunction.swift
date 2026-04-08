import MLX

/// A type that computes a differentiable loss between predictions and targets.
///
/// Conform to this protocol to create custom loss functions that integrate with
/// MLXKit's training engine. Every conforming type must specify a ``reduction``
/// strategy and implement ``callAsFunction(predictions:targets:)``.
///
/// ```swift
/// let loss: any LossFunction = .crossEntropy
/// let value = loss(predictions: logits, targets: labels)
/// ```
public protocol LossFunction: Sendable {
    /// The reduction strategy applied to per-sample losses.
    var reduction: Reduction { get }

    /// Computes the loss between predictions and targets.
    ///
    /// - Parameters:
    ///   - predictions: Model outputs (logits or probabilities).
    ///   - targets: Ground-truth values or class indices.
    /// - Returns: The reduced loss value, shaped according to ``reduction``.
    func callAsFunction(predictions: MLXArray, targets: MLXArray) -> MLXArray
}
