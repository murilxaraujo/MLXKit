import MLX

/// Specifies how per-sample losses are aggregated into a scalar.
///
/// Use a `Reduction` value when constructing any ``LossFunction`` to control
/// the output shape:
///
/// ```swift
/// let loss = CrossEntropyLoss(reduction: .mean)
/// ```
///
/// - ``mean``: Average over all elements (default for most losses).
/// - ``sum``: Sum over all elements.
/// - ``none``: Return per-sample losses without aggregation.
public enum Reduction: Sendable, Equatable {
    /// Average the loss over all elements.
    case mean
    /// Sum the loss over all elements.
    case sum
    /// Return per-sample losses without reduction.
    case none

    /// Applies this reduction to a per-sample loss tensor.
    ///
    /// - Parameter loss: An unreduced loss array.
    /// - Returns: The reduced scalar, or the original array if ``none``.
    public func apply(to loss: MLXArray) -> MLXArray {
        switch self {
        case .mean: loss.mean()
        case .sum: loss.sum()
        case .none: loss
        }
    }
}
