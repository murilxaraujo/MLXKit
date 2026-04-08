import MLX

/// Tracks mean absolute error across batches.
///
/// Accumulates the sum of absolute errors and total sample count, then
/// computes the running average on ``compute()``.
///
/// ```swift
/// let mae = MeanAbsoluteError()
/// mae.update(predictions: predicted, targets: actual)
/// print(mae.compute()) // e.g. 0.5
/// ```
public final class MeanAbsoluteError: Metric, @unchecked Sendable {
    public let name = "mae"

    private var sumAbsoluteError: Float = 0
    private var total: Int = 0

    public init() {}

    public func update(predictions: MLXArray, targets: MLXArray) {
        let ae = abs(predictions - targets).sum()
        sumAbsoluteError += ae.item(Float.self)
        total += targets.size
    }

    public func compute() -> Float {
        guard total > 0 else { return 0 }
        return sumAbsoluteError / Float(total)
    }

    public func reset() {
        sumAbsoluteError = 0
        total = 0
    }
}
