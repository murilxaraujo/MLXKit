import MLX

/// Tracks mean squared error across batches.
///
/// Accumulates the sum of squared errors and total sample count, then
/// computes the running average on ``compute()``.
///
/// ```swift
/// let mse = MeanSquaredError()
/// mse.update(predictions: predicted, targets: actual)
/// print(mse.compute()) // e.g. 0.25
/// ```
public final class MeanSquaredError: Metric, @unchecked Sendable {
    public let name = "mse"

    private var sumSquaredError: Float = 0
    private var total: Int = 0

    public init() {}

    public func update(predictions: MLXArray, targets: MLXArray) {
        let diff = predictions - targets
        let se = (diff * diff).sum()
        sumSquaredError += se.item(Float.self)
        total += targets.size
    }

    public func compute() -> Float {
        guard total > 0 else { return 0 }
        return sumSquaredError / Float(total)
    }

    public func reset() {
        sumSquaredError = 0
        total = 0
    }
}
