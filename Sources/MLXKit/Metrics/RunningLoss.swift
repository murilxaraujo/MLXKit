import MLX

/// Tracks the running average of a loss value across batches.
///
/// This is a standalone utility, not a ``Metric``. The training engine owns
/// training-loss reporting.
///
/// ```swift
/// let runningLoss = RunningLoss()
/// runningLoss.update(loss: lossValue, count: batchSize)
/// print(runningLoss.compute()) // e.g. 1.23
/// ```
public final class RunningLoss: @unchecked Sendable {
    public let name: String

    private var sumLoss: Float = 0
    private var total: Int = 0

    public init(name: String = "loss") {
        self.name = name
    }

    /// Updates with a pre-computed scalar loss value.
    ///
    /// - Parameters:
    ///   - loss: The batch loss value (scalar MLXArray).
    ///   - count: The number of samples in the batch.
    public func update(loss: MLXArray, count: Int) {
        sumLoss += loss.item(Float.self) * Float(count)
        total += count
    }

    public func compute() -> Float {
        guard total > 0 else { return 0 }
        return sumLoss / Float(total)
    }

    public func reset() {
        sumLoss = 0
        total = 0
    }
}
