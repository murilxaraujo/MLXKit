import MLX

/// Tracks the running average of a loss value across batches.
///
/// Unlike other metrics, ``RunningLoss`` takes the loss value directly
/// via ``update(loss:count:)`` rather than predictions and targets.
/// The standard ``update(predictions:targets:)`` treats predictions as
/// pre-computed loss values.
///
/// ```swift
/// let runningLoss = RunningLoss()
/// runningLoss.update(loss: lossValue, count: batchSize)
/// print(runningLoss.compute()) // e.g. 1.23
/// ```
public final class RunningLoss: Metric, @unchecked Sendable {
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
        sumLoss += loss.item(Float.self)
        total += count
    }

    /// Convenience conformance — treats `predictions` as a scalar loss value.
    /// The `targets` array's size is used as the batch count.
    public func update(predictions: MLXArray, targets: MLXArray) {
        update(loss: predictions, count: targets.size)
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
