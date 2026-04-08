/// Stops training if the loss becomes NaN or infinite.
///
/// Checks the `"loss"` key (or a custom key) in ``TrainingState/logs``
/// at the end of each batch.
///
/// ```swift
/// let nanGuard = TerminateOnNaN()
/// ```
public final class TerminateOnNaN: Callback, @unchecked Sendable {
    /// The log key to monitor for NaN/Inf values.
    public let monitor: String

    public init(monitor: String = "loss") {
        self.monitor = monitor
    }

    public func onBatchEnd(state: TrainingState) {
        guard let value = state.logs[monitor] else { return }
        if value.isNaN || value.isInfinite {
            state.shouldStop = true
        }
    }
}
