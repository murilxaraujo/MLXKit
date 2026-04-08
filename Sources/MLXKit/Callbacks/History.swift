/// Records per-epoch metric values over a training run.
///
/// After training, access the full history of any metric:
///
/// ```swift
/// let history = History()
/// // ... used as a callback during training ...
/// let losses = history["loss"]       // [0.5, 0.3, 0.2, ...]
/// let accuracies = history["accuracy"] // [0.8, 0.9, 0.92, ...]
/// ```
public final class History: Callback, @unchecked Sendable {
    /// All recorded epoch logs, in order.
    public private(set) var epochs: [[String: Float]] = []

    public init() {}

    /// Returns the series of values for the given metric key across all epochs.
    ///
    /// - Parameter key: The metric name (e.g. `"loss"`, `"accuracy"`).
    /// - Returns: An array of values, one per epoch, or `nil` entries where
    ///   the metric was absent.
    public subscript(key: String) -> [Float?] {
        epochs.map { $0[key] }
    }

    public func onEpochEnd(state: TrainingState) {
        epochs.append(state.logs)
    }

    /// Clears all recorded history.
    public func reset() {
        epochs.removeAll()
    }
}
