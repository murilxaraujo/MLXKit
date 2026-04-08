import MLX

/// A stateful metric that accumulates values across batches and computes
/// an aggregate result.
///
/// Metrics follow an `update` / `compute` / `reset` lifecycle:
///
/// 1. **Update** — call ``update(predictions:targets:)`` after each batch to
///    feed new observations into the metric's internal accumulators.
/// 2. **Compute** — call ``compute()`` at any point to read the current
///    aggregate value (e.g. running accuracy).
/// 3. **Reset** — call ``reset()`` at the start of each epoch (or whenever
///    you want to clear accumulated state).
///
/// ```swift
/// var accuracy = Accuracy()
/// for batch in dataLoader {
///     let logits = model(batch.inputs)
///     accuracy.update(predictions: logits, targets: batch.labels)
/// }
/// print("Epoch accuracy: \(accuracy.compute())")
/// accuracy.reset()
/// ```
public protocol Metric: AnyObject, Sendable {
    /// A human-readable name for this metric (e.g. `"accuracy"`).
    var name: String { get }

    /// Incorporates a new batch of predictions and targets into the
    /// metric's running state.
    ///
    /// - Parameters:
    ///   - predictions: Model outputs (logits, probabilities, or values).
    ///   - targets: Ground-truth values or class indices.
    func update(predictions: MLXArray, targets: MLXArray)

    /// Returns the current aggregate metric value.
    ///
    /// - Returns: The computed metric as a scalar.
    func compute() -> Float

    /// Clears all accumulated state, preparing the metric for a new epoch.
    func reset()
}
