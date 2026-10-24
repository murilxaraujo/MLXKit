import MLX

/// A stateful metric over typed model outputs and targets.
///
/// The training engine resets metrics before each independent epoch or
/// evaluation run, calls `update(output:target:)` once per batch, and reads
/// `compute()` at the reporting boundary. Implementations own their aggregation
/// across examples and should not assume that every batch has the same size.
public protocol Metric<Output, Target>: AnyObject {
    associatedtype Output
    associatedtype Target

    /// A human-readable name for this metric (e.g. `"accuracy"`).
    var name: String { get }

    /// Incorporates one batch of model outputs and targets.
    func update(output: Output, target: Target)

    /// Returns the current aggregate metric value.
    ///
    /// - Returns: The computed metric as a scalar.
    func compute() -> Float

    /// Clears all accumulated state, preparing the metric for a new epoch.
    func reset()
}
