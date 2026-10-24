import MLX

/// A stateful metric over typed model outputs and targets.
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
