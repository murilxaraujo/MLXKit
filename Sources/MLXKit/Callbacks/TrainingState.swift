import MLX
import MLXNN

/// Mutable state shared between the training loop and callbacks.
///
/// The training engine updates this state at each lifecycle point, and
/// callbacks can read or modify it (e.g. setting ``shouldStop`` to end
/// training early).
public final class TrainingState: @unchecked Sendable {
    /// The current epoch (zero-indexed).
    public var epoch: Int = 0

    /// The total number of epochs requested.
    public var totalEpochs: Int = 0

    /// The current batch within the epoch (zero-indexed).
    public var batch: Int = 0

    /// The total number of batches in the current epoch, if known.
    public var totalBatches: Int?

    /// Key-value log entries for the current step (e.g. `"loss"`, `"accuracy"`).
    public var logs: [String: Float] = [:]

    /// Set to `true` from a callback to request early termination.
    public var shouldStop: Bool = false

    public init() {}
}
