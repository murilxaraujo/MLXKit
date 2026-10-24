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

    /// Number of successful optimizer updates completed so far. Unlike
    /// ``batch``, this advances once per optimizer update and can be restored
    /// when resuming from an epoch-boundary checkpoint.
    public var optimizerStep: Int = 0

    /// Whether the current epoch consumed its complete data sequence. A
    /// callback-requested stop during a batch leaves this false, so callers do
    /// not mistake a batch-boundary stop for epoch-boundary resumable progress.
    public var epochCompleted: Bool = false

    /// The total number of batches in the current epoch, if known.
    public var totalBatches: Int?

    /// Key-value log entries for the current step (e.g. `"loss"`, `"accuracy"`).
    public var logs: [String: Float] = [:]

    /// Set in `onBatchEnd` or `onEpochEnd` to request orderly early termination.
    /// The engine completes the current batch/epoch boundary before stopping.
    public var shouldStop: Bool = false

    public init() {}
}
