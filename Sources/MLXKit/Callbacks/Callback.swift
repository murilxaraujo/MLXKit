/// A hook into the training lifecycle.
///
/// Implement any subset of the lifecycle methods to respond to training
/// events. All methods have default no-op implementations, so you only
/// override what you need.
///
/// ```swift
/// final class PrintCallback: Callback {
///     func onEpochEnd(state: TrainingState) {
///         print("Epoch \(state.epoch) done — loss: \(state.logs["loss"] ?? 0)")
///     }
/// }
/// ```
///
/// Lifecycle order per training run:
/// 1. ``onTrainBegin(state:)``
/// 2. For each epoch:
///    1. ``onEpochBegin(state:)``
///    2. For each batch:
///       1. ``onBatchBegin(state:)``
///       2. ``onBatchEnd(state:)``
///    3. ``onEpochEnd(state:)``
/// 3. ``onTrainEnd(state:)``
public protocol Callback: AnyObject, Sendable {
    /// Called once at the start of training.
    func onTrainBegin(state: TrainingState)
    /// Called at the start of each epoch.
    func onEpochBegin(state: TrainingState)
    /// Called at the start of each batch.
    func onBatchBegin(state: TrainingState)
    /// Called at the end of each batch.
    func onBatchEnd(state: TrainingState)
    /// Called at the end of each epoch.
    func onEpochEnd(state: TrainingState)
    /// Called once at the end of training.
    func onTrainEnd(state: TrainingState)
}

// Default no-op implementations.
extension Callback {
    public func onTrainBegin(state: TrainingState) {}
    public func onEpochBegin(state: TrainingState) {}
    public func onBatchBegin(state: TrainingState) {}
    public func onBatchEnd(state: TrainingState) {}
    public func onEpochEnd(state: TrainingState) {}
    public func onTrainEnd(state: TrainingState) {}
}
