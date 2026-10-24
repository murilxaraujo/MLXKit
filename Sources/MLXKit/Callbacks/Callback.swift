/// A hook into the training lifecycle.
///
/// Implement any subset of the lifecycle methods to respond to training
/// events. Methods are synchronous and throwing. A callback error propagates
/// from `fit`; callbacks are dispatched serially in the training task.
/// Once train-begin dispatch starts, train-end dispatch runs exactly once on
/// normal completion, early stop, cancellation, or failure. If both the run and
/// train-end callback fail, the original run failure is propagated.
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
    func onTrainBegin(state: TrainingState) throws
    /// Called at the start of each epoch.
    func onEpochBegin(state: TrainingState) throws
    /// Called at the start of each batch.
    func onBatchBegin(state: TrainingState) throws
    /// Called at the end of each batch.
    func onBatchEnd(state: TrainingState) throws
    /// Called at the end of each epoch.
    func onEpochEnd(state: TrainingState) throws
    /// Called once at the end of training.
    func onTrainEnd(state: TrainingState) throws
}

// Default no-op implementations.
extension Callback {
    public func onTrainBegin(state: TrainingState) throws {}
    public func onEpochBegin(state: TrainingState) throws {}
    public func onBatchBegin(state: TrainingState) throws {}
    public func onBatchEnd(state: TrainingState) throws {}
    public func onEpochEnd(state: TrainingState) throws {}
    public func onTrainEnd(state: TrainingState) throws {}
}
