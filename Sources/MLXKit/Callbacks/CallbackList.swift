/// Dispatches lifecycle hooks to an ordered list of callbacks.
///
/// The training engine uses a `CallbackList` to invoke all registered
/// callbacks in the order they were provided.
///
/// ```swift
/// let callbacks = CallbackList([history, earlyStopping, progress])
/// callbacks.onEpochEnd(state: state)
/// ```
public final class CallbackList: @unchecked Sendable {
    /// The callbacks, dispatched in order.
    public let callbacks: [any Callback]

    public init(_ callbacks: [any Callback]) {
        self.callbacks = callbacks
    }

    public func onTrainBegin(state: TrainingState) {
        for callback in callbacks { callback.onTrainBegin(state: state) }
    }

    public func onEpochBegin(state: TrainingState) {
        for callback in callbacks { callback.onEpochBegin(state: state) }
    }

    public func onBatchBegin(state: TrainingState) {
        for callback in callbacks { callback.onBatchBegin(state: state) }
    }

    public func onBatchEnd(state: TrainingState) {
        for callback in callbacks { callback.onBatchEnd(state: state) }
    }

    public func onEpochEnd(state: TrainingState) {
        for callback in callbacks { callback.onEpochEnd(state: state) }
    }

    public func onTrainEnd(state: TrainingState) {
        for callback in callbacks { callback.onTrainEnd(state: state) }
    }
}
