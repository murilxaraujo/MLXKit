/// A callback that wraps closures for each lifecycle hook.
///
/// Use this for quick, inline callbacks without defining a new class:
///
/// ```swift
/// let logger = LambdaCallback(
///     onEpochEnd: { state in
///         print("Epoch \(state.epoch) done")
///     }
/// )
/// ```
public final class LambdaCallback: Callback, @unchecked Sendable {
    private let _onTrainBegin: (@Sendable (TrainingState) -> Void)?
    private let _onTrainEnd: (@Sendable (TrainingState) -> Void)?
    private let _onEpochBegin: (@Sendable (TrainingState) -> Void)?
    private let _onEpochEnd: (@Sendable (TrainingState) -> Void)?
    private let _onBatchBegin: (@Sendable (TrainingState) -> Void)?
    private let _onBatchEnd: (@Sendable (TrainingState) -> Void)?

    public init(
        onTrainBegin: (@Sendable (TrainingState) -> Void)? = nil,
        onTrainEnd: (@Sendable (TrainingState) -> Void)? = nil,
        onEpochBegin: (@Sendable (TrainingState) -> Void)? = nil,
        onEpochEnd: (@Sendable (TrainingState) -> Void)? = nil,
        onBatchBegin: (@Sendable (TrainingState) -> Void)? = nil,
        onBatchEnd: (@Sendable (TrainingState) -> Void)? = nil
    ) {
        self._onTrainBegin = onTrainBegin
        self._onTrainEnd = onTrainEnd
        self._onEpochBegin = onEpochBegin
        self._onEpochEnd = onEpochEnd
        self._onBatchBegin = onBatchBegin
        self._onBatchEnd = onBatchEnd
    }

    public func onTrainBegin(state: TrainingState) { _onTrainBegin?(state) }
    public func onTrainEnd(state: TrainingState) { _onTrainEnd?(state) }
    public func onEpochBegin(state: TrainingState) { _onEpochBegin?(state) }
    public func onEpochEnd(state: TrainingState) { _onEpochEnd?(state) }
    public func onBatchBegin(state: TrainingState) { _onBatchBegin?(state) }
    public func onBatchEnd(state: TrainingState) { _onBatchEnd?(state) }
}
