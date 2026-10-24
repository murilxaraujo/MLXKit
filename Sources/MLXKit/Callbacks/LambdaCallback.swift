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
    private let _onTrainBegin: (@Sendable (TrainingState) throws -> Void)?
    private let _onTrainEnd: (@Sendable (TrainingState) throws -> Void)?
    private let _onEpochBegin: (@Sendable (TrainingState) throws -> Void)?
    private let _onEpochEnd: (@Sendable (TrainingState) throws -> Void)?
    private let _onBatchBegin: (@Sendable (TrainingState) throws -> Void)?
    private let _onBatchEnd: (@Sendable (TrainingState) throws -> Void)?

    public init(
        onTrainBegin: (@Sendable (TrainingState) throws -> Void)? = nil,
        onTrainEnd: (@Sendable (TrainingState) throws -> Void)? = nil,
        onEpochBegin: (@Sendable (TrainingState) throws -> Void)? = nil,
        onEpochEnd: (@Sendable (TrainingState) throws -> Void)? = nil,
        onBatchBegin: (@Sendable (TrainingState) throws -> Void)? = nil,
        onBatchEnd: (@Sendable (TrainingState) throws -> Void)? = nil
    ) {
        self._onTrainBegin = onTrainBegin
        self._onTrainEnd = onTrainEnd
        self._onEpochBegin = onEpochBegin
        self._onEpochEnd = onEpochEnd
        self._onBatchBegin = onBatchBegin
        self._onBatchEnd = onBatchEnd
    }

    public func onTrainBegin(state: TrainingState) throws { try _onTrainBegin?(state) }
    public func onTrainEnd(state: TrainingState) throws { try _onTrainEnd?(state) }
    public func onEpochBegin(state: TrainingState) throws { try _onEpochBegin?(state) }
    public func onEpochEnd(state: TrainingState) throws { try _onEpochEnd?(state) }
    public func onBatchBegin(state: TrainingState) throws { try _onBatchBegin?(state) }
    public func onBatchEnd(state: TrainingState) throws { try _onBatchEnd?(state) }
}
