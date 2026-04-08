import Foundation

/// Stops training when a monitored metric stops improving.
///
/// Tracks the monitored metric across epochs and sets
/// ``TrainingState/shouldStop`` if no improvement is seen for `patience`
/// consecutive epochs.
///
/// ```swift
/// let earlyStopping = EarlyStopping(monitor: "val_loss", patience: 5)
/// ```
///
/// - Parameters:
///   - monitor: The metric key to watch in ``TrainingState/logs``.
///   - patience: Number of epochs with no improvement before stopping.
///   - minDelta: Minimum change to qualify as an improvement.
///   - mode: Whether to minimize (`.min`) or maximize (`.max`) the metric.
///     If `.auto`, inferred from the metric name ("loss" → `.min`, otherwise `.max`).
public final class EarlyStopping: Callback, @unchecked Sendable {

    /// Whether the metric should be minimized or maximized.
    public enum Mode: Sendable {
        case min, max, auto
    }

    public let monitor: String
    public let patience: Int
    public let minDelta: Float
    public let mode: Mode

    private var bestValue: Float?
    private var waitCount: Int = 0
    private var resolvedMode: ResolvedMode = .min

    private enum ResolvedMode {
        case min, max

        func isBetter(_ current: Float, than best: Float, by delta: Float) -> Bool {
            switch self {
            case .min: return current < best - delta
            case .max: return current > best + delta
            }
        }
    }

    public init(
        monitor: String = "val_loss",
        patience: Int = 5,
        minDelta: Float = 0,
        mode: Mode = .auto
    ) {
        self.monitor = monitor
        self.patience = patience
        self.minDelta = minDelta
        self.mode = mode
    }

    public func onTrainBegin(state: TrainingState) {
        resolvedMode = resolveMode()
        bestValue = nil
        waitCount = 0
    }

    public func onEpochEnd(state: TrainingState) {
        guard let current = state.logs[monitor] else { return }

        guard let best = bestValue else {
            bestValue = current
            return
        }

        if resolvedMode.isBetter(current, than: best, by: minDelta) {
            bestValue = current
            waitCount = 0
        } else {
            waitCount += 1
            if waitCount >= patience {
                state.shouldStop = true
            }
        }
    }

    private func resolveMode() -> ResolvedMode {
        switch mode {
        case .min: return .min
        case .max: return .max
        case .auto:
            let lowered = monitor.lowercased()
            if lowered.contains("loss") || lowered.contains("error") {
                return .min
            }
            return .max
        }
    }
}
