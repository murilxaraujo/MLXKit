import Foundation

/// Prints training progress to standard output.
///
/// Reports epoch and batch progress with metric values from the training
/// state logs.
///
/// ```swift
/// let progress = ProgressReporter()
/// ```
public final class ProgressReporter: Callback, @unchecked Sendable {

    public init() {}

    public func onEpochEnd(state: TrainingState) {
        var parts = ["Epoch \(state.epoch + 1)/\(state.totalEpochs)"]
        for (key, value) in state.logs.sorted(by: { $0.key < $1.key }) {
            parts.append("\(key): \(String(format: "%.4f", value))")
        }
        print(parts.joined(separator: " — "))
    }
}
