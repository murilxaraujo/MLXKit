import Foundation
import MLX
import MLXNN

/// Saves model weights when a monitored metric improves.
///
/// Uses `MLX.save` to write the model's trainable parameters to the
/// specified directory.
///
/// ```swift
/// let checkpoint = ModelCheckpoint(
///     directory: URL(filePath: "/tmp/checkpoints"),
///     monitor: "val_loss"
/// )
/// ```
public final class ModelCheckpoint: Callback, @unchecked Sendable {

    public let directory: URL
    public let monitor: String
    public let mode: EarlyStopping.Mode

    /// The model to checkpoint. Set by the training engine before training begins.
    public var model: Module?

    private var bestValue: Float?
    private var resolvedMode: ResolvedMode = .min

    private enum ResolvedMode {
        case min, max

        func isBetter(_ current: Float, than best: Float) -> Bool {
            switch self {
            case .min: return current < best
            case .max: return current > best
            }
        }
    }

    public init(
        directory: URL,
        monitor: String = "val_loss",
        mode: EarlyStopping.Mode = .auto
    ) {
        self.directory = directory
        self.monitor = monitor
        self.mode = mode
    }

    public func onTrainBegin(state: TrainingState) {
        resolvedMode = resolveMode()
        bestValue = nil
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    public func onEpochEnd(state: TrainingState) {
        guard let current = state.logs[monitor] else { return }

        let shouldSave: Bool
        if let best = bestValue {
            shouldSave = resolvedMode.isBetter(current, than: best)
        } else {
            shouldSave = true
        }

        if shouldSave {
            bestValue = current
            saveCheckpoint(epoch: state.epoch)
        }
    }

    private func saveCheckpoint(epoch: Int) {
        guard let model else { return }
        let flatParams = model.trainableParameters().flattened()
        let arrays = Dictionary(flatParams, uniquingKeysWith: { _, b in b })
        let url = directory.appendingPathComponent("checkpoint_epoch\(epoch).safetensors")
        try? save(arrays: arrays, url: url)
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
