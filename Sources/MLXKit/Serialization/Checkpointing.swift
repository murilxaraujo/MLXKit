import Foundation
import MLX
import MLXNN

/// Utilities for saving and loading model weights and training checkpoints.
///
/// Weights are stored as safetensors files. Full checkpoints include model
/// parameters and training metadata (epoch, step) in the safetensors metadata.
///
/// ```swift
/// // Save and load weights only
/// try Checkpointing.saveWeights(model, to: weightsURL)
/// try Checkpointing.loadWeights(into: model, from: weightsURL)
///
/// // Save and load a full training checkpoint
/// try Checkpointing.save(model: model, epoch: 10, step: 5000, to: checkpointURL)
/// let info = try Checkpointing.load(into: model, from: checkpointURL)
/// print(info.epoch) // 10
/// ```
public enum Checkpointing {

    /// Information restored from a full training checkpoint.
    public struct CheckpointInfo: Sendable {
        /// The epoch at which the checkpoint was saved.
        public let epoch: Int
        /// The training step at which the checkpoint was saved.
        public let step: Int
        /// Any additional metadata stored in the checkpoint.
        public let metadata: [String: String]
    }

    // MARK: - Weights only

    /// Saves a model's trainable parameters to a safetensors file.
    ///
    /// - Parameters:
    ///   - model: The model whose parameters to save.
    ///   - url: The file URL to write to (must have `.safetensors` extension).
    /// - Throws: If the file cannot be written.
    public static func saveWeights(_ model: Module, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        let flatParams = model.trainableParameters().flattened()
        let arrays = Dictionary(flatParams, uniquingKeysWith: { _, b in b })
        try MLX.save(arrays: arrays, url: url)
    }

    /// Loads parameters from a safetensors file into a model.
    ///
    /// - Parameters:
    ///   - model: The model to update with loaded parameters.
    ///   - url: The file URL to read from.
    /// - Throws: ``CheckpointError/fileNotFound(_:)`` if the file doesn't exist,
    ///   or if the file format is invalid.
    @discardableResult
    public static func loadWeights(into model: Module, from url: URL) throws -> Module {
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
            throw CheckpointError.fileNotFound(url)
        }
        let arrays = try loadArrays(url: url)
        let nested = unflatten(arrays)
        return model.update(parameters: nested)
    }

    // MARK: - Full checkpoint (model + metadata)

    /// Saves a full training checkpoint including model weights and training state.
    ///
    /// - Parameters:
    ///   - model: The model whose parameters to save.
    ///   - epoch: The current epoch number.
    ///   - step: The current training step.
    ///   - additionalMetadata: Extra key-value pairs to store.
    ///   - url: The file URL to write to.
    /// - Throws: If the file cannot be written.
    public static func save(
        model: Module,
        epoch: Int,
        step: Int = 0,
        additionalMetadata: [String: String] = [:],
        to url: URL
    ) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        let flatParams = model.trainableParameters().flattened()
        let arrays = Dictionary(flatParams, uniquingKeysWith: { _, b in b })
        var metadata = additionalMetadata
        metadata["mlxkit.epoch"] = String(epoch)
        metadata["mlxkit.step"] = String(step)
        try MLX.save(arrays: arrays, metadata: metadata, url: url)
    }

    /// Loads a full training checkpoint, restoring model weights and returning
    /// the saved training state.
    ///
    /// - Parameters:
    ///   - model: The model to update with loaded parameters.
    ///   - url: The file URL to read from.
    /// - Returns: A ``CheckpointInfo`` with the restored epoch, step, and metadata.
    /// - Throws: ``CheckpointError/fileNotFound(_:)`` if the file doesn't exist.
    @discardableResult
    public static func load(into model: Module, from url: URL) throws -> CheckpointInfo {
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
            throw CheckpointError.fileNotFound(url)
        }
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        let nested = unflatten(arrays)
        _ = model.update(parameters: nested)

        let epoch = metadata["mlxkit.epoch"].flatMap(Int.init) ?? 0
        let step = metadata["mlxkit.step"].flatMap(Int.init) ?? 0

        // Filter out internal keys for the public metadata
        var publicMetadata = metadata
        publicMetadata.removeValue(forKey: "mlxkit.epoch")
        publicMetadata.removeValue(forKey: "mlxkit.step")

        return CheckpointInfo(epoch: epoch, step: step, metadata: publicMetadata)
    }

    // MARK: - Helpers

    /// Converts a flat `[String: MLXArray]` dictionary (with dotted keys like
    /// `"layers.0.weight"`) back into a `ModuleParameters` nested dictionary.
    private static func unflatten(_ flat: [String: MLXArray]) -> ModuleParameters {
        // Build a nested [String: Any] tree, then convert to ModuleParameters
        var tree = [String: Any]()
        for (dottedKey, value) in flat {
            let parts = dottedKey.split(separator: ".").map(String.init)
            insertIntoTree(&tree, keys: parts, value: value)
        }
        return convertToModuleParameters(tree)
    }

    private static func insertIntoTree(_ tree: inout [String: Any], keys: [String], value: MLXArray) {
        guard let first = keys.first else { return }
        if keys.count == 1 {
            tree[first] = value
        } else {
            var subTree = (tree[first] as? [String: Any]) ?? [:]
            insertIntoTree(&subTree, keys: Array(keys.dropFirst()), value: value)
            tree[first] = subTree
        }
    }

    private static func convertToModuleParameters(_ tree: [String: Any]) -> ModuleParameters {
        var items = [String: NestedItem<String, MLXArray>]()
        for (key, value) in tree {
            items[key] = convertToNestedItem(value)
        }
        return ModuleParameters(values: items)
    }

    private static func convertToNestedItem(_ value: Any) -> NestedItem<String, MLXArray> {
        if let array = value as? MLXArray {
            return .value(array)
        } else if let dict = value as? [String: Any] {
            var items = [String: NestedItem<String, MLXArray>]()
            for (k, v) in dict {
                items[k] = convertToNestedItem(v)
            }
            return .dictionary(items)
        }
        return .none
    }
}

/// Errors that can occur during checkpoint operations.
public enum CheckpointError: Error, LocalizedError {
    /// The specified checkpoint file was not found.
    case fileNotFound(URL)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let url):
            return "Checkpoint file not found at: \(url.path(percentEncoded: false))"
        }
    }
}
