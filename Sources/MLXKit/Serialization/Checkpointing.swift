import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// The MLXKit-owned progress restored at an epoch boundary.
///
/// `optimizerStep` counts successful optimizer updates, not input batches. The
/// caller must recreate the data sequence when resuming. MLXKit does not capture
/// arbitrary asynchronous iterator state.
public struct TrainingProgress: Sendable, Equatable {
    /// Number of fully completed epochs.
    public let completedEpochs: Int

    /// Number of successful optimizer updates completed so far.
    public let optimizerStep: Int

    public init(completedEpochs: Int, optimizerStep: Int) {
        self.completedEpochs = completedEpochs
        self.optimizerStep = optimizerStep
    }

    /// Progress before the first training epoch and optimizer update.
    public static let zero = TrainingProgress(completedEpochs: 0, optimizerStep: 0)
}

/// Serializable state supplied by an optimizer that supports training recovery.
///
/// Array keys and metadata are owned by the optimizer implementation. A
/// conformer must validate all state before mutating itself in
/// `restoreCheckpointState(_:)`; it should include relevant optimizer
/// hyperparameters in metadata so incompatible configurations can be rejected.
public struct OptimizerCheckpointState {
    public let arrays: [String: MLXArray]
    public let metadata: [String: String]

    public init(arrays: [String: MLXArray], metadata: [String: String] = [:]) {
        self.arrays = arrays
        self.metadata = metadata
    }
}

/// Optional capability for optimizers whose mutable state can be restored.
///
/// `MLXOptimizers.Optimizer` exposes state for evaluation but does not currently
/// expose a general state-restore API. Built-in optimizers therefore do not
/// conform automatically. Custom optimizers can opt in by implementing a
/// stable identifier and a validation-first state restore.
public protocol CheckpointableOptimizer: Optimizer {
    /// Stable identifier for this optimizer's state format and implementation.
    var checkpointIdentifier: String { get }

    /// Returns the optimizer's complete state needed to continue updating.
    func makeCheckpointState() throws -> OptimizerCheckpointState

    /// Restores previously validated state. Implementations must not partially
    /// mutate optimizer state if validation fails.
    func restoreCheckpointState(_ state: OptimizerCheckpointState) throws
}

/// Utilities for saving model weights, inference metadata, and resumable
/// epoch-boundary training checkpoints.
///
/// New MLXKit artifacts use a versioned safetensors envelope. Inference
/// artifacts contain trainable module parameters and caller-provided metadata;
/// MLXKit does not serialize model code, model construction, tokenizers, or
/// inference caches. Training checkpoints additionally contain optimizer state
/// and epoch/update progress, when the optimizer conforms to
/// ``CheckpointableOptimizer``.
public enum Checkpointing {

    private static let formatName = "mlxkit.artifact"
    private static let formatVersion = "1"
    private static let formatKey = "mlxkit.format"
    private static let versionKey = "mlxkit.version"
    private static let kindKey = "mlxkit.kind"
    private static let modelPrefix = "model."
    private static let optimizerPrefix = "optimizer."

    /// Information stored in an inference artifact.
    public struct InferenceArtifactInfo: Sendable {
        /// Caller-provided metadata. This does not include executable model or
        /// tokenizer state.
        public let metadata: [String: String]
    }

    /// Information restored from an epoch-boundary training checkpoint.
    public struct TrainingCheckpointInfo: Sendable {
        public let progress: TrainingProgress
        public let metadata: [String: String]
        public let optimizerIdentifier: String
    }

    /// Information stored by the legacy model-plus-progress API.
    ///
    /// This is not a resumable training checkpoint: it contains model
    /// parameters and counters only, not optimizer state.
    @available(*, deprecated, message: "Use TrainingProgress and saveTrainingCheckpoint when optimizer recovery is required")
    public struct CheckpointInfo: Sendable {
        public let epoch: Int
        public let step: Int
        public let metadata: [String: String]
    }

    // MARK: Weights

    /// Saves a model's trainable parameters only.
    public static func saveWeights(_ model: Module, to url: URL) throws {
        try saveArtifact(
            arrays: prefixed(modelArrays(model), with: modelPrefix),
            metadata: [kindKey: "weights"],
            to: url
        )
    }

    /// Loads model trainable parameters from a weights, inference, or training
    /// artifact. The model is unchanged if artifact validation fails.
    @discardableResult
    public static func loadWeights(into model: Module, from url: URL) throws -> Module {
        let artifact = try readArtifact(from: url)
        guard artifact.kind == nil || ["weights", "inference", "training", "progress", "legacy-progress"].contains(artifact.kind!) else {
            throw CheckpointError.incompatibleArtifact(expected: "model weights", actual: artifact.kind!)
        }
        let parameters = try modelParameters(from: artifact.modelArrays, for: model, at: url)
        _ = model.update(parameters: parameters)
        return model
    }

    // MARK: Inference artifact

    /// Saves trainable parameters and caller-supplied inference metadata.
    ///
    /// This format does not serialize model construction, executable code,
    /// tokenizer files, or inference caches. Applications own those assets.
    public static func saveInferenceArtifact(
        _ model: Module,
        metadata: [String: String] = [:],
        to url: URL
    ) throws {
        try validateUserMetadata(metadata)
        var artifactMetadata = metadata
        artifactMetadata[kindKey] = "inference"
        try saveArtifact(
            arrays: prefixed(modelArrays(model), with: modelPrefix),
            metadata: artifactMetadata,
            to: url
        )
    }

    /// Loads an inference artifact's parameters and returns its caller metadata.
    /// Model parameters are not changed if validation fails.
    @discardableResult
    public static func loadInferenceArtifact(
        into model: Module,
        from url: URL
    ) throws -> InferenceArtifactInfo {
        let artifact = try readArtifact(from: url)
        guard artifact.kind == "inference" else {
            throw CheckpointError.incompatibleArtifact(
                expected: "inference artifact", actual: artifact.kind ?? "legacy untyped weights"
            )
        }
        let parameters = try modelParameters(from: artifact.modelArrays, for: model, at: url)
        _ = model.update(parameters: parameters)
        return InferenceArtifactInfo(metadata: artifact.publicMetadata)
    }

    // MARK: Resumable training checkpoint

    /// Saves model parameters, optimizer state, and epoch-boundary progress.
    ///
    /// The caller must use the same model/objective/optimizer configuration and
    /// learning-rate schedule when resuming, and recreate the data sequence at
    /// the next epoch. Optimizer hyperparameters and custom state should be
    /// validated by the optimizer's `restoreCheckpointState(_:)` implementation.
    public static func saveTrainingCheckpoint(
        model: Module,
        optimizer: any CheckpointableOptimizer,
        progress: TrainingProgress,
        metadata: [String: String] = [:],
        to url: URL
    ) throws {
        try validate(progress: progress)
        try validateUserMetadata(metadata)
        guard !optimizer.checkpointIdentifier.isEmpty else {
            throw CheckpointError.invalidOptimizerIdentifier
        }

        let optimizerState = try optimizer.makeCheckpointState()
        var arrays = prefixed(modelArrays(model), with: modelPrefix)
        let optimizerArrays = prefixed(optimizerState.arrays, with: optimizerPrefix)
        for (key, value) in optimizerArrays {
            guard arrays.updateValue(value, forKey: key) == nil else {
                throw CheckpointError.duplicateArrayKey(key)
            }
        }

        var artifactMetadata = metadata
        artifactMetadata[kindKey] = "training"
        artifactMetadata["mlxkit.completedEpochs"] = String(progress.completedEpochs)
        artifactMetadata["mlxkit.optimizerStep"] = String(progress.optimizerStep)
        artifactMetadata["mlxkit.optimizerIdentifier"] = optimizer.checkpointIdentifier
        for (key, value) in optimizerState.metadata {
            let prefixedKey = "mlxkit.optimizer.\(key)"
            guard artifactMetadata[prefixedKey] == nil else {
                throw CheckpointError.reservedMetadataKey(prefixedKey)
            }
            artifactMetadata[prefixedKey] = value
        }
        try saveArtifact(arrays: arrays, metadata: artifactMetadata, to: url)
    }

    /// Saves a training checkpoint or throws when the optimizer does not
    /// implement the checkpoint-state capability.
    public static func saveTrainingCheckpoint(
        model: Module,
        optimizer: any Optimizer,
        progress: TrainingProgress,
        metadata: [String: String] = [:],
        to url: URL
    ) throws {
        guard let checkpointable = optimizer as? any CheckpointableOptimizer else {
            throw CheckpointError.unsupportedOptimizerRecovery(
                String(reflecting: type(of: optimizer))
            )
        }
        try saveTrainingCheckpoint(
            model: model,
            optimizer: checkpointable,
            progress: progress,
            metadata: metadata,
            to: url
        )
    }

    /// Restores a training checkpoint into a model and a checkpoint-capable
    /// optimizer. The optimizer identifier is checked before either is changed.
    @discardableResult
    public static func loadTrainingCheckpoint(
        into model: Module,
        optimizer: any CheckpointableOptimizer,
        from url: URL
    ) throws -> TrainingCheckpointInfo {
        let artifact = try readArtifact(from: url)
        guard artifact.kind == "training" else {
            throw CheckpointError.incompatibleArtifact(
                expected: "resumable training checkpoint", actual: artifact.kind ?? "legacy artifact"
            )
        }

        let expectedIdentifier = artifact.rawMetadata["mlxkit.optimizerIdentifier"] ?? ""
        guard expectedIdentifier == optimizer.checkpointIdentifier else {
            throw CheckpointError.optimizerMismatch(
                expected: expectedIdentifier, actual: optimizer.checkpointIdentifier
            )
        }

        let progress = TrainingProgress(
            completedEpochs: try integerMetadata("mlxkit.completedEpochs", in: artifact.rawMetadata),
            optimizerStep: try integerMetadata("mlxkit.optimizerStep", in: artifact.rawMetadata)
        )
        try validate(progress: progress)
        let modelState = try modelParameters(from: artifact.modelArrays, for: model, at: url)
        let optimizerState = OptimizerCheckpointState(
            arrays: artifact.optimizerArrays,
            metadata: artifact.rawMetadata.reduce(into: [:]) { result, entry in
                let prefix = "mlxkit.optimizer."
                if entry.key.hasPrefix(prefix) {
                    result[String(entry.key.dropFirst(prefix.count))] = entry.value
                }
            }
        )

        // The optimizer contract requires validation-before-mutation. The model
        // update is performed only after that potentially throwing operation.
        try optimizer.restoreCheckpointState(optimizerState)
        _ = model.update(parameters: modelState)
        return TrainingCheckpointInfo(
            progress: progress,
            metadata: artifact.publicMetadata,
            optimizerIdentifier: expectedIdentifier
        )
    }

    /// Restores a training checkpoint or throws when the optimizer does not
    /// implement the checkpoint-state capability.
    @discardableResult
    public static func loadTrainingCheckpoint(
        into model: Module,
        optimizer: any Optimizer,
        from url: URL
    ) throws -> TrainingCheckpointInfo {
        guard let checkpointable = optimizer as? any CheckpointableOptimizer else {
            throw CheckpointError.unsupportedOptimizerRecovery(
                String(reflecting: type(of: optimizer))
            )
        }
        return try loadTrainingCheckpoint(
            into: model,
            optimizer: checkpointable,
            from: url
        )
    }

    // MARK: Progress metadata compatibility

    /// Saves model parameters with progress metadata only.
    ///
    /// This is useful for reporting or application-managed epoch restarts. It
    /// does not save optimizer state and is not sufficient for faithful training
    /// recovery.
    public static func saveTrainingProgress(
        model: Module,
        progress: TrainingProgress,
        metadata: [String: String] = [:],
        to url: URL
    ) throws {
        try validate(progress: progress)
        try validateUserMetadata(metadata)
        var artifactMetadata = metadata
        artifactMetadata[kindKey] = "progress"
        artifactMetadata["mlxkit.completedEpochs"] = String(progress.completedEpochs)
        artifactMetadata["mlxkit.optimizerStep"] = String(progress.optimizerStep)
        try saveArtifact(arrays: prefixed(modelArrays(model), with: modelPrefix), metadata: artifactMetadata, to: url)
    }

    /// Loads model parameters and progress metadata without restoring an
    /// optimizer. This does not resume training with optimizer-equivalent state.
    @discardableResult
    public static func loadTrainingProgress(
        into model: Module,
        from url: URL
    ) throws -> (progress: TrainingProgress, metadata: [String: String]) {
        let artifact = try readArtifact(from: url)
        guard artifact.kind == "progress" || artifact.kind == "legacy-progress" else {
            throw CheckpointError.incompatibleArtifact(
                expected: "training progress artifact", actual: artifact.kind ?? "legacy artifact"
            )
        }
        let progress: TrainingProgress
        if artifact.kind == "legacy-progress" {
            progress = TrainingProgress(
                completedEpochs: Int(artifact.rawMetadata["mlxkit.epoch"] ?? "0") ?? 0,
                optimizerStep: Int(artifact.rawMetadata["mlxkit.step"] ?? "0") ?? 0
            )
        } else {
            progress = TrainingProgress(
                completedEpochs: try integerMetadata("mlxkit.completedEpochs", in: artifact.rawMetadata),
                optimizerStep: try integerMetadata("mlxkit.optimizerStep", in: artifact.rawMetadata)
            )
        }
        try validate(progress: progress)
        let parameters = try modelParameters(from: artifact.modelArrays, for: model, at: url)
        _ = model.update(parameters: parameters)
        return (progress, artifact.publicMetadata)
    }

    /// Legacy compatibility API. Saves parameters and counters, not optimizer
    /// state; use `saveTrainingCheckpoint` for recovery.
    @available(*, deprecated, message: "This saves weights and progress metadata only; use saveTrainingCheckpoint for optimizer recovery")
    public static func save(
        model: Module,
        epoch: Int,
        step: Int = 0,
        additionalMetadata: [String: String] = [:],
        to url: URL
    ) throws {
        try saveTrainingProgress(
            model: model,
            progress: TrainingProgress(completedEpochs: epoch, optimizerStep: step),
            metadata: additionalMetadata,
            to: url
        )
    }

    /// Legacy compatibility API. The returned counters do not imply optimizer
    /// state was restored.
    @available(*, deprecated, message: "This restores weights and progress metadata only; use loadTrainingCheckpoint for optimizer recovery")
    @discardableResult
    public static func load(into model: Module, from url: URL) throws -> CheckpointInfo {
        let restored = try loadTrainingProgress(into: model, from: url)
        return CheckpointInfo(
            epoch: restored.progress.completedEpochs,
            step: restored.progress.optimizerStep,
            metadata: restored.metadata
        )
    }

    // MARK: Validation and serialization helpers

    private struct Artifact {
        let kind: String?
        let modelArrays: [String: MLXArray]
        let optimizerArrays: [String: MLXArray]
        let rawMetadata: [String: String]
        let publicMetadata: [String: String]
    }

    private static func validate(progress: TrainingProgress) throws {
        guard progress.completedEpochs >= 0, progress.optimizerStep >= 0 else {
            throw CheckpointError.invalidProgress(
                completedEpochs: progress.completedEpochs,
                optimizerStep: progress.optimizerStep
            )
        }
    }

    private static func validateUserMetadata(_ metadata: [String: String]) throws {
        if let key = metadata.keys.first(where: { $0.hasPrefix("mlxkit.") }) {
            throw CheckpointError.reservedMetadataKey(key)
        }
    }

    private static func modelArrays(_ model: Module) -> [String: MLXArray] {
        Dictionary(model.trainableParameters().flattened(), uniquingKeysWith: { _, newer in newer })
    }

    private static func prefixed(
        _ arrays: [String: MLXArray],
        with prefix: String
    ) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: arrays.map { (prefix + $0.key, $0.value) })
    }

    private static func saveArtifact(
        arrays: [String: MLXArray],
        metadata: [String: String],
        to url: URL
    ) throws {
        guard url.isFileURL else { throw CheckpointError.nonFileURL(url) }
        var completeMetadata = metadata
        completeMetadata[formatKey] = formatName
        completeMetadata[versionKey] = formatVersion

        let manager = FileManager.default
        let directory = url.deletingLastPathComponent()
        let temporaryURL = directory.appendingPathComponent(
            ".\(url.deletingPathExtension().lastPathComponent).\(UUID().uuidString).safetensors"
        )
        defer { try? manager.removeItem(at: temporaryURL) }

        do {
            try manager.createDirectory(at: directory, withIntermediateDirectories: true)
            try MLX.save(arrays: arrays, metadata: completeMetadata, url: temporaryURL)
            if manager.fileExists(atPath: url.path(percentEncoded: false)) {
                _ = try manager.replaceItemAt(url, withItemAt: temporaryURL)
            } else {
                try manager.moveItem(at: temporaryURL, to: url)
            }
        } catch {
            throw CheckpointError.saveFailed(url, reason: error.localizedDescription)
        }
    }

    private static func readArtifact(from url: URL) throws -> Artifact {
        guard url.isFileURL else { throw CheckpointError.nonFileURL(url) }
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
            throw CheckpointError.fileNotFound(url)
        }

        let arrays: [String: MLXArray]
        let metadata: [String: String]
        do {
            (arrays, metadata) = try loadArraysAndMetadata(url: url)
        } catch {
            throw CheckpointError.corruptArtifact(url, reason: error.localizedDescription)
        }

        let format = metadata[formatKey]
        let version = metadata[versionKey]
        if format == nil && version == nil {
            // Legacy weight files stored raw parameter keys. Legacy progress
            // files stored the same arrays plus mlxkit.epoch/step metadata.
            let isLegacyProgress = metadata["mlxkit.epoch"] != nil || metadata["mlxkit.step"] != nil
            let kind = isLegacyProgress ? "legacy-progress" : nil
            var publicMetadata = metadata
            publicMetadata.removeValue(forKey: "mlxkit.epoch")
            publicMetadata.removeValue(forKey: "mlxkit.step")
            return Artifact(
                kind: kind,
                modelArrays: arrays,
                optimizerArrays: [:],
                rawMetadata: metadata,
                publicMetadata: publicMetadata
            )
        }
        guard format == formatName else {
            throw CheckpointError.incompatibleFormat(expected: formatName, actual: format ?? "missing")
        }
        guard version == formatVersion else {
            throw CheckpointError.unsupportedVersion(version ?? "missing")
        }
        guard let kind = metadata[kindKey] else {
            throw CheckpointError.corruptArtifact(url, reason: "Missing artifact kind metadata.")
        }

        var modelArrays: [String: MLXArray] = [:]
        var optimizerArrays: [String: MLXArray] = [:]
        for (key, value) in arrays {
            if key.hasPrefix(modelPrefix) {
                modelArrays[String(key.dropFirst(modelPrefix.count))] = value
            } else if key.hasPrefix(optimizerPrefix) {
                optimizerArrays[String(key.dropFirst(optimizerPrefix.count))] = value
            } else {
                throw CheckpointError.corruptArtifact(
                    url, reason: "Array key '\(key)' has no recognized artifact namespace."
                )
            }
        }

        var publicMetadata = metadata
        for key in [formatKey, versionKey, kindKey, "mlxkit.completedEpochs", "mlxkit.optimizerStep", "mlxkit.optimizerIdentifier"] {
            publicMetadata.removeValue(forKey: key)
        }
        for key in metadata.keys where key.hasPrefix("mlxkit.optimizer.") {
            publicMetadata.removeValue(forKey: key)
        }
        return Artifact(
            kind: kind,
            modelArrays: modelArrays,
            optimizerArrays: optimizerArrays,
            rawMetadata: metadata,
            publicMetadata: publicMetadata
        )
    }

    private static func modelParameters(
        from saved: [String: MLXArray],
        for model: Module,
        at url: URL
    ) throws -> ModuleParameters {
        let current = modelArrays(model)
        let expectedKeys = current.keys.sorted()
        let actualKeys = saved.keys.sorted()
        guard expectedKeys == actualKeys else {
            throw CheckpointError.parameterKeyMismatch(
                url, expected: expectedKeys, actual: actualKeys
            )
        }
        for key in expectedKeys {
            let expected = current[key]!
            let actual = saved[key]!
            guard expected.shape == actual.shape else {
                throw CheckpointError.parameterShapeMismatch(
                    url, parameter: key, expected: expected.shape, actual: actual.shape
                )
            }
            guard expected.dtype == actual.dtype else {
                throw CheckpointError.parameterDTypeMismatch(
                    url, parameter: key, expected: String(describing: expected.dtype),
                    actual: String(describing: actual.dtype)
                )
            }
        }
        return unflatten(saved)
    }

    private static func integerMetadata(_ key: String, in metadata: [String: String]) throws -> Int {
        guard let rawValue = metadata[key], let value = Int(rawValue) else {
            throw CheckpointError.invalidMetadata(key)
        }
        return value
    }

    private static func unflatten(_ flat: [String: MLXArray]) -> ModuleParameters {
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
            for (key, value) in dict {
                items[key] = convertToNestedItem(value)
            }
            return .dictionary(items)
        }
        return .none
    }
}

/// Errors reported by artifact persistence and restore operations.
public enum CheckpointError: Error, LocalizedError {
    case fileNotFound(URL)
    case nonFileURL(URL)
    case corruptArtifact(URL, reason: String)
    case saveFailed(URL, reason: String)
    case incompatibleFormat(expected: String, actual: String)
    case incompatibleArtifact(expected: String, actual: String)
    case unsupportedVersion(String)
    case invalidMetadata(String)
    case invalidProgress(completedEpochs: Int, optimizerStep: Int)
    case invalidOptimizerIdentifier
    case unsupportedOptimizerRecovery(String)
    case optimizerMismatch(expected: String, actual: String)
    case reservedMetadataKey(String)
    case duplicateArrayKey(String)
    case parameterKeyMismatch(URL, expected: [String], actual: [String])
    case parameterShapeMismatch(URL, parameter: String, expected: [Int], actual: [Int])
    case parameterDTypeMismatch(URL, parameter: String, expected: String, actual: String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let url):
            "Artifact file not found at: \(url.path(percentEncoded: false))"
        case .nonFileURL(let url):
            "Artifact URL must use the file scheme: \(url.absoluteString)"
        case .corruptArtifact(let url, let reason):
            "Artifact at \(url.path(percentEncoded: false)) is corrupt: \(reason)"
        case .saveFailed(let url, let reason):
            "Could not save artifact to \(url.path(percentEncoded: false)): \(reason)"
        case .incompatibleFormat(let expected, let actual):
            "Expected artifact format '\(expected)', received '\(actual)'."
        case .incompatibleArtifact(let expected, let actual):
            "Expected a \(expected), received \(actual)."
        case .unsupportedVersion(let version):
            "Artifact format version '\(version)' is not supported."
        case .invalidMetadata(let key):
            "Artifact metadata '\(key)' is missing or invalid."
        case .invalidProgress(let completedEpochs, let optimizerStep):
            "Training progress must be nonnegative; received epochs=\(completedEpochs), optimizerStep=\(optimizerStep)."
        case .invalidOptimizerIdentifier:
            "Checkpointable optimizers must provide a nonempty checkpoint identifier."
        case .unsupportedOptimizerRecovery(let type):
            "Optimizer '\(type)' does not support checkpoint recovery. Conform it to CheckpointableOptimizer or use a weights/progress artifact."
        case .optimizerMismatch(let expected, let actual):
            "Checkpoint requires optimizer '\(expected)', received '\(actual)'."
        case .reservedMetadataKey(let key):
            "Metadata key '\(key)' is reserved by MLXKit or the optimizer checkpoint."
        case .duplicateArrayKey(let key):
            "Artifact contains duplicate tensor key '\(key)'."
        case .parameterKeyMismatch(let url, let expected, let actual):
            "Model parameters in \(url.lastPathComponent) do not match. Expected keys \(expected), received \(actual)."
        case .parameterShapeMismatch(let url, let parameter, let expected, let actual):
            "Parameter '\(parameter)' in \(url.lastPathComponent) has shape \(actual); expected \(expected)."
        case .parameterDTypeMismatch(let url, let parameter, let expected, let actual):
            "Parameter '\(parameter)' in \(url.lastPathComponent) has dtype \(actual); expected \(expected)."
        }
    }
}
