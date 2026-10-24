// GPU-dependent tests — run via Xcode (xcodebuild test), not `swift test`.

#if MLX_GPU_TESTS
import Testing
import Foundation
import MLX
import MLXNN
import MLXOptimizers
@testable import MLXKit

private final class TinyModel: Module {
    let linear1 = Linear(inputDimensions: 4, outputDimensions: 3)
    let linear2 = Linear(inputDimensions: 3, outputDimensions: 2)

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(linear1(x))
    }
}

private final class WrongShapeModel: Module {
    let linear = Linear(inputDimensions: 4, outputDimensions: 5)
}

private final class ScalarCheckpointModel: Module, TrainableModel {
    typealias Input = MLXArray
    typealias Output = MLXArray

    let weight = MLXArray(Float32(1))

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        input * weight
    }
}

/// Small test optimizer with explicit, restorable momentum state. It also makes
/// the schedule-step continuation observable in the resume parity test.
private final class MomentumCheckpointOptimizer: CheckpointableOptimizer, LearningRateControllable {
    let checkpointIdentifier: String
    let initialLearningRate: Float
    let momentum: Float
    var learningRate: Float
    private var velocity: [String: MLXArray] = [:]
    private(set) var updateCount = 0

    init(
        learningRate: Float,
        momentum: Float,
        checkpointIdentifier: String = "tests.momentum.v1"
    ) {
        self.initialLearningRate = learningRate
        self.learningRate = learningRate
        self.momentum = momentum
        self.checkpointIdentifier = checkpointIdentifier
    }

    func setLearningRate(_ value: Float) throws {
        learningRate = value
    }

    func innerState() -> [MLXArray] {
        velocity.keys.sorted().compactMap { velocity[$0] }
    }

    func update(model: Module, gradients: ModuleParameters) {
        let parameters = Dictionary(model.trainableParameters().flattened(), uniquingKeysWith: { _, b in b })
        let gradientValues = Dictionary(gradients.flattened(), uniquingKeysWith: { _, b in b })
        var nextParameters: [(String, MLXArray)] = []
        var nextVelocity: [String: MLXArray] = [:]

        for (key, parameter) in parameters {
            let gradient = gradientValues[key]!
            let currentVelocity = velocity[key] ?? MLXArray.zeros(like: parameter)
            let updatedVelocity = momentum * currentVelocity + gradient
            nextVelocity[key] = updatedVelocity
            nextParameters.append((key, parameter - learningRate * updatedVelocity))
        }

        velocity = nextVelocity
        updateCount += 1
        _ = model.update(parameters: .unflattened(nextParameters))
    }

    func makeCheckpointState() throws -> OptimizerCheckpointState {
        OptimizerCheckpointState(
            arrays: velocity,
            metadata: [
                "updates": String(updateCount),
                "initialLearningRate": String(initialLearningRate),
                "learningRate": String(learningRate),
                "momentum": String(momentum),
            ]
        )
    }

    func restoreCheckpointState(_ state: OptimizerCheckpointState) throws {
        guard let updates = state.metadata["updates"].flatMap(Int.init), updates >= 0,
              let initialLR = state.metadata["initialLearningRate"].flatMap(Float.init),
              initialLR == initialLearningRate,
              let restoredLR = state.metadata["learningRate"].flatMap(Float.init),
              restoredLR.isFinite, restoredLR >= 0,
              let restoredMomentum = state.metadata["momentum"].flatMap(Float.init),
              restoredMomentum == momentum
        else {
            throw TestOptimizerError.incompatibleState
        }
        guard state.arrays.values.allSatisfy({ array in
            eval(isFinite(array).all())
            return isFinite(array).all().item(Bool.self)
        }) else {
            throw TestOptimizerError.incompatibleState
        }

        // All checks happen before this commit.
        velocity = state.arrays
        updateCount = updates
        learningRate = restoredLR
    }
}

private enum TestOptimizerError: Error {
    case incompatibleState
}

@Suite("Checkpointing (GPU)")
struct CheckpointGPUTests {
    private func tempURL(name: String = "artifact") -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent("\(name).safetensors")
    }

    private func expectParametersEqual(_ lhs: Module, _ rhs: Module) {
        let left = Dictionary(lhs.trainableParameters().flattened(), uniquingKeysWith: { _, b in b })
        let right = Dictionary(rhs.trainableParameters().flattened(), uniquingKeysWith: { _, b in b })
        #expect(Set(left.keys) == Set(right.keys))
        for key in left.keys {
            #expect(left[key]!.allClose(right[key]!, rtol: 1e-6, atol: 1e-6).all().item(Bool.self))
        }
    }

    @Test func weightsRoundTripAndStrictlyValidateModelBeforeMutation() throws {
        let source = TinyModel()
        let url = tempURL(name: "weights")
        try Checkpointing.saveWeights(source, to: url)

        let restored = TinyModel()
        try Checkpointing.loadWeights(into: restored, from: url)
        expectParametersEqual(source, restored)

        let incompatible = WrongShapeModel()
        let before = Dictionary(incompatible.trainableParameters().flattened(), uniquingKeysWith: { _, b in b })
        #expect(throws: CheckpointError.self) {
            try Checkpointing.loadWeights(into: incompatible, from: url)
        }
        let after = Dictionary(incompatible.trainableParameters().flattened(), uniquingKeysWith: { _, b in b })
        for key in before.keys {
            #expect(before[key]!.allClose(after[key]!, rtol: 0, atol: 0).all().item(Bool.self))
        }
    }

    @Test func inferenceArtifactRoundTripPreservesCallerMetadata() throws {
        let source = TinyModel()
        let url = tempURL(name: "inference")
        try Checkpointing.saveInferenceArtifact(
            source,
            metadata: ["architecture": "tiny-v1", "labels": "a,b"],
            to: url
        )

        let restored = TinyModel()
        let info = try Checkpointing.loadInferenceArtifact(into: restored, from: url)
        #expect(info.metadata == ["architecture": "tiny-v1", "labels": "a,b"])
        expectParametersEqual(source, restored)
    }

    @Test func savingReplacesExistingArtifactAtomically() throws {
        let source = TinyModel()
        let restored = TinyModel()
        let url = tempURL(name: "replace")
        try Checkpointing.saveWeights(source, to: url)
        try Checkpointing.saveInferenceArtifact(restored, metadata: ["replacement": "yes"], to: url)

        let destination = TinyModel()
        let info = try Checkpointing.loadInferenceArtifact(into: destination, from: url)
        #expect(info.metadata["replacement"] == "yes")
        expectParametersEqual(restored, destination)
    }

    @Test func corruptAndUnknownVersionArtifactsFailWithoutChangingModel() throws {
        let source = TinyModel()
        let corruptURL = tempURL(name: "corrupt")
        try FileManager.default.createDirectory(
            at: corruptURL.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try Data([0, 1, 2, 3, 4]).write(to: corruptURL)
        let beforeCorruptLoad = TinyModel()
        let beforeCorruptValues = Dictionary(
            beforeCorruptLoad.trainableParameters().flattened().map { ($0.0, $0.1.asArray(Float.self)) },
            uniquingKeysWith: { _, b in b }
        )
        #expect(throws: CheckpointError.self) {
            try Checkpointing.loadWeights(into: beforeCorruptLoad, from: corruptURL)
        }
        for (key, value) in beforeCorruptValues {
            let after = Dictionary(
                beforeCorruptLoad.trainableParameters().flattened(),
                uniquingKeysWith: { _, b in b }
            )[key]!
            #expect(after.asArray(Float.self) == value)
        }

        let versionURL = tempURL(name: "future-version")
        let modelArrays = Dictionary(source.trainableParameters().flattened(), uniquingKeysWith: { _, b in b })
        let namespaced = Dictionary(modelArrays.map { ("model.\($0.key)", $0.value) }, uniquingKeysWith: { _, b in b })
        try FileManager.default.createDirectory(
            at: versionURL.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try MLX.save(
            arrays: namespaced,
            metadata: [
                "mlxkit.format": "mlxkit.artifact",
                "mlxkit.version": "999",
                "mlxkit.kind": "weights",
            ],
            url: versionURL
        )

        let target = TinyModel()
        let before = Dictionary(target.trainableParameters().flattened(), uniquingKeysWith: { _, b in b })
        #expect(throws: CheckpointError.self) {
            try Checkpointing.loadWeights(into: target, from: versionURL)
        }
        let after = Dictionary(target.trainableParameters().flattened(), uniquingKeysWith: { _, b in b })
        for key in before.keys {
            #expect(before[key]!.allClose(after[key]!, rtol: 0, atol: 0).all().item(Bool.self))
        }
    }

    @Test func trainingCheckpointRoundTripRestoresOptimizerAndProgress() throws {
        let source = ScalarCheckpointModel()
        let optimizer = MomentumCheckpointOptimizer(learningRate: 0.1, momentum: 0.8)
        optimizer.update(model: source, gradients: .unflattened([("weight", MLXArray(Float32(0.5)))]))
        optimizer.update(model: source, gradients: .unflattened([("weight", MLXArray(Float32(0.25)))]))

        let url = tempURL(name: "training")
        try Checkpointing.saveTrainingCheckpoint(
            model: source,
            optimizer: optimizer,
            progress: TrainingProgress(completedEpochs: 3, optimizerStep: 8),
            metadata: ["run": "parity"],
            to: url
        )

        let restoredModel = ScalarCheckpointModel()
        let restoredOptimizer = MomentumCheckpointOptimizer(learningRate: 0.1, momentum: 0.8)
        let info = try Checkpointing.loadTrainingCheckpoint(
            into: restoredModel,
            optimizer: restoredOptimizer,
            from: url
        )
        #expect(info.progress == TrainingProgress(completedEpochs: 3, optimizerStep: 8))
        #expect(info.metadata["run"] == "parity")
        #expect(restoredOptimizer.updateCount == 2)
        #expect(restoredOptimizer.innerState().count == 1)
        expectParametersEqual(source, restoredModel)
    }

    @Test func optimizerMismatchAndRejectedOptimizerStateDoNotMutateModel() throws {
        let source = ScalarCheckpointModel()
        let sourceOptimizer = MomentumCheckpointOptimizer(learningRate: 0.1, momentum: 0.8)
        sourceOptimizer.update(
            model: source,
            gradients: .unflattened([("weight", MLXArray(Float32(0.5)))])
        )
        let url = tempURL(name: "optimizer-validation")
        try Checkpointing.saveTrainingCheckpoint(
            model: source,
            optimizer: sourceOptimizer,
            progress: TrainingProgress(completedEpochs: 1, optimizerStep: 1),
            to: url
        )

        let target = ScalarCheckpointModel()
        let originalWeight = target.weight.item(Float.self)
        let incompatibleOptimizer = MomentumCheckpointOptimizer(learningRate: 0.1, momentum: 0.9)
        #expect(throws: TestOptimizerError.self) {
            try Checkpointing.loadTrainingCheckpoint(
                into: target,
                optimizer: incompatibleOptimizer,
                from: url
            )
        }
        #expect(target.weight.item(Float.self) == originalWeight)
        #expect(incompatibleOptimizer.updateCount == 0)

        let mismatchedOptimizer = MomentumCheckpointOptimizer(
            learningRate: 0.1,
            momentum: 0.8,
            checkpointIdentifier: "tests.other.v1"
        )
        let wrongURL = tempURL(name: "wrong-optimizer")
        let untouchedModel = ScalarCheckpointModel()
        let untouchedWeight = untouchedModel.weight.item(Float.self)
        try Checkpointing.saveTrainingCheckpoint(
            model: source,
            optimizer: sourceOptimizer,
            progress: TrainingProgress(completedEpochs: 1, optimizerStep: 1),
            to: wrongURL
        )
        let otherModel = ScalarCheckpointModel()
        #expect(throws: CheckpointError.self) {
            try Checkpointing.loadTrainingCheckpoint(
                into: otherModel,
                optimizer: mismatchedOptimizer as any Optimizer,
                from: wrongURL
            )
        }
        #expect(untouchedModel.weight.item(Float.self) == untouchedWeight)
        #expect(mismatchedOptimizer.updateCount == 0)
    }

    @Test func unsupportedBuiltInOptimizerFailsClearly() throws {
        let url = tempURL(name: "unsupported-optimizer")
        #expect(throws: CheckpointError.self) {
            try Checkpointing.saveTrainingCheckpoint(
                model: ScalarCheckpointModel(),
                optimizer: SGD(learningRate: 0.1) as any Optimizer,
                progress: TrainingProgress(completedEpochs: 0, optimizerStep: 0),
                to: url
            )
        }
    }

    @Test func malformedProgressAndFilesystemErrorsAreReported() throws {
        let model = TinyModel()
        let malformedURL = tempURL(name: "malformed-progress")
        try FileManager.default.createDirectory(
            at: malformedURL.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        let arrays = Dictionary(
            model.trainableParameters().flattened().map { ("model.\($0.0)", $0.1) },
            uniquingKeysWith: { _, b in b }
        )
        try MLX.save(
            arrays: arrays,
            metadata: [
                "mlxkit.format": "mlxkit.artifact",
                "mlxkit.version": "1",
                "mlxkit.kind": "progress",
                "mlxkit.completedEpochs": "not-an-integer",
                "mlxkit.optimizerStep": "4",
            ],
            url: malformedURL
        )
        #expect(throws: CheckpointError.self) {
            try Checkpointing.loadTrainingProgress(into: TinyModel(), from: malformedURL)
        }

        let blockedParent = tempURL(name: "parent-is-a-file")
        try FileManager.default.createDirectory(
            at: blockedParent.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try Data([1]).write(to: blockedParent)
        let childURL = blockedParent.appendingPathComponent("weights.safetensors")
        #expect(throws: CheckpointError.self) {
            try Checkpointing.saveWeights(model, to: childURL)
        }
        #expect(FileManager.default.fileExists(atPath: blockedParent.path(percentEncoded: false)))
    }

    @Test func legacyUnversionedWeightsAndProgressRemainReadable() throws {
        let source = TinyModel()
        let rawArrays = Dictionary(
            source.trainableParameters().flattened(), uniquingKeysWith: { _, b in b }
        )
        let weightsURL = tempURL(name: "legacy-weights")
        try FileManager.default.createDirectory(
            at: weightsURL.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try MLX.save(arrays: rawArrays, url: weightsURL)
        let restoredWeights = TinyModel()
        try Checkpointing.loadWeights(into: restoredWeights, from: weightsURL)
        expectParametersEqual(source, restoredWeights)

        let progressURL = tempURL(name: "legacy-progress")
        try FileManager.default.createDirectory(
            at: progressURL.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try MLX.save(
            arrays: rawArrays,
            metadata: ["mlxkit.epoch": "4", "mlxkit.step": "12", "note": "old"],
            url: progressURL
        )
        let restoredProgressModel = TinyModel()
        let progress = try Checkpointing.loadTrainingProgress(into: restoredProgressModel, from: progressURL)
        #expect(progress.progress == TrainingProgress(completedEpochs: 4, optimizerStep: 12))
        #expect(progress.metadata["note"] == "old")
    }

    @Test func trainingResumeMatchesUninterruptedTrainingIncludingScheduleStep() async throws {
        func loader() throws -> DataLoader<MLXArray, MLXArray> {
            let features = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float32]).reshaped([4, 1])
            let targets = MLXArray([0.5, 1.0, 1.5, 2.0] as [Float32]).reshaped([4, 1])
            return try DataLoader(
                source: InMemoryDataset(features: features, labels: targets),
                batchSize: 2
            )
        }

        let uninterrupted = ScalarCheckpointModel()
        let uninterruptedOptimizer = MomentumCheckpointOptimizer(learningRate: 0.08, momentum: 0.7)
        let schedule = CosineDecay(initialLR: 0.08, totalSteps: 4)
        let uninterruptedConfig = TrainingConfiguration(
            optimizer: uninterruptedOptimizer,
            objective: MSELoss(),
            learningRateSchedule: schedule
        )
        _ = try await TrainingEngine().fit(
            uninterrupted,
            configuration: uninterruptedConfig,
            data: try loader(),
            epochs: 2
        )

        let firstPart = ScalarCheckpointModel()
        let firstOptimizer = MomentumCheckpointOptimizer(learningRate: 0.08, momentum: 0.7)
        let firstConfig = TrainingConfiguration(
            optimizer: firstOptimizer,
            objective: MSELoss(),
            learningRateSchedule: schedule
        )
        _ = try await TrainingEngine().fit(
            firstPart,
            configuration: firstConfig,
            data: try loader(),
            epochs: 1
        )

        let url = tempURL(name: "resume")
        try Checkpointing.saveTrainingCheckpoint(
            model: firstPart,
            optimizer: firstOptimizer,
            progress: TrainingProgress(completedEpochs: 1, optimizerStep: firstOptimizer.updateCount),
            to: url
        )

        let resumedModel = ScalarCheckpointModel()
        let resumedOptimizer = MomentumCheckpointOptimizer(learningRate: 0.08, momentum: 0.7)
        let checkpoint = try Checkpointing.loadTrainingCheckpoint(
            into: resumedModel,
            optimizer: resumedOptimizer,
            from: url
        )
        let resumedConfig = TrainingConfiguration(
            optimizer: resumedOptimizer,
            objective: MSELoss(),
            learningRateSchedule: schedule
        )
        _ = try await TrainingEngine().fit(
            resumedModel,
            configuration: resumedConfig,
            data: try loader(),
            epochs: 2,
            resumeFrom: checkpoint.progress
        )

        expectParametersEqual(uninterrupted, resumedModel)
        #expect(resumedOptimizer.updateCount == uninterruptedOptimizer.updateCount)
        #expect(abs(resumedOptimizer.learningRate - uninterruptedOptimizer.learningRate) < 1e-7)
    }

    @Test func legacyProgressAPIIsExplicitlyNotAcceptedAsTrainingRecovery() throws {
        let model = TinyModel()
        let url = tempURL(name: "progress")
        try Checkpointing.saveTrainingProgress(
            model: model,
            progress: TrainingProgress(completedEpochs: 2, optimizerStep: 4),
            to: url
        )

        let target = TinyModel()
        #expect(throws: CheckpointError.self) {
            try Checkpointing.loadTrainingCheckpoint(
                into: target,
                optimizer: MomentumCheckpointOptimizer(learningRate: 0.1, momentum: 0.8),
                from: url
            )
        }
        let restored = try Checkpointing.loadTrainingProgress(into: target, from: url)
        #expect(restored.progress == TrainingProgress(completedEpochs: 2, optimizerStep: 4))
    }

    @Test func loadFromNonexistentPathThrows() {
        let model = TinyModel()
        let url = URL(filePath: "/nonexistent/checkpoint-\(UUID().uuidString).safetensors")
        #expect(throws: CheckpointError.self) {
            try Checkpointing.loadWeights(into: model, from: url)
        }
    }
}
#endif
