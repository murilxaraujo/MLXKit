// GPU-dependent tests — run via Xcode (xcodebuild test), not `swift test`.

#if MLX_GPU_TESTS
import Testing
import Foundation
import MLX
import MLXNN
@testable import MLXKit

/// Simple two-layer model for testing checkpoint round-trips.
private final class TinyModel: Module {
    let linear1 = Linear(inputDimensions: 4, outputDimensions: 3)
    let linear2 = Linear(inputDimensions: 3, outputDimensions: 2)

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(linear1(x))
    }
}

@Suite("Checkpointing (GPU)")
struct CheckpointGPUTests {
    private func tempURL(name: String = "test_checkpoint") -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent("\(name).safetensors")
    }

    @Test func weightRoundTrip() throws {
        let model1 = TinyModel()
        let url = tempURL(name: "weights")
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        // Save weights from model1
        try Checkpointing.saveWeights(model1, to: url)

        // Load into a fresh model
        let model2 = TinyModel()
        try Checkpointing.loadWeights(into: model2, from: url)

        // Compare parameters
        let params1 = model1.trainableParameters().flattened()
        let params2 = model2.trainableParameters().flattened()
        #expect(params1.count == params2.count)

        for (p1, p2) in zip(params1, params2) {
            #expect(p1.0 == p2.0) // same key
            #expect(p1.1.shape == p2.1.shape) // same shape
        }

        // Cleanup
        try? FileManager.default.removeItem(at: url.deletingLastPathComponent())
    }

    @Test func fullCheckpointRoundTrip() throws {
        let model = TinyModel()
        let url = tempURL(name: "full_checkpoint")
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        // Save full checkpoint
        try Checkpointing.save(
            model: model,
            epoch: 10,
            step: 5000,
            additionalMetadata: ["best_loss": "0.123"],
            to: url
        )

        // Load into a fresh model
        let model2 = TinyModel()
        let info = try Checkpointing.load(into: model2, from: url)

        #expect(info.epoch == 10)
        #expect(info.step == 5000)
        #expect(info.metadata["best_loss"] == "0.123")

        // Cleanup
        try? FileManager.default.removeItem(at: url.deletingLastPathComponent())
    }

    @Test func loadFromNonexistentPathThrows() throws {
        let model = TinyModel()
        let url = URL(filePath: "/tmp/nonexistent_\(UUID().uuidString).safetensors")

        #expect(throws: CheckpointError.self) {
            try Checkpointing.loadWeights(into: model, from: url)
        }
    }
}
#endif
