// GPU-dependent tiny-decoder correctness tests — run via `make test-gpu`.

#if MLX_GPU_TESTS
import Testing
import MLX
import MLXNN
import MLXOptimizers
import MLXKit
@testable import TinyDecoderCore

@Suite("Tiny decoder correctness (GPU)", .serialized)
struct TinyDecoderGPUTests {
    @Test func futureTokensDoNotChangeEarlierLogits() {
        seed(22)
        let model = TinyDecoder(
            vocabularySize: 10,
            width: 8,
            headCount: 2,
            maximumSequenceLength: 4
        )
        let prefix = MLXArray([[Int32(1), 2, 3, 4]])
        let changedFuture = MLXArray([[Int32(1), 2, 8, 9]])

        let originalLogits = model(prefix)
        let changedLogits = model(changedFuture)
        eval(originalLogits, changedLogits)

        // Both positions precede the first changed token. Compare each vector
        // directly so the test does not depend on an array-slicing API.
        #expect(originalLogits[0, 0].allClose(changedLogits[0, 0], rtol: 1e-5, atol: 1e-6).item(Bool.self))
        #expect(originalLogits[0, 1].allClose(changedLogits[0, 1], rtol: 1e-5, atol: 1e-6).item(Bool.self))
    }

    @Test func logitsPreserveBatchSequenceAndVocabularyAxes() {
        seed(24)
        let model = TinyDecoder(
            vocabularySize: 11,
            width: 8,
            headCount: 2,
            maximumSequenceLength: 5
        )
        let tokens = MLXArray([
            [Int32(0), 1, 2, 3, 4],
            [Int32(5), 6, 7, 8, 9],
        ])

        let logits = model(tokens)
        eval(logits)

        #expect(logits.shape == [2, 5, 11])
        #expect(isFinite(logits).all().item(Bool.self))
    }

    @Test func objectiveGradientsAndTrainingUpdateMatchDirectMLX() async throws {
        seed(23)
        let trainingModel = makeModel()
        let directModel = makeModel()
        directModel.update(parameters: trainingModel.parameters())

        let inputs = MLXArray([
            [Int32(0), 1, 2, 3],
            [Int32(2), 3, 4, 5],
        ])
        let targets = MLXArray([
            [Int32(1), 2, 3, 4],
            [Int32(3), 4, 5, 6],
        ])
        let initialParameters = trainingModel.trainableParameters().flattened().sorted { $0.0 < $1.0 }
        eval(initialParameters.map(\.1))
        let initialValues = Dictionary(
            uniqueKeysWithValues: initialParameters.map { ($0.0, $0.1.asArray(Float.self)) }
        )
        let objective = NextTokenObjective()
        let directLossAndGrad = valueAndGrad(model: directModel) {
            (model: TinyDecoder, tokens: MLXArray, labels: MLXArray) -> MLXArray in
            objective(output: model(tokens), target: labels)
        }
        let (directLoss, gradients) = directLossAndGrad(directModel, inputs, targets)
        eval(directLoss, gradients)

        #expect(directLoss.item(Float.self).isFinite)
        #expect(!initialParameters.isEmpty, "Model parameters: \(trainingModel.parameters().flattened().map(\.0))")
        let flattenedGradients = gradients.flattened()
        #expect(!flattenedGradients.isEmpty)
        for (_, gradient) in flattenedGradients {
            #expect(isFinite(gradient).all().item(Bool.self))
        }

        let directOptimizer = SGD(learningRate: 0.01)
        directOptimizer.update(model: directModel, gradients: gradients)
        eval(directModel, directOptimizer)

        let batches = try DataLoader(
            source: InMemoryDataset(features: inputs, labels: targets),
            batchSize: inputs.dim(0)
        )
        let engineOptimizer = SGD(learningRate: 0.01)
        let configuration = TrainingConfiguration(
            optimizer: engineOptimizer,
            objective: objective
        )
        let history = try await TrainingEngine().fit(
            trainingModel,
            configuration: configuration,
            data: batches,
            epochs: 1
        )
        let engineLoss = history["loss"].compactMap { $0 }.first
        #expect(engineLoss != nil)
        if let engineLoss {
            #expect(abs(engineLoss - directLoss.item(Float.self)) < 1e-5)
        }

        let actualParameters = trainingModel.trainableParameters().flattened().sorted { $0.0 < $1.0 }
        let expectedParameters = directModel.trainableParameters().flattened().sorted { $0.0 < $1.0 }
        #expect(actualParameters.map(\.0) == expectedParameters.map(\.0))
        for (actual, expected) in zip(actualParameters, expectedParameters) {
            #expect(actual.1.allClose(expected.1, rtol: 0, atol: 1e-5).all().item(Bool.self))
        }

        let changedParameterNames = actualParameters.compactMap { updated -> String? in
            guard let initial = initialValues[updated.0] else { return nil }
            let updatedValues = updated.1.asArray(Float.self)
            let maximumChange = zip(updatedValues, initial).map { abs($0 - $1) }.max() ?? 0
            return maximumChange > 1e-7 ? updated.0 : nil
        }
        #expect(changedParameterNames.contains { $0.contains("tokenEmbedding") }, "Changed parameters: \(changedParameterNames)")
        #expect(changedParameterNames.contains { $0.contains("block") }, "Changed parameters: \(changedParameterNames)")
    }

    private func makeModel() -> TinyDecoder {
        TinyDecoder(
            vocabularySize: 8,
            width: 8,
            headCount: 2,
            maximumSequenceLength: 4
        )
    }
}
#endif
