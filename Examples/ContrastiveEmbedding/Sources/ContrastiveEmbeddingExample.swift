// A deterministic, local example of paired-view contrastive representation learning.

import Foundation
import ArgumentParser
import MLX
import MLXNN
import MLXOptimizers
import MLXKit

private let inputDimensions = 128
private let embeddingDimensions = 64
private let contrastiveTemperature: Float = 0.1
private let trainingBatchSize = 32

private enum EmbeddingExampleError: Error, LocalizedError {
    case nonFiniteValue(String, Float)
    case validationLossDidNotImprove(initial: Float, final: Float)
    case retrievalDidNotImprove(initial: Float, final: Float)
    case recallAtOrBelowChance(Float)
    case missingGradient
    case parameterUpdateMissing
    case embeddingsNotNormalized(Float)

    var errorDescription: String? {
        switch self {
        case .nonFiniteValue(let name, let value):
            "Expected finite \(name), received \(value)."
        case .validationLossDidNotImprove(let initial, let final):
            "Validation contrastive loss did not decrease: initial=\(initial), final=\(final)."
        case .retrievalDidNotImprove(let initial, let final):
            "Validation paired Recall@1 did not improve: initial=\(initial), final=\(final)."
        case .recallAtOrBelowChance(let recall):
            "Paired Recall@1 must exceed the 1/32 random-retrieval baseline; received \(recall)."
        case .missingGradient:
            "The contrastive objective did not produce finite, nonzero trainable gradients."
        case .parameterUpdateMissing:
            "Training completed without changing any trainable parameter."
        case .embeddingsNotNormalized(let deviation):
            "Embedding L2 norm differs from one by \(deviation), exceeding 1e-4."
        }
    }
}

struct RunOptions: ParsableArguments {
    @Flag(help: "Run the smaller deterministic fixture for CI and quick checks.")
    var smoke = false

    var epochs: Int { smoke ? 30 : 40 }
    var trainingPairs: Int { smoke ? 96 : 256 }
    var validationPairs: Int { smoke ? 64 : 128 }
}

// MARK: - Encoder

// The example calls the async engine methods sequentially and does not access
// the module while an awaited engine operation is mutating its parameters.
private final class EmbeddingEncoder: Module, TrainableModel, @unchecked Sendable {
    typealias Input = MLXArray
    typealias Output = MLXArray

    let hidden = Linear(inputDimensions: inputDimensions, outputDimensions: 128)
    let projection = Linear(inputDimensions: 128, outputDimensions: embeddingDimensions)

    /// Encodes independent feature vectors as unit-length embeddings.
    func encode(_ input: MLXArray) -> MLXArray {
        projection(maximum(hidden(input), 0)).l2Normalized(axis: -1)
    }

    /// Applies the same encoder to each view of each paired example.
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let pairCount = input.dim(0)
        let viewCount = input.dim(1)
        return encode(input.reshaped([-1, inputDimensions]))
            .reshaped([pairCount, viewCount, embeddingDimensions])
    }
}

// MARK: - Local paired-view fixture

/// A fixed-width generator makes both fixture creation and data splits
/// independent of MLX's global random state.
private struct FixtureRandom {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func unit() -> Float {
        state &+= 0x9E3779B97F4A7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        value ^= value >> 31
        return Float(value >> 40) / Float(1 << 24)
    }

    mutating func signed() -> Float {
        unit() * 2 - 1
    }
}

/// Creates two noisy views of a shared 64-dimensional identity signal. Each
/// view also has independent nuisance features in the remaining dimensions.
private func makePairedData(pairCount: Int, seed: UInt64, identityOffset: Int32) -> InMemoryDataset {
    var random = FixtureRandom(seed: seed)
    var features: [Float32] = []
    var identities: [Int32] = []
    features.reserveCapacity(pairCount * 2 * inputDimensions)
    identities.reserveCapacity(pairCount)

    for pair in 0..<pairCount {
        let identity = (0..<64).map { _ in random.signed() }
        for _ in 0..<2 {
            var view = [Float32](repeating: 0, count: inputDimensions)
            for dimension in 0..<64 {
                view[dimension] = identity[dimension] + random.signed() * 0.12
            }
            for dimension in 64..<inputDimensions {
                view[dimension] = random.signed() * 2.0
            }
            features.append(contentsOf: view)
        }
        identities.append(identityOffset + Int32(pair))
    }

    return InMemoryDataset(
        features: MLXArray(features).reshaped([pairCount, 2, inputDimensions]),
        labels: MLXArray(identities)
    )
}

// MARK: - Contrastive objective and retrieval metric

/// Symmetric NT-Xent over a batch of paired views.
///
/// Views are flattened in pair order, so each anchor's positive is the other
/// adjacent view. Its denominator includes every other view in the batch and
/// excludes only the anchor itself.
private func ntXentLoss(_ embeddings: MLXArray, temperature: Float) -> MLXArray {
    let pairCount = embeddings.dim(0)
    let anchorCount = pairCount * 2
    let vectors = embeddings.reshaped([anchorCount, embeddingDimensions])
    let similarity = matmul(vectors, vectors.transposed()) / temperature
    let logits = similarity - eye(anchorCount) * 1e9
    let positiveIndices = MLXArray((0..<anchorCount).map { Int32($0 ^ 1) })
        .reshaped([anchorCount, 1])
    let positiveLogits = takeAlong(logits, positiveIndices, axis: 1).squeezed(axis: 1)
    return (logits.logSumExp(axis: 1) - positiveLogits).mean()
}

private final class PairedRecallAt1: Metric, @unchecked Sendable {
    typealias Output = MLXArray
    typealias Target = MLXArray

    let name = "paired_recall_at_1"
    private var correct = 0
    private var total = 0

    func update(output: MLXArray, target: MLXArray) {
        let queries = output[0..., 0, 0...]
        let candidates = output[0..., 1, 0...]
        let nearest = matmul(queries, candidates.transposed()).argMax(axis: 1)
        let retrievedIdentities = takeAlong(target, nearest, axis: 0)
        correct += (retrievedIdentities .== target).sum().item(Int.self)
        total += target.size
    }

    func compute() -> Float {
        guard total > 0 else { return 0 }
        return Float(correct) / Float(total)
    }

    func reset() {
        correct = 0
        total = 0
    }
}

private struct NTXentObjective: Objective {
    let temperature: Float

    func callAsFunction(output: MLXArray, target: MLXArray) -> MLXArray {
        ntXentLoss(output, temperature: temperature)
    }
}

private func contrastiveTrainingStep(
    _ model: EmbeddingEncoder,
    _ batch: Batch<MLXArray, MLXArray>
) -> MLXArray {
    ntXentLoss(model(batch.input), temperature: contrastiveTemperature)
}

private func requireFinite(_ name: String, _ value: Float) throws {
    guard value.isFinite else {
        throw EmbeddingExampleError.nonFiniteValue(name, value)
    }
}

private func requireUnitNorms(_ embeddings: MLXArray) throws {
    let norms = sqrt((embeddings * embeddings).sum(axis: -1))
    let maximumDeviation = abs(norms - 1).max().item(Float.self)
    guard maximumDeviation.isFinite, maximumDeviation <= 1e-4 else {
        throw EmbeddingExampleError.embeddingsNotNormalized(maximumDeviation)
    }
}

// MARK: - Run

@main
struct ContrastiveEmbeddingExample: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Train and evaluate a deterministic paired-view contrastive encoder."
    )

    @OptionGroup var options: RunOptions

    mutating func run() async throws {
        seed(42)
        let gradientCheckModel = EmbeddingEncoder()
        seed(42)
        let model = EmbeddingEncoder()

        let trainingData = makePairedData(pairCount: options.trainingPairs, seed: 2026, identityOffset: 0)
        let validationData = makePairedData(
            pairCount: options.validationPairs,
            seed: 90210,
            identityOffset: Int32(options.trainingPairs + 1_000)
        )
        let trainingLoader = try DataLoader(
            source: trainingData,
            batchSize: trainingBatchSize,
            shuffle: .full(seed: 42)
        )
        let validationLoader = try DataLoader(source: validationData, batchSize: trainingBatchSize)

        let configuration = TrainingConfiguration(
            optimizer: Adam(learningRate: 0.003),
            objective: NTXentObjective(temperature: contrastiveTemperature),
            metrics: [PairedRecallAt1()]
        )
        let engine = TrainingEngine()

        let initialParameters = Dictionary(
            uniqueKeysWithValues: model.trainableParameters().flattened().map {
                ($0.0, $0.1.asArray(Float.self))
            }
        )

        // Exercise autodiff on a real batch before training. The training engine
        // repeats this objective and also checks gradients before optimizer updates.
        var iterator = trainingLoader.makeAsyncIterator()
        guard let firstBatch = try await iterator.next() else {
            throw TrainingError.emptyTrainingData
        }
        let lossAndGradient = valueAndGrad(model: gradientCheckModel) {
            (differentiableModel: EmbeddingEncoder, inputs: MLXArray, _: MLXArray) -> MLXArray in
            ntXentLoss(
                differentiableModel(inputs),
                temperature: contrastiveTemperature
            )
        }
        let (initialLossArray, initialGradients) = lossAndGradient(
            gradientCheckModel,
            firstBatch.input,
            firstBatch.target
        )
        eval(initialLossArray, initialGradients)
        let initialGradientValue = initialLossArray.item(Float.self)
        try requireFinite("initial contrastive loss", initialGradientValue)
        let gradientLeaves = initialGradients.flattened()
        guard !gradientLeaves.isEmpty else {
            throw EmbeddingExampleError.missingGradient
        }
        var foundNonzeroGradient = false
        for (name, gradient) in gradientLeaves {
            guard isFinite(gradient).all().item(Bool.self) else {
                throw EmbeddingExampleError.nonFiniteValue("gradient for \(name)", .nan)
            }
            foundNonzeroGradient = foundNonzeroGradient || abs(gradient).sum().item(Float.self) > 0
        }
        guard foundNonzeroGradient else {
            throw EmbeddingExampleError.missingGradient
        }

        let initialValidation = try await engine.evaluate(
            model,
            configuration: configuration,
            data: validationLoader,
            trainingStep: contrastiveTrainingStep
        )
        try Self.validateMetrics(initialValidation)

        print("Contrastive paired-view embedding example\(options.smoke ? " — smoke" : "")")
        print("input=[batch, 2, 128] embedding=[batch, 2, 64] temperature=\(contrastiveTemperature)")
        print("trainingPairs=\(options.trainingPairs) validationPairs=\(options.validationPairs) epochs=\(options.epochs)")
        print("Initial validation: loss=\(Self.formatted(initialValidation["loss"])), paired Recall@1=\(Self.formatted(initialValidation["paired_recall_at_1"]))")

        let history = try await engine.fit(
            model,
            configuration: configuration,
            data: trainingLoader,
            epochs: options.epochs,
            validationData: validationLoader,
            trainingStep: contrastiveTrainingStep,
            callbacks: [ProgressReporter()]
        )

        let finalValidation = try await engine.evaluate(
            model,
            configuration: configuration,
            data: validationLoader,
            trainingStep: contrastiveTrainingStep
        )
        try Self.validateMetrics(finalValidation)

        for epoch in history.epochs {
            for (name, value) in epoch {
                try requireFinite("epoch \(name)", value)
            }
        }
        guard let initialValidationLoss = initialValidation["loss"],
              let finalValidationLoss = finalValidation["loss"],
              finalValidationLoss < initialValidationLoss
        else {
            throw EmbeddingExampleError.validationLossDidNotImprove(
                initial: initialValidation["loss"] ?? .nan,
                final: finalValidation["loss"] ?? .nan
            )
        }
        let recall = finalValidation["paired_recall_at_1"] ?? .nan
        let initialRecall = initialValidation["paired_recall_at_1"] ?? .nan
        guard recall > initialRecall else {
            throw EmbeddingExampleError.retrievalDidNotImprove(
                initial: initialRecall,
                final: recall
            )
        }
        guard recall > 1 / Float(trainingBatchSize) else {
            throw EmbeddingExampleError.recallAtOrBelowChance(recall)
        }

        let updatedParameters = Dictionary(
            uniqueKeysWithValues: model.trainableParameters().flattened().map {
                ($0.0, $0.1.asArray(Float.self))
            }
        )
        let parameterChanged = initialParameters.contains { name, initial in
            guard let updated = updatedParameters[name], updated.count == initial.count else {
                return false
            }
            return zip(initial, updated).contains { abs($0 - $1) > 1e-7 }
        }
        guard parameterChanged else {
            throw EmbeddingExampleError.parameterUpdateMissing
        }

        let validationEmbeddings = model(validationData.features)
        eval(validationEmbeddings)
        guard isFinite(validationEmbeddings).all().item(Bool.self) else {
            throw EmbeddingExampleError.nonFiniteValue("validation embedding", .nan)
        }
        try requireUnitNorms(validationEmbeddings)

        let newInputs = makePairedData(pairCount: 4, seed: 424242, identityOffset: 90_000)
            .features[0..., 0, 0...]
        let newVectors = model.encode(newInputs)
        eval(newVectors)
        guard isFinite(newVectors).all().item(Bool.self) else {
            throw EmbeddingExampleError.nonFiniteValue("generated embedding", .nan)
        }
        try requireUnitNorms(newVectors)

        print("Final validation: loss=\(Self.formatted(finalValidationLoss)), paired Recall@1=\(Self.formatted(recall))")
        print("Autodiff: finite, nonzero gradients · optimizer: parameters updated")
        print("Embedding norms: unit length within 1e-4 · new vectors shape: \(newVectors.shape)")
        print("Completed \(history.epochs.count) epochs; random retrieval baseline for batch 32 is 0.03125.")
    }

    private static func validateMetrics(_ metrics: [String: Float]) throws {
        for (name, value) in metrics {
            try requireFinite(name, value)
        }
    }

    private static func formatted(_ value: Float?) -> String {
        guard let value else { return "missing" }
        return String(format: "%.4f", value)
    }
}
