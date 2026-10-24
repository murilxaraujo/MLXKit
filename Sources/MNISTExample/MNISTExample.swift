// A deterministic, local supervised-classification example.
//
// The fixture has MNIST-like tensor dimensions, but it is generated synthetic
// data. It is deliberately learnable and does not download or claim to use MNIST.

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXKit

private enum ClassifierExampleError: Error, LocalizedError {
    case invalidArguments(String)
    case nonFiniteMetric(String, Float)
    case validationDidNotImprove(initial: Float, final: Float)
    case validationAccuracyTooLow(Float)
    case artifactMetadataMismatch
    case artifactPredictionsChanged

    var errorDescription: String? {
        switch self {
        case .invalidArguments(let message): message
        case .nonFiniteMetric(let name, let value):
            "Expected a finite \(name), received \(value)."
        case .validationDidNotImprove(let initial, let final):
            "Validation loss did not improve: initial=\(initial), final=\(final)."
        case .validationAccuracyTooLow(let accuracy):
            "Validation accuracy did not exceed the fixture's 0.50 smoke floor: \(accuracy)."
        case .artifactMetadataMismatch:
            "The inference artifact did not preserve its caller-owned metadata."
        case .artifactPredictionsChanged:
            "Predictions changed after reloading the inference artifact (rtol=1e-5, atol=1e-5)."
        }
    }
}

private struct RunOptions {
    let epochs: Int
    let samplesPerClass: Int
    let validationSamplesPerClass: Int
    let smoke: Bool

    static func parse(_ arguments: [String]) throws -> RunOptions {
        guard arguments.allSatisfy({ $0 == "--smoke" || $0 == "--help" }) else {
            let invalid = arguments.first { $0 != "--smoke" && $0 != "--help" }!
            throw ClassifierExampleError.invalidArguments(
                "Unknown argument '\(invalid)'. Run with --help for usage."
            )
        }

        if arguments.contains("--help") {
            print("Usage: swift run MNISTExample [--smoke] [--help]")
            print("  --smoke  Run the smaller deterministic fixture for CI and quick checks.")
            exit(EXIT_SUCCESS)
        }

        let smoke = arguments.contains("--smoke")
        return RunOptions(
            epochs: smoke ? 8 : 12,
            samplesPerClass: smoke ? 24 : 40,
            validationSamplesPerClass: smoke ? 8 : 12,
            smoke: smoke
        )
    }
}

// MARK: - Model

private final class Classifier: Module, TrainableModel {
    typealias Input = MLXArray
    typealias Output = MLXArray

    static let inputDimensions = 784
    static let classCount = 10

    let hidden = Linear(inputDimensions: 784, outputDimensions: 32)
    let output = Linear(inputDimensions: 32, outputDimensions: 10)

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        output(maximum(hidden(input), 0))
    }
}

// MARK: - Deterministic local fixture

/// A tiny fixed-seed generator keeps fixture creation independent of MLX's
/// global random state, which is reserved for model initialization.
private struct FixtureRandom {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func nextUnitFloat() -> Float {
        state &+= 0x9E3779B97F4A7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        value ^= value >> 31
        return Float(value >> 40) / Float(1 << 24)
    }
}

/// Builds balanced synthetic vectors with class-specific feature signatures.
/// Each class owns eight strong coordinates; small deterministic distractors
/// make the fixture less trivial while keeping its signal easy to learn.
private func makeSyntheticData(samplesPerClass: Int, seed: UInt64) -> InMemoryDataset {
    let classCount = Classifier.classCount
    var random = FixtureRandom(seed: seed)
    var features: [Float32] = []
    var labels: [Int32] = []
    features.reserveCapacity(samplesPerClass * classCount * Classifier.inputDimensions)
    labels.reserveCapacity(samplesPerClass * classCount)

    for label in 0..<classCount {
        for _ in 0..<samplesPerClass {
            var sample = [Float32](repeating: 0, count: Classifier.inputDimensions)

            for offset in 0..<8 {
                sample[label * 8 + offset] = 0.7 + random.nextUnitFloat() * 0.3
            }
            for _ in 0..<4 {
                let distractor = 80 + Int(
                    random.nextUnitFloat() * Float(Classifier.inputDimensions - 80)
                )
                sample[distractor] = random.nextUnitFloat() * 0.04
            }

            features.append(contentsOf: sample)
            labels.append(Int32(label))
        }
    }

    return InMemoryDataset(
        features: MLXArray(features).reshaped([labels.count, Classifier.inputDimensions]),
        labels: MLXArray(labels)
    )
}

private func requireFinite(_ metrics: [String: Float]) throws {
    for (name, value) in metrics where !value.isFinite {
        throw ClassifierExampleError.nonFiniteMetric(name, value)
    }
}

// MARK: - Run

@main
struct MNISTExample {
    static func main() async throws {
        let options = try RunOptions.parse(Array(CommandLine.arguments.dropFirst()))
        seed(42)

        let mode = options.smoke ? "Smoke" : "Instructional"
        print("MLXKit Supervised Classifier — \(mode) run")
        print("Synthetic 784-feature fixture · 10 balanced classes · seed 42\n")

        let model = Classifier()
        print("Model Summary:")
        model.printSummary()
        print()

        let trainData = makeSyntheticData(samplesPerClass: options.samplesPerClass, seed: 2026)
        let validationData = makeSyntheticData(
            samplesPerClass: options.validationSamplesPerClass,
            seed: 90210
        )
        let trainLoader = try DataLoader(
            source: trainData,
            batchSize: 32,
            shuffle: .full(seed: 42)
        )
        let validationLoader = try DataLoader(source: validationData, batchSize: 32)

        let configuration = TrainingConfiguration(
            optimizer: Adam(learningRate: 0.01),
            objective: CrossEntropyLoss(),
            metrics: [Accuracy()]
        )
        let engine = TrainingEngine()
        let initialValidation = try await engine.evaluate(
            model,
            configuration: configuration,
            data: validationLoader
        )
        try requireFinite(initialValidation)

        print("Training for \(options.epochs) epochs...")
        let history = try await engine.fit(
            model,
            configuration: configuration,
            data: trainLoader,
            epochs: options.epochs,
            validationData: validationLoader,
            callbacks: [ProgressReporter()]
        )
        print()

        let finalValidation = try await engine.evaluate(
            model,
            configuration: configuration,
            data: validationLoader
        )
        try requireFinite(finalValidation)

        guard let initialLoss = initialValidation["loss"],
              let finalLoss = finalValidation["loss"],
              finalLoss < initialLoss
        else {
            throw ClassifierExampleError.validationDidNotImprove(
                initial: initialValidation["loss"] ?? .nan,
                final: finalValidation["loss"] ?? .nan
            )
        }
        let finalAccuracy = finalValidation["accuracy"] ?? .nan
        guard finalAccuracy > 0.5 else {
            throw ClassifierExampleError.validationAccuracyTooLow(finalAccuracy)
        }

        print(String(format: "Validation: loss %.4f → %.4f, accuracy %.1f%%",
                     initialLoss, finalLoss, finalAccuracy * 100))
        print("Recorded \(history.epochs.count) epochs; all epoch metrics are finite: \(try historyAreFinite(history))")

        let exampleInputs = validationData.features[0..<min(8, validationData.count)]
        let predictions = engine.predict(model, input: exampleInputs)
        eval(predictions)
        guard isFinite(predictions).all().item(Bool.self) else {
            throw ClassifierExampleError.nonFiniteMetric("prediction", .nan)
        }
        print("Prediction logits shape: \(predictions.shape)")

        try verifyInferenceArtifact(
            model: model,
            inputs: exampleInputs,
            expectedPredictions: predictions
        )
        print("Inference artifact round trip: predictions match (rtol=1e-5, atol=1e-5)")
        print("\nDone!")
    }

    private static func historyAreFinite(_ history: History) throws -> Bool {
        for epoch in history.epochs {
            try requireFinite(epoch)
        }
        return true
    }

    private static func verifyInferenceArtifact(
        model: Classifier,
        inputs: MLXArray,
        expectedPredictions: MLXArray
    ) throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxkit-classifier-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let artifactURL = directory.appendingPathComponent("classifier.safetensors")

        let metadata = [
            "architecture": "MLXKitSyntheticClassifier-v1",
            "inputDimensions": String(Classifier.inputDimensions),
            "classCount": String(Classifier.classCount),
            "labelMapping": (0..<Classifier.classCount).map(String.init).joined(separator: ","),
            "fixture": "balanced-class-signatures-seed-2026",
        ]
        try Checkpointing.saveInferenceArtifact(model, metadata: metadata, to: artifactURL)

        let restoredModel = Classifier()
        eval(restoredModel)
        let artifact = try Checkpointing.loadInferenceArtifact(into: restoredModel, from: artifactURL)
        guard artifact.metadata == metadata else {
            throw ClassifierExampleError.artifactMetadataMismatch
        }

        let actualPredictions = restoredModel(inputs)
        eval(actualPredictions)
        guard actualPredictions.allClose(expectedPredictions, rtol: 1e-5, atol: 1e-5)
            .all().item(Bool.self)
        else {
            throw ClassifierExampleError.artifactPredictionsChanged
        }
    }
}
