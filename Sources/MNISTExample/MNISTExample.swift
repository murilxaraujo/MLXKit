// MNIST-style end-to-end example exercising the full MLXKit API.
//
// Trains a simple classifier on synthetic MNIST-like data (28x28 flattened
// to 784 features, 10 classes) to demonstrate the complete workflow:
//   1. Define a model conforming to TrainableModel
//   2. Configure optimizer, objective, metrics, and LR schedule
//   3. Fit with callbacks (progress, early stopping, history)
//   4. Evaluate on validation data
//   5. Predict on new inputs
//   6. Print model summary
//   7. Save and load checkpoint

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import MLXKit

// MARK: - Model

final class MNISTModel: Module, TrainableModel {
    typealias Input = MLXArray
    typealias Output = MLXArray

    let linear1 = Linear(inputDimensions: 784, outputDimensions: 128)
    let linear2 = Linear(inputDimensions: 128, outputDimensions: 64)
    let linear3 = Linear(inputDimensions: 64, outputDimensions: 10)

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = linear1(input)
        x = maximum(x, 0) // ReLU
        x = linear2(x)
        x = maximum(x, 0) // ReLU
        return linear3(x)
    }
}

// MARK: - Synthetic Data

func makeSyntheticData(count: Int) -> InMemoryDataset {
    // Random features simulating flattened 28x28 images
    let features = MLXRandom.uniform(low: 0.0, high: 1.0, [count, 784])
    // Random class labels 0-9
    let labels = MLXRandom.randInt(low: 0, high: 10, [count])
    return InMemoryDataset(features: features, labels: labels)
}

// MARK: - Main

@main
struct MNISTExample {
    static func main() async throws {
        seed(42)
        print("MLXKit MNIST-style Example")
        print("==========================\n")

        // 1. Create model
        let model = MNISTModel()

        // 2. Print summary
        print("Model Summary:")
        model.printSummary()
        print()

        // 3. Configure training
        let engine = TrainingEngine()
        let config = TrainingConfiguration(
            optimizer: Adam(learningRate: 1e-3),
            objective: CrossEntropyLoss(),
            metrics: [Accuracy()],
            learningRateSchedule: CosineDecay(initialLR: 1e-3, totalSteps: 40),
            gradientProcessing: GradientProcessingConfiguration(
                transforms: [{ gradients in
                    // Demonstrate ordered tree-wide processing before clipping.
                    gradients.mapValues { $0 * 0.9 }
                }],
                globalNormClip: 1.0
            ),
            gradientAccumulationSteps: 2
        )

        // 4. Create data loaders
        let trainData = makeSyntheticData(count: 500)
        let valData = makeSyntheticData(count: 100)
        let trainLoader = try DataLoader(
            source: trainData,
            batchSize: 32,
            shuffle: .full(seed: 42)
        )
        let valLoader = try DataLoader(source: valData, batchSize: 32)

        // 5. Fit with callbacks
        print("Training...")
        let history = try await engine.fit(
            model, configuration: config, data: trainLoader,
            epochs: 5,
            validationData: valLoader,
            callbacks: [ProgressReporter()]
        )
        print()

        // 6. Print history
        print("Training History:")
        for (i, logs) in history.epochs.enumerated() {
            let loss = logs["loss"].map { String(format: "%.4f", $0) } ?? "N/A"
            let acc = logs["accuracy"].map { String(format: "%.4f", $0) } ?? "N/A"
            let valLoss = logs["val_loss"].map { String(format: "%.4f", $0) } ?? "N/A"
            let valAcc = logs["val_accuracy"].map { String(format: "%.4f", $0) } ?? "N/A"
            print("  Epoch \(i + 1): loss=\(loss), accuracy=\(acc), val_loss=\(valLoss), val_accuracy=\(valAcc)")
        }
        print()

        // 7. Evaluate
        let evalMetrics = try await engine.evaluate(model, configuration: config, data: valLoader)
        print("Evaluation: \(evalMetrics)")
        print()

        // 8. Predict
        let testInput = MLXRandom.uniform(low: 0.0, high: 1.0, [3, 784])
        let predictions = engine.predict(model, input: testInput)
        print("Predictions shape: \(predictions.shape)") // [3, 10]
        print()

        // 9. Save an inference artifact (weights plus caller-owned metadata).
        let checkpointDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxkit_example")
        let checkpointURL = checkpointDir.appendingPathComponent("model.safetensors")

        try Checkpointing.saveInferenceArtifact(
            model,
            metadata: ["architecture": "MNISTModel", "classCount": "10"],
            to: checkpointURL
        )
        print("Inference artifact saved to: \(checkpointURL.path(percentEncoded: false))")

        let newModel = MNISTModel()
        eval(newModel)
        let artifact = try Checkpointing.loadInferenceArtifact(into: newModel, from: checkpointURL)
        print("Inference artifact loaded: \(artifact.metadata)")

        // Cleanup
        try? FileManager.default.removeItem(at: checkpointDir)

        print("\nDone!")
    }
}
