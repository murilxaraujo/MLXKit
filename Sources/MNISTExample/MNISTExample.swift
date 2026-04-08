// MNIST-style end-to-end example exercising the full MLXKit API.
//
// Trains a simple classifier on synthetic MNIST-like data (28x28 flattened
// to 784 features, 10 classes) to demonstrate the complete workflow:
//   1. Define a model conforming to TrainableModel
//   2. Compile with optimizer, loss, metrics, and LR schedule
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
    let linear1 = Linear(inputDimensions: 784, outputDimensions: 128)
    let linear2 = Linear(inputDimensions: 128, outputDimensions: 64)
    let linear3 = Linear(inputDimensions: 64, outputDimensions: 10)

    func forward(_ input: MLXArray) -> MLXArray {
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
        print("MLXKit MNIST-style Example")
        print("==========================\n")

        // 1. Create model
        let model = MNISTModel()

        // 2. Print summary
        print("Model Summary:")
        model.printSummary()
        print()

        // 3. Compile
        let engine = TrainingEngine()
        let config = engine.compile(
            model: model,
            optimizer: Adam(learningRate: 1e-3),
            loss: CrossEntropyLoss(),
            metrics: [Accuracy()],
            lrSchedule: CosineDecay(initialLR: 1e-3, totalSteps: 500)
        )

        // 4. Create data loaders
        let trainData = makeSyntheticData(count: 500)
        let valData = makeSyntheticData(count: 100)
        let trainLoader = DataLoader(dataset: trainData, batchSize: 32, shuffle: true)
        let valLoader = DataLoader(dataset: valData, batchSize: 32)

        // 5. Fit with callbacks
        print("Training...")
        let history = try await engine.fit(
            model, config: config, data: trainLoader,
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
        let evalMetrics = try await engine.evaluate(model, config: config, data: valLoader)
        print("Evaluation: \(evalMetrics)")
        print()

        // 8. Predict
        let testInput = MLXRandom.uniform(low: 0.0, high: 1.0, [3, 784])
        let predictions = engine.predict(model, input: testInput)
        print("Predictions shape: \(predictions.shape)") // [3, 10]
        print()

        // 9. Checkpoint
        let checkpointDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxkit_example")
        let checkpointURL = checkpointDir.appendingPathComponent("model.safetensors")

        try Checkpointing.save(model: model, epoch: 5, step: 500, to: checkpointURL)
        print("Checkpoint saved to: \(checkpointURL.path(percentEncoded: false))")

        let newModel = MNISTModel()
        eval(newModel)
        let info = try Checkpointing.load(into: newModel, from: checkpointURL)
        print("Checkpoint loaded: epoch=\(info.epoch), step=\(info.step)")

        // Cleanup
        try? FileManager.default.removeItem(at: checkpointDir)

        print("\nDone!")
    }
}
