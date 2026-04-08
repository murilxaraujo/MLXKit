import MLX

/// Tracks classification accuracy across batches.
///
/// For multi-class predictions (logits or probabilities), the predicted class
/// is the argmax along the last axis. For pre-computed class indices, predictions
/// are compared directly.
///
/// ```swift
/// let accuracy = Accuracy()
/// accuracy.update(predictions: logits, targets: labels)
/// print(accuracy.compute()) // e.g. 0.85
/// ```
public final class Accuracy: Metric, @unchecked Sendable {
    public let name = "accuracy"

    private var correct: Int = 0
    private var total: Int = 0

    public init() {}

    public func update(predictions: MLXArray, targets: MLXArray) {
        let predicted: MLXArray
        if predictions.ndim > 1 {
            predicted = predictions.argMax(axis: -1)
        } else {
            predicted = predictions
        }
        let matches = predicted .== targets
        correct += matches.sum().item(Int.self)
        total += targets.size
    }

    public func compute() -> Float {
        guard total > 0 else { return 0 }
        return Float(correct) / Float(total)
    }

    public func reset() {
        correct = 0
        total = 0
    }
}
