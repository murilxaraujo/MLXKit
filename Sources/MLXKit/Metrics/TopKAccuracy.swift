import MLX

/// Tracks top-k classification accuracy across batches.
///
/// A prediction is considered correct if the true class is among the top `k`
/// predicted classes (by logit/probability value).
///
/// ```swift
/// let topK = TopKAccuracy(k: 5)
/// topK.update(predictions: logits, targets: labels)
/// print(topK.compute()) // e.g. 0.95
/// ```
public final class TopKAccuracy: Metric, @unchecked Sendable {
    public let name: String
    public let k: Int

    private var correct: Int = 0
    private var total: Int = 0

    public init(k: Int = 5) {
        self.k = k
        self.name = "top\(k)_accuracy"
    }

    public func update(predictions: MLXArray, targets: MLXArray) {
        precondition(predictions.ndim >= 2, "TopKAccuracy requires logits with shape [batch, classes]")
        let numClasses = predictions.dim(-1)
        let effectiveK = min(k, numClasses)

        // Sort indices in descending order along last axis and take top-k
        let sorted = argSort(predictions, axis: -1)
        let topKIndices = sorted[0..., (numClasses - effectiveK)...]

        // Check if targets appear in top-k
        let expandedTargets = targets.expandedDimensions(axis: -1)
        let matches = (topKIndices .== expandedTargets).asType(.int32).sum(axis: -1)
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
