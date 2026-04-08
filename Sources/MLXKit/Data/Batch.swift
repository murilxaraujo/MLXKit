@preconcurrency import MLX

/// A single data point consisting of input features and a target label.
public struct Sample: @unchecked Sendable {
    /// The input features for this sample.
    public let features: MLXArray
    /// The target label for this sample.
    public let labels: MLXArray

    public init(features: MLXArray, labels: MLXArray) {
        self.features = features
        self.labels = labels
    }
}

/// A batched collection of samples, ready for model consumption.
public struct Batch: @unchecked Sendable {
    /// The batched input features with shape `[batchSize, ...]`.
    public let features: MLXArray
    /// The batched target labels with shape `[batchSize, ...]`.
    public let labels: MLXArray
    /// The number of samples in this batch.
    public let size: Int

    public init(features: MLXArray, labels: MLXArray, size: Int) {
        self.features = features
        self.labels = labels
        self.size = size
    }
}
