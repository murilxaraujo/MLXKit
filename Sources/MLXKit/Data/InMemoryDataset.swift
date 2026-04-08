@preconcurrency import MLX

/// An array-backed dataset that holds all features and labels in memory.
///
/// Both arrays must have the same leading dimension (number of samples).
///
/// ```swift
/// let dataset = InMemoryDataset(
///     features: MLXArray(randomNormal: [1000, 784]),
///     labels: MLXArray(randomNormal: [1000])
/// )
/// print(dataset.count) // 1000
/// let sample = dataset[0] // Sample with features [784], labels []
/// ```
public struct InMemoryDataset: Dataset, @unchecked Sendable {
    /// All features, with shape `[N, ...]`.
    public let features: MLXArray
    /// All labels, with shape `[N, ...]`.
    public let labels: MLXArray

    public var count: Int { features.dim(0) }

    public init(features: MLXArray, labels: MLXArray) {
        precondition(
            features.dim(0) == labels.dim(0),
            "Features and labels must have the same number of samples"
        )
        self.features = features
        self.labels = labels
    }

    public subscript(index: Int) -> Sample {
        Sample(features: features[index], labels: labels[index])
    }
}
