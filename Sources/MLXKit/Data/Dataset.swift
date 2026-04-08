import MLX

/// A collection of samples that can be randomly accessed by integer index.
///
/// Conforming types provide indexed access to individual samples, enabling
/// the ``DataLoader`` to batch, shuffle, and iterate over data efficiently.
///
/// ```swift
/// struct MyDataset: Dataset {
///     let features: MLXArray
///     let labels: MLXArray
///     var count: Int { features.dim(0) }
///     subscript(index: Int) -> Sample { ... }
/// }
/// ```
public protocol Dataset: RandomAccessCollection where Index == Int, Element == Sample {
    /// The number of samples in the dataset.
    var count: Int { get }

    /// All features as a single array with shape `[N, ...]`.
    var features: MLXArray { get }

    /// All labels as a single array with shape `[N, ...]`.
    var labels: MLXArray { get }
}

extension Dataset {
    public var startIndex: Int { 0 }
    public var endIndex: Int { count }
}
