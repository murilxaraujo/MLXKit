@preconcurrency import MLX

/// A single logical example with typed model input and target values.
public struct Sample<Input, Target> {
    public let input: Input
    public let target: Target

    public init(input: Input, target: Target) {
        self.input = input
        self.target = target
    }
}

/// A batch ready for one training or evaluation update.
public struct Batch<Input, Target> {
    public let input: Input
    public let target: Target
    public let sampleCount: Int

    public init(input: Input, target: Target, sampleCount: Int) {
        self.input = input
        self.target = target
        self.sampleCount = sampleCount
    }
}

// The array-backed 1.x data pipeline remains available while M4 replaces it.
// These conveniences keep its existing Dataset/DataLoader implementation
// source-compatible internally without restricting the 2.0 batch contract.
extension Sample where Input == MLXArray, Target == MLXArray {
    public init(features: MLXArray, labels: MLXArray) {
        self.init(input: features, target: labels)
    }

    public var features: MLXArray { input }
    public var labels: MLXArray { target }
}

extension Batch where Input == MLXArray, Target == MLXArray {
    public init(features: MLXArray, labels: MLXArray, size: Int) {
        self.init(input: features, target: labels, sampleCount: size)
    }

    public var features: MLXArray { input }
    public var labels: MLXArray { target }
    public var size: Int { sampleCount }
}
