import MLX

/// Computes a differentiable scalar objective from the complete model output
/// and a typed target.
public protocol Objective<Output, Target> {
    associatedtype Output
    associatedtype Target

    func callAsFunction(output: Output, target: Target) -> MLXArray
}

/// An objective backed by a closure.
public struct ClosureObjective<ModelOutput, TargetValue>: Objective {
    public typealias Output = ModelOutput
    public typealias Target = TargetValue

    private let body: (ModelOutput, TargetValue) -> MLXArray

    public init(_ body: @escaping (ModelOutput, TargetValue) -> MLXArray) {
        self.body = body
    }

    public func callAsFunction(output: ModelOutput, target: TargetValue) -> MLXArray {
        body(output, target)
    }
}
