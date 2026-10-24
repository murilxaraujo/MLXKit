import MLX
import MLXNN

/// A `Module` that can be trained with ``TrainingEngine``.
///
/// Conform to this protocol by declaring the input and output types and
/// implementing ``callAsFunction(_:)``. The training engine uses this method
/// for training and inference.
///
/// ```swift
/// class MyModel: Module, TrainableModel {
///     let linear1 = Linear(inputDimensions: 784, outputDimensions: 128)
///     let linear2 = Linear(inputDimensions: 128, outputDimensions: 10)
///
///     func callAsFunction(_ input: MLXArray) -> MLXArray {
///         var x = linear1(input)
///         x = MLX.maximum(x, 0) // ReLU
///         return linear2(x)
///     }
/// }
/// ```
public protocol TrainableModel: Module {
    /// The type accepted by the model.
    associatedtype Input

    /// The complete value returned by the model.
    associatedtype Output

    /// Performs the forward pass of the model.
    func callAsFunction(_ input: Input) -> Output
}
