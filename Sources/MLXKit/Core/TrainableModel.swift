import MLX
import MLXNN

/// A `Module` that can be trained with ``TrainingEngine``.
///
/// Conform to this protocol by implementing ``forward(_:)`` on your
/// `Module` subclass. The training engine uses this method for both
/// gradient computation and inference.
///
/// ```swift
/// class MyModel: Module, TrainableModel {
///     let linear1 = Linear(inputDimensions: 784, outputDimensions: 128)
///     let linear2 = Linear(inputDimensions: 128, outputDimensions: 10)
///
///     func forward(_ input: MLXArray) -> MLXArray {
///         var x = linear1(input)
///         x = MLX.maximum(x, 0) // ReLU
///         return linear2(x)
///     }
/// }
/// ```
public protocol TrainableModel: Module {
    /// Performs the forward pass of the model.
    ///
    /// - Parameter input: The input tensor.
    /// - Returns: The model's output (logits, probabilities, or predictions).
    func forward(_ input: MLXArray) -> MLXArray
}
