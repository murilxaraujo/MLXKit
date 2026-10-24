import MLX
import MLXNN

/// Computes the cross-entropy loss between logits and integer class targets.
///
/// Wraps the MLXNN cross-entropy operation with the ``LossFunction`` protocol.
///
/// ```swift
/// let loss = CrossEntropyLoss()
/// let value = loss(predictions: logits, targets: labels)
/// ```
///
/// - Parameters:
///   - labelSmoothing: Smoothing factor in `[0, 1)`. When greater than zero,
///     the target distribution is mixed with a uniform distribution.
///   - reduction: How to aggregate per-sample losses (default: `.mean`).
public struct CrossEntropyLoss: LossFunction {
    public let reduction: Reduction
    /// Label smoothing factor in `[0, 1)`.
    public let labelSmoothing: Float

    public init(labelSmoothing: Float = 0, reduction: Reduction = .mean) {
        self.labelSmoothing = labelSmoothing
        self.reduction = reduction
    }

    public func callAsFunction(predictions: MLXArray, targets: MLXArray) -> MLXArray {
        let unreduced = MLXNN.crossEntropy(
            logits: predictions,
            targets: targets,
            labelSmoothing: labelSmoothing,
            reduction: .none
        )
        return reduction.apply(to: unreduced)
    }

    /// Computes integer-target cross entropy with optional sample and class
    /// weights, using the actual batch size as the denominator.
    ///
    /// This operation supports logits shaped `[batch, classes]` and targets
    /// shaped `[batch]`. Supplied sample weights must have shape `[batch]` and
    /// class weights must have shape `[classes]`. Weights must be finite and
    /// nonnegative. Invalid shapes or values produce a non-finite scalar
    /// objective, which ``TrainingEngine`` rejects before an optimizer update.
    ///
    /// The weighted operation always returns a scalar mean, independent of
    /// this loss value's configured ``reduction``. Class weights multiply the
    /// per-example loss after label smoothing.
    ///
    /// ```swift
    /// let loss = CrossEntropyLoss(labelSmoothing: 0.1)
    /// let value = loss.weightedMean(
    ///     predictions: logits,
    ///     targets: labels,
    ///     sampleWeights: exampleWeights,
    ///     classWeights: classWeights
    /// )
    /// ```
    public func weightedMean(
        predictions: MLXArray,
        targets: MLXArray,
        sampleWeights: MLXArray? = nil,
        classWeights: MLXArray? = nil
    ) -> MLXArray {
        guard predictions.ndim == 2,
              targets.ndim == 1,
              predictions.shape[0] == targets.shape[0],
              predictions.shape[0] > 0,
              (sampleWeights == nil || sampleWeights?.shape == [targets.shape[0]]),
              (classWeights == nil || classWeights?.shape == [predictions.shape[1]]),
              sampleWeights.map(Self.hasValidWeights) ?? true,
              classWeights.map(Self.hasValidWeights) ?? true
        else {
            return MLXArray(Float.nan)
        }

        let unreduced = MLXNN.crossEntropy(
            logits: predictions,
            targets: targets,
            labelSmoothing: labelSmoothing,
            reduction: .none
        )

        var weights = MLXArray.ones([targets.shape[0]], dtype: unreduced.dtype)
        if let sampleWeights {
            weights = weights * sampleWeights.asType(unreduced.dtype)
        }
        if let classWeights {
            let targetClassWeights = classWeights.take(targets, axis: 0)
            weights = weights * targetClassWeights.asType(unreduced.dtype)
        }

        return (unreduced * weights).sum() / Float(targets.shape[0])
    }

    private static func hasValidWeights(_ weights: MLXArray) -> Bool {
        (isFinite(weights).all() & (weights .>= 0).all()).item(Bool.self)
    }
}
