import MLX
import MLXOptimizers

/// Optional optimizer capability required to apply an `LRSchedule`.
///
/// MLXOptimizers' base `Optimizer` protocol does not expose a common
/// learning-rate property. Conformers make that capability explicit so the
/// training engine never needs to inspect concrete optimizer types.
public protocol LearningRateControllable: Optimizer {
    func setLearningRate(_ value: Float) throws
}

private func validateLearningRate(_ value: Float) throws {
    guard value.isFinite, value >= 0 else {
        throw TrainingError.invalidLearningRate(value)
    }
}

extension SGD: LearningRateControllable {
    public func setLearningRate(_ value: Float) throws {
        try validateLearningRate(value)
        learningRate = value
    }
}

extension RMSprop: LearningRateControllable {
    public func setLearningRate(_ value: Float) throws {
        try validateLearningRate(value)
        learningRate = value
    }
}

extension AdaGrad: LearningRateControllable {
    public func setLearningRate(_ value: Float) throws {
        try validateLearningRate(value)
        learningRate = value
    }
}

extension AdaDelta: LearningRateControllable {
    public func setLearningRate(_ value: Float) throws {
        try validateLearningRate(value)
        learningRate = value
    }
}

extension Adam: LearningRateControllable {
    public func setLearningRate(_ value: Float) throws {
        try validateLearningRate(value)
        learningRate = value
    }
}

extension Adamax: LearningRateControllable {
    public func setLearningRate(_ value: Float) throws {
        try validateLearningRate(value)
        learningRate = value
    }
}

extension Lion: LearningRateControllable {
    public func setLearningRate(_ value: Float) throws {
        try validateLearningRate(value)
        learningRate = value
    }
}

extension Adafactor: LearningRateControllable {
    public func setLearningRate(_ value: Float) throws {
        try validateLearningRate(value)
        // Adafactor defaults to internally computed relative rates. An
        // external schedule takes ownership of the rate when configured.
        relativeStep = false
        learningRate = value
    }
}

extension Muon: LearningRateControllable {
    public func setLearningRate(_ value: Float) throws {
        try validateLearningRate(value)
        learningRate = value
    }
}
