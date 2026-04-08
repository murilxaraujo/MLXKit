import Foundation

/// Decays the learning rate exponentially every step.
///
/// The learning rate at step `t` is:
/// ```
/// lr = initialLR * gamma ^ t
/// ```
///
/// ```swift
/// let schedule = ExponentialDecay(initialLR: 0.01, gamma: 0.999)
/// ```
public struct ExponentialDecay: LRSchedule {
    public let initialLR: Float
    public let gamma: Float

    public init(initialLR: Float, gamma: Float) {
        self.initialLR = initialLR
        self.gamma = gamma
    }

    public func learningRate(atStep step: Int) -> Float {
        initialLR * pow(gamma, Float(step))
    }
}
