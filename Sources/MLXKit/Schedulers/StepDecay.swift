import Foundation

/// Multiplies the learning rate by a factor at fixed step intervals.
///
/// The learning rate at step `t` is:
/// ```
/// lr = initialLR * gamma ^ floor(t / stepSize)
/// ```
///
/// ```swift
/// let schedule = StepDecay(initialLR: 0.1, stepSize: 30, gamma: 0.1)
/// // Step 0-29: 0.1, Step 30-59: 0.01, Step 60-89: 0.001, ...
/// ```
public struct StepDecay: LRSchedule {
    public let initialLR: Float
    public let stepSize: Int
    public let gamma: Float

    public init(initialLR: Float, stepSize: Int, gamma: Float = 0.1) {
        self.initialLR = initialLR
        self.stepSize = stepSize
        self.gamma = gamma
    }

    public func learningRate(atStep step: Int) -> Float {
        let numDecays = step / stepSize
        return initialLR * pow(gamma, Float(numDecays))
    }
}
