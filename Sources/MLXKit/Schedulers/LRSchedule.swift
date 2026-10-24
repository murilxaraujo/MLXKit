import Foundation

/// A learning rate schedule that maps a training step to a learning rate.
///
/// Conform to this protocol to create custom schedules. The training engine
/// calls ``learningRate(atStep:)`` before each optimizer update. The configured
/// optimizer must conform to ``LearningRateControllable``; unsupported
/// optimizer/schedule combinations fail before training begins.
///
/// ```swift
/// let schedule = CosineDecay(initialLR: 1e-3, totalSteps: 10000)
/// let lr = schedule.learningRate(atStep: 500)
/// ```
public protocol LRSchedule: Sendable {
    /// Returns the learning rate for the given training step.
    ///
    /// - Parameter step: The current training step (zero-indexed).
    /// - Returns: The learning rate to use at this step.
    func learningRate(atStep step: Int) -> Float
}
