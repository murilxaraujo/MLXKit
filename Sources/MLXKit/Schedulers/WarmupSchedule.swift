import Foundation

/// Composes a linear warmup ramp with an inner schedule.
///
/// During the first `warmupSteps`, the learning rate increases linearly from
/// 0 to the inner schedule's value at `warmupSteps`. After warmup, the inner
/// schedule takes over.
///
/// ```swift
/// let inner = CosineDecay(initialLR: 1e-3, totalSteps: 10000)
/// let schedule = WarmupSchedule(inner: inner, warmupSteps: 500)
/// ```
public struct WarmupSchedule<Inner: LRSchedule>: LRSchedule {
    public let inner: Inner
    public let warmupSteps: Int

    public init(inner: Inner, warmupSteps: Int) {
        self.inner = inner
        self.warmupSteps = warmupSteps
    }

    public func learningRate(atStep step: Int) -> Float {
        let innerLR = inner.learningRate(atStep: step)
        guard step < warmupSteps else { return innerLR }
        let warmupFactor = Float(step) / Float(warmupSteps)
        return innerLR * warmupFactor
    }
}
