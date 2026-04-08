import Foundation

/// Decays the learning rate following a cosine curve from `initialLR` to `minLR`.
///
/// The learning rate at step `t` is:
/// ```
/// lr = minLR + 0.5 * (initialLR - minLR) * (1 + cos(π * t / totalSteps))
/// ```
///
/// After `totalSteps`, the learning rate remains at `minLR`.
///
/// ```swift
/// let schedule = CosineDecay(initialLR: 1e-3, totalSteps: 10000, minLR: 1e-6)
/// ```
public struct CosineDecay: LRSchedule {
    public let initialLR: Float
    public let totalSteps: Int
    public let minLR: Float

    public init(initialLR: Float, totalSteps: Int, minLR: Float = 0) {
        self.initialLR = initialLR
        self.totalSteps = totalSteps
        self.minLR = minLR
    }

    public func learningRate(atStep step: Int) -> Float {
        guard step < totalSteps else { return minLR }
        let progress = Float(step) / Float(totalSteps)
        return minLR + 0.5 * (initialLR - minLR) * (1 + cos(Float.pi * progress))
    }
}
