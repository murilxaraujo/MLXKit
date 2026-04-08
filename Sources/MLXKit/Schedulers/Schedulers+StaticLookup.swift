extension LRSchedule where Self == CosineDecay {
    /// Cosine decay schedule.
    public static func cosineDecay(
        initialLR: Float,
        totalSteps: Int,
        minLR: Float = 0
    ) -> CosineDecay {
        CosineDecay(initialLR: initialLR, totalSteps: totalSteps, minLR: minLR)
    }
}

extension LRSchedule where Self == StepDecay {
    /// Step decay schedule.
    public static func stepDecay(
        initialLR: Float,
        stepSize: Int,
        gamma: Float = 0.1
    ) -> StepDecay {
        StepDecay(initialLR: initialLR, stepSize: stepSize, gamma: gamma)
    }
}

extension LRSchedule where Self == ExponentialDecay {
    /// Exponential decay schedule.
    public static func exponentialDecay(
        initialLR: Float,
        gamma: Float
    ) -> ExponentialDecay {
        ExponentialDecay(initialLR: initialLR, gamma: gamma)
    }
}
