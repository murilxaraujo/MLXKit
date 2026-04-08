import Testing
@testable import MLXKit

@Suite("CosineDecay")
struct CosineDecayTests {
    @Test func step0ReturnsInitialLR() {
        let schedule = CosineDecay(initialLR: 0.001, totalSteps: 1000)
        let lr = schedule.learningRate(atStep: 0)
        #expect(abs(lr - 0.001) < 1e-6)
    }

    @Test func stepTotalReturnsMinLR() {
        let schedule = CosineDecay(initialLR: 0.001, totalSteps: 1000, minLR: 1e-6)
        let lr = schedule.learningRate(atStep: 1000)
        #expect(abs(lr - 1e-6) < 1e-7)
    }

    @Test func midpointValue() {
        let schedule = CosineDecay(initialLR: 0.001, totalSteps: 1000, minLR: 0)
        let lr = schedule.learningRate(atStep: 500)
        // cos(π * 0.5) = 0, so lr = 0 + 0.5 * 0.001 * (1 + 0) = 0.0005
        #expect(abs(lr - 0.0005) < 1e-6)
    }

    @Test func beyondTotalStepsReturnsMinLR() {
        let schedule = CosineDecay(initialLR: 0.01, totalSteps: 100, minLR: 0.001)
        let lr = schedule.learningRate(atStep: 200)
        #expect(abs(lr - 0.001) < 1e-6)
    }
}

@Suite("StepDecay")
struct StepDecayTests {
    @Test func noDecayBeforeFirstStep() {
        let schedule = StepDecay(initialLR: 0.1, stepSize: 30, gamma: 0.1)
        let lr = schedule.learningRate(atStep: 0)
        #expect(abs(lr - 0.1) < 1e-6)
    }

    @Test func dropsAtBoundary() {
        let schedule = StepDecay(initialLR: 0.1, stepSize: 30, gamma: 0.1)

        let lr29 = schedule.learningRate(atStep: 29)
        #expect(abs(lr29 - 0.1) < 1e-6)

        let lr30 = schedule.learningRate(atStep: 30)
        #expect(abs(lr30 - 0.01) < 1e-6)

        let lr60 = schedule.learningRate(atStep: 60)
        #expect(abs(lr60 - 0.001) < 1e-6)
    }

    @Test func customGamma() {
        let schedule = StepDecay(initialLR: 1.0, stepSize: 10, gamma: 0.5)
        let lr = schedule.learningRate(atStep: 20)
        // 1.0 * 0.5^2 = 0.25
        #expect(abs(lr - 0.25) < 1e-6)
    }
}

@Suite("ExponentialDecay")
struct ExponentialDecayTests {
    @Test func step0ReturnsInitialLR() {
        let schedule = ExponentialDecay(initialLR: 0.01, gamma: 0.999)
        let lr = schedule.learningRate(atStep: 0)
        #expect(abs(lr - 0.01) < 1e-6)
    }

    @Test func formulaVerified() {
        let schedule = ExponentialDecay(initialLR: 0.01, gamma: 0.5)
        let lr = schedule.learningRate(atStep: 3)
        // 0.01 * 0.5^3 = 0.00125
        #expect(abs(lr - 0.00125) < 1e-6)
    }
}

@Suite("WarmupSchedule")
struct WarmupScheduleTests {
    @Test func step0IsZero() {
        let inner = CosineDecay(initialLR: 0.001, totalSteps: 1000)
        let schedule = WarmupSchedule(inner: inner, warmupSteps: 100)
        let lr = schedule.learningRate(atStep: 0)
        #expect(abs(lr) < 1e-8)
    }

    @Test func linearRampDuringWarmup() {
        let inner = CosineDecay(initialLR: 0.001, totalSteps: 1000)
        let schedule = WarmupSchedule(inner: inner, warmupSteps: 100)

        let lr50 = schedule.learningRate(atStep: 50)
        // warmupFactor = 50/100 = 0.5
        // inner at step 50 ≈ initialLR (very close since 50/1000 is small)
        // lr ≈ 0.5 * inner(50)
        let innerLR50 = inner.learningRate(atStep: 50)
        #expect(abs(lr50 - 0.5 * innerLR50) < 1e-6)
    }

    @Test func innerScheduleTakesOverAfterWarmup() {
        let inner = CosineDecay(initialLR: 0.001, totalSteps: 1000)
        let schedule = WarmupSchedule(inner: inner, warmupSteps: 100)

        let lr100 = schedule.learningRate(atStep: 100)
        let innerLR100 = inner.learningRate(atStep: 100)
        #expect(abs(lr100 - innerLR100) < 1e-6)

        let lr500 = schedule.learningRate(atStep: 500)
        let innerLR500 = inner.learningRate(atStep: 500)
        #expect(abs(lr500 - innerLR500) < 1e-6)
    }
}

@Suite("Scheduler Static Lookup")
struct SchedulerStaticLookupTests {
    @Test func cosineDecayLookup() {
        let schedule: CosineDecay = .cosineDecay(initialLR: 0.001, totalSteps: 1000)
        #expect(schedule.initialLR == 0.001)
        #expect(schedule.totalSteps == 1000)
    }

    @Test func stepDecayLookup() {
        let schedule: StepDecay = .stepDecay(initialLR: 0.1, stepSize: 30)
        #expect(schedule.initialLR == 0.1)
        #expect(schedule.stepSize == 30)
    }

    @Test func exponentialDecayLookup() {
        let schedule: ExponentialDecay = .exponentialDecay(initialLR: 0.01, gamma: 0.999)
        #expect(schedule.initialLR == 0.01)
        #expect(schedule.gamma == 0.999)
    }
}
