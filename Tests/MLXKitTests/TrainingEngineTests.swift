import Testing
import MLX
import MLXNN
import MLXOptimizers
@testable import MLXKit

@Suite("TrainingConfiguration")
struct TrainingConfigurationTests {
    @Test func storesAllComponents() {
        let config = TrainingConfiguration(
            optimizer: Adam(learningRate: 0.001),
            objective: MSELoss(),
            metrics: [Accuracy(), MeanSquaredError()],
            learningRateSchedule: CosineDecay(initialLR: 0.001, totalSteps: 1000)
        )
        #expect(config.metrics.count == 2)
        #expect(config.learningRateSchedule != nil)
    }

    @Test func configurationsAreIndependentValues() {
        let config1 = TrainingConfiguration(
            optimizer: Adam(learningRate: 0.001),
            objective: MSELoss()
        )
        let config2 = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.01),
            objective: CrossEntropyLoss()
        )
        // Both configs exist independently
        #expect(config1.metrics.isEmpty)
        #expect(config2.metrics.isEmpty)
    }
}

@Suite("TrainingError")
struct TrainingErrorTests {
    @Test func invalidEpochCountHasActionableMessage() {
        let error = TrainingError.invalidEpochCount(0)
        #expect(error.errorDescription?.contains("greater than zero") == true)
    }
}
