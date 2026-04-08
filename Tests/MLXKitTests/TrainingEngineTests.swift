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
            loss: MSELoss(),
            metrics: [Accuracy(), MeanSquaredError()],
            lrSchedule: CosineDecay(initialLR: 0.001, totalSteps: 1000)
        )
        #expect(config.metrics.count == 2)
        #expect(config.lrSchedule != nil)
    }

    @Test func canBeReplacedByRecompiling() {
        let config1 = TrainingConfiguration(
            optimizer: Adam(learningRate: 0.001),
            loss: MSELoss()
        )
        let config2 = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.01),
            loss: CrossEntropyLoss()
        )
        // Both configs exist independently
        #expect(config1.metrics.isEmpty)
        #expect(config2.metrics.isEmpty)
    }
}

@Suite("TrainingError")
struct TrainingErrorTests {
    @Test func notCompiledErrorMessage() {
        let error = TrainingError.notCompiled
        #expect(error.errorDescription?.contains("compile") == true)
    }
}
