// GPU-dependent tests — run via Xcode (xcodebuild test), not `swift test`.

#if MLX_GPU_TESTS
import Testing
import MLX
import MLXNN
@testable import MLXKit

/// Two-layer model with known parameter counts.
/// linear1: 4*3 + 3 = 15 params (weight + bias)
/// linear2: 3*2 + 2 = 8 params
/// Total: 23 params
private final class KnownModel: Module {
    let linear1 = Linear(inputDimensions: 4, outputDimensions: 3)
    let linear2 = Linear(inputDimensions: 3, outputDimensions: 2)
}

@Suite("Module.summary() (GPU)")
struct ModuleSummaryGPUTests {
    @Test func paramCountsMatchManualCalc() {
        let model = KnownModel()
        eval(model)
        let summary = model.summary()
        let total = summary.totalParameters
        // linear1: 4*3 + 3 = 15, linear2: 3*2 + 2 = 8, total = 23
        #expect(total == 23)
    }

    @Test func containsLayerTypeNames() {
        let model = KnownModel()
        eval(model)
        let summary = model.summary()
        let text = summary.formatted()
        #expect(text.contains("Linear"))
    }

    @Test func printSummaryDoesNotCrash() {
        let model = KnownModel()
        eval(model)
        model.printSummary()
    }
}
#endif
