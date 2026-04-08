import Testing
@testable import MLXKit

@Suite("ModelSummary")
struct ModelSummaryTests {
    @Test func formattedContainsHeaders() {
        let summary = ModelSummary(layers: [
            SummaryRow(name: "linear1", type: "Linear", parameterCount: 100),
            SummaryRow(name: "linear2", type: "Linear", parameterCount: 50),
        ])
        let text = summary.formatted()
        #expect(text.contains("Layer"))
        #expect(text.contains("Type"))
        #expect(text.contains("Params"))
    }

    @Test func totalParametersIsSumOfParts() {
        let summary = ModelSummary(layers: [
            SummaryRow(name: "a", type: "Linear", parameterCount: 100),
            SummaryRow(name: "b", type: "Linear", parameterCount: 200),
            SummaryRow(name: "c", type: "Embedding", parameterCount: 300),
        ])
        #expect(summary.totalParameters == 600)
    }

    @Test func estimatedSizeBytesIsParamsTimes4() {
        let summary = ModelSummary(layers: [
            SummaryRow(name: "a", type: "Linear", parameterCount: 1000),
        ])
        #expect(summary.estimatedSizeBytes == 4000)
    }

    @Test func formattedContainsLayerTypes() {
        let summary = ModelSummary(layers: [
            SummaryRow(name: "embed", type: "Embedding", parameterCount: 500),
            SummaryRow(name: "dense", type: "Linear", parameterCount: 100),
        ])
        let text = summary.formatted()
        #expect(text.contains("Embedding"))
        #expect(text.contains("Linear"))
    }

    @Test func formattedContainsTotalLine() {
        let summary = ModelSummary(layers: [
            SummaryRow(name: "a", type: "Linear", parameterCount: 1234),
        ])
        let text = summary.formatted()
        #expect(text.contains("Total params"))
    }

    @Test func emptyModelProducesValidSummary() {
        let summary = ModelSummary(layers: [])
        #expect(summary.totalParameters == 0)
        #expect(summary.estimatedSizeBytes == 0)
        let text = summary.formatted()
        #expect(text.contains("Total params"))
    }
}
