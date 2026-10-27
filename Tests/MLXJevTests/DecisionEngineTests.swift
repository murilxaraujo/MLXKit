import Foundation
import Testing
@testable import MLXJev

@Suite("MLXJev decision strategies")
struct DecisionEngineTests {
    @Test func systemOneChoosesTheHighestLetterSlot() async throws {
        let engine = DecisionEngine(model: FakeTokenLogitsModel(), strategy: .systemOne)
        let result = try await engine.decide(choiceRequest())

        #expect(result.strategy == "systemOne")
        #expect(result.answers["next"]?.outcome == .choice("first"))
        #expect(result.answers["next"]?.probabilities.count == 2)
        #expect(result.telemetry.modelCalls == 1)
        #expect(result.telemetry.promptVersion == "systemone-qev-swift-v1")
    }

    @Test func jevMLXSlotsUseTheSameRequestAndReturnCommonAnswerTypes() async throws {
        let engine = DecisionEngine(
            model: FakeTokenLogitsModel(), strategy: .jevMLX(scoring: .slots)
        )
        let result = try await engine.decide(choiceRequest())

        #expect(result.strategy == "jevMLX.slots")
        #expect(result.answers["next"]?.outcome == .choice("first"))
        #expect(result.answers["next"]?.confidence == nil)
        #expect(result.answers["next"]?.probabilities.reduce(0) { $0 + $1.probability } ?? 0 > 0.999)
        #expect(result.telemetry.promptVersion == "jevmlx-swift-v1")
    }

    @Test func jevMLXLabelsScoreAutoregressiveCandidateContinuations() async throws {
        let model = FakeTokenLogitsModel()
        let engine = DecisionEngine(model: model, strategy: .jevMLX(scoring: .labels))
        let result = try await engine.decide(choiceRequest())

        #expect(result.answers["next"]?.outcome == .choice("first"))
        #expect((result.telemetry.modelCalls) > 2)
    }

    @Test func systemOneComputesExpectedOrdinalLevelAndConfidence() async throws {
        let request = DecisionRequest(
            state: .string("The task is nearly complete."),
            questions: [
                DecisionQuestion(
                    id: "progress",
                    instructions: .string("How far along is the task?"),
                    kind: .ordinal(levels: [.string("started"), .string("done")])
                ),
            ]
        )
        let result = try await DecisionEngine(
            model: FakeTokenLogitsModel(), strategy: .systemOne
        ).decide(request)

        guard let answer = result.answers["progress"] else {
            Issue.record("Missing ordinal answer")
            return
        }
        #expect(answer.outcome == .ordinal(answer.probabilities[1].probability))
        #expect(answer.confidence != nil)
        #expect(answer.confidenceSemantics == .systemOneOrdinalMAD)
    }

    @Test func systemOneRejectsMultiSelectInsteadOfChangingItsMeaning() async throws {
        let request = DecisionRequest(
            state: .string("Refund and login issue."),
            questions: [
                DecisionQuestion(
                    id: "tags",
                    instructions: .string("Which tags apply?"),
                    kind: .multiSelect(options: [DecisionOption(id: "refund"), DecisionOption(id: "login")])
                ),
            ]
        )

        await #expect(throws: DecisionError.self) {
            try await DecisionEngine(model: FakeTokenLogitsModel(), strategy: .systemOne).decide(request)
        }
    }

    @Test func benchmarkRunnerKeepsStrategyFailuresIsolated() async throws {
        let runs = await StrategyBenchmarkRunner.compare(
            multiSelectRequest(),
            model: FakeTokenLogitsModel(),
            strategies: [.systemOne, .jevMLX(scoring: .slots)]
        )

        #expect(runs.count == 2)
        #expect(runs[0].response == nil)
        #expect(runs[0].failureDescription != nil)
        #expect(runs[1].response?.answers["tags"]?.outcome != nil)
    }

    @Test func validatesQuestionAndOptionIdentifiers() async throws {
        let duplicateOptionRequest = DecisionRequest(
            state: .string("state"),
            questions: [
                DecisionQuestion(
                    id: "q",
                    instructions: .string("choose"),
                    kind: .choice(options: [DecisionOption(id: "same"), DecisionOption(id: "same")])
                ),
            ]
        )
        await #expect(throws: DecisionError.self) {
            try await DecisionEngine(model: FakeTokenLogitsModel(), strategy: .systemOne)
                .decide(duplicateOptionRequest)
        }
    }

    private func choiceRequest() -> DecisionRequest {
        DecisionRequest(
            state: .object(["ticket": .string("A duplicate charge.")]),
            questions: [
                DecisionQuestion(
                    id: "next",
                    instructions: .string("Which team should handle this?"),
                    kind: .choice(options: [
                        DecisionOption(id: "first", description: .string("billing")),
                        DecisionOption(id: "second", description: .string("technical")),
                    ])
                ),
            ]
        )
    }

    private func multiSelectRequest() -> DecisionRequest {
        DecisionRequest(
            state: .string("The customer needs a refund and has a login problem."),
            questions: [
                DecisionQuestion(
                    id: "tags",
                    instructions: .string("Which tags apply?"),
                    kind: .multiSelect(options: [DecisionOption(id: "refund"), DecisionOption(id: "login")])
                ),
            ]
        )
    }
}

private struct FakeTokenLogitsModel: TokenLogitsModel {
    let modelID = "fake-model"
    let supportsMedia = false

    func tokenize(messages: [DecisionPromptMessage], media: [DecisionMedia]) async throws -> [Int] {
        [1, 2, 3]
    }

    func tokenize(text: String) async throws -> [Int] {
        text.unicodeScalars.map { Int($0.value) }
    }

    func decode(tokenIDs: [Int]) async throws -> String {
        String(String.UnicodeScalarView(tokenIDs.compactMap(UnicodeScalar.init)))
    }

    func nextTokenLogits(after tokenIDs: [Int], media: [DecisionMedia]) async throws -> [Float] {
        var logits = [Float](repeating: 0, count: 512)
        logits[65] = 4
        logits[66] = 1
        return logits
    }
}
