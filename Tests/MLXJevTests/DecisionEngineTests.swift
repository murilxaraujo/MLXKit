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

    @Test func mediaAdapterMustExplicitlySupportImageConditioning() async throws {
        let request = visualQuestion(
            instruction: "Is there an error?",
            media: [
                .image(data: Data([1]), mimeType: "image/png")
            ])
        await #expect(throws: DecisionError.self) {
            try await DecisionEngine(
                model: FakeTokenLogitsModel(supportsMedia: true), strategy: .systemOne
            )
            .decide(request)
        }
    }

    @Test func rejectsOversizedRequestsBeforeInference() async throws {
        let limits = DecisionLimits(maxQuestions: 1, maxOptions: 2, maxInputBytes: 20)
        await #expect(throws: DecisionError.self) {
            try await DecisionEngine(model: FakeTokenLogitsModel(), strategy: .systemOne, limits: limits)
                .decide(choiceRequest())
        }
        let options = (0..<11).map { DecisionOption(id: "option\($0)") }
        let request = DecisionRequest(
            state: .string("state"),
            questions: [
                DecisionQuestion(id: "q", instructions: .string("choose"), kind: .choice(options: options))
            ])
        await #expect(throws: DecisionError.self) {
            try await DecisionEngine(
                model: FakeTokenLogitsModel(), strategy: .systemOne,
                limits: .init(maxOptions: 32)
            ).decide(request)
        }
    }

    @Test func systemOneComputesExpectedOrdinalLevelAndConfidence() async throws {
        let request = DecisionRequest(
            state: .string("The task is nearly complete."),
            questions: [
                DecisionQuestion(
                    id: "progress",
                    instructions: .string("How far along is the task?"),
                    kind: .ordinal(levels: [.string("started"), .string("done")])
                )
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
                )
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
                )
            ]
        )
        await #expect(throws: DecisionError.self) {
            try await DecisionEngine(model: FakeTokenLogitsModel(), strategy: .systemOne)
                .decide(duplicateOptionRequest)
        }
    }

    @Test func realOneJevCheckpointLoadsAndScoresAChoice() async throws {
        guard ProcessInfo.processInfo.environment["MLXJEV_REAL_MODEL"] == "1" else {
            return
        }

        let model = try await MLXLanguageModelAdapter.load(
            modelID: "OmniJev/OneJev1.6-0.8B"
        )
        let response = try await DecisionEngine(
            model: model,
            strategy: .systemOne
        ).decide(choiceRequest())

        #expect(response.modelID == "OmniJev/OneJev1.6-0.8B")
        #expect(response.answers["next"]?.probabilities.count == 2)
        #expect(
            response.answers["next"]?.probabilities.allSatisfy {
                $0.probability.isFinite && (0.0...1.0).contains($0.probability)
            } == true)
        #expect(response.telemetry.modelCalls == 1)
        #expect(response.telemetry.elapsedMilliseconds > 0)
        #expect(response.telemetry.promptVersion == "systemone-qev-swift-v1")
    }

    @Test func realOneJevCheckpointScoresAttachedScreenshots() async throws {
        guard ProcessInfo.processInfo.environment["MLXJEV_REAL_MODEL"] == "1" else {
            return
        }

        let model = try await MLXLanguageModelAdapter.load(modelID: "OmniJev/OneJev1.6-0.8B")
        let receipt = try imageMedia(named: "IMG_1495.png")
        let errorScreen = try imageMedia(named: "IMG_1583.png")
        let engine = DecisionEngine(model: model, strategy: .systemOne)

        let receiptResult = try await engine.decide(
            visualQuestion(
                instruction: "Does this screenshot show an order payment summary and total?",
                media: [receipt]
            )
        )
        let errorResult = try await engine.decide(
            visualQuestion(
                instruction: "Does this screenshot show a Something went wrong message and Retry button?",
                media: [errorScreen]
            )
        )

        #expect(receiptResult.modelID == "OmniJev/OneJev1.6-0.8B")
        #expect(receiptResult.answers["visual"]?.outcome == .boolean(true))
        #expect(errorResult.answers["visual"]?.outcome == .boolean(true))
        #expect(receiptResult.answers["visual"]?.probabilities.count == 2)
        #expect(errorResult.answers["visual"]?.probabilities.count == 2)
        #expect(receiptResult.telemetry.modelCalls == 1)
        #expect(errorResult.telemetry.modelCalls == 1)
        print("OneJev receipt screenshot: \(String(describing: receiptResult.answers["visual"]))")
        print("OneJev error screenshot: \(String(describing: errorResult.answers["visual"]))")
    }

    @Test func realQwen25InstructCheckpointScoresChoice() async throws {
        guard ProcessInfo.processInfo.environment["MLXJEV_REAL_MODEL"] == "1" else {
            return
        }

        let model = try await MLXLanguageModelAdapter.loadLanguageModel(
            modelID: "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        )
        let response = try await DecisionEngine(
            model: model,
            strategy: .jevMLX(scoring: .slots)
        ).decide(choiceRequest())

        #expect(response.modelID == "mlx-community/Qwen2.5-1.5B-Instruct-4bit")
        #expect(response.answers["next"]?.probabilities.count == 2)
        #expect(
            response.answers["next"]?.probabilities.allSatisfy {
                $0.probability.isFinite && (0.0...1.0).contains($0.probability)
            } == true)
        #expect(response.telemetry.modelCalls == 1)
        print("Qwen2.5 JevMLX slot result: \(String(describing: response.answers["next"]))")
    }

    @Test func realQwen25ComparesStrategiesOnSameRequest() async throws {
        guard ProcessInfo.processInfo.environment["MLXJEV_REAL_MODEL"] == "1" else { return }
        let model = try await MLXLanguageModelAdapter.loadLanguageModel(
            modelID: "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        )
        let runs = await StrategyBenchmarkRunner.compare(
            choiceRequest(), model: model,
            strategies: [.systemOne, .jevMLX(scoring: .slots), .jevMLX(scoring: .labels)])
        for run in runs {
            #expect(
                run.response?.answers["next"]?.probabilities.count == 2,
                "\(run.strategy.identifier): \(run.failureDescription ?? "unknown failure")")
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
                )
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
                )
            ]
        )
    }

    private func imageMedia(named filename: String) throws -> DecisionMedia {
        let url = URL(fileURLWithPath: NSHomeDirectory())
            .appending(path: "Downloads")
            .appending(path: filename)
        return .image(data: try Data(contentsOf: url), mimeType: "image/png")
    }

    private func visualQuestion(instruction: String, media: [DecisionMedia]) -> DecisionRequest {
        DecisionRequest(
            state: .string("A screenshot from a food delivery app."),
            questions: [
                DecisionQuestion(
                    id: "visual",
                    instructions: .string(instruction),
                    kind: .boolean()
                )
            ],
            media: media
        )
    }
}

private struct FakeTokenLogitsModel: TokenLogitsModel {
    let modelID = "fake-model"
    var supportsMedia = false

    func continuationTokens(after prefix: [Int], label: String) async throws -> [Int] {
        Array(" \(label)".unicodeScalars.map { Int($0.value) })
    }

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
