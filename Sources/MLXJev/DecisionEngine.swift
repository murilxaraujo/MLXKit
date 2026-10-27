import Foundation

public enum JevMLXScoring: String, Codable, Sendable {
    /// Score each allowed option string as an autoregressive continuation.
    case labels
    /// Read one-token option slots from the next-token logits.
    case slots
}

/// Selects the inference semantics while keeping requests and results common.
public enum DecisionStrategy: Codable, Sendable, Equatable {
    /// OneJev / qev-style labelled answer-slot scoring.
    case systemOne
    /// JevMLX-style constrained option scoring.
    case jevMLX(scoring: JevMLXScoring = .labels)

    public var identifier: String {
        switch self {
        case .systemOne:
            "systemOne"
        case .jevMLX(.labels):
            "jevMLX.labels"
        case .jevMLX(.slots):
            "jevMLX.slots"
        }
    }

    var promptVersion: String {
        switch self {
        case .systemOne:
            DecisionPromptRenderer.systemOnePromptVersion
        case .jevMLX:
            DecisionPromptRenderer.jevMLXPromptVersion
        }
    }
}

/// Executes a typed request with one of the supported scoring strategies.
///
/// This is a correctness-first baseline: it runs questions and candidate
/// continuations sequentially. It does not share prefix caches yet. Both
/// strategies use the same model instance, request schema, and result types.
public struct DecisionEngine: Sendable {
    private let model: any TokenLogitsModel
    public let strategy: DecisionStrategy

    public init(model: any TokenLogitsModel, strategy: DecisionStrategy) {
        self.model = model
        self.strategy = strategy
    }

    public func decide(_ request: DecisionRequest) async throws -> DecisionResponse {
        try validate(request)
        if !request.media.isEmpty, !model.supportsMedia {
            throw DecisionError.unsupportedMedia(
                "Model '\(model.modelID)' does not support media inputs."
            )
        }

        let start = Date.timeIntervalSinceReferenceDate
        let slotLabels: [String]
        switch strategy {
        case .systemOne, .jevMLX(.slots):
            slotLabels = try await verifiedSlotLabels(from: model)
        case .jevMLX(.labels):
            slotLabels = []
        }

        var answers: [String: DecisionAnswer] = [:]
        var modelCalls = 0
        for question in request.questions {
            switch strategy {
            case .systemOne:
                let (answer, calls) = try await scoreSystemOne(
                    question, request: request, slotLabels: slotLabels
                )
                answers[question.id] = answer
                modelCalls += calls
            case .jevMLX(let scoring):
                let (answer, calls) = try await scoreJevMLX(
                    question, request: request, scoring: scoring, slotLabels: slotLabels
                )
                answers[question.id] = answer
                modelCalls += calls
            }
        }

        let elapsed = (Date.timeIntervalSinceReferenceDate - start) * 1_000
        return DecisionResponse(
            modelID: model.modelID,
            strategy: strategy.identifier,
            answers: answers,
            telemetry: DecisionTelemetry(
                elapsedMilliseconds: elapsed,
                modelCalls: modelCalls,
                promptVersion: strategy.promptVersion
            )
        )
    }

    private func validate(_ request: DecisionRequest) throws {
        guard !request.questions.isEmpty else { throw DecisionError.noQuestions }
        var questionIDs = Set<String>()
        for question in request.questions {
            guard !question.id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw DecisionError.emptyQuestionID
            }
            guard questionIDs.insert(question.id).inserted else {
                throw DecisionError.duplicateQuestionID(question.id)
            }
            guard !(try question.instructions.renderedText())
                .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                throw DecisionError.emptyInstructions(question.id)
            }

            let options: [DecisionOption]
            switch question.kind {
            case .boolean:
                continue
            case .choice(let values), .multiSelect(let values):
                options = values
            case .ordinal(let values):
                guard !values.isEmpty else { throw DecisionError.noOptions(question.id) }
                if case .systemOne = strategy, values.count > 10 {
                    throw DecisionError.tooManyOptions(questionID: question.id, count: values.count, maximum: 10)
                }
                continue
            }

            guard !options.isEmpty else { throw DecisionError.noOptions(question.id) }
            var optionIDs = Set<String>()
            for option in options {
                guard optionIDs.insert(option.id).inserted else {
                    throw DecisionError.duplicateOptionID(questionID: question.id, optionID: option.id)
                }
            }

            if case .systemOne = strategy, case .multiSelect = question.kind {
                throw DecisionError.unsupportedQuestion(questionID: question.id, strategy: strategy.identifier)
            }
        }
    }

    private func scoreSystemOne(
        _ question: DecisionQuestion,
        request: DecisionRequest,
        slotLabels: [String]
    ) async throws -> (DecisionAnswer, Int) {
        let candidateCount = numberOfCandidates(in: question.kind)
        guard candidateCount <= slotLabels.count else {
            throw DecisionError.tooManyOptions(
                questionID: question.id, count: candidateCount, maximum: slotLabels.count
            )
        }
        let messages = try DecisionPromptRenderer.messages(
            for: request, question: question, strategy: .systemOne,
            slotLabels: Array(slotLabels.prefix(candidateCount))
        )
        let prefix = try await model.tokenize(messages: messages, media: request.media)
        let logits = try await model.nextTokenLogits(after: prefix, media: request.media)
        let labels = Array(slotLabels.prefix(candidateCount))
        let slotIDs = try await oneTokenIDs(labels, from: model)
        let candidateLogits = try gather(logits, tokenIDs: slotIDs)
        let probabilities = try softmax(candidateLogits)
        return (try assemble(question.kind, probabilities: probabilities, confidence: true), 1)
    }

    private func scoreJevMLX(
        _ question: DecisionQuestion,
        request: DecisionRequest,
        scoring: JevMLXScoring,
        slotLabels: [String]
    ) async throws -> (DecisionAnswer, Int) {
        if case .multiSelect(let options) = question.kind {
            var values: [DecisionProbability] = []
            var calls = 0
            for option in options {
                let description = try option.description?.renderedText() ?? ""
                let text = description.isEmpty
                    ? "\(try question.instructions.renderedText())\nOption: \(option.id)\nShould this option be selected?"
                    : "\(try question.instructions.renderedText())\nOption: \(option.id): \(description)\nShould this option be selected?"
                let booleanQuestion = DecisionQuestion(
                    id: question.id,
                    instructions: .string(text),
                    kind: .boolean()
                )
                let (answer, usedCalls) = try await scoreJevMLX(
                    booleanQuestion, request: request, scoring: scoring, slotLabels: slotLabels
                )
                guard case .boolean = answer.outcome,
                      let yes = answer.probabilities.first(where: { $0.id == "yes" })?.probability
                else { throw DecisionError.optionScoreFailure(option.id) }
                values.append(DecisionProbability(optionID: option.id, probability: yes))
                calls += usedCalls
            }
            let selected = values.filter { $0.probability >= 0.5 }.map(\.id)
            return (
                DecisionAnswer(
                    outcome: .multiSelect(selected), probabilities: values,
                    confidenceSemantics: .notProvided
                ),
                calls
            )
        }

        let labels: [String]
        switch question.kind {
        case .boolean:
            labels = scoring == .slots ? ["A", "B"] : ["yes", "no"]
        case .choice(let options):
            labels = scoring == .slots ? Array(slotLabels.prefix(options.count)) : options.map(\.id)
        case .ordinal(let levels):
            labels = scoring == .slots
                ? Array(slotLabels.prefix(levels.count))
                : levels.indices.map(String.init)
        case .multiSelect:
            fatalError("multiSelect handled above")
        }

        guard labels.count == numberOfCandidates(in: question.kind) else {
            throw DecisionError.tooManyOptions(
                questionID: question.id, count: numberOfCandidates(in: question.kind), maximum: slotLabels.count
            )
        }

        let promptLabels = scoring == .slots ? labels : []
        let messages = try DecisionPromptRenderer.messages(
            for: request, question: question, strategy: .jevMLX(scoring: scoring), slotLabels: promptLabels
        )
        let prefix = try await model.tokenize(messages: messages, media: request.media)

        let candidateScores: [Double]
        let calls: Int
        switch scoring {
        case .slots:
            let logits = try await model.nextTokenLogits(after: prefix, media: request.media)
            let slotIDs = try await oneTokenIDs(labels, from: model)
            candidateScores = try gather(logits, tokenIDs: slotIDs).map(Double.init)
            calls = 1
        case .labels:
            var scores: [Double] = []
            var callCount = 0
            for label in labels {
                let continuation = try await model.tokenize(text: " \(label)")
                guard !continuation.isEmpty else {
                    throw DecisionError.invalidAnswerSlot(label)
                }
                let result = try await continuationLogProbability(
                    continuation, after: prefix, media: request.media
                )
                scores.append(result.logProbability)
                callCount += result.calls
            }
            candidateScores = scores
            calls = callCount
        }

        let probabilities = try softmax(candidateScores)
        return (try assemble(question.kind, probabilities: probabilities, confidence: false), calls)
    }

    private func continuationLogProbability(
        _ continuation: [Int],
        after prefix: [Int],
        media: [DecisionMedia]
    ) async throws -> (logProbability: Double, calls: Int) {
        var context = prefix
        var score = 0.0
        for token in continuation {
            let logits = try await model.nextTokenLogits(after: context, media: media)
            let logProbabilities = try logSoftmax(logits)
            guard token >= 0, token < logProbabilities.count else {
                throw DecisionError.invalidAnswerSlot("token \(token)")
            }
            score += logProbabilities[token]
            context.append(token)
        }
        return (score, continuation.count)
    }

    private func assemble(
        _ kind: DecisionQuestionKind,
        probabilities: [Double],
        confidence includeConfidence: Bool
    ) throws -> DecisionAnswer {
        switch kind {
        case .boolean:
            guard probabilities.count == 2 else { throw DecisionError.invalidLogits }
            return DecisionAnswer(
                outcome: .boolean(probabilities[0] >= 0.5),
                probabilities: [
                    DecisionProbability(optionID: "yes", probability: probabilities[0]),
                    DecisionProbability(optionID: "no", probability: probabilities[1]),
                ]
            )
        case .choice(let options):
            guard probabilities.count == options.count else { throw DecisionError.invalidLogits }
            let winner = probabilities.indices.max(by: { probabilities[$0] < probabilities[$1] }) ?? 0
            let confidence = includeConfidence ? peakConfidence(probabilities) : nil
            return DecisionAnswer(
                outcome: .choice(options[winner].id),
                probabilities: zip(options, probabilities).map {
                    DecisionProbability(optionID: $0.0.id, probability: $0.1)
                },
                confidence: confidence,
                confidenceSemantics: includeConfidence ? .systemOnePeak : .notProvided
            )
        case .ordinal(let levels):
            guard probabilities.count == levels.count else { throw DecisionError.invalidLogits }
            let value = probabilities.enumerated().reduce(0.0) { $0 + Double($1.offset) * $1.element }
            let values = probabilities.enumerated().map {
                DecisionProbability(optionID: String($0.offset), probability: $0.element)
            }
            let confidence = includeConfidence ? ordinalConfidence(probabilities) : nil
            return DecisionAnswer(
                outcome: .ordinal(value), probabilities: values, confidence: confidence,
                confidenceSemantics: includeConfidence ? .systemOneOrdinalMAD : .notProvided
            )
        case .multiSelect:
            throw DecisionError.invalidLogits
        }
    }

    private func numberOfCandidates(in kind: DecisionQuestionKind) -> Int {
        switch kind {
        case .boolean:
            2
        case .choice(let options), .multiSelect(let options):
            options.count
        case .ordinal(let levels):
            levels.count
        }
    }
}

private func verifiedSlotLabels(from model: any TokenLogitsModel) async throws -> [String] {
    let letters = (65...90).map { String(UnicodeScalar($0)!) }
    let letterIDs = try await oneTokenIDs(letters, from: model)
    guard Set(letterIDs).count == letterIDs.count else {
        throw DecisionError.invalidAnswerSlot("A-Z token IDs collide")
    }

    let excluded = Set([
        "BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ", "GJ", "GK",
        "GQ", "GZ", "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ",
    ])
    let pairs = (65...90).flatMap { first in
        (65...90).map { second in
            String(UnicodeScalar(first)!) + String(UnicodeScalar(second)!)
        }
    }.filter { !excluded.contains($0) }
    let extras = Array(pairs.prefix(230))
    do {
        let extraIDs = try await oneTokenIDs(extras, from: model)
        guard Set(letterIDs + extraIDs).count == letterIDs.count + extraIDs.count else {
            return letters
        }
        return letters + extras
    } catch {
        return letters
    }
}

private func oneTokenIDs(_ labels: [String], from model: any TokenLogitsModel) async throws -> [Int] {
    var ids: [Int] = []
    for label in labels {
        let tokenIDs = try await model.tokenize(text: label)
        guard tokenIDs.count == 1,
              try await model.decode(tokenIDs: tokenIDs) == label
        else {
            throw DecisionError.invalidAnswerSlot(label)
        }
        ids.append(tokenIDs[0])
    }
    return ids
}

private func gather(_ logits: [Float], tokenIDs: [Int]) throws -> [Float] {
    try validateLogits(logits)
    return try tokenIDs.map { id in
        guard id >= 0, id < logits.count else { throw DecisionError.invalidAnswerSlot("token \(id)") }
        return logits[id]
    }
}

private func validateLogits(_ logits: [Float]) throws {
    guard !logits.isEmpty,
          !logits.contains(where: { $0.isNaN || $0 == .infinity }),
          logits.contains(where: { $0.isFinite })
    else { throw DecisionError.invalidLogits }
}

private func logSoftmax(_ logits: [Float]) throws -> [Double] {
    try validateLogits(logits)
    let maxValue = Double(logits.filter(\.isFinite).max()!)
    let denominator = logits.reduce(0.0) { partial, value in
        value.isFinite ? partial + exp(Double(value) - maxValue) : partial
    }
    let logNormalizer = maxValue + log(denominator)
    return logits.map { $0.isFinite ? Double($0) - logNormalizer : -.infinity }
}

private func softmax(_ scores: [Double]) throws -> [Double] {
    guard !scores.isEmpty,
          scores.allSatisfy({ !$0.isNaN && $0 != .infinity }),
          let maximum = scores.filter(\.isFinite).max()
    else { throw DecisionError.invalidLogits }
    let values = scores.map { $0.isFinite ? exp($0 - maximum) : 0 }
    let total = values.reduce(0, +)
    guard total.isFinite, total > 0 else { throw DecisionError.invalidLogits }
    return values.map { $0 / total }
}

private func peakConfidence(_ probabilities: [Double]) -> Double {
    guard probabilities.count > 1 else { return 1 }
    let uniform = 1 / Double(probabilities.count)
    return min(1, max(0, (probabilities.max()! - uniform) / (1 - uniform)))
}

private func ordinalConfidence(_ probabilities: [Double]) -> Double {
    guard probabilities.count > 1 else { return 1 }
    let mode = probabilities.indices.max(by: { probabilities[$0] < probabilities[$1] }) ?? 0
    let distance = probabilities.enumerated().reduce(0.0) {
        $0 + $1.element * Double(abs($1.offset - mode))
    }
    let center = Double(probabilities.count - 1) / 2
    let uniformMAD = probabilities.indices.reduce(0.0) {
        $0 + abs(Double($1) - center) / Double(probabilities.count)
    }
    guard uniformMAD > 0 else { return 1 }
    return max(0, 1 - distance / uniformMAD)
}
