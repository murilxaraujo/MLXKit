import Foundation

/// A JSON-compatible value used for decision state and option descriptions.
public indirect enum DecisionValue: Codable, Sendable, Equatable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([DecisionValue])
    case object([String: DecisionValue])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([DecisionValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: DecisionValue].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null:
            try container.encodeNil()
        case .bool(let value):
            try container.encode(value)
        case .number(let value):
            try container.encode(value)
        case .string(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        }
    }

    func renderedJSON(prettyPrinted: Bool = true) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        if prettyPrinted {
            encoder.outputFormatting.insert(.prettyPrinted)
        }
        return String(decoding: try encoder.encode(self), as: UTF8.self)
    }

    func renderedText() throws -> String {
        switch self {
        case .string(let value):
            value
        case .null:
            ""
        default:
            try renderedJSON()
        }
    }
}

/// A named and ordered candidate. Array order is part of the prompt contract.
public struct DecisionOption: Codable, Sendable, Equatable, Identifiable {
    public let id: String
    public let description: DecisionValue?

    public init(id: String, description: DecisionValue? = nil) {
        self.id = id
        self.description = description
    }
}

/// A question in the common decision API.
public enum DecisionQuestionKind: Codable, Sendable, Equatable {
    /// A yes/no decision (`noul` in System One).
    case boolean(trueDescription: DecisionValue? = nil, falseDescription: DecisionValue? = nil)
    /// Select exactly one named option.
    case choice(options: [DecisionOption])
    /// Select one ordered level; the result is the expected zero-based level.
    case ordinal(levels: [DecisionValue])
    /// Decide independently whether each option applies.
    case multiSelect(options: [DecisionOption])
}

public struct DecisionQuestion: Codable, Sendable, Equatable, Identifiable {
    public let id: String
    public let instructions: DecisionValue
    public let kind: DecisionQuestionKind

    public init(id: String, instructions: DecisionValue, kind: DecisionQuestionKind) {
        self.id = id
        self.instructions = instructions
        self.kind = kind
    }
}

/// Media attached to a request. The selected model adapter decides which media
/// types it can process.
public enum DecisionMedia: Sendable, Equatable {
    case image(data: Data, mimeType: String)
    case video(data: Data, mimeType: String)

    public var mimeType: String {
        switch self {
        case .image(_, let mimeType), .video(_, let mimeType):
            mimeType
        }
    }
}

public struct DecisionRequest: Sendable, Equatable {
    public let state: DecisionValue
    public let questions: [DecisionQuestion]
    public let media: [DecisionMedia]

    public init(
        state: DecisionValue,
        questions: [DecisionQuestion],
        media: [DecisionMedia] = []
    ) {
        self.state = state
        self.questions = questions
        self.media = media
    }
}

public struct DecisionProbability: Codable, Sendable, Equatable, Identifiable {
    public let id: String
    public let probability: Double

    public init(optionID: String, probability: Double) {
        self.id = optionID
        self.probability = probability
    }
}

public enum DecisionOutcome: Codable, Sendable, Equatable {
    case boolean(Bool)
    case choice(String)
    case ordinal(Double)
    case multiSelect([String])
}

public enum ConfidenceSemantics: String, Codable, Sendable {
    /// Peak probability rescaled from uniform (0) to certainty (1).
    case systemOnePeak
    /// One minus the probability-weighted mean absolute deviation from the modal level.
    case systemOneOrdinalMAD
    /// The strategy does not claim a confidence score.
    case notProvided
}

public struct DecisionAnswer: Codable, Sendable, Equatable {
    public let outcome: DecisionOutcome
    public let probabilities: [DecisionProbability]
    public let confidence: Double?
    public let confidenceSemantics: ConfidenceSemantics

    public init(
        outcome: DecisionOutcome,
        probabilities: [DecisionProbability],
        confidence: Double? = nil,
        confidenceSemantics: ConfidenceSemantics = .notProvided
    ) {
        self.outcome = outcome
        self.probabilities = probabilities
        self.confidence = confidence
        self.confidenceSemantics = confidenceSemantics
    }
}

public struct DecisionTelemetry: Codable, Sendable, Equatable {
    public let elapsedMilliseconds: Double
    public let modelCalls: Int
    public let promptVersion: String

    public init(elapsedMilliseconds: Double, modelCalls: Int, promptVersion: String) {
        self.elapsedMilliseconds = elapsedMilliseconds
        self.modelCalls = modelCalls
        self.promptVersion = promptVersion
    }
}

public struct DecisionResponse: Codable, Sendable, Equatable {
    public let modelID: String
    public let strategy: String
    public let answers: [String: DecisionAnswer]
    public let telemetry: DecisionTelemetry

    public init(
        modelID: String,
        strategy: String,
        answers: [String: DecisionAnswer],
        telemetry: DecisionTelemetry
    ) {
        self.modelID = modelID
        self.strategy = strategy
        self.answers = answers
        self.telemetry = telemetry
    }
}

/// A prompt message passed to a model adapter. Adapters can apply the model's
/// native chat template when converting these messages to tokens.
public struct DecisionPromptMessage: Sendable, Equatable {
    public enum Role: String, Sendable {
        case system
        case user
        case assistant
    }

    public let role: Role
    public let content: String

    public init(role: Role, content: String) {
        self.role = role
        self.content = content
    }
}
