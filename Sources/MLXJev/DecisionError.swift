import Foundation

public enum DecisionError: Error, LocalizedError, Sendable {
    case noQuestions
    case duplicateQuestionID(String)
    case emptyQuestionID
    case emptyInstructions(String)
    case noOptions(String)
    case tooManyOptions(questionID: String, count: Int, maximum: Int)
    case duplicateOptionID(questionID: String, optionID: String)
    case unsupportedQuestion(questionID: String, strategy: String)
    case unsupportedMedia(String)
    case emptyTokenPrefix
    case invalidLogitsShape([Int])
    case invalidLogits
    case invalidTemperature(Double)
    case invalidAnswerSlot(String)
    case optionScoreFailure(String)

    public var errorDescription: String? {
        switch self {
        case .noQuestions:
            "A decision request must contain at least one question."
        case .duplicateQuestionID(let id):
            "Question identifier '\(id)' is repeated."
        case .emptyQuestionID:
            "Question identifiers must not be empty."
        case .emptyInstructions(let id):
            "Question '\(id)' must have non-empty instructions."
        case .noOptions(let id):
            "Question '\(id)' must contain at least one option or level."
        case .tooManyOptions(let id, let count, let maximum):
            "Question '\(id)' has \(count) options; the selected strategy supports at most \(maximum)."
        case .duplicateOptionID(let id, let optionID):
            "Question '\(id)' contains duplicate option identifier '\(optionID)'."
        case .unsupportedQuestion(let id, let strategy):
            "Question '\(id)' is not supported by the '\(strategy)' strategy."
        case .unsupportedMedia(let message):
            message
        case .emptyTokenPrefix:
            "The model cannot score an empty token prefix."
        case .invalidLogitsShape(let shape):
            "Expected model logits with shape [1, sequence, vocabulary], received \(shape)."
        case .invalidLogits:
            "The model returned an empty or non-finite logits vector."
        case .invalidTemperature(let value):
            "Temperature must be finite and greater than zero; received \(value)."
        case .invalidAnswerSlot(let label):
            "Answer label '\(label)' is not one round-trip tokenizer token."
        case .optionScoreFailure(let option):
            "Could not score option '\(option)'."
        }
    }
}
