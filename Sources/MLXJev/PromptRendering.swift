import Foundation

enum DecisionPromptRenderer {
    static let systemOnePromptVersion = "systemone-qev-swift-v1"
    static let jevMLXPromptVersion = "jevmlx-swift-v1"

    static let systemOneSystem =
        "Apply the question to the state. Choose exactly one of the listed options. "
        + "Respond with only its uppercase letter, with no explanation or reasoning."
    static let defaultYesDescription = "the statement is true / the answer is yes"
    static let defaultNoDescription = "the statement is false / the answer is no"

    static func messages(
        for request: DecisionRequest,
        question: DecisionQuestion,
        strategy: DecisionStrategy,
        slotLabels: [String] = []
    ) throws -> [DecisionPromptMessage] {
        switch strategy {
        case .systemOne:
            return try systemOneMessages(for: request, question: question, slotLabels: slotLabels)
        case .jevMLX:
            return try jevMLXMessages(for: request, question: question, slotLabels: slotLabels)
        }
    }

    private static func systemOneMessages(
        for request: DecisionRequest,
        question: DecisionQuestion,
        slotLabels: [String]
    ) throws -> [DecisionPromptMessage] {
        let options = try displayOptions(for: question, labels: slotLabels)
        let labels = options.map(\.label)
        let instructions = try renderEntry(question.instructions)
        let optionsText = options.map { option in
            option.description.isEmpty
                ? "\(option.label). \(option.name)"
                : "\(option.label). \(option.name): \(option.description)"
        }.joined(separator: "\n")
        let answerLine =
            labels.count <= 26
            ? "Answer with one letter: \(labels.joined(separator: ", "))."
            : "Answer with one label: \(labels.joined(separator: ", "))."
        let suffix: String
        if case .ordinal = question.kind {
            suffix = "Rate the state: \(instructions)\n\nOptions:\n\(optionsText)\n\n\(answerLine)"
        } else {
            suffix = "Question: \(instructions)\n\nOptions:\n\(optionsText)\n\n\(answerLine)"
        }
        let stateText = try request.state.renderedText()
        let mediaHint = request.media.isEmpty ? "" : "\n\nUse the attached screenshot(s) as visual evidence."
        let userContent = "<state>\n\(stateText)\n</state>\n\n\(suffix)\(mediaHint)"
        return [
            DecisionPromptMessage(role: .system, content: systemOneSystem),
            DecisionPromptMessage(role: .user, content: userContent),
        ]
    }

    private static func jevMLXMessages(
        for request: DecisionRequest,
        question: DecisionQuestion,
        slotLabels: [String]
    ) throws -> [DecisionPromptMessage] {
        let instructions = try renderEntry(question.instructions)
        let options = try displayOptions(for: question, labels: slotLabels)
        let optionsText = options.map { option in
            if slotLabels.isEmpty {
                return option.description.isEmpty
                    ? "- \(option.name)"
                    : "- \(option.name): \(option.description)"
            }
            return option.description.isEmpty
                ? "\(option.label). \(option.name)"
                : "\(option.label). \(option.name): \(option.description)"
        }.joined(separator: "\n")

        let system =
            "You are a judgment engine. Choose the option that best matches the state. "
            + "Use only the listed options."
        let stateText = try request.state.renderedText()
        let mediaHint = request.media.isEmpty ? "" : "\n\nUse the attached screenshot(s) as visual evidence."
        let answerInstruction =
            slotLabels.isEmpty
            ? "Return only the selected option name."
            : "Return only the option label: \(slotLabels.joined(separator: ", "))."
        let user =
            "State:\n\(stateText)\n\nQuestion:\n\(instructions)\n\nOptions:\n\(optionsText)\n\n\(answerInstruction)\(mediaHint)\nAnswer:"
        return [
            DecisionPromptMessage(role: .system, content: system),
            DecisionPromptMessage(role: .user, content: user),
        ]
    }

    private struct DisplayOption {
        let label: String
        let name: String
        let description: String
    }

    private static func displayOptions(
        for question: DecisionQuestion,
        labels: [String]
    ) throws -> [DisplayOption] {
        let rawOptions: [(String, DecisionValue?)]
        switch question.kind {
        case .boolean(let trueDescription, let falseDescription):
            rawOptions = [
                ("yes", trueDescription ?? .string(defaultYesDescription)),
                ("no", falseDescription ?? .string(defaultNoDescription)),
            ]
        case .choice(let options), .multiSelect(let options):
            rawOptions = options.map { ($0.id, $0.description) }
        case .ordinal(let levels):
            rawOptions = levels.enumerated().map { (String($0.offset), $0.element) }
        }

        guard labels.isEmpty || rawOptions.count <= labels.count else {
            throw DecisionError.tooManyOptions(
                questionID: question.id, count: rawOptions.count, maximum: labels.count
            )
        }

        return try rawOptions.enumerated().map { index, entry in
            let description = try entry.1.map(renderEntry) ?? ""
            return DisplayOption(
                label: labels.isEmpty ? "" : labels[index], name: entry.0, description: description)
        }
    }

    private static func renderEntry(_ value: DecisionValue) throws -> String {
        switch value {
        case .string(let text):
            text.trimmingCharacters(in: .whitespacesAndNewlines)
        case .null:
            ""
        default:
            try value.renderedJSON()
        }
    }
}
