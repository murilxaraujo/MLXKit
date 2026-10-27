import CoreImage
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers

/// The small model surface needed by the decision strategies.
///
/// Implementations own tokenizer/chat-template behavior and inference state.
/// They must return logits for the next token in vocabulary order. Returned
/// values are ordinary Swift values so MLX arrays do not cross actor boundaries.
public protocol TokenLogitsModel: Sendable {
    var modelID: String { get }
    var supportsMedia: Bool { get }

    func tokenize(
        messages: [DecisionPromptMessage],
        media: [DecisionMedia]
    ) async throws -> [Int]

    func tokenize(text: String) async throws -> [Int]
    func decode(tokenIDs: [Int]) async throws -> String
    func nextTokenLogits(after tokenIDs: [Int], media: [DecisionMedia]) async throws -> [Float]
    func nextTokenLogits(
        for messages: [DecisionPromptMessage], media: [DecisionMedia]
    ) async throws -> [Float]
    func continuationTokens(after prefix: [Int], label: String) async throws -> [Int]
}

extension TokenLogitsModel {
    public func nextTokenLogits(
        for messages: [DecisionPromptMessage], media: [DecisionMedia]
    ) async throws -> [Float] {
        guard media.isEmpty else {
            throw DecisionError.unsupportedMedia(
                "This model does not implement image-conditioned logits.")
        }
        let tokens = try await tokenize(messages: messages, media: media)
        return try await nextTokenLogits(after: tokens, media: media)
    }

    public func continuationTokens(after prefix: [Int], label: String) async throws -> [Int] {
        throw DecisionError.invalidAnswerSlot(
            "\(label): tokenizer boundary not verified by model adapter")
    }
}

/// A logits adapter for a loaded `mlx-swift-lm` model container.
///
/// This baseline evaluates a complete token prefix for each read. It avoids
/// generation and sampling, which is essential for constrained decisions, but
/// does not yet share a prefix cache across questions. It is useful for parity
/// tests and small benchmark cases. Image inputs are prepared through the model's
/// native processor for one-shot slot scoring.
public struct MLXLanguageModelAdapter: TokenLogitsModel {
    private let container: ModelContainer

    public let modelID: String
    public let supportsMedia: Bool

    public init(container: ModelContainer, modelID: String, supportsMedia: Bool = false) {
        self.container = container
        self.modelID = modelID
        self.supportsMedia = supportsMedia
    }

    /// Loads a model configuration through the MLX Swift language/vision model
    /// factories. OneJev nests Qwen image settings in `image_processor`, so the
    /// processor registry normalizes that file shape before decoding it.
    public static func load(modelID: String) async throws -> Self {
        let factory = VLMModelFactory(
            typeRegistry: VLMTypeRegistry.shared,
            processorRegistry: qwenProcessorRegistry(),
            modelRegistry: VLMRegistry.shared
        )
        let container = try await factory.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: modelID)
        )
        return Self(container: container, modelID: modelID, supportsMedia: true)
    }

    /// Loads a text-only model through the LLM factory.
    public static func loadLanguageModel(modelID: String) async throws -> Self {
        let container = try await LLMModelFactory.shared.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: modelID)
        )
        return Self(container: container, modelID: modelID)
    }

    public func tokenize(
        messages: [DecisionPromptMessage],
        media: [DecisionMedia]
    ) async throws -> [Int] {
        guard media.isEmpty else {
            throw DecisionError.unsupportedMedia(
                "Token-only prompt rendering cannot represent media; use slot scoring."
            )
        }
        let messages = messages.map { ["role": $0.role.rawValue, "content": $0.content] }
        return try await container.perform { context in
            try context.tokenizer.applyChatTemplate(
                messages: messages,
                tools: nil,
                additionalContext: ["enable_thinking": false]
            )
        }
    }

    public func tokenize(text: String) async throws -> [Int] {
        await container.perform { context in
            context.tokenizer.encode(text: text, addSpecialTokens: false)
        }
    }

    public func decode(tokenIDs: [Int]) async throws -> String {
        await container.perform { context in
            context.tokenizer.decode(tokenIds: tokenIDs)
        }
    }

    public func continuationTokens(after prefix: [Int], label: String) async throws -> [Int] {
        try await container.perform { context in
            let tokenizer = context.tokenizer
            let renderedPrefix = tokenizer.decode(tokenIds: prefix)
            let combined = tokenizer.encode(text: renderedPrefix + " " + label, addSpecialTokens: false)
            guard combined.starts(with: prefix), combined.count > prefix.count else {
                throw DecisionError.invalidAnswerSlot("\(label): tokenizer changes the prompt boundary")
            }
            return Array(combined.dropFirst(prefix.count))
        }
    }

    public func nextTokenLogits(after tokenIDs: [Int], media: [DecisionMedia]) async throws -> [Float] {
        guard media.isEmpty else {
            throw DecisionError.unsupportedMedia(
                "Media-conditioned continuation scoring is not supported; use slot scoring."
            )
        }
        guard !tokenIDs.isEmpty else {
            throw DecisionError.emptyTokenPrefix
        }

        return try await container.perform { context in
            let input = MLXArray(tokenIDs).expandedDimensions(axis: 0)
            let modelInput = LMInput(tokens: input)
            return try Self.logits(for: modelInput, context: context)
        }
    }

    public func nextTokenLogits(
        for messages: [DecisionPromptMessage], media: [DecisionMedia]
    ) async throws -> [Float] {
        guard !messages.isEmpty else { throw DecisionError.emptyTokenPrefix }
        let images = try Self.images(from: media)
        let chat = messages.map { message in
            let role: Chat.Message.Role
            switch message.role {
            case .system:
                role = .system
            case .user:
                role = .user
            case .assistant:
                role = .assistant
            }
            return Chat.Message(
                role: role,
                content: message.content,
                images: message.role == .user ? images : []
            )
        }
        let userInput = UserInput(
            chat: chat,
            additionalContext: ["enable_thinking": false]
        )
        let input = try await container.prepare(input: userInput)
        return try await container.perform(nonSendable: input) { context, preparedInput in
            try Self.logits(for: preparedInput, context: context)
        }
    }

    private static func logits(for input: LMInput, context: ModelContext) throws -> [Float] {
        let cache = try context.model.newCache(parameters: nil)
        let result = try context.model.prepare(
            input,
            cache: cache,
            state: nil,
            prefill: PrefillParameters(chunking: .unchunked)
        )
        let logits: MLXArray
        switch result {
        case .logits(let output):
            logits = output.logits
        case .tokens(let remainingTokens):
            logits = context.model(remainingTokens, cache: cache, state: nil).logits
        }
        guard logits.ndim == 3, logits.dim(0) == 1, logits.dim(1) > 0 else {
            throw DecisionError.invalidLogitsShape(logits.shape)
        }
        let lastLogits = logits[0, -1]
        eval(lastLogits)
        return lastLogits.asArray(Float.self)
    }

    private static func images(from media: [DecisionMedia]) throws -> [UserInput.Image] {
        try media.map { item in
            switch item {
            case .image(let data, _):
                guard let image = CIImage(data: data, options: [.applyOrientationProperty: true]) else {
                    throw DecisionError.unsupportedMedia("An image attachment could not be decoded.")
                }
                return .ciImage(image)
            case .video:
                throw DecisionError.unsupportedMedia("Video inputs are not supported by this adapter.")
            }
        }
    }

    private static func qwenProcessorRegistry() -> ProcessorTypeRegistry {
        ProcessorTypeRegistry(creators: [
            "Qwen3VLProcessor": { data, tokenizer in
                let normalized = try normalizeQwenProcessorConfiguration(data)
                let configuration = try JSONDecoder().decode(
                    Qwen3VLProcessorConfiguration.self,
                    from: normalized
                )
                return Qwen3VLProcessor(configuration, tokenizer: tokenizer)
            }
        ])
    }

    private static func normalizeQwenProcessorConfiguration(_ data: Data) throws -> Data {
        guard
            var configuration = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let imageProcessor = configuration["image_processor"] as? [String: Any]
        else {
            return data
        }

        for (key, value) in imageProcessor where configuration[key] == nil {
            configuration[key] = value
        }
        return try JSONSerialization.data(withJSONObject: configuration)
    }
}
