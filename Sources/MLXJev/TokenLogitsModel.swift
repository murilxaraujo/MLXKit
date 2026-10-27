import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM

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
}

/// A text-only logits adapter for a loaded `mlx-swift-lm` model container.
///
/// This baseline evaluates a complete token prefix for each read. It avoids
/// generation and sampling, which is essential for constrained decisions, but
/// does not yet share a prefix cache across questions. It is useful for parity
/// tests and small benchmark cases. Multimodal inputs require a processor-aware
/// adapter and are rejected here rather than silently ignored.
public struct MLXLanguageModelAdapter: TokenLogitsModel {
    private let container: ModelContainer

    public let modelID: String
    public let supportsMedia = false

    public init(container: ModelContainer, modelID: String) {
        self.container = container
        self.modelID = modelID
    }

    /// Loads a model configuration through the MLX Swift language/vision model
    /// factories. Checkpoint compatibility depends on the model artifact being
    /// supported by the selected `mlx-swift-lm` release.
    public static func load(modelID: String) async throws -> Self {
        let container = try await VLMModelFactory.shared.loadContainer(
            configuration: ModelConfiguration(id: modelID)
        )
        return Self(container: container, modelID: modelID)
    }

    public func tokenize(
        messages: [DecisionPromptMessage],
        media: [DecisionMedia]
    ) async throws -> [Int] {
        try Self.rejectMedia(media)
        let messages = messages.map { ["role": $0.role.rawValue, "content": $0.content] }
        return try await container.perform { context in
            try context.tokenizer.applyChatTemplate(messages: messages)
        }
    }

    public func tokenize(text: String) async throws -> [Int] {
        await container.perform { context in
            context.tokenizer.encode(text: text)
        }
    }

    public func decode(tokenIDs: [Int]) async throws -> String {
        await container.perform { context in
            context.tokenizer.decode(tokenIDs: tokenIDs)
        }
    }

    public func nextTokenLogits(after tokenIDs: [Int], media: [DecisionMedia]) async throws -> [Float] {
        try Self.rejectMedia(media)
        guard !tokenIDs.isEmpty else {
            throw DecisionError.emptyTokenPrefix
        }

        return try await container.perform { context in
            let input = MLXArray(tokenIDs).expandedDimensions(axis: 0)
            let logits = context.model.callAsFunction(input, cache: nil)
            guard logits.ndim == 3, logits.dim(0) == 1, logits.dim(1) > 0 else {
                throw DecisionError.invalidLogitsShape(logits.shape)
            }
            let lastLogits = logits[0, -1]
            eval(lastLogits)
            return lastLogits.asArray(Float.self)
        }
    }

    private static func rejectMedia(_ media: [DecisionMedia]) throws {
        guard media.isEmpty else {
            throw DecisionError.unsupportedMedia("The baseline MLX language-model adapter is text-only.")
        }
    }
}
