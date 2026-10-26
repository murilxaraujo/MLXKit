import Foundation
import ArgumentParser
import MLX
import MLXNN
import MLXOptimizers
import MLXKit

struct ExampleOptions: ParsableArguments {
    @Option(name: .customLong("vocab"), help: "Vocabulary size.")
    var vocabularySize = 256
    @Option(help: "Model width.")
    var width = 128
    @Option(name: .customLong("heads"), help: "Attention head count.")
    var headCount = 4
    @Option(help: "Input tokens per row.")
    var sequenceLength = 64
    @Option(help: "Training epochs.")
    var epochs = 12
    var trainingRows = 32
    var validationRows = 8
    var batchSize = 4
    @Flag(help: "Use a smaller fixed fixture and model for CI.")
    var smoke = false
}

private enum OptionsError: Error, LocalizedError {
    case invalidArgument(String)
    case invalidConfiguration

    var errorDescription: String? {
        switch self {
        case .invalidArgument(let argument):
            "Invalid argument: \(argument). Use --help for usage."
        case .invalidConfiguration:
            "Use vocab > 1, width > 0 divisible by heads, sequence length >= 5, and epochs > 0."
        }
    }
}

private enum DecoderExampleError: Error, LocalizedError {
    case nonFinite(String)
    case missingGradients
    case parameterUpdateMissing
    case noLearningSignal(initial: Float, final: Float, split: String)
    case unexpectedOutputShape([Int])
    case invalidGeneratedTokens
    case artifactMetadataMismatch
    case artifactInferenceMismatch

    var errorDescription: String? {
        switch self {
        case .nonFinite(let name): "Expected finite values for \(name)."
        case .missingGradients: "The next-token objective did not produce finite, nonzero gradients."
        case .parameterUpdateMissing: "Training did not change any model parameter."
        case .noLearningSignal(let initial, let final, let split):
            "\(split) loss did not decrease: initial=\(initial), final=\(final)."
        case .unexpectedOutputShape(let shape): "Unexpected decoder logits shape: \(shape)."
        case .invalidGeneratedTokens: "Greedy generation returned an invalid prefix, length, or token ID."
        case .artifactMetadataMismatch: "Inference artifact metadata did not round-trip exactly."
        case .artifactInferenceMismatch: "Restored model logits differ from the trained model."
        }
    }
}

private func makeTokenRows(
    count: Int,
    sequenceLength: Int,
    vocabularySize: Int,
    startOffset: Int
) -> [[Int32]] {
    precondition(count > 0 && sequenceLength > 0 && vocabularySize > 1)
    return (0..<count).map { row in
        let start = (startOffset + row * 73) % vocabularySize
        return (0...sequenceLength).map { position in
            Int32((start + position) % vocabularySize)
        }
    }
}

private func greedyGenerate(
    model: TinyDecoder,
    prompt: [Int32],
    newTokenCount: Int,
    maximumSequenceLength: Int
) -> [Int32] {
    precondition(!prompt.isEmpty && newTokenCount >= 0)
    precondition(prompt.count + newTokenCount <= maximumSequenceLength)
    var tokens = prompt

    for _ in 0..<newTokenCount {
        let context = MLXArray([tokens])
        let logits = model(context)
        let nextPosition = tokens.count - 1
        let nextToken = logits[0, nextPosition].argMax().item(Int32.self)
        tokens.append(nextToken)
    }
    return tokens
}

private func requireFinite(_ array: MLXArray, name: String) throws {
    eval(array)
    guard isFinite(array).all().item(Bool.self) else {
        throw DecoderExampleError.nonFinite(name)
    }
}

private func requireFiniteMetrics(_ metrics: [String: Float]) throws {
    for (name, value) in metrics where !value.isFinite {
        throw DecoderExampleError.nonFinite("metric \(name)")
    }
}

private func snapshotParameters(_ model: TinyDecoder) -> [String: [Float]] {
    Dictionary(
        uniqueKeysWithValues: model.trainableParameters().flattened().map {
            ($0.0, $0.1.asArray(Float.self))
        }
    )
}

private func hasParameterUpdate(from initial: [String: [Float]], model: TinyDecoder) -> Bool {
    let updated = snapshotParameters(model)
    return initial.contains { name, values in
        guard let current = updated[name], current.count == values.count else { return false }
        return zip(values, current).contains { abs($0 - $1) > 1e-7 }
    }
}

private func makeArtifactMetadata(options: ExampleOptions) -> [String: String] {
    [
        "architecture": "TinyDecoder-PreNorm-TiedEmbedding-v1",
        "vocabularySize": String(options.vocabularySize),
        "width": String(options.width),
        "headCount": String(options.headCount),
        "blockCount": "2",
        "maximumSequenceLength": String(options.sequenceLength),
        "tokenMapping": "integer-token-ids-0..<vocabularySize",
        "positionEncoding": "learned-absolute",
        "fixture": "next-integer-mod-vocabulary; local-deterministic-v1",
        "initialization": "MLX-Linear-and-Embedding-defaults; seed-42",
        "reference": "Vaswani-et-al-2017-Attention-Is-All-You-Need",
    ]
}

@main
struct TinyDecoderExample: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Train and evaluate a small causal decoder on a local token fixture."
    )

    @OptionGroup var parsedOptions: ExampleOptions

    mutating func run() async throws {
        var options = parsedOptions
        if options.smoke {
            guard options.vocabularySize == 256,
                  options.width == 128,
                  options.headCount == 4,
                  options.sequenceLength == 64,
                  options.epochs == 12
            else {
                throw OptionsError.invalidArgument("--smoke cannot be combined with model options")
            }
            options.vocabularySize = 32
            options.width = 16
            options.headCount = 2
            options.sequenceLength = 16
            options.trainingRows = 16
            options.validationRows = 4
        }

        guard options.vocabularySize > 1,
              options.width > 0,
              options.headCount > 0,
              options.width % options.headCount == 0,
              options.sequenceLength >= 5,
              options.epochs > 0
        else {
            throw OptionsError.invalidConfiguration
        }

        seed(42)
        let model = Self.makeModel(options)
        let objective = NextTokenObjective()
        let optimizer = Adam(learningRate: options.smoke ? 0.01 : 0.003)
        let configuration = TrainingConfiguration(optimizer: optimizer, objective: objective)
        let trainingData = Self.makeBatches(
            count: options.trainingRows,
            offset: 0,
            options: options
        )
        let validationData = Self.makeBatches(
            count: options.validationRows,
            offset: options.trainingRows * 73,
            options: options
        )
        let initialParameters = snapshotParameters(model)
        let engine = TrainingEngine()

        let firstBatch = trainingData.batches[0]
        let firstLogits = model(firstBatch.input)
        guard firstLogits.shape == [options.batchSize, options.sequenceLength, options.vocabularySize] else {
            throw DecoderExampleError.unexpectedOutputShape(firstLogits.shape)
        }
        try requireFinite(firstLogits, name: "initial logits")

        let gradientFunction = valueAndGrad(model: model) {
            (differentiableModel: TinyDecoder, tokens: MLXArray, targets: MLXArray) -> MLXArray in
            objective(output: differentiableModel(tokens), target: targets)
        }
        let (initialLoss, gradients) = gradientFunction(model, firstBatch.input, firstBatch.target)
        eval(initialLoss, gradients)
        guard initialLoss.item(Float.self).isFinite else {
            throw DecoderExampleError.nonFinite("initial next-token objective")
        }
        let gradientLeaves = gradients.flattened()
        guard !gradientLeaves.isEmpty else {
            throw DecoderExampleError.missingGradients
        }
        var hasNonzeroGradient = false
        for (_, gradient) in gradientLeaves {
            guard isFinite(gradient).all().item(Bool.self) else {
                throw DecoderExampleError.nonFinite("gradient")
            }
            hasNonzeroGradient = hasNonzeroGradient || abs(gradient).sum().item(Float.self) > 0
        }
        guard hasNonzeroGradient else {
            throw DecoderExampleError.missingGradients
        }

        let initialTraining = try await engine.evaluate(
            model,
            configuration: configuration,
            data: trainingData,
            trainingStep: nextTokenTrainingStep
        )
        let initialValidation = try await engine.evaluate(
            model,
            configuration: configuration,
            data: validationData,
            trainingStep: nextTokenTrainingStep
        )
        try requireFiniteMetrics(initialTraining)
        try requireFiniteMetrics(initialValidation)

        print("Tiny causal decoder (local implementation example; not a large-scale LLM)")
        print("mode=\(options.smoke ? "smoke" : "full") input=[\(options.batchSize), \(options.sequenceLength)] vocab=\(options.vocabularySize) width=\(options.width) heads=\(options.headCount) blocks=2")
        print("Initial loss: train=\(Self.formatted(initialTraining["loss"])), held-out=\(Self.formatted(initialValidation["loss"]))")

        let history = try await engine.fit(
            model,
            configuration: configuration,
            data: trainingData,
            epochs: options.epochs,
            trainingStep: nextTokenTrainingStep
        )
        let finalTraining = try await engine.evaluate(
            model,
            configuration: configuration,
            data: trainingData,
            trainingStep: nextTokenTrainingStep
        )
        let finalValidation = try await engine.evaluate(
            model,
            configuration: configuration,
            data: validationData,
            trainingStep: nextTokenTrainingStep
        )
        try requireFiniteMetrics(finalTraining)
        try requireFiniteMetrics(finalValidation)
        for epoch in history.epochs {
            try requireFiniteMetrics(epoch)
        }

        let initialTrainLoss = initialTraining["loss"] ?? .nan
        let finalTrainLoss = finalTraining["loss"] ?? .nan
        guard finalTrainLoss < initialTrainLoss else {
            throw DecoderExampleError.noLearningSignal(
                initial: initialTrainLoss,
                final: finalTrainLoss,
                split: "Training"
            )
        }
        let initialHeldOutLoss = initialValidation["loss"] ?? .nan
        let finalHeldOutLoss = finalValidation["loss"] ?? .nan
        guard finalHeldOutLoss < initialHeldOutLoss else {
            throw DecoderExampleError.noLearningSignal(
                initial: initialHeldOutLoss,
                final: finalHeldOutLoss,
                split: "Held-out"
            )
        }
        guard hasParameterUpdate(from: initialParameters, model: model) else {
            throw DecoderExampleError.parameterUpdateMissing
        }

        let generationPrompt = Array((0..<min(4, options.sequenceLength)).map {
            Int32($0 % options.vocabularySize)
        })
        let generatedTokenCount = min(8, options.sequenceLength - generationPrompt.count)
        model.train(false)
        let generated = greedyGenerate(
            model: model,
            prompt: generationPrompt,
            newTokenCount: generatedTokenCount,
            maximumSequenceLength: options.sequenceLength
        )
        guard generatedTokenCount > 0,
              generated.count == generationPrompt.count + generatedTokenCount,
              Array(generated.prefix(generationPrompt.count)) == generationPrompt,
              generated.allSatisfy({ $0 >= 0 && $0 < Int32(options.vocabularySize) })
        else {
            throw DecoderExampleError.invalidGeneratedTokens
        }

        try Self.verifyInferenceArtifact(
            model: model,
            validationInput: validationData.batches[0].input,
            generatedTokens: generated,
            options: options
        )

        print("Training loss: \(Self.formatted(finalTrainLoss)) · held-out loss: \(Self.formatted(finalHeldOutLoss))")
        print("Logits shape: \(firstLogits.shape) · generated token IDs: \(generated)")
        print("Finite, nonzero gradients · parameters updated · inference artifact round trip passed")
        print("Trained \(history.epochs.count) epochs using local fixed token rows; Adam optimizer state is not included in the inference artifact.")
    }

    private static func makeModel(_ options: ExampleOptions) -> TinyDecoder {
        TinyDecoder(
            vocabularySize: options.vocabularySize,
            width: options.width,
            headCount: options.headCount,
            maximumSequenceLength: options.sequenceLength
        )
    }

    private static func makeBatches(
        count: Int,
        offset: Int,
        options: ExampleOptions
    ) -> FixedTokenBatches {
        let rows = makeTokenRows(
            count: count,
            sequenceLength: options.sequenceLength,
            vocabularySize: options.vocabularySize,
            startOffset: offset
        )
        return FixedTokenBatches(
            tokenRows: rows,
            sequenceLength: options.sequenceLength,
            batchSize: options.batchSize
        )
    }

    private static func verifyInferenceArtifact(
        model: TinyDecoder,
        validationInput: MLXArray,
        generatedTokens: [Int32],
        options: ExampleOptions
    ) throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxkit-tiny-decoder-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let artifactURL = directory.appendingPathComponent("decoder.safetensors")
        let metadata = makeArtifactMetadata(options: options)
        try Checkpointing.saveInferenceArtifact(model, metadata: metadata, to: artifactURL)

        let restoredModel = makeModel(options)
        let artifactInfo = try Checkpointing.loadInferenceArtifact(into: restoredModel, from: artifactURL)
        restoredModel.train(false)
        guard artifactInfo.metadata == metadata else {
            throw DecoderExampleError.artifactMetadataMismatch
        }

        let originalLogits = model(validationInput)
        let restoredLogits = restoredModel(validationInput)
        let originalGenerationLogits = model(MLXArray([generatedTokens]))
        let restoredGenerationLogits = restoredModel(MLXArray([generatedTokens]))
        eval(originalLogits, restoredLogits, originalGenerationLogits, restoredGenerationLogits)
        let logitsMatch = originalLogits.allClose(restoredLogits, rtol: 1e-5, atol: 1e-5)
            .all().item(Bool.self)
        let generationMatches = originalGenerationLogits.allClose(
            restoredGenerationLogits,
            rtol: 1e-5,
            atol: 1e-5
        ).all().item(Bool.self)
        guard logitsMatch && generationMatches else {
            throw DecoderExampleError.artifactInferenceMismatch
        }
    }

    private static func formatted(_ value: Float?) -> String {
        guard let value else { return "missing" }
        return String(format: "%.4f", value)
    }
}

private func nextTokenTrainingStep(
    _ model: TinyDecoder,
    _ batch: Batch<MLXArray, MLXArray>
) -> MLXArray {
    NextTokenObjective()(output: model(batch.input), target: batch.target)
}
