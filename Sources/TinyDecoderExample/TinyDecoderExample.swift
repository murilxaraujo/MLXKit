import Foundation
import MLX
import MLXOptimizers
import MLXKit
import TinyDecoderCore

private struct ExampleOptions {
    var vocabularySize = 16
    var width = 32
    var headCount = 4
    var sequenceLength = 8
    var epochs = 4

    static func parse(_ arguments: [String]) throws -> ExampleOptions {
        var options = ExampleOptions()
        var index = 0

        while index < arguments.count {
            let flag = arguments[index]
            guard index + 1 < arguments.count, let value = Int(arguments[index + 1]) else {
                throw OptionsError.invalidArgument(flag)
            }
            switch flag {
            case "--vocab": options.vocabularySize = value
            case "--width": options.width = value
            case "--heads": options.headCount = value
            case "--sequence-length": options.sequenceLength = value
            case "--epochs": options.epochs = value
            default: throw OptionsError.invalidArgument(flag)
            }
            index += 2
        }

        guard options.vocabularySize > 1,
              options.width > 0,
              options.headCount > 0,
              options.width % options.headCount == 0,
              options.sequenceLength > 1,
              options.epochs > 0
        else {
            throw OptionsError.invalidConfiguration
        }
        return options
    }
}

private enum OptionsError: Error, LocalizedError {
    case invalidArgument(String)
    case invalidConfiguration

    var errorDescription: String? {
        switch self {
        case .invalidArgument(let argument):
            "Invalid argument: \(argument). Use --vocab, --width, --heads, --sequence-length, or --epochs with an integer value."
        case .invalidConfiguration:
            "Use vocab > 1, width > 0 divisible by heads, sequence length > 1, and epochs > 0."
        }
    }
}

@main
struct TinyDecoderExample {
    static func main() async throws {
        let options = try ExampleOptions.parse(Array(CommandLine.arguments.dropFirst()))
        seed(42)

        let model = TinyDecoder(
            vocabularySize: options.vocabularySize,
            width: options.width,
            headCount: options.headCount,
            maximumSequenceLength: options.sequenceLength
        )
        let engine = TrainingEngine()
        let configuration = TrainingConfiguration(
            optimizer: Adam(learningRate: 0.003),
            objective: NextTokenObjective()
        )

        // A fixed local pattern: the target token is always the next vocabulary
        // item, wrapping at the vocabulary boundary. No data is downloaded.
        let rows = makeTokenRows(
            count: 10,
            sequenceLength: options.sequenceLength,
            vocabularySize: options.vocabularySize
        )
        let trainingBatches = FixedTokenBatches(
            tokenRows: Array(rows.prefix(8)),
            sequenceLength: options.sequenceLength,
            batchSize: 2
        )
        let evaluationBatches = FixedTokenBatches(
            tokenRows: Array(rows.suffix(2)),
            sequenceLength: options.sequenceLength,
            batchSize: 2
        )

        print("Tiny causal decoder (implementation example; not a large-scale LLM)")
        print("vocab=\(options.vocabularySize) width=\(options.width) heads=\(options.headCount) sequenceLength=\(options.sequenceLength)")

        let history = try await engine.fit(
            model,
            configuration: configuration,
            data: trainingBatches,
            epochs: options.epochs
        )
        let evaluation = try await engine.evaluate(
            model,
            configuration: configuration,
            data: evaluationBatches
        )

        if let finalEpoch = history.epochs.last {
            print(String(format: "training loss: %.4f", finalEpoch["loss"] ?? .nan))
        }
        print(String(format: "evaluation loss: %.4f", evaluation["loss"] ?? .nan))
    }
}
