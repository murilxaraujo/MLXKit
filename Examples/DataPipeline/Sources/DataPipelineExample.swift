import ArgumentParser
import MLX
import MLXNN
import MLXOptimizers
import MLXKit

private final class ScalarRegressionModel: Module, TrainableModel {
    typealias Input = MLXArray
    typealias Output = MLXArray

    let weight = MLXArray(Float32(1))

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        input * weight
    }
}

/// A local generator with no collection interface or advertised length.
private struct SyntheticSampleStream: AsyncSequence {
    typealias Element = Sample<MLXArray, MLXArray>
    typealias Failure = Never

    let sampleLimit: Int

    func makeAsyncIterator() -> Iterator {
        Iterator(sampleLimit: sampleLimit)
    }

    struct Iterator: AsyncIteratorProtocol {
        typealias Element = Sample<MLXArray, MLXArray>
        typealias Failure = Never

        let sampleLimit: Int
        var offset = 0

        mutating func next() async -> Element? {
            guard offset < sampleLimit else { return nil }
            defer { offset += 1 }

            let input = Float32(offset + 1) / Float32(sampleLimit)
            return Sample(
                input: MLXArray([input]),
                target: MLXArray([2 * input])
            )
        }
    }
}

@main
struct DataPipelineExample: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Demonstrate indexed in-memory and streaming MLXKit data pipelines."
    )

    mutating func run() async throws {
        let model = ScalarRegressionModel()
        let configuration = TrainingConfiguration(
            optimizer: SGD(learningRate: 0.05),
            objective: MSELoss()
        )
        let engine = TrainingEngine()

        let inputs = MLXArray((1...32).map { Float32($0) / 32 }).reshaped([32, 1])
        let targets = inputs * 2
        let inMemoryLoader = try DataLoader(
            source: InMemoryDataset(features: inputs, labels: targets),
            batchSize: 8,
            shuffle: .full(seed: 42)
        )

        let inMemoryHistory = try await engine.fit(
            model,
            configuration: configuration,
            data: inMemoryLoader,
            epochs: 1
        )
        let inMemoryLoss = inMemoryHistory["loss"].compactMap { $0 }.last ?? .nan
        print("In-memory training loss: \(inMemoryLoss)")

        let preparedStream = SyntheticSampleStream(sampleLimit: 32)
            .map { sample in
                Sample(input: sample.input, target: sample.target)
            }
            .filter { $0.input.item(Float.self) > 0 }

        let streamedBatches = try DataLoader(
            source: preparedStream,
            batchSize: 8,
            shuffle: .buffered(windowSize: 6, seed: 42),
            collate: { samples in
                Batch(
                    input: stacked(samples.map(\.input)),
                    target: stacked(samples.map(\.target)),
                    sampleCount: samples.count
                )
            }
        )
        let prefetchedStream = try streamedBatches.prefetched(bufferSize: 2)

        let streamHistory = try await engine.fit(
            model,
            configuration: configuration,
            data: prefetchedStream,
            epochs: 1
        )
        let streamLoss = streamHistory["loss"].compactMap { $0 }.last ?? .nan
        print("Streaming training loss: \(streamLoss)")
        print("Streaming uses a bounded shuffle window of 6 and prefetch capacity of 2.")
    }
}
