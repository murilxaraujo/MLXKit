import Darwin
import Foundation
import MLX
import MLXKit
import MLXNN
import MLXOptimizers

// TrainingEngine's async APIs transfer this value into their isolated training
// operation. The benchmark uses each configuration for one awaited operation
// at a time, so its optimizer and metric state are never accessed concurrently.
extension TrainingConfiguration: @unchecked Sendable {}

private let embeddingSize = 100
private let hiddenSize = 100
private let sequenceLength = 30
private let defaultBatchSize = 20

private enum BenchmarkError: Error, LocalizedError {
    case invalidData(String)
    case missingMetric(String)

    var errorDescription: String? {
        switch self {
        case .invalidData(let message): message
        case .missingMetric(let name): "Training did not produce the expected metric '\(name)'."
        }
    }
}

private struct Options {
    let dataDirectory: URL
    let initializationDirectory: URL
    let output: URL
    let runID: String
    let epochs: Int
    let batchSize: Int
    let seed: Int
    let warmupSteps: Int
    let maxTrainBatches: Int

    static func parse(_ arguments: [String]) throws -> Options {
        var values: [String: String] = [:]
        var index = 1
        while index < arguments.count {
            let key = arguments[index]
            if key == "--help" || key == "-h" {
                print("""
                Usage: WikiText2RNNBenchmark --data-dir PATH --initialization-dir PATH --output PATH --run-id ID [options]
                  --epochs N              Training epochs (default: 20)
                  --batch-size N          Batch size (default: 20)
                  --seed N                Random seed (default: 42)
                  --warmup-steps N        Steps omitted from latency statistics (default: 10)
                  --max-train-batches N   Bounded smoke-test limit (default: 0 = full split)
                """)
                exit(0)
            }
            guard key.hasPrefix("--"), index + 1 < arguments.count else {
                throw BenchmarkError.invalidData("Expected a value after \(key).")
            }
            values[key] = arguments[index + 1]
            index += 2
        }
        func int(_ key: String, default defaultValue: Int) throws -> Int {
            guard let value = values[key] else { return defaultValue }
            guard let parsed = Int(value) else { throw BenchmarkError.invalidData("\(key) must be an integer.") }
            return parsed
        }
        guard let data = values["--data-dir"], let initDir = values["--initialization-dir"],
              let output = values["--output"], let runID = values["--run-id"] else {
            throw BenchmarkError.invalidData("--data-dir, --initialization-dir, --output, and --run-id are required.")
        }
        let options = Options(
            dataDirectory: URL(fileURLWithPath: data, isDirectory: true),
            initializationDirectory: URL(fileURLWithPath: initDir, isDirectory: true),
            output: URL(fileURLWithPath: output),
            runID: runID,
            epochs: try int("--epochs", default: 20),
            batchSize: try int("--batch-size", default: defaultBatchSize),
            seed: try int("--seed", default: 42),
            warmupSteps: try int("--warmup-steps", default: 10),
            maxTrainBatches: try int("--max-train-batches", default: 0)
        )
        guard options.epochs > 0, options.batchSize > 0, options.seed >= 0,
              options.warmupSteps >= 0, options.maxTrainBatches >= 0 else {
            throw BenchmarkError.invalidData("Epochs and batch size must be positive; other counts cannot be negative.")
        }
        return options
    }
}

private func float32Values(_ path: URL, count: Int) throws -> [Float32] {
    let data = try Data(contentsOf: path, options: .mappedIfSafe)
    guard data.count == count * MemoryLayout<Float32>.size else {
        throw BenchmarkError.invalidData("Unexpected float32 byte count in \(path.path).")
    }
    return data.withUnsafeBytes { raw in
        (0..<count).map { raw.loadUnaligned(fromByteOffset: $0 * 4, as: Float32.self) }
    }
}

private func int32Values(_ path: URL, count: Int) throws -> [Int32] {
    let data = try Data(contentsOf: path, options: .mappedIfSafe)
    guard data.count == count * MemoryLayout<Int32>.size else {
        throw BenchmarkError.invalidData("Unexpected int32 byte count in \(path.path).")
    }
    return data.withUnsafeBytes { raw in
        (0..<count).map { raw.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self) }
    }
}

private final class TanhRNNLayer: Module {
    let inputProjection: Linear
    let recurrentProjection: Linear

    init(layer: Int, inputSize: Int, directory: URL) throws {
        let prefix = "rnn-layer\(layer)"
        inputProjection = Linear(
            weight: MLXArray(try float32Values(
                directory.appendingPathComponent("\(prefix)-weight-ih-f32le.bin"),
                count: hiddenSize * inputSize
            )).reshaped([hiddenSize, inputSize]),
            bias: MLXArray(try float32Values(
                directory.appendingPathComponent("\(prefix)-bias-ih-f32le.bin"), count: hiddenSize
            ))
        )
        recurrentProjection = Linear(
            weight: MLXArray(try float32Values(
                directory.appendingPathComponent("\(prefix)-weight-hh-f32le.bin"),
                count: hiddenSize * hiddenSize
            )).reshaped([hiddenSize, hiddenSize]),
            bias: MLXArray(try float32Values(
                directory.appendingPathComponent("\(prefix)-bias-hh-f32le.bin"), count: hiddenSize
            ))
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let projected = inputProjection(input)
        var hidden = MLXArray.zeros([input.dim(0), hiddenSize])
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(input.dim(1))
        for step in 0..<input.dim(1) {
            hidden = tanh(projected[.ellipsis, step, 0...] + recurrentProjection(hidden))
            outputs.append(hidden)
        }
        return stacked(outputs, axis: 1)
    }
}

private final class WikiTextRNN: Module, TrainableModel {
    typealias Input = MLXArray
    typealias Output = MLXArray

    let embedding: Embedding
    let dropout = Dropout(p: 0.25)
    let recurrent1: TanhRNNLayer
    let recurrent2: TanhRNNLayer
    let output: Linear
    let vocabularySize: Int

    init(vocabularySize: Int, initializationDirectory: URL) throws {
        self.vocabularySize = vocabularySize
        embedding = Embedding(weight: MLXArray(try float32Values(
            initializationDirectory.appendingPathComponent("embedding-weight-f32le.bin"),
            count: vocabularySize * embeddingSize
        )).reshaped([vocabularySize, embeddingSize]))
        recurrent1 = try TanhRNNLayer(layer: 0, inputSize: embeddingSize, directory: initializationDirectory)
        recurrent2 = try TanhRNNLayer(layer: 1, inputSize: hiddenSize, directory: initializationDirectory)
        output = Linear(
            weight: MLXArray(try float32Values(
                initializationDirectory.appendingPathComponent("output-weight-f32le.bin"),
                count: vocabularySize * hiddenSize
            )).reshaped([vocabularySize, hiddenSize]),
            bias: MLXArray(try float32Values(
                initializationDirectory.appendingPathComponent("output-bias-f32le.bin"), count: vocabularySize
            ))
        )
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        var values = dropout(embedding(tokens))
        values = recurrent1(values)
        values = recurrent2(values)
        values = output(values)
        return values.reshaped([tokens.dim(0) * tokens.dim(1), vocabularySize])
    }
}

private final class TokenAccuracy: Metric, @unchecked Sendable {
    typealias Output = MLXArray
    typealias Target = MLXArray

    let name = "token_accuracy"
    private var correct = 0
    private var total = 0

    func update(output: MLXArray, target: MLXArray) {
        let flattenedTargets = target.reshaped([-1])
        correct += (output.argMax(axis: -1) .== flattenedTargets).sum().item(Int.self)
        total += flattenedTargets.size
    }

    func compute() -> Float { total == 0 ? 0 : Float(correct) / Float(total) }
    func reset() { correct = 0; total = 0 }
}

private struct SequenceCrossEntropy: Objective, Sendable {
    typealias Output = MLXArray
    typealias Target = MLXArray

    func callAsFunction(output: MLXArray, target: MLXArray) -> MLXArray {
        CrossEntropyLoss()(predictions: output, targets: target.reshaped([-1]))
    }
}

private final class EpochTiming: Callback, @unchecked Sendable {
    private var batchStart = 0.0
    private var epochStart = 0.0
    private let displayEpoch: Int
    private(set) var batchDurations: [Double] = []
    private(set) var epochDuration = 0.0

    init(displayEpoch: Int) { self.displayEpoch = displayEpoch }

    func onBatchBegin(state: TrainingState) { batchStart = ProcessInfo.processInfo.systemUptime }
    func onBatchEnd(state: TrainingState) { batchDurations.append(ProcessInfo.processInfo.systemUptime - batchStart) }
    func onEpochBegin(state: TrainingState) { epochStart = ProcessInfo.processInfo.systemUptime }
    func onEpochEnd(state: TrainingState) {
        epochDuration = ProcessInfo.processInfo.systemUptime - epochStart
        if let loss = state.logs["loss"], let accuracy = state.logs["token_accuracy"],
           let validationLoss = state.logs["val_loss"],
           let validationAccuracy = state.logs["val_token_accuracy"] {
            print(String(format: "Epoch %d — loss %.5f, ppl %.2f, token_accuracy %.3f, val_loss %.5f, val_ppl %.2f, val_accuracy %.3f, %.2fs",
                         displayEpoch, loss, exp(min(loss, 80)), accuracy, validationLoss,
                         exp(min(validationLoss, 80)), validationAccuracy,
                         epochDuration))
            fflush(stdout)
        }
    }
}

private func loadSplit(_ directory: URL, split: String, count: Int) throws -> InMemoryDataset {
    let values = count * sequenceLength
    let features = MLXArray(try int32Values(
        directory.appendingPathComponent("\(split)-inputs-i32le.bin"), count: values
    )).reshaped([count, sequenceLength])
    let labels = MLXArray(try int32Values(
        directory.appendingPathComponent("\(split)-targets-i32le.bin"), count: values
    )).reshaped([count, sequenceLength])
    return InMemoryDataset(features: features, labels: labels)
}

private func percentile(_ values: [Double], _ fraction: Double) -> Double? {
    guard !values.isEmpty else { return nil }
    let sorted = values.sorted()
    let index = min(sorted.count - 1, max(0, Int((Double(sorted.count - 1) * fraction).rounded())))
    return sorted[index]
}

private func sysctlString(_ name: String) -> String? {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
    var bytes = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &bytes, &size, nil, 0) == 0 else { return nil }
    return String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

private func peakResidentBytes() -> UInt64 {
    var usage = rusage()
    return getrusage(RUSAGE_SELF, &usage) == 0 ? UInt64(usage.ru_maxrss) : 0
}

private func writeReport(_ report: [String: Any], epochs: [[String: Any]], to url: URL) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys, .fragmentsAllowed])
    try data.write(to: url, options: .atomic)
    var csv = "epoch,train_loss,train_perplexity,train_token_accuracy,validation_loss,validation_perplexity,validation_token_accuracy,tokens,elapsed_seconds\n"
    for row in epochs {
        csv += "\(row["epoch"]!),\(row["train_loss"]!),\(row["train_perplexity"]!),\(row["train_token_accuracy"]!),\(row["validation_loss"]!),\(row["validation_perplexity"]!),\(row["validation_token_accuracy"]!),\(row["tokens"]!),\(row["elapsed_seconds"]!)\n"
    }
    try csv.write(to: url.deletingLastPathComponent().appendingPathComponent("mlxkit-epochs.csv"), atomically: true, encoding: .utf8)
}

@main
private struct WikiText2RNNBenchmark {
    static func main() async throws {
        let options = try Options.parse(CommandLine.arguments)
        let dataURL = options.dataDirectory
        let dataManifestURL = dataURL.appendingPathComponent("manifest.json")
        guard let dataManifest = try JSONSerialization.jsonObject(with: Data(contentsOf: dataManifestURL)) as? [String: Any],
              let vocabularySize = dataManifest["vocabulary_size"] as? Int,
              let counts = dataManifest["sequence_counts"] as? [String: Int],
              let targetTokenCounts = dataManifest["target_token_counts"] as? [String: Int],
              let sourceHashes = dataManifest["source_file_sha256"] as? [String: String],
              let preparedFiles = dataManifest["files"] as? [String: [String: Any]],
              let vocabularyHash = dataManifest["vocabulary_sha256"] as? String else {
            throw BenchmarkError.invalidData("Invalid prepared WikiText-2 manifest.")
        }
        var initializationManifest = (try JSONSerialization.jsonObject(with: Data(
            contentsOf: options.initializationDirectory.appendingPathComponent("manifest.json")
        ))) as? [String: Any] ?? [:]
        guard initializationManifest["vocabulary_size"] as? Int == vocabularySize,
              initializationManifest["dataset_vocabulary_sha256"] as? String == vocabularyHash else {
            throw BenchmarkError.invalidData("Dataset and initialization vocabulary do not match.")
        }
        initializationManifest["manifest_path"] = options.initializationDirectory.appendingPathComponent("manifest.json").path

        var training = try loadSplit(dataURL, split: "train", count: counts["train"] ?? 0)
        if options.maxTrainBatches > 0 {
            let limitedCount = min(training.count, options.maxTrainBatches * options.batchSize)
            training = InMemoryDataset(
                features: training.features[0..<limitedCount],
                labels: training.labels[0..<limitedCount]
            )
        }
        let validation = try loadSplit(dataURL, split: "validation", count: counts["validation"] ?? 0)
        let test = try loadSplit(dataURL, split: "test", count: counts["test"] ?? 0)

        seed(UInt64(options.seed))
        let model = try WikiTextRNN(vocabularySize: vocabularySize, initializationDirectory: options.initializationDirectory)
        func makeConfiguration() -> TrainingConfiguration<MLXArray, MLXArray> {
            TrainingConfiguration(
                optimizer: SGD(learningRate: 0.5),
                objective: SequenceCrossEntropy(),
                metrics: [TokenAccuracy()]
            )
        }
        let engine = TrainingEngine()
        let validationLoader = try DataLoader(
            source: validation, batchSize: options.batchSize, shuffle: .none, dropLast: true
        )
        let testLoader = try DataLoader(
            source: test, batchSize: options.batchSize, shuffle: .none, dropLast: true
        )
        let initialMetrics = try await engine.evaluate(model, configuration: makeConfiguration(), data: validationLoader)
        guard let initialValidationLoss = initialMetrics["loss"],
              let initialValidationAccuracy = initialMetrics["token_accuracy"] else {
            throw BenchmarkError.missingMetric("initial validation loss/token accuracy")
        }

        print("Training \(options.epochs) epochs on mlx-\(Device.defaultDevice().deviceType.rawValue) (vocab=\(vocabularySize), sequences=\(training.count))...")
        fflush(stdout)
        let fitStarted = ProcessInfo.processInfo.systemUptime
        var allBatchDurations: [Double] = []
        var epochs: [[String: Any]] = []
        var totalTrainingTokens = 0
        for epoch in 0..<options.epochs {
            let trainLoader = try DataLoader(
                source: training,
                batchSize: options.batchSize,
                shuffle: .full(seed: UInt64(options.seed + epoch)),
                dropLast: true
            )
            let timing = EpochTiming(displayEpoch: epoch + 1)
            let history = try await engine.fit(
                model,
                configuration: makeConfiguration(),
                data: trainLoader,
                epochs: 1,
                validationData: validationLoader,
                callbacks: [timing]
            )
            guard let logs = history.epochs.first,
                  let trainLoss = logs["loss"],
                  let trainAccuracy = logs["token_accuracy"],
                  let validationLoss = logs["val_loss"],
                  let validationAccuracy = logs["val_token_accuracy"] else {
                throw BenchmarkError.missingMetric("epoch loss, validation loss, and token accuracy")
            }
            let epochTokens = (training.count / options.batchSize) * options.batchSize * sequenceLength
            totalTrainingTokens += epochTokens
            allBatchDurations.append(contentsOf: timing.batchDurations)
            epochs.append([
                "epoch": epoch + 1,
                "train_loss": trainLoss,
                "train_perplexity": exp(min(trainLoss, 80)),
                "train_token_accuracy": trainAccuracy,
                "validation_loss": validationLoss,
                "validation_perplexity": exp(min(validationLoss, 80)),
                "validation_token_accuracy": validationAccuracy,
                "tokens": epochTokens,
                "elapsed_seconds": timing.epochDuration,
            ])
        }
        let fitWallSeconds = ProcessInfo.processInfo.systemUptime - fitStarted
        let testMetrics = try await engine.evaluate(model, configuration: makeConfiguration(), data: testLoader)
        guard let testLoss = testMetrics["loss"], let testAccuracy = testMetrics["token_accuracy"] else {
            throw BenchmarkError.missingMetric("test loss and token accuracy")
        }
        let testPerplexity = exp(min(testLoss, 80))
        let validationLoss = epochs.last?["validation_loss"] as? Float ?? Float.infinity
        let validationPerplexity = exp(min(validationLoss, 80))
        let batchSeconds = allBatchDurations.reduce(0, +)
        let steadySteps = Array(allBatchDurations.dropFirst(options.warmupSteps))
        let memory = Memory.snapshot()
        let parameters = model.parameters().flattenedValues().reduce(0) { $0 + $1.size }
        let gitRevision = ProcessInfo.processInfo.environment["MLXKIT_GIT_REVISION"] ?? "unknown"
        let upstreamRevision = dataManifest["upstream_revision"] as? String ?? "unknown"
        let tokenization = dataManifest["tokenizer"] as? String ?? "unknown"
        let vocabularyPolicy = dataManifest["vocabulary_policy"] as? String ?? "unknown"
        let windowing = dataManifest["windowing"] as? String ?? "unknown"
        let physicalMemory = ProcessInfo.processInfo.physicalMemory

        let report: [String: Any] = [
            "schema_version": 1,
            "run_id": options.runID,
            "implementation": "mlxkit",
            "created_at_utc": ISO8601DateFormatter().string(from: Date()),
            "git_revision": gitRevision,
            "git_worktree_dirty": ProcessInfo.processInfo.environment["MLXKIT_GIT_DIRTY"] == "true",
            "upstream": [
                "repository": "https://github.com/liux2/RNN-on-wikitext2",
                "revision": upstreamRevision,
                "source_file_sha256": sourceHashes,
            ],
            "system": [
                "platform": "macOS",
                "machine": sysctlString("hw.model") ?? ProcessInfo.processInfo.hostName,
                "hardware_model": sysctlString("hw.model") ?? "unknown",
                "processor": sysctlString("machdep.cpu.brand_string") ?? "unknown",
                "macos_version": ProcessInfo.processInfo.operatingSystemVersionString,
                "physical_memory_bytes": physicalMemory,
                "peak_process_resident_bytes": peakResidentBytes(),
            ],
            "software": [
                "swift": ProcessInfo.processInfo.environment["WIKITEXT_SWIFT_VERSION"] ?? "unknown",
                "mlxkit": gitRevision,
                "mlx_swift": "0.32.3",
            ],
            "backend": [
                "name": "mlx-\(Device.defaultDevice().deviceType.rawValue)",
                "mlx_peak_active_memory_bytes": memory.peakMemory,
                "mlx_cache_memory_bytes": memory.cacheMemory,
            ],
            "dataset": [
                "source_url": "https://github.com/liux2/RNN-on-wikitext2",
                "upstream_revision": upstreamRevision,
                "source_file_sha256": sourceHashes,
                "manifest": dataManifestURL.path,
                "prepared_file_sha256": preparedFiles.compactMapValues { $0["sha256"] as? String },
                "vocabulary_size": vocabularySize,
                "vocabulary_sha256": vocabularyHash,
                "sequence_length": sequenceLength,
                "train_sequences": training.count,
                "validation_sequences": counts["validation"] ?? 0,
                "test_sequences": counts["test"] ?? 0,
                "initial_validation_tokens": (counts["validation"] ?? 0) / options.batchSize * options.batchSize * sequenceLength,
                "test_tokens": (counts["test"] ?? 0) / options.batchSize * options.batchSize * sequenceLength,
                "preprocessing": tokenization,
                "vocabulary_policy": vocabularyPolicy,
                "windowing": windowing,
                "target_token_counts": targetTokenCounts,
            ],
            "initialization": initializationManifest,
            "configuration": [
                "architecture": "Embedding -> Dropout -> 2-layer tanh RNN -> Linear vocabulary projection",
                "embedding_size": embeddingSize,
                "hidden_size": hiddenSize,
                "recurrent_layers": 2,
                "parameters": parameters,
                "embedding_initializer": "shared uniform[-0.1,0.1]",
                "rnn_and_output_initializer": "shared uniform[-1/sqrt(hidden_size),+1/sqrt(hidden_size)]",
                "dropout_probability": 0.25,
                "dropout_location": "embedding output only",
                "initial_hidden_state": "zeros per batch; no state carried between windows",
                "loss": "mean cross entropy over next-token targets",
                "optimizer": "SGD",
                "learning_rate": 0.5,
                "momentum": 0.0,
                "gradient_clipping": NSNull(),
                "batch_size": options.batchSize,
                "sequence_length": sequenceLength,
                "epochs": options.epochs,
                "seed": options.seed,
                "train_shuffle": "shared SplitMix64 Fisher-Yates permutation per epoch; same window order in both implementations",
                "drop_last": true,
                "max_train_batches": options.maxTrainBatches,
            ],
            "measurements": [
                "initial_validation_loss": initialValidationLoss,
                "initial_validation_perplexity": exp(min(initialValidationLoss, 80)),
                "initial_validation_token_accuracy": initialValidationAccuracy,
                "fit_wall_seconds_including_validation": fitWallSeconds,
                "training_batch_seconds": batchSeconds,
                "training_tokens": totalTrainingTokens,
                "training_tokens_per_second_batch_time": batchSeconds > 0 ? Double(totalTrainingTokens) / batchSeconds : 0,
                "training_tokens_per_second_fit_wall": fitWallSeconds > 0 ? Double(totalTrainingTokens) / fitWallSeconds : 0,
                "training_step_latency_median_seconds_excluding_warmup": percentile(steadySteps, 0.5) ?? 0,
                "training_step_latency_p90_seconds_excluding_warmup": percentile(steadySteps, 0.9) ?? 0,
                "warmup_steps_excluded_from_latency": min(options.warmupSteps, allBatchDurations.count),
                "validation_loss": validationLoss,
                "validation_perplexity": validationPerplexity,
                "validation_token_accuracy": epochs.last?["validation_token_accuracy"] ?? 0,
                "test_loss": testLoss,
                "test_perplexity": testPerplexity,
                "test_token_accuracy": testAccuracy,
                "peak_process_resident_memory_method": "getrusage(RUSAGE_SELF).ru_maxrss; macOS bytes; MLX active-memory peak reported separately",
            ],
            "epochs": epochs,
        ]
        try writeReport(report, epochs: epochs, to: options.output)
        print(String(format: "Validation perplexity: %.3f", Double(validationPerplexity)))
        print(String(format: "Test perplexity: %.3f", testPerplexity))
        print(String(format: "Training throughput: %.1f tokens/s", Double(totalTrainingTokens) / max(batchSeconds, 1e-9)))
        print("MLX backend: mlx-\(Device.defaultDevice().deviceType.rawValue)")
        print("Report: \(options.output.path)")
        fflush(stdout)
    }
}
