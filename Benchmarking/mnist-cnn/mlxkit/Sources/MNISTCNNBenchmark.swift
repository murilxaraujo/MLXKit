import Darwin
import Foundation
import MLX
import MLXKit
import MLXNN
import MLXOptimizers

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

    static func parse(_ arguments: [String]) throws -> Options {
        var values: [String: String] = [:]
        var index = 1
        while index < arguments.count {
            let key = arguments[index]
            if key == "--help" || key == "-h" {
                print("""
                Usage: MNISTCNNBenchmark --data-dir PATH --initialization-dir PATH --output PATH --run-id ID [options]
                  --epochs N          Training epochs (default: 5)
                  --batch-size N      Batch size (default: 32)
                  --seed N            Random seed (default: 42)
                  --warmup-steps N    Steps omitted from latency statistics (default: 10)
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
            epochs: try int("--epochs", default: 5),
            batchSize: try int("--batch-size", default: 32),
            seed: try int("--seed", default: 42),
            warmupSteps: try int("--warmup-steps", default: 10)
        )
        guard options.epochs > 0, options.batchSize > 0, options.seed >= 0, options.warmupSteps >= 0 else {
            throw BenchmarkError.invalidData("Epochs and batch size must be positive; seed and warmup steps cannot be negative.")
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

private func byteValues(_ path: URL, count: Int) throws -> [UInt8] {
    let data = try Data(contentsOf: path, options: .mappedIfSafe)
    guard data.count == count else { throw BenchmarkError.invalidData("Unexpected byte count in \(path.path).") }
    return Array(data)
}

private final class CNNConvolution: Module {
    let weight: MLXArray
    let bias: MLXArray
    private let stride: Int

    init(name: String, inputChannels: Int, outputChannels: Int, kernel: Int, directory: URL) throws {
        let canonical = try float32Values(
            directory.appendingPathComponent("\(name)-weight-f32le.bin"),
            count: outputChannels * inputChannels * kernel * kernel
        )
        weight = MLXArray(canonical)
            .reshaped([outputChannels, inputChannels, kernel, kernel])
            .transposed(0, 2, 3, 1)
        bias = MLXArray(try float32Values(
            directory.appendingPathComponent("\(name)-bias-f32le.bin"), count: outputChannels
        ))
        stride = 1
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        conv2d(input, weight, stride: .init((stride, stride)), padding: 0) + bias
    }
}

private final class MNISTCNN: Module, TrainableModel {
    typealias Input = MLXArray
    typealias Output = MLXArray

    let conv1: CNNConvolution
    let conv2: CNNConvolution
    let conv3: CNNConvolution
    let pool = MaxPool2d(kernelSize: 2, stride: 2)
    let dropout1 = Dropout(p: 0.5)
    let dropout2 = Dropout(p: 0.5)
    let dropout3 = Dropout(p: 0.5)
    let fc1: Linear
    let fc2: Linear

    init(initializationDirectory: URL) throws {
        conv1 = try CNNConvolution(name: "conv1", inputChannels: 1, outputChannels: 32, kernel: 5, directory: initializationDirectory)
        conv2 = try CNNConvolution(name: "conv2", inputChannels: 32, outputChannels: 32, kernel: 5, directory: initializationDirectory)
        conv3 = try CNNConvolution(name: "conv3", inputChannels: 32, outputChannels: 64, kernel: 5, directory: initializationDirectory)
        let canonicalFC1 = MLXArray(try float32Values(
            initializationDirectory.appendingPathComponent("fc1-weight-f32le.bin"), count: 256 * 576
        ))
        // PyTorch flattens NCHW features as C,H,W; MLX flattens NHWC as H,W,C.
        let mlxFC1 = canonicalFC1.reshaped([256, 64, 3, 3]).transposed(0, 2, 3, 1).reshaped([256, 576])
        fc1 = Linear(
            weight: mlxFC1,
            bias: MLXArray(try float32Values(initializationDirectory.appendingPathComponent("fc1-bias-f32le.bin"), count: 256))
        )
        fc2 = Linear(
            weight: MLXArray(try float32Values(initializationDirectory.appendingPathComponent("fc2-weight-f32le.bin"), count: 10 * 256)).reshaped([10, 256]),
            bias: MLXArray(try float32Values(initializationDirectory.appendingPathComponent("fc2-bias-f32le.bin"), count: 10))
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var values = maximum(conv1(input), 0)
        values = dropout1(maximum(pool(conv2(values)), 0))
        values = dropout2(maximum(pool(conv3(values)), 0))
        values = values.reshaped([values.dim(0), 576])
        values = dropout3(maximum(fc1(values), 0))
        return fc2(values)
    }
}

private final class CNNBenchmarkTiming: Callback, @unchecked Sendable {
    private var batchStart = 0.0
    private var epochStart = 0.0
    private(set) var batchDurations: [Double] = []
    private(set) var epochDurations: [Double] = []

    func onBatchBegin(state: TrainingState) { batchStart = ProcessInfo.processInfo.systemUptime }
    func onBatchEnd(state: TrainingState) { batchDurations.append(ProcessInfo.processInfo.systemUptime - batchStart) }
    func onEpochBegin(state: TrainingState) { epochStart = ProcessInfo.processInfo.systemUptime }
    func onEpochEnd(state: TrainingState) {
        epochDurations.append(ProcessInfo.processInfo.systemUptime - epochStart)
        if let loss = state.logs["loss"], let accuracy = state.logs["accuracy"],
           let validationLoss = state.logs["val_loss"], let validationAccuracy = state.logs["val_accuracy"] {
            print(String(format: "Epoch %d — loss %.5f, accuracy %.3f, val_loss %.5f, val_accuracy %.3f, %.2fs",
                         state.epoch + 1, loss, accuracy, validationLoss, validationAccuracy,
                         ProcessInfo.processInfo.systemUptime - epochStart))
            fflush(stdout)
        }
    }
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

private func loadData(_ directory: URL) throws -> (InMemoryDataset, InMemoryDataset, InMemoryDataset, [String: Any]) {
    let manifestURL = directory.appendingPathComponent("manifest.json")
    guard let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as? [String: Any],
          let sourceURL = manifest["source_url"] as? String,
          let archiveHash = manifest["source_archive_sha256"] as? String,
          let files = manifest["files"] as? [String: [String: Any]] else {
        throw BenchmarkError.invalidData("Invalid MNIST dataset manifest.")
    }
    let rawTrain = try byteValues(directory.appendingPathComponent("train-images-uint8.bin"), count: 60_000 * 784)
    let rawTrainLabels = try byteValues(directory.appendingPathComponent("train-labels-uint8.bin"), count: 60_000)
    let rawTest = try byteValues(directory.appendingPathComponent("test-images-uint8.bin"), count: 10_000 * 784)
    let rawTestLabels = try byteValues(directory.appendingPathComponent("test-labels-uint8.bin"), count: 10_000)
    let trainImages = MLXArray(rawTrain.map { Float32($0) / 255 }).reshaped([60_000, 28, 28, 1])
    let testImages = MLXArray(rawTest.map { Float32($0) / 255 }).reshaped([10_000, 28, 28, 1])
    let trainLabels = MLXArray(rawTrainLabels.map(Int32.init))
    let testLabels = MLXArray(rawTestLabels.map(Int32.init))
    let training = InMemoryDataset(features: trainImages[0..<48_000], labels: trainLabels[0..<48_000])
    let validation = InMemoryDataset(features: trainImages[48_000..<60_000], labels: trainLabels[48_000..<60_000])
    let test = InMemoryDataset(features: testImages, labels: testLabels)
    return (training, validation, test, [
        "source_url": sourceURL,
        "source_archive_sha256": archiveHash,
        "manifest": manifestURL.path,
        "preparation_python_version": manifest["preparation_python_version"] as? String ?? "unknown",
        "prepared_file_sha256": files.compactMapValues { $0["sha256"] as? String },
    ])
}

private func saveReport(_ report: [String: Any], epochs: [[String: Any]], to url: URL) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys, .fragmentsAllowed])
    try data.write(to: url, options: .atomic)
    var csv = "epoch,train_loss,train_accuracy,validation_loss,validation_accuracy,elapsed_seconds\n"
    for row in epochs {
        csv += "\(row["epoch"]!),\(row["train_loss"]!),\(row["train_accuracy"]!),\(row["validation_loss"]!),\(row["validation_accuracy"]!),\(row["elapsed_seconds"]!)\n"
    }
    try csv.write(to: url.deletingLastPathComponent().appendingPathComponent("mlxkit-epochs.csv"), atomically: true, encoding: .utf8)
}

@main
private struct MNISTCNNBenchmark {
    static func main() async throws {
        let options = try Options.parse(CommandLine.arguments)
        let (training, validation, test, datasetMetadata) = try loadData(options.dataDirectory)
        seed(UInt64(options.seed))
        let model = try MNISTCNN(initializationDirectory: options.initializationDirectory)
        let configuration = TrainingConfiguration(
            optimizer: Adam(learningRate: 0.001, betas: (0.9, 0.999), eps: 1e-8, biasCorrection: true),
            objective: CrossEntropyLoss(),
            metrics: [Accuracy()]
        )
        let trainingLoader = try DataLoader(source: training, batchSize: options.batchSize, shuffle: .none)
        let validationLoader = try DataLoader(source: validation, batchSize: options.batchSize)
        let testLoader = try DataLoader(source: test, batchSize: options.batchSize)
        let engine = TrainingEngine()
        let initialMetrics = try await engine.evaluate(model, configuration: configuration, data: validationLoader)
        guard let initialLoss = initialMetrics["loss"], let initialAccuracy = initialMetrics["accuracy"] else {
            throw BenchmarkError.missingMetric("initial validation loss/accuracy")
        }
        let timing = CNNBenchmarkTiming()
        let backend = Device.defaultDevice().deviceType.rawValue
        print("Training \(options.epochs) epochs on mlx-\(backend)...")
        fflush(stdout)
        let started = ProcessInfo.processInfo.systemUptime
        let history = try await engine.fit(
            model,
            configuration: configuration,
            data: trainingLoader,
            epochs: options.epochs,
            validationData: validationLoader,
            callbacks: [timing]
        )
        let fitSeconds = ProcessInfo.processInfo.systemUptime - started
        let testMetrics = try await engine.evaluate(model, configuration: configuration, data: testLoader)
        guard let testLoss = testMetrics["loss"], let testAccuracy = testMetrics["accuracy"] else {
            throw BenchmarkError.missingMetric("test loss/accuracy")
        }
        let epochRows: [[String: Any]] = history.epochs.enumerated().map { index, logs in
            [
                "epoch": index + 1,
                "train_loss": logs["loss"] ?? .nan,
                "train_accuracy": logs["accuracy"] ?? .nan,
                "validation_loss": logs["val_loss"] ?? .nan,
                "validation_accuracy": logs["val_accuracy"] ?? .nan,
                "elapsed_seconds": timing.epochDurations.indices.contains(index) ? timing.epochDurations[index] : 0,
            ]
        }
        let batchSeconds = timing.batchDurations.reduce(0, +)
        let steadySteps = Array(timing.batchDurations.dropFirst(options.warmupSteps))
        let memory = Memory.snapshot()
        let revision = ProcessInfo.processInfo.environment["MLXKIT_GIT_REVISION"] ?? "unknown"
        let fileBytes = (try? Data(contentsOf: options.initializationDirectory.appendingPathComponent("manifest.json"))) ?? Data()
        var initManifest = (try? JSONSerialization.jsonObject(with: fileBytes)) as? [String: Any] ?? [:]
        initManifest["manifest_path"] = options.initializationDirectory.appendingPathComponent("manifest.json").path
        let report: [String: Any] = [
            "schema_version": 1,
            "run_id": options.runID,
            "implementation": "mlxkit",
            "created_at_utc": ISO8601DateFormatter().string(from: Date()),
            "git_revision": revision,
            "git_worktree_dirty": ProcessInfo.processInfo.environment["MLXKIT_GIT_DIRTY"] == "true",
            "system": [
                "platform": "macOS",
                "machine": sysctlString("hw.model") ?? ProcessInfo.processInfo.hostName,
                "hardware_model": sysctlString("hw.model") ?? "unknown",
                "processor": sysctlString("machdep.cpu.brand_string") ?? "unknown",
                "macos_version": ProcessInfo.processInfo.operatingSystemVersionString,
                "physical_memory_bytes": ProcessInfo.processInfo.physicalMemory,
                "peak_process_resident_bytes": peakResidentBytes(),
            ],
            "software": [
                "swift": ProcessInfo.processInfo.environment["MNIST_SWIFT_VERSION"] ?? "unknown",
                "mlxkit": revision,
                "mlx_swift": "0.32.3",
            ],
            "backend": [
                "name": "mlx-\(backend)",
                "mlx_peak_active_memory_bytes": memory.peakMemory,
                "mlx_cache_memory_bytes": memory.cacheMemory,
            ],
            "dataset": [
                "source_url": datasetMetadata["source_url"] ?? "unknown",
                "source_archive_sha256": datasetMetadata["source_archive_sha256"] ?? "unknown",
                "manifest": datasetMetadata["manifest"] ?? "unknown",
                "preparation_python_version": datasetMetadata["preparation_python_version"] ?? "unknown",
                "prepared_file_sha256": datasetMetadata["prepared_file_sha256"] ?? [:],
                "train_examples": 48_000,
                "validation_examples": 12_000,
                "test_examples": 10_000,
                "input_shape": [1, 28, 28],
                "preprocessing": "uint8 pixels cast to float32, divided by 255; NHWC for MLX",
                "validation_split": "last 12000 of original 60000 training examples",
            ],
            "initialization": initManifest,
            "configuration": [
                "architecture": ["Conv2d(1,32,5)", "Conv2d(32,32,5)", "Conv2d(32,64,5)", "Linear(576,256)", "Linear(256,10)"],
                "convolution_padding": "valid",
                "pooling": "2x2 max pooling after conv2 and conv3",
                "hidden_activation": "relu",
                "dropout_probability": 0.5,
                "dropout_locations": ["after second max-pool", "after third max-pool", "after fc1 ReLU"],
                "output": "10 logits",
                "parameters": 228_010,
                "kernel_initializer": "shared seeded Glorot uniform",
                "bias_initializer": "zeros",
                "loss": "sparse categorical cross-entropy from logits",
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1e-8,
                "bias_correction": true,
                "batch_size": options.batchSize,
                "epochs": options.epochs,
                "seed": options.seed,
                "shuffle": false,
            ],
            "measurements": [
                "initial_validation_loss": initialLoss,
                "initial_validation_accuracy": initialAccuracy,
                "fit_wall_seconds_including_validation": fitSeconds,
                "training_batch_seconds": batchSeconds,
                "training_examples_per_second_batch_time": batchSeconds > 0 ? Double(48_000 * options.epochs) / batchSeconds : 0,
                "training_step_latency_median_seconds_excluding_warmup": percentile(steadySteps, 0.5) ?? 0,
                "training_step_latency_p90_seconds_excluding_warmup": percentile(steadySteps, 0.9) ?? 0,
                "warmup_steps_excluded_from_latency": min(options.warmupSteps, timing.batchDurations.count),
                "test_loss": testLoss,
                "test_accuracy": testAccuracy,
                "peak_process_resident_memory_method": "getrusage(RUSAGE_SELF).ru_maxrss; macOS bytes; MLX active-memory peak reported separately",
            ],
            "epochs": epochRows,
        ]
        try saveReport(report, epochs: epochRows, to: options.output)
        print(String(format: "Test loss: %.6f", testLoss))
        print(String(format: "Test accuracy: %.4f%%", testAccuracy * 100))
        print(String(format: "Training throughput: %.1f examples/s", Double(48_000 * options.epochs) / max(batchSeconds, 1e-9)))
        print("MLX backend: mlx-\(backend)")
        print("Report: \(options.output.path)")
        fflush(stdout)
    }
}
