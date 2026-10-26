import Darwin
import Foundation
import MLX
import MLXKit
import MLXNN
import MLXOptimizers

private enum BenchmarkError: Error, LocalizedError {
    case invalidArgument(String)
    case invalidData(String)
    case missingMetric(String)

    var errorDescription: String? {
        switch self {
        case .invalidArgument(let message), .invalidData(let message): message
        case .missingMetric(let name): "Training did not produce the expected metric '\(name)'."
        }
    }
}

private struct Options {
    var dataDirectory: URL
    var initializationDirectory: URL
    var output: URL
    var runID: String
    var epochs = 10
    var batchSize = 32
    var seed = 42
    var warmupSteps = 10

    static func parse(_ arguments: [String]) throws -> Options {
        var values: [String: String] = [:]
        var index = 1
        while index < arguments.count {
            let key = arguments[index]
            guard key.hasPrefix("--"), index + 1 < arguments.count else {
                if key == "--help" || key == "-h" {
                    print("""
                    Usage: MNISTMLPBenchmark --data-dir PATH --initialization-dir PATH --output PATH --run-id ID [options]
                      --epochs N          Training epochs (default: 10)
                      --batch-size N      Batch size (default: 32)
                      --seed N            Random seed (default: 42)
                      --warmup-steps N    Steps omitted from latency statistics (default: 10)
                    """)
                    exit(0)
                }
                throw BenchmarkError.invalidArgument("Expected a value after \(key).")
            }
            values[key] = arguments[index + 1]
            index += 2
        }

        guard let dataPath = values["--data-dir"],
              let initializationPath = values["--initialization-dir"],
              let outputPath = values["--output"],
              let runID = values["--run-id"]
        else {
            throw BenchmarkError.invalidArgument("--data-dir, --output, and --run-id are required; use --help.")
        }

        var options = Options(
            dataDirectory: URL(fileURLWithPath: dataPath, isDirectory: true),
            initializationDirectory: URL(fileURLWithPath: initializationPath, isDirectory: true),
            output: URL(fileURLWithPath: outputPath),
            runID: runID
        )
        options.epochs = try integer(values["--epochs"], name: "--epochs", default: options.epochs)
        options.batchSize = try integer(values["--batch-size"], name: "--batch-size", default: options.batchSize)
        options.seed = try integer(values["--seed"], name: "--seed", default: options.seed)
        options.warmupSteps = try integer(values["--warmup-steps"], name: "--warmup-steps", default: options.warmupSteps)
        guard options.epochs > 0, options.batchSize > 0, options.seed >= 0, options.warmupSteps >= 0 else {
            throw BenchmarkError.invalidArgument("Epochs and batch size must be positive; seed and warmup steps cannot be negative.")
        }
        return options
    }

    private static func integer(_ value: String?, name: String, default defaultValue: Int) throws -> Int {
        guard let value else { return defaultValue }
        guard let parsed = Int(value) else {
            throw BenchmarkError.invalidArgument("\(name) must be an integer, got '\(value)'.")
        }
        return parsed
    }
}

private final class MNISTMLP: Module, TrainableModel {
    typealias Input = MLXArray
    typealias Output = MLXArray

    let first: Linear
    let second: Linear
    let output: Linear

    init(initializationDirectory: URL) throws {
        first = try Self.linear(name: "dense1", input: 784, output: 256, directory: initializationDirectory)
        second = try Self.linear(name: "dense2", input: 256, output: 256, directory: initializationDirectory)
        output = try Self.linear(name: "dense3", input: 256, output: 10, directory: initializationDirectory)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        output(maximum(second(maximum(first(input), 0)), 0))
    }

    private static func linear(name: String, input: Int, output: Int, directory: URL) throws -> Linear {
        let weights = try float32Values(
            at: directory.appendingPathComponent("\(name)-weight-f32le.bin"),
            count: input * output
        )
        let biases = try float32Values(
            at: directory.appendingPathComponent("\(name)-bias-f32le.bin"),
            count: output
        )
        return Linear(
            weight: MLXArray(weights).reshaped([output, input]),
            bias: MLXArray(biases)
        )
    }
}

private final class BenchmarkTiming: Callback, @unchecked Sendable {
    private var batchStart: TimeInterval = 0
    private var epochStart: TimeInterval = 0
    private(set) var batchDurations: [Double] = []
    private(set) var epochDurations: [Double] = []

    func onBatchBegin(state: TrainingState) {
        batchStart = ProcessInfo.processInfo.systemUptime
    }

    func onBatchEnd(state: TrainingState) {
        batchDurations.append(ProcessInfo.processInfo.systemUptime - batchStart)
    }

    func onEpochBegin(state: TrainingState) {
        epochStart = ProcessInfo.processInfo.systemUptime
    }

    func onEpochEnd(state: TrainingState) {
        epochDurations.append(ProcessInfo.processInfo.systemUptime - epochStart)
    }
}

private struct EpochRecord: Codable {
    let epoch: Int
    let trainLoss: Float
    let trainAccuracy: Float
    let validationLoss: Float
    let validationAccuracy: Float
    let elapsedSeconds: Double

    enum CodingKeys: String, CodingKey {
        case epoch
        case trainLoss = "train_loss"
        case trainAccuracy = "train_accuracy"
        case validationLoss = "validation_loss"
        case validationAccuracy = "validation_accuracy"
        case elapsedSeconds = "elapsed_seconds"
    }
}

private struct RunReport: Codable {
    struct InitializationFile: Codable {
        let shape: [Int]
        let bytes: Int
        let sha256: String
    }

    struct Initialization: Codable {
        let seed: Int
        let algorithm: String
        let biasInitializer: String
        let weightLayout: String
        let generator: String
        let pythonVersion: String
        let manifestPath: String
        let files: [String: InitializationFile]

        enum CodingKeys: String, CodingKey {
            case seed, algorithm, generator, files
            case biasInitializer = "bias_initializer"
            case weightLayout = "weight_layout"
            case pythonVersion = "python_version"
            case manifestPath = "manifest_path"
        }
    }

    struct System: Codable {
        let platform: String
        let machine: String
        let hardwareModel: String?
        let processor: String?
        let macOSVersion: String
        let physicalMemoryBytes: UInt64
        let peakProcessResidentBytes: UInt64

        enum CodingKeys: String, CodingKey {
            case platform, machine, processor
            case hardwareModel = "hardware_model"
            case macOSVersion = "macos_version"
            case physicalMemoryBytes = "physical_memory_bytes"
            case peakProcessResidentBytes = "peak_process_resident_bytes"
        }
    }

    struct Backend: Codable {
        let name: String
        let mlxPeakActiveMemoryBytes: Int
        let mlxCacheMemoryBytes: Int

        enum CodingKeys: String, CodingKey {
            case name
            case mlxPeakActiveMemoryBytes = "mlx_peak_active_memory_bytes"
            case mlxCacheMemoryBytes = "mlx_cache_memory_bytes"
        }
    }

    struct Software: Codable {
        let swift: String
        let mlxKit: String
        let mlxSwift: String

        enum CodingKeys: String, CodingKey {
            case swift
            case mlxKit = "mlxkit"
            case mlxSwift = "mlx_swift"
        }
    }

    struct Dataset: Codable {
        let sourceURL: String
        let sourceArchiveSHA256: String
        let manifest: String
        let preparationPythonVersion: String
        let preparedFileSHA256: [String: String]
        let trainExamples: Int
        let validationExamples: Int
        let testExamples: Int
        let inputShape: [Int]
        let preprocessing: String
        let validationSplit: String

        enum CodingKeys: String, CodingKey {
            case manifest, preprocessing
            case sourceURL = "source_url"
            case sourceArchiveSHA256 = "source_archive_sha256"
            case preparationPythonVersion = "preparation_python_version"
            case preparedFileSHA256 = "prepared_file_sha256"
            case trainExamples = "train_examples"
            case validationExamples = "validation_examples"
            case testExamples = "test_examples"
            case inputShape = "input_shape"
            case validationSplit = "validation_split"
        }
    }

    struct Configuration: Codable {
        let architecture: [Int]
        let hiddenActivation: String
        let output: String
        let parameters: Int
        let kernelInitializer: String
        let biasInitializer: String
        let loss: String
        let optimizer: String
        let learningRate: Float
        let beta1: Float
        let beta2: Float
        let epsilon: Float
        let biasCorrection: Bool
        let batchSize: Int
        let epochs: Int
        let seed: Int
        let shuffle: Bool

        enum CodingKeys: String, CodingKey {
            case architecture, output, parameters, loss, optimizer, epsilon, epochs, seed, shuffle
            case hiddenActivation = "hidden_activation"
            case kernelInitializer = "kernel_initializer"
            case biasInitializer = "bias_initializer"
            case learningRate = "learning_rate"
            case beta1 = "beta_1"
            case beta2 = "beta_2"
            case biasCorrection = "bias_correction"
            case batchSize = "batch_size"
        }
    }

    struct Measurements: Codable {
        let initialValidationLoss: Float
        let initialValidationAccuracy: Float
        let fitWallSecondsIncludingValidation: Double
        let trainingBatchSeconds: Double
        let trainingExamplesPerSecondBatchTime: Double?
        let trainingStepLatencyMedianSecondsExcludingWarmup: Double?
        let trainingStepLatencyP90SecondsExcludingWarmup: Double?
        let warmupStepsExcludedFromLatency: Int
        let testLoss: Float
        let testAccuracy: Float
        let peakProcessResidentMemoryMethod: String

        enum CodingKeys: String, CodingKey {
            case initialValidationLoss = "initial_validation_loss"
            case initialValidationAccuracy = "initial_validation_accuracy"
            case fitWallSecondsIncludingValidation = "fit_wall_seconds_including_validation"
            case trainingBatchSeconds = "training_batch_seconds"
            case trainingExamplesPerSecondBatchTime = "training_examples_per_second_batch_time"
            case trainingStepLatencyMedianSecondsExcludingWarmup = "training_step_latency_median_seconds_excluding_warmup"
            case trainingStepLatencyP90SecondsExcludingWarmup = "training_step_latency_p90_seconds_excluding_warmup"
            case warmupStepsExcludedFromLatency = "warmup_steps_excluded_from_latency"
            case testLoss = "test_loss"
            case testAccuracy = "test_accuracy"
            case peakProcessResidentMemoryMethod = "peak_process_resident_memory_method"
        }
    }

    let schemaVersion: Int
    let runID: String
    let implementation: String
    let createdAtUTC: String
    let gitRevision: String
    let gitWorktreeDirty: Bool
    let system: System
    let software: Software
    let backend: Backend
    let dataset: Dataset
    let initialization: Initialization
    let configuration: Configuration
    let measurements: Measurements
    let epochs: [EpochRecord]

    enum CodingKeys: String, CodingKey {
        case system, software, backend, dataset, initialization, configuration, measurements, epochs
        case schemaVersion = "schema_version"
        case runID = "run_id"
        case implementation
        case createdAtUTC = "created_at_utc"
        case gitRevision = "git_revision"
        case gitWorktreeDirty = "git_worktree_dirty"
    }
}

private func loadBytes(_ path: URL, expectedCount: Int) throws -> [UInt8] {
    let data = try Data(contentsOf: path, options: .mappedIfSafe)
    guard data.count == expectedCount else {
        throw BenchmarkError.invalidData("Unexpected byte count in \(path.path): \(data.count); expected \(expectedCount). Run prepare-data.py again.")
    }
    return Array(data)
}

private func float32Values(at path: URL, count: Int) throws -> [Float32] {
    let data = try Data(contentsOf: path, options: .mappedIfSafe)
    guard data.count == count * MemoryLayout<Float32>.size else {
        throw BenchmarkError.invalidData("Unexpected float32 byte count in \(path.path).")
    }
    return data.withUnsafeBytes { rawBuffer in
        (0..<count).map { index in
            rawBuffer.loadUnaligned(fromByteOffset: index * MemoryLayout<Float32>.size, as: Float32.self)
        }
    }
}

private func loadInitializationManifest(at directory: URL) throws -> RunReport.Initialization {
    struct Manifest: Decodable {
        let seed: Int
        let algorithm: String
        let biasInitializer: String
        let weightLayout: String
        let generator: String
        let pythonVersion: String
        let files: [String: RunReport.InitializationFile]

        enum CodingKeys: String, CodingKey {
            case seed, algorithm, generator, files
            case biasInitializer = "bias_initializer"
            case weightLayout = "weight_layout"
            case pythonVersion = "python_version"
        }
    }

    let manifestURL = directory.appendingPathComponent("manifest.json")
    let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
    return RunReport.Initialization(
        seed: manifest.seed,
        algorithm: manifest.algorithm,
        biasInitializer: manifest.biasInitializer,
        weightLayout: manifest.weightLayout,
        generator: manifest.generator,
        pythonVersion: manifest.pythonVersion,
        manifestPath: manifestURL.path,
        files: manifest.files
    )
}

private func loadDataset(directory: URL) throws -> (InMemoryDataset, InMemoryDataset, InMemoryDataset, [String: Any]) {
    let manifestURL = directory.appendingPathComponent("manifest.json")
    let manifestData = try Data(contentsOf: manifestURL)
    guard let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
          let sourceURL = manifest["source_url"] as? String,
          let archiveHash = manifest["source_archive_sha256"] as? String,
          let files = manifest["files"] as? [String: [String: Any]]
    else {
        throw BenchmarkError.invalidData("Invalid MNIST manifest at \(manifestURL.path).")
    }

    let rawTrainImages = try loadBytes(directory.appendingPathComponent("train-images-uint8.bin"), expectedCount: 60_000 * 784)
    let rawTrainLabels = try loadBytes(directory.appendingPathComponent("train-labels-uint8.bin"), expectedCount: 60_000)
    let rawTestImages = try loadBytes(directory.appendingPathComponent("test-images-uint8.bin"), expectedCount: 10_000 * 784)
    let rawTestLabels = try loadBytes(directory.appendingPathComponent("test-labels-uint8.bin"), expectedCount: 10_000)

    let trainFeatures = MLXArray(rawTrainImages.map { Float32($0) / 255 }).reshaped([60_000, 784])
    let trainTargets = MLXArray(rawTrainLabels.map(Int32.init))
    let testFeatures = MLXArray(rawTestImages.map { Float32($0) / 255 }).reshaped([10_000, 784])
    let testTargets = MLXArray(rawTestLabels.map(Int32.init))

    let training = InMemoryDataset(
        features: trainFeatures[0..<48_000],
        labels: trainTargets[0..<48_000]
    )
    let validation = InMemoryDataset(
        features: trainFeatures[48_000..<60_000],
        labels: trainTargets[48_000..<60_000]
    )
    let test = InMemoryDataset(features: testFeatures, labels: testTargets)
    let datasetReport: [String: Any] = [
        "source_url": sourceURL,
        "source_archive_sha256": archiveHash,
        "manifest": manifestURL.path,
        "preparation_python_version": manifest["preparation_python_version"] as? String ?? "unknown",
        "prepared_file_sha256": files.compactMapValues { $0["sha256"] as? String },
    ]
    return (training, validation, test, datasetReport)
}

private func percentile(_ values: [Double], fraction: Double) -> Double? {
    guard !values.isEmpty else { return nil }
    let sorted = values.sorted()
    let index = min(sorted.count - 1, max(0, Int((Double(sorted.count - 1) * fraction).rounded())))
    return sorted[index]
}

private func systemString(_ name: String) -> String? {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
    var buffer = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
    let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
    return String(decoding: bytes, as: UTF8.self)
}

private func peakResidentBytes() -> UInt64 {
    var usage = rusage()
    guard getrusage(RUSAGE_SELF, &usage) == 0 else { return 0 }
    return UInt64(usage.ru_maxrss)
}

private func writeReport(_ report: RunReport, to url: URL) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(report)
    try data.write(to: url, options: .atomic)

    let csvURL = url.deletingLastPathComponent()
        .appendingPathComponent(url.deletingPathExtension().lastPathComponent + "-epochs.csv")
    var csv = "epoch,train_loss,train_accuracy,validation_loss,validation_accuracy,elapsed_seconds\n"
    for epoch in report.epochs {
        csv += "\(epoch.epoch),\(epoch.trainLoss),\(epoch.trainAccuracy),\(epoch.validationLoss),\(epoch.validationAccuracy),\(epoch.elapsedSeconds)\n"
    }
    try csv.write(to: csvURL, atomically: true, encoding: .utf8)
}

@main
private struct MNISTMLPBenchmark {
    static func main() async throws {
        do {
            try await run(Options.parse(CommandLine.arguments))
        } catch {
            fputs("MNIST MLXKit benchmark failed: \(error.localizedDescription)\n", stderr)
            throw error
        }
    }

    private static func run(_ options: Options) async throws {
        let (trainingData, validationData, testData, datasetMetadata) = try loadDataset(directory: options.dataDirectory)
        seed(UInt64(options.seed))
        let model = try MNISTMLP(initializationDirectory: options.initializationDirectory)
        let optimizer = Adam(
            learningRate: 0.001,
            betas: (0.9, 0.999),
            eps: 1e-7,
            biasCorrection: true
        )
        let configuration = TrainingConfiguration(
            optimizer: optimizer,
            objective: CrossEntropyLoss(),
            metrics: [Accuracy()]
        )
        let engine = TrainingEngine()
        let validationLoader = try DataLoader(source: validationData, batchSize: options.batchSize)
        let testLoader = try DataLoader(source: testData, batchSize: options.batchSize)
        let timing = BenchmarkTiming()
        var epochRecords: [EpochRecord] = []
        let backendName = Device.defaultDevice().deviceType.rawValue
        let initialValidationMetrics = try await engine.evaluate(
            model,
            configuration: configuration,
            data: validationLoader
        )
        guard let initialValidationLoss = initialValidationMetrics["loss"],
              let initialValidationAccuracy = initialValidationMetrics["accuracy"] else {
            throw BenchmarkError.missingMetric("initial validation loss/accuracy")
        }
        let trainingStarted = ProcessInfo.processInfo.systemUptime
        print("Training \(options.epochs) epochs on mlx-\(backendName)...")
        fflush(stdout)

        for epoch in 0..<options.epochs {
            let loader = try DataLoader(
                source: trainingData,
                batchSize: options.batchSize,
                shuffle: .full(seed: UInt64(options.seed + epoch))
            )
            let history = try await engine.fit(
                model,
                configuration: configuration,
                data: loader,
                epochs: 1,
                validationData: validationLoader,
                callbacks: [timing]
            )
            guard let values = history.epochs.first else {
                throw BenchmarkError.invalidData("Training returned no metrics for epoch \(epoch + 1).")
            }
            guard let loss = values["loss"], let accuracy = values["accuracy"],
                  let validationLoss = values["val_loss"], let validationAccuracy = values["val_accuracy"]
            else {
                throw BenchmarkError.missingMetric("loss/accuracy/val_loss/val_accuracy")
            }
            let record = EpochRecord(
                epoch: epoch + 1,
                trainLoss: loss,
                trainAccuracy: accuracy,
                validationLoss: validationLoss,
                validationAccuracy: validationAccuracy,
                elapsedSeconds: timing.epochDurations[epoch]
            )
            epochRecords.append(record)
            print(String(format: "Epoch %d/%d — loss %.5f, accuracy %.3f, val_loss %.5f, val_accuracy %.3f, %.2fs",
                         record.epoch, options.epochs, record.trainLoss, record.trainAccuracy,
                         record.validationLoss, record.validationAccuracy, record.elapsedSeconds))
            fflush(stdout)
        }
        let fitWallSeconds = ProcessInfo.processInfo.systemUptime - trainingStarted

        let testMetrics = try await engine.evaluate(
            model,
            configuration: configuration,
            data: testLoader
        )
        guard let testLoss = testMetrics["loss"], let testAccuracy = testMetrics["accuracy"] else {
            throw BenchmarkError.missingMetric("test loss/accuracy")
        }

        let allBatchSeconds = timing.batchDurations.reduce(0, +)
        let steadySteps = Array(timing.batchDurations.dropFirst(options.warmupSteps))
        let memory = Memory.snapshot()
        let revision = ProcessInfo.processInfo.environment["MLXKIT_GIT_REVISION"] ?? "unknown"
        let version = ProcessInfo.processInfo.operatingSystemVersionString
        let report = RunReport(
            schemaVersion: 1,
            runID: options.runID,
            implementation: "mlxkit",
            createdAtUTC: ISO8601DateFormatter().string(from: Date()),
            gitRevision: revision,
            gitWorktreeDirty: ProcessInfo.processInfo.environment["MLXKIT_GIT_DIRTY"] == "true",
            system: .init(
                platform: "macOS",
                machine: systemString("hw.model") ?? ProcessInfo.processInfo.hostName,
                hardwareModel: systemString("hw.model"),
                processor: systemString("machdep.cpu.brand_string") ?? systemString("hw.model"),
                macOSVersion: version,
                physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
                peakProcessResidentBytes: peakResidentBytes()
            ),
            software: .init(
                swift: ProcessInfo.processInfo.environment["MNIST_SWIFT_VERSION"] ?? "unknown",
                mlxKit: revision,
                mlxSwift: "0.32.3"
            ),
            backend: .init(
                name: "mlx-\(backendName)",
                mlxPeakActiveMemoryBytes: memory.peakMemory,
                mlxCacheMemoryBytes: memory.cacheMemory
            ),
            dataset: .init(
                sourceURL: datasetMetadata["source_url"] as? String ?? "unknown",
                sourceArchiveSHA256: datasetMetadata["source_archive_sha256"] as? String ?? "unknown",
                manifest: datasetMetadata["manifest"] as? String ?? "unknown",
                preparationPythonVersion: datasetMetadata["preparation_python_version"] as? String ?? "unknown",
                preparedFileSHA256: datasetMetadata["prepared_file_sha256"] as? [String: String] ?? [:],
                trainExamples: 48_000,
                validationExamples: 12_000,
                testExamples: 10_000,
                inputShape: [784],
                preprocessing: "uint8 pixels flattened and cast to float32, divided by 255",
                validationSplit: "last 12000 of the original 60000 training examples"
            ),
            initialization: try loadInitializationManifest(at: options.initializationDirectory),
            configuration: .init(
                architecture: [784, 256, 256, 10],
                hiddenActivation: "relu",
                output: "10 logits",
                parameters: 269_322,
                kernelInitializer: "shared seeded Glorot uniform",
                biasInitializer: "zeros",
                loss: "sparse categorical cross-entropy from logits",
                optimizer: "Adam",
                learningRate: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-7,
                biasCorrection: true,
                batchSize: options.batchSize,
                epochs: options.epochs,
                seed: options.seed,
                shuffle: true
            ),
            measurements: .init(
                initialValidationLoss: initialValidationLoss,
                initialValidationAccuracy: initialValidationAccuracy,
                fitWallSecondsIncludingValidation: fitWallSeconds,
                trainingBatchSeconds: allBatchSeconds,
                trainingExamplesPerSecondBatchTime: allBatchSeconds > 0
                    ? Double(48_000 * options.epochs) / allBatchSeconds : nil,
                trainingStepLatencyMedianSecondsExcludingWarmup: percentile(steadySteps, fraction: 0.5),
                trainingStepLatencyP90SecondsExcludingWarmup: percentile(steadySteps, fraction: 0.9),
                warmupStepsExcludedFromLatency: min(options.warmupSteps, timing.batchDurations.count),
                testLoss: testLoss,
                testAccuracy: testAccuracy,
                peakProcessResidentMemoryMethod: "getrusage(RUSAGE_SELF).ru_maxrss; macOS bytes; MLX active-memory peak reported separately"
            ),
            epochs: epochRecords
        )

        try writeReport(report, to: options.output)
        print(String(format: "Test loss: %.6f", testLoss))
        print(String(format: "Test accuracy: %.4f%%", testAccuracy * 100))
        if let throughput = report.measurements.trainingExamplesPerSecondBatchTime {
            print(String(format: "Training throughput: %.1f examples/s", throughput))
        }
        print("MLX backend: \(report.backend.name)")
        print("Report: \(options.output.path)")
        fflush(stdout)
    }
}
