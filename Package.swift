// swift-tools-version: 6.4

import PackageDescription

/// Set to `true` to enable GPU-dependent tests (requires Apple Silicon + Metal).
/// Use `make test-all` or pass `-Xswiftc -DMLX_GPU_TESTS` to enable.
let enableGPUTests = ProcessInfo.processInfo.environment["MLX_GPU_TESTS"] != nil

import Foundation

// Executable examples live in the standalone Tuist project under Examples/.
let package = Package(
    name: "MLXKit",
    platforms: [
        .macOS(.v27)
    ],
    products: [
        .library(
            name: "MLXKit",
            targets: ["MLXKit"]
        ),
        .library(
            name: "MLXJev",
            targets: ["MLXJev"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.32.3"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.32.3"),
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "MLXKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "MLXJev",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
            ]
        ),
        .executableTarget(
            name: "MNISTMLPBenchmark",
            dependencies: [
                "MLXKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            path: "Benchmarking/mnist-mlp/mlxkit",
            exclude: ["README.md"],
            sources: ["Sources/MNISTMLPBenchmark.swift"]
        ),
        .executableTarget(
            name: "MNISTCNNBenchmark",
            dependencies: [
                "MLXKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            path: "Benchmarking/mnist-cnn/mlxkit",
            exclude: ["README.md"],
            sources: ["Sources/MNISTCNNBenchmark.swift"]
        ),
        .executableTarget(
            name: "WikiText2RNNBenchmark",
            dependencies: [
                "MLXKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            path: "Benchmarking/wikitext2-rnn/mlxkit",
            exclude: ["README.md"],
            sources: ["Sources/WikiText2RNNBenchmark.swift"]
        ),
        .testTarget(
            name: "MLXKitTests",
            dependencies: ["MLXKit"],
            swiftSettings: enableGPUTests ? [.define("MLX_GPU_TESTS")] : []
        ),
        .testTarget(
            name: "MLXJevTests",
            dependencies: ["MLXJev"]
        ),
    ]
)
