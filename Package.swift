// swift-tools-version: 6.4

import PackageDescription

/// Set to `true` to enable GPU-dependent tests (requires Apple Silicon + Metal).
/// Use `make test-all` or pass `-Xswiftc -DMLX_GPU_TESTS` to enable.
let enableGPUTests = ProcessInfo.processInfo.environment["MLX_GPU_TESTS"] != nil

import Foundation

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
        .executable(
            name: "MNISTExample",
            targets: ["MNISTExample"]
        ),
        .executable(
            name: "ContrastiveEmbeddingExample",
            targets: ["ContrastiveEmbeddingExample"]
        ),
        .executable(
            name: "TinyDecoderExample",
            targets: ["TinyDecoderExample"]
        ),
        .executable(
            name: "DataPipelineExample",
            targets: ["DataPipelineExample"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.32.3"),
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
        .executableTarget(
            name: "MNISTExample",
            dependencies: ["MLXKit"]
        ),
        .executableTarget(
            name: "ContrastiveEmbeddingExample",
            dependencies: ["MLXKit"]
        ),
        .target(
            name: "TinyDecoderCore",
            dependencies: [
                "MLXKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "TinyDecoderExample",
            dependencies: ["MLXKit", "TinyDecoderCore"]
        ),
        .executableTarget(
            name: "DataPipelineExample",
            dependencies: [
                "MLXKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MLXKitTests",
            dependencies: ["MLXKit", "TinyDecoderCore"],
            swiftSettings: enableGPUTests ? [.define("MLX_GPU_TESTS")] : []
        ),
    ]
)
