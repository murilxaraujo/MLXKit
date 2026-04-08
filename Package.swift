// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "MLXKit",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
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
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
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
        .testTarget(
            name: "MLXKitTests",
            dependencies: ["MLXKit"]
        ),
    ]
)
