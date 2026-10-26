import ProjectDescription

let project = Project(
    name: "MLXKitExamples",
    packages: [
        .local(path: ".."),
        .remote(
            url: "https://github.com/apple/swift-argument-parser",
            requirement: .exact("1.8.2")
        ),
        .remote(
            url: "https://github.com/ml-explore/mlx-swift",
            requirement: .upToNextMinor(from: "0.32.3")
        ),
    ],
    targets: [
        .target(
            name: "MNISTExample",
            destinations: [.mac],
            product: .commandLineTool,
            bundleId: "com.mlxkit.examples.mnist",
            deploymentTargets: .macOS("27.0"),
            infoPlist: .default,
            sources: ["SupervisedClassifier/Sources/**"],
            dependencies: [
                .package(product: "MLXKit"),
                .package(product: "MLX"),
                .package(product: "MLXNN"),
                .package(product: "MLXOptimizers"),
                .package(product: "ArgumentParser"),
            ]
        ),
        .target(
            name: "ContrastiveEmbeddingExample",
            destinations: [.mac],
            product: .commandLineTool,
            bundleId: "com.mlxkit.examples.contrastive-embedding",
            deploymentTargets: .macOS("27.0"),
            infoPlist: .default,
            sources: ["ContrastiveEmbedding/Sources/**"],
            dependencies: [
                .package(product: "MLXKit"),
                .package(product: "MLX"),
                .package(product: "MLXNN"),
                .package(product: "MLXOptimizers"),
                .package(product: "ArgumentParser"),
            ]
        ),
        .target(
            name: "TinyDecoderExample",
            destinations: [.mac],
            product: .commandLineTool,
            bundleId: "com.mlxkit.examples.tiny-decoder",
            deploymentTargets: .macOS("27.0"),
            infoPlist: .default,
            sources: ["TinyDecoder/Sources/**", "TinyDecoder/Core/**"],
            dependencies: [
                .package(product: "MLXKit"),
                .package(product: "MLX"),
                .package(product: "MLXNN"),
                .package(product: "MLXOptimizers"),
                .package(product: "ArgumentParser"),
            ]
        ),
        .target(
            name: "DataPipelineExample",
            destinations: [.mac],
            product: .commandLineTool,
            bundleId: "com.mlxkit.examples.data-pipeline",
            deploymentTargets: .macOS("27.0"),
            infoPlist: .default,
            sources: ["DataPipeline/Sources/**"],
            dependencies: [
                .package(product: "MLXKit"),
                .package(product: "MLX"),
                .package(product: "MLXNN"),
                .package(product: "MLXOptimizers"),
                .package(product: "ArgumentParser"),
            ]
        ),
        .target(
            name: "TinyDecoderTests",
            destinations: [.mac],
            product: .unitTests,
            bundleId: "com.mlxkit.examples.tiny-decoder-tests",
            deploymentTargets: .macOS("27.0"),
            infoPlist: .default,
            sources: ["TinyDecoder/Tests/**", "TinyDecoder/Core/**"],
            dependencies: [
                .package(product: "MLXKit"),
                .package(product: "MLX"),
                .package(product: "MLXNN"),
                .package(product: "MLXOptimizers"),
            ]
        ),
    ]
)
