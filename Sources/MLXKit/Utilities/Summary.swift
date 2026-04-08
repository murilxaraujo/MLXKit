import Foundation
import MLX
import MLXNN

/// A row in the model summary table.
public struct SummaryRow: Sendable {
    /// The name/path of the layer (e.g. `"linear1"`, `"layers.0.attention"`).
    public let name: String
    /// The type of the module (e.g. `"Linear"`, `"Embedding"`).
    public let type: String
    /// The number of trainable parameters in this module (not counting children).
    public let parameterCount: Int
}

/// A complete model summary.
public struct ModelSummary: Sendable {
    /// All layers in the model, in traversal order.
    public let layers: [SummaryRow]
    /// Total number of trainable parameters across all layers.
    public var totalParameters: Int { layers.reduce(0) { $0 + $1.parameterCount } }

    /// Estimated memory usage in bytes (assuming Float32).
    public var estimatedSizeBytes: Int { totalParameters * 4 }

    /// Formats the summary as an ASCII table string.
    public func formatted() -> String {
        let nameHeader = "Layer"
        let typeHeader = "Type"
        let paramsHeader = "Params"

        let nameWidth = max(nameHeader.count, layers.map(\.name.count).max() ?? 0)
        let typeWidth = max(typeHeader.count, layers.map(\.type.count).max() ?? 0)
        let paramsWidth = max(paramsHeader.count, layers.map { formatNumber($0.parameterCount).count }.max() ?? 0)

        let totalWidth = nameWidth + typeWidth + paramsWidth + 10
        let separator = String(repeating: "─", count: totalWidth)

        var lines: [String] = []
        lines.append(separator)
        lines.append(formatRow(nameHeader, typeHeader, paramsHeader, nameWidth, typeWidth, paramsWidth))
        lines.append(separator)

        for layer in layers {
            lines.append(formatRow(
                layer.name,
                layer.type,
                formatNumber(layer.parameterCount),
                nameWidth, typeWidth, paramsWidth
            ))
        }

        lines.append(separator)
        lines.append("Total params: \(formatNumber(totalParameters))")
        lines.append("Estimated size: \(formatBytes(estimatedSizeBytes))")
        lines.append(separator)

        return lines.joined(separator: "\n")
    }
}

extension Module {
    /// Generates a summary of the model's layers and parameter counts.
    ///
    /// ```swift
    /// let model = MyModel()
    /// let summary = model.summary()
    /// print(summary.formatted())
    /// ```
    ///
    /// - Returns: A ``ModelSummary`` with layer details.
    public func summary() -> ModelSummary {
        let named = namedModules()
        var layers: [SummaryRow] = []

        for (name, module) in named {
            let displayName = name.isEmpty ? String(describing: Swift.type(of: module)) : name
            let typeName = String(describing: Swift.type(of: module))
                .components(separatedBy: "<").first ?? String(describing: Swift.type(of: module))

            // Count only this module's own parameters (not children's)
            let ownParams = countOwnParameters(module)

            layers.append(SummaryRow(
                name: displayName,
                type: typeName,
                parameterCount: ownParams
            ))
        }

        return ModelSummary(layers: layers)
    }

    /// Prints the model summary to stdout.
    public func printSummary() {
        print(summary().formatted())
    }
}

// MARK: - Helpers

private func countOwnParameters(_ module: Module) -> Int {
    // Get this module's trainable parameters (flat, own level only)
    let params = module.trainableParameters()
    var count = 0
    for (_, item) in params {
        if case .value(let array) = item {
            count += array.size
        }
    }
    return count
}

private func formatRow(
    _ col1: String, _ col2: String, _ col3: String,
    _ w1: Int, _ w2: Int, _ w3: Int
) -> String {
    " \(col1.padding(toLength: w1, withPad: " ", startingAt: 0)) │ \(col2.padding(toLength: w2, withPad: " ", startingAt: 0)) │ \(col3.padding(toLength: w3, withPad: " ", startingAt: 0)) "
}

private func formatNumber(_ n: Int) -> String {
    let formatter = NumberFormatter()
    formatter.numberStyle = .decimal
    formatter.groupingSeparator = ","
    return formatter.string(from: NSNumber(value: n)) ?? "\(n)"
}

private func formatBytes(_ bytes: Int) -> String {
    if bytes < 1024 {
        return "\(bytes) B"
    } else if bytes < 1024 * 1024 {
        return String(format: "%.1f KB", Double(bytes) / 1024)
    } else if bytes < 1024 * 1024 * 1024 {
        return String(format: "%.1f MB", Double(bytes) / (1024 * 1024))
    } else {
        return String(format: "%.1f GB", Double(bytes) / (1024 * 1024 * 1024))
    }
}
