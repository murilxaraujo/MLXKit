import Foundation

/// One strategy's result for a benchmark request. A failure is retained so a
/// strategy that does not support a question type does not prevent the other
/// strategy from completing the comparison.
public struct StrategyBenchmarkRun: Sendable {
    public let strategy: DecisionStrategy
    public let response: DecisionResponse?
    public let failureDescription: String?

    public init(
        strategy: DecisionStrategy,
        response: DecisionResponse? = nil,
        failureDescription: String? = nil
    ) {
        self.strategy = strategy
        self.response = response
        self.failureDescription = failureDescription
    }
}

/// Runs the same request and model sequentially through each strategy.
///
/// Sequential execution avoids concurrent model access and makes the first
/// comparison a correctness-first baseline. It measures elapsed model and
/// scoring time reported by each `DecisionEngine`; it does not claim parity or
/// control for differences in prompts and candidate counts.
public enum StrategyBenchmarkRunner {
    public static func compare(
        _ request: DecisionRequest,
        model: any TokenLogitsModel,
        strategies: [DecisionStrategy] = [.systemOne, .jevMLX(scoring: .labels)]
    ) async -> [StrategyBenchmarkRun] {
        var runs: [StrategyBenchmarkRun] = []
        for strategy in strategies {
            do {
                let response = try await DecisionEngine(model: model, strategy: strategy).decide(request)
                runs.append(StrategyBenchmarkRun(strategy: strategy, response: response))
            } catch {
                runs.append(
                    StrategyBenchmarkRun(
                        strategy: strategy,
                        failureDescription: error.localizedDescription
                    )
                )
            }
        }
        return runs
    }
}
