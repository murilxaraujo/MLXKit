import Foundation

extension Callback where Self == EarlyStopping {
    /// Early stopping with default settings (monitors `val_loss`, patience 5).
    public static var earlyStopping: EarlyStopping { EarlyStopping() }
}

extension Callback where Self == ProgressReporter {
    /// Progress reporter that prints epoch summaries to stdout.
    public static var progressReporter: ProgressReporter { ProgressReporter() }
}

extension Callback where Self == TerminateOnNaN {
    /// Terminates training if loss becomes NaN or Inf.
    public static var terminateOnNaN: TerminateOnNaN { TerminateOnNaN() }
}

extension Callback where Self == History {
    /// History recorder that stores per-epoch metrics.
    public static var history: History { History() }
}
