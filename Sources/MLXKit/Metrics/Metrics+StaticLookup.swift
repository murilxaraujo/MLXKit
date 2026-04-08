extension Metric where Self == Accuracy {
    /// Classification accuracy metric.
    public static var accuracy: Accuracy { Accuracy() }
}

extension Metric where Self == TopKAccuracy {
    /// Top-5 accuracy metric.
    public static var top5Accuracy: TopKAccuracy { TopKAccuracy(k: 5) }
}

extension Metric where Self == MeanSquaredError {
    /// Mean squared error metric.
    public static var meanSquaredError: MeanSquaredError { MeanSquaredError() }
}

extension Metric where Self == MeanAbsoluteError {
    /// Mean absolute error metric.
    public static var meanAbsoluteError: MeanAbsoluteError { MeanAbsoluteError() }
}

extension Metric where Self == RunningLoss {
    /// Running loss metric.
    public static var runningLoss: RunningLoss { RunningLoss() }
}
