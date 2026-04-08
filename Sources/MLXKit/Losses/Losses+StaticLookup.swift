extension LossFunction where Self == CrossEntropyLoss {
    /// Cross-entropy loss with default settings.
    public static var crossEntropy: CrossEntropyLoss { CrossEntropyLoss() }
}

extension LossFunction where Self == BinaryCrossEntropyLoss {
    /// Binary cross-entropy loss with default settings.
    public static var binaryCrossEntropy: BinaryCrossEntropyLoss { BinaryCrossEntropyLoss() }
}

extension LossFunction where Self == MSELoss {
    /// Mean squared error loss with default settings.
    public static var mse: MSELoss { MSELoss() }
}

extension LossFunction where Self == L1Loss {
    /// L1 (mean absolute error) loss with default settings.
    public static var l1: L1Loss { L1Loss() }
}

extension LossFunction where Self == HuberLoss {
    /// Huber loss with default settings.
    public static var huber: HuberLoss { HuberLoss() }
}

extension LossFunction where Self == KLDivLoss {
    /// KL divergence loss with default settings.
    public static var klDiv: KLDivLoss { KLDivLoss() }
}
