import Testing
@testable import MLXKit

/// Thread-safe collector for test assertions.
private final class Collector: @unchecked Sendable {
    var items: [String] = []
    func append(_ item: String) { items.append(item) }
}

// MARK: - CallbackList dispatch order

@Suite("CallbackList")
struct CallbackListTests {
    @Test func dispatchesInOrder() {
        let order = Collector()

        let a = LambdaCallback(onEpochEnd: { _ in order.append("A") })
        let b = LambdaCallback(onEpochEnd: { _ in order.append("B") })
        let c = LambdaCallback(onEpochEnd: { _ in order.append("C") })

        let list = CallbackList([a, b, c])
        let state = TrainingState()
        list.onEpochEnd(state: state)

        #expect(order.items == ["A", "B", "C"])
    }

    @Test func dispatchesAllHooks() {
        let hooks = Collector()

        let cb = LambdaCallback(
            onTrainBegin: { _ in hooks.append("trainBegin") },
            onTrainEnd: { _ in hooks.append("trainEnd") },
            onEpochBegin: { _ in hooks.append("epochBegin") },
            onEpochEnd: { _ in hooks.append("epochEnd") },
            onBatchBegin: { _ in hooks.append("batchBegin") },
            onBatchEnd: { _ in hooks.append("batchEnd") }
        )

        let list = CallbackList([cb])
        let state = TrainingState()

        list.onTrainBegin(state: state)
        list.onEpochBegin(state: state)
        list.onBatchBegin(state: state)
        list.onBatchEnd(state: state)
        list.onEpochEnd(state: state)
        list.onTrainEnd(state: state)

        #expect(hooks.items == ["trainBegin", "epochBegin", "batchBegin", "batchEnd", "epochEnd", "trainEnd"])
    }

    @Test func shouldStopPropagates() {
        let stopper = LambdaCallback(onBatchEnd: { state in
            state.shouldStop = true
        })

        let list = CallbackList([stopper])
        let state = TrainingState()
        #expect(state.shouldStop == false)

        list.onBatchEnd(state: state)
        #expect(state.shouldStop == true)
    }
}

// MARK: - Default no-op implementations

final class NoOpCallback: Callback {}

@Suite("Default No-Op Callback")
struct DefaultNoOpCallbackTests {
    @Test func defaultImplementationsDontCrash() {
        let cb = NoOpCallback()
        let state = TrainingState()
        cb.onTrainBegin(state: state)
        cb.onEpochBegin(state: state)
        cb.onBatchBegin(state: state)
        cb.onBatchEnd(state: state)
        cb.onEpochEnd(state: state)
        cb.onTrainEnd(state: state)
    }
}

// MARK: - History

@Suite("History")
struct HistoryTests {
    @Test func recordsEpochLogs() {
        let history = History()
        let state = TrainingState()

        state.logs = ["loss": 0.5, "accuracy": 0.8]
        history.onEpochEnd(state: state)

        state.logs = ["loss": 0.3, "accuracy": 0.9]
        history.onEpochEnd(state: state)

        #expect(history.epochs.count == 2)
        #expect(history["loss"] == [0.5, 0.3])
        #expect(history["accuracy"] == [0.8, 0.9])
    }

    @Test func missingMetricReturnsNil() {
        let history = History()
        let state = TrainingState()

        state.logs = ["loss": 0.5]
        history.onEpochEnd(state: state)

        let accuracyValues = history["accuracy"]
        #expect(accuracyValues == [nil])
    }

    @Test func resetClearsHistory() {
        let history = History()
        let state = TrainingState()

        state.logs = ["loss": 0.5]
        history.onEpochEnd(state: state)
        #expect(history.epochs.count == 1)

        history.reset()
        #expect(history.epochs.count == 0)
    }
}

// MARK: - EarlyStopping

@Suite("EarlyStopping")
struct EarlyStoppingTests {
    @Test func stopsAfterPatience() {
        let es = EarlyStopping(monitor: "val_loss", patience: 3, mode: .min)
        let state = TrainingState()

        es.onTrainBegin(state: state)

        state.logs = ["val_loss": 1.0]
        es.onEpochEnd(state: state)
        #expect(state.shouldStop == false)

        for i in 1...3 {
            state.logs = ["val_loss": 1.0 + Float(i) * 0.1]
            es.onEpochEnd(state: state)
        }

        #expect(state.shouldStop == true)
    }

    @Test func improvementResetsPatienceCounter() {
        let es = EarlyStopping(monitor: "val_loss", patience: 2, mode: .min)
        let state = TrainingState()

        es.onTrainBegin(state: state)

        state.logs = ["val_loss": 1.0]
        es.onEpochEnd(state: state)

        state.logs = ["val_loss": 1.1]
        es.onEpochEnd(state: state)
        #expect(state.shouldStop == false)

        state.logs = ["val_loss": 0.8]
        es.onEpochEnd(state: state)
        #expect(state.shouldStop == false)

        state.logs = ["val_loss": 0.9]
        es.onEpochEnd(state: state)
        #expect(state.shouldStop == false)

        state.logs = ["val_loss": 0.9]
        es.onEpochEnd(state: state)
        #expect(state.shouldStop == true)
    }

    @Test func autoModeInfersMinForLoss() {
        let es = EarlyStopping(monitor: "val_loss", patience: 1, mode: .auto)
        let state = TrainingState()
        es.onTrainBegin(state: state)

        state.logs = ["val_loss": 0.5]
        es.onEpochEnd(state: state)

        state.logs = ["val_loss": 0.6]
        es.onEpochEnd(state: state)
        #expect(state.shouldStop == true)
    }

    @Test func autoModeInfersMaxForAccuracy() {
        let es = EarlyStopping(monitor: "accuracy", patience: 1, mode: .auto)
        let state = TrainingState()
        es.onTrainBegin(state: state)

        state.logs = ["accuracy": 0.9]
        es.onEpochEnd(state: state)

        state.logs = ["accuracy": 0.8]
        es.onEpochEnd(state: state)
        #expect(state.shouldStop == true)
    }
}

// MARK: - TerminateOnNaN

@Suite("TerminateOnNaN")
struct TerminateOnNaNTests {
    @Test func stopsOnNaN() {
        let guard_ = TerminateOnNaN()
        let state = TrainingState()

        state.logs = ["loss": Float.nan]
        guard_.onBatchEnd(state: state)
        #expect(state.shouldStop == true)
    }

    @Test func stopsOnInfinite() {
        let guard_ = TerminateOnNaN()
        let state = TrainingState()

        state.logs = ["loss": Float.infinity]
        guard_.onBatchEnd(state: state)
        #expect(state.shouldStop == true)
    }

    @Test func doesNotStopOnValidLoss() {
        let guard_ = TerminateOnNaN()
        let state = TrainingState()

        state.logs = ["loss": 0.5]
        guard_.onBatchEnd(state: state)
        #expect(state.shouldStop == false)
    }

    @Test func customMonitorKey() {
        let guard_ = TerminateOnNaN(monitor: "custom_loss")
        let state = TrainingState()

        state.logs = ["custom_loss": Float.nan]
        guard_.onBatchEnd(state: state)
        #expect(state.shouldStop == true)
    }
}

// MARK: - LambdaCallback

@Suite("LambdaCallback")
struct LambdaCallbackTests {
    @Test func closuresFireAtCorrectHooks() {
        let fired = Collector()

        let cb = LambdaCallback(
            onTrainBegin: { _ in fired.append("trainBegin") },
            onEpochEnd: { _ in fired.append("epochEnd") }
        )

        let state = TrainingState()
        cb.onTrainBegin(state: state)
        cb.onEpochBegin(state: state)
        cb.onEpochEnd(state: state)

        #expect(fired.items == ["trainBegin", "epochEnd"])
    }
}

// MARK: - Static Lookup

@Suite("Callback Static Lookup")
struct CallbackStaticLookupTests {
    @Test func earlyStoppingLookup() {
        let cb: EarlyStopping = .earlyStopping
        #expect(cb.monitor == "val_loss")
        #expect(cb.patience == 5)
    }

    @Test func progressReporterLookup() {
        let _: ProgressReporter = .progressReporter
    }

    @Test func terminateOnNaNLookup() {
        let cb: TerminateOnNaN = .terminateOnNaN
        #expect(cb.monitor == "loss")
    }

    @Test func historyLookup() {
        let cb: History = .history
        #expect(cb.epochs.isEmpty)
    }
}

// MARK: - TrainingState

@Suite("TrainingState")
struct TrainingStateTests {
    @Test func defaultValues() {
        let state = TrainingState()
        #expect(state.epoch == 0)
        #expect(state.batch == 0)
        #expect(state.totalEpochs == 0)
        #expect(state.totalBatches == nil)
        #expect(state.logs.isEmpty)
        #expect(state.shouldStop == false)
    }
}
