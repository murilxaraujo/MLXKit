import Testing
@testable import MLXKit

private enum PrefetchTestFailure: Error, Equatable {
    case upstream
}

@Suite("Bounded prefetch")
struct PrefetchedSequenceTests {
    @Test func prefetchPreservesOrderAndEveryElement() async throws {
        let source = AsyncThrowingStream<Int, Error> { continuation in
            for value in 0..<25 {
                continuation.yield(value)
            }
            continuation.finish()
        }
        let prefetched = try source.prefetched(bufferSize: 3)

        #expect(try await collect(prefetched) == Array(0..<25))
    }

    @Test func dataLoaderBatchesCanBePrefetched() async throws {
        let samples = (0..<5).map { Sample(input: $0, target: -$0) }
        let loader = try DataLoader(
            source: samples,
            batchSize: 2,
            collate: { samples in
                Batch(
                    input: samples.map(\.input).reduce(0, +),
                    target: samples.map(\.target).reduce(0, +),
                    sampleCount: samples.count
                )
            }
        )
        let prefetched = try loader.prefetched(bufferSize: 1)
        let batches = try await collect(prefetched)

        #expect(batches.map(\.input) == [1, 5, 4])
        #expect(batches.map(\.sampleCount) == [2, 2, 1])
    }

    @Test func producerDoesNotReadPastCapacityWithoutConsumerDemand() async throws {
        let tracker = PullTracker()
        let prefetched = try TrackedSequence(count: 20, tracker: tracker)
            .prefetched(bufferSize: 3)
        var iterator = prefetched.makeAsyncIterator()

        #expect(await tracker.waitUntilPullCount(3) == 3)
        #expect(await tracker.currentPullCount() == 3)

        #expect(try await iterator.next() == 0)
        #expect(await tracker.currentPullCount() == 3)

        #expect(try await iterator.next() == 1)
        #expect(await tracker.waitUntilPullCount(4) == 4)
        #expect(await tracker.currentPullCount() == 4)
    }

    @Test func upstreamFailurePropagatesAfterEarlierElements() async throws {
        let source = AsyncThrowingStream<Int, Error> { continuation in
            continuation.yield(1)
            continuation.finish(throwing: PrefetchTestFailure.upstream)
        }
        let prefetched = try source.prefetched(bufferSize: 2)
        var iterator = prefetched.makeAsyncIterator()

        #expect(try await iterator.next() == 1)
        do {
            _ = try await iterator.next()
            Issue.record("Expected the upstream error to propagate")
        } catch let error as PrefetchTestFailure {
            #expect(error == .upstream)
        }

        #expect(try await iterator.next() == nil)
    }

    @Test func invalidCapacityFailsBeforeCreatingThePrefetcher() {
        let source = AsyncStream<Int> { _ in }
        #expect(throws: PrefetchError.invalidBufferSize(0)) {
            try source.prefetched(bufferSize: 0)
        }
    }

    @Test func cancellationReachesACooperativeUpstream() async throws {
        let probe = CancellationProbe()
        let prefetched = try CancellationAwareSequence(probe: probe).prefetched(bufferSize: 1)

        let consumer = Task {
            var iterator = prefetched.makeAsyncIterator()
            return try await iterator.next()
        }

        await probe.waitUntilSourceStarts()
        consumer.cancel()

        do {
            _ = try await consumer.value
            Issue.record("Expected cancellation to reach the consumer")
        } catch is CancellationError {
            // Expected: cancelling the consumer cancels its prefetch producer.
        }
        await probe.waitUntilUpstreamCancelled()
    }

    @Test func droppingTheIteratorCancelsUpstream() async throws {
        let probe = CancellationProbe()
        let prefetched = try CancellationAwareSequence(probe: probe).prefetched(bufferSize: 1)
        var iterator: PrefetchedSequence<CancellationAwareSequence>.Iterator? =
            prefetched.makeAsyncIterator()

        _ = iterator
        await probe.waitUntilSourceStarts()
        iterator = nil
        await probe.waitUntilUpstreamCancelled()
    }

    private func collect<Source: AsyncSequence>(_ source: Source) async throws -> [Source.Element]
    where Source.Failure: Error {
        var values: [Source.Element] = []
        for try await value in source {
            values.append(value)
        }
        return values
    }
}

private actor PullTracker {
    private var pullCount = 0
    private var waiters: [(target: Int, continuation: CheckedContinuation<Int, Never>)] = []

    func recordPull() {
        pullCount += 1
        var remaining: [(target: Int, continuation: CheckedContinuation<Int, Never>)] = []
        for waiter in waiters {
            if pullCount >= waiter.target {
                waiter.continuation.resume(returning: pullCount)
            } else {
                remaining.append(waiter)
            }
        }
        waiters = remaining
    }

    func currentPullCount() -> Int {
        pullCount
    }

    func waitUntilPullCount(_ target: Int) async -> Int {
        if pullCount >= target { return pullCount }
        return await withCheckedContinuation { continuation in
            waiters.append((target, continuation))
        }
    }
}

private struct TrackedSequence: AsyncSequence {
    typealias Element = Int
    typealias Failure = Never

    let count: Int
    let tracker: PullTracker

    func makeAsyncIterator() -> Iterator {
        Iterator(count: count, tracker: tracker)
    }

    struct Iterator: AsyncIteratorProtocol {
        typealias Element = Int
        typealias Failure = Never

        let count: Int
        let tracker: PullTracker
        var offset = 0

        mutating func next() async -> Element? {
            guard offset < count else { return nil }
            let value = offset
            offset += 1
            await tracker.recordPull()
            return value
        }
    }
}

private actor CancellationProbe {
    private var sourceStarted = false
    private var upstreamCancelled = false
    private var startWaiters: [CheckedContinuation<Void, Never>] = []
    private var cancellationWaiters: [CheckedContinuation<Void, Never>] = []

    func markSourceStarted() {
        sourceStarted = true
        let waiters = startWaiters
        startWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }

    func markUpstreamCancelled() {
        upstreamCancelled = true
        let waiters = cancellationWaiters
        cancellationWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }

    func waitUntilSourceStarts() async {
        guard !sourceStarted else { return }
        await withCheckedContinuation { startWaiters.append($0) }
    }

    func waitUntilUpstreamCancelled() async {
        guard !upstreamCancelled else { return }
        await withCheckedContinuation { cancellationWaiters.append($0) }
    }
}

private actor CancellationGate {
    private var isOpen = false
    private var waiter: CheckedContinuation<Void, Never>?

    func wait() async {
        guard !isOpen else { return }
        await withCheckedContinuation { waiter = $0 }
    }

    func open() {
        isOpen = true
        waiter?.resume()
        waiter = nil
    }
}

private struct CancellationAwareSequence: AsyncSequence {
    typealias Element = Int
    typealias Failure = any Error

    let probe: CancellationProbe

    func makeAsyncIterator() -> Iterator {
        Iterator(probe: probe)
    }

    struct Iterator: AsyncIteratorProtocol {
        typealias Element = Int
        typealias Failure = any Error

        let probe: CancellationProbe

        mutating func next() async throws -> Element? {
            let probe = self.probe
            await probe.markSourceStarted()
            let gate = CancellationGate()
            let result: Element? = try await withTaskCancellationHandler {
                await gate.wait()
                try Task.checkCancellation()
                return nil
            } onCancel: {
                Task {
                    await gate.open()
                    await probe.markUpstreamCancelled()
                }
            }
            return result
        }
    }
}
