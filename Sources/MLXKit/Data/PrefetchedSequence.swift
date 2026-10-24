import Foundation

/// Errors raised by bounded prefetch sequences.
public enum PrefetchError: Error, LocalizedError, Equatable {
    case invalidBufferSize(Int)
    case bufferInvariantViolation
    case elementTypeMismatch

    public var errorDescription: String? {
        switch self {
        case .invalidBufferSize(let value):
            "Prefetch buffer size must be greater than zero; received \(value)."
        case .bufferInvariantViolation:
            "The prefetch producer exceeded its configured buffer capacity."
        case .elementTypeMismatch:
            "The prefetch producer returned an element with an unexpected type."
        }
    }
}

/// An asynchronous sequence that prefetches source elements into a bounded
/// buffer while preserving source order.
///
/// The buffer capacity counts elements that the consumer has not requested
/// yet. The producer reserves one buffer slot before it advances the source,
/// so an in-flight source read is included in the bound. When iteration ends
/// early or is cancelled, the producer task is cancelled and the bounded
/// demand stream is closed. Prompt upstream cleanup requires a source whose
/// `next()` operation cooperates with task cancellation.
public struct PrefetchedSequence<Source: AsyncSequence>: AsyncSequence {
    public typealias Element = Source.Element
    public typealias Failure = any Error

    public let source: Source
    public let bufferSize: Int

    public init(source: Source, bufferSize: Int) throws {
        guard bufferSize > 0 else {
            throw PrefetchError.invalidBufferSize(bufferSize)
        }
        self.source = source
        self.bufferSize = bufferSize
    }

    public func makeAsyncIterator() -> Iterator {
        Iterator(source: source, bufferSize: bufferSize)
    }

    public struct Iterator: AsyncIteratorProtocol {
        public typealias Element = Source.Element
        public typealias Failure = any Error

        private let state: PrefetchIteratorState<Element>

        fileprivate init(source: Source, bufferSize: Int) {
            let (outputStream, outputContinuation) = AsyncThrowingStream<Any, Error>.makeStream(
                of: Any.self,
                throwing: Error.self,
                bufferingPolicy: .bufferingOldest(bufferSize)
            )
            let (demandStream, demandContinuation) = AsyncStream<Void>.makeStream(
                of: Void.self,
                bufferingPolicy: .bufferingOldest(bufferSize)
            )
            for _ in 0..<bufferSize {
                demandContinuation.yield(())
            }

            let producer = PrefetchProducer(
                source: source,
                demand: demandStream,
                continuation: outputContinuation
            )
            let producerTask = Task.detached { @Sendable [producer] in
                await producer.run()
            }
            outputContinuation.onTermination = { @Sendable _ in
                producerTask.cancel()
                demandContinuation.finish()
            }

            self.state = PrefetchIteratorState(
                stream: outputStream,
                demandContinuation: demandContinuation,
                producerTask: producerTask
            )
        }

        public mutating func next() async throws -> Element? {
            try Task.checkCancellation()
            do {
                let element = try await state.next()
                try Task.checkCancellation()
                if element == nil {
                    state.cancel()
                }
                return element
            } catch {
                state.cancel()
                throw error
            }
        }
    }
}

/// Adds bounded prefetch to any asynchronous sequence.
public extension AsyncSequence {
    func prefetched(bufferSize: Int) throws -> PrefetchedSequence<Self> {
        try PrefetchedSequence(source: self, bufferSize: bufferSize)
    }
}

private final class PrefetchIteratorState<Element> {
    private var iterator: AsyncThrowingStream<Any, Error>.Iterator
    private let demandContinuation: AsyncStream<Void>.Continuation
    private let producerTask: Task<Void, Never>
    private var hasDeliveredElement = false
    private var isTerminated = false

    init(
        stream: AsyncThrowingStream<Any, Error>,
        demandContinuation: AsyncStream<Void>.Continuation,
        producerTask: Task<Void, Never>
    ) {
        self.iterator = stream.makeAsyncIterator()
        self.demandContinuation = demandContinuation
        self.producerTask = producerTask
    }

    func next() async throws -> Element? {
        guard !isTerminated else { return nil }
        // A request for the next element acknowledges that the previously
        // returned element has left the prefetch buffer and frees one slot.
        if hasDeliveredElement {
            demandContinuation.yield(())
            hasDeliveredElement = false
        }
        do {
            guard let value = try await iterator.next() else {
                isTerminated = true
                return nil
            }
            guard let element = value as? Element else {
                throw PrefetchError.elementTypeMismatch
            }
            hasDeliveredElement = true
            return element
        } catch {
            isTerminated = true
            cancel()
            throw error
        }
    }

    func cancel() {
        producerTask.cancel()
        demandContinuation.finish()
    }

    deinit {
        cancel()
    }
}

// The detached producer task is the sole owner of these closures and the
// upstream iterator after initialization. The output continuation is the only
// handoff to the consumer, and it preserves sequence order.
private final class PrefetchProducer: @unchecked Sendable {
    private let nextSourceElement: () async throws -> Any?
    private var demand: AsyncStream<Void>.Iterator
    private let continuation: AsyncThrowingStream<Any, Error>.Continuation

    init<Source: AsyncSequence>(
        source: Source,
        demand: AsyncStream<Void>,
        continuation: AsyncThrowingStream<Any, Error>.Continuation
    ) {
        let sourceIterator = PrefetchSourceIterator(source: source)
        self.nextSourceElement = {
            guard let element = try await sourceIterator.next() else { return nil }
            return element as Any
        }
        self.demand = demand.makeAsyncIterator()
        self.continuation = continuation
    }

    func run() async {
        do {
            while await demand.next() != nil {
                try Task.checkCancellation()
                guard let element = try await nextSourceElement() else {
                    continuation.finish()
                    return
                }

                switch continuation.yield(element) {
                case .enqueued:
                    break
                case .dropped:
                    continuation.finish(throwing: PrefetchError.bufferInvariantViolation)
                    return
                case .terminated:
                    return
                @unknown default:
                    continuation.finish(throwing: PrefetchError.bufferInvariantViolation)
                    return
                }
            }

            if Task.isCancelled {
                continuation.finish(throwing: CancellationError())
            } else {
                continuation.finish()
            }
        } catch {
            continuation.finish(throwing: error)
        }
    }
}

private final class PrefetchSourceIterator<Source: AsyncSequence> {
    private var iterator: Source.AsyncIterator

    init(source: Source) {
        self.iterator = source.makeAsyncIterator()
    }

    func next() async throws -> Source.Element? {
        try await iterator.next()
    }
}
