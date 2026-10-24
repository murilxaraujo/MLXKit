import Testing
@testable import MLXKit

private enum PipelineTestFailure: Error, Equatable {
    case transform
    case source
}

@Suite("Composable data pipeline")
struct DataPipelineTests {
    @Test func sampleTransformsRunInDeclaredOrderBeforeBatching() async throws {
        let source = AsyncThrowingStream<Sample<Int, Int>, Error> { continuation in
            for value in 0..<10 {
                continuation.yield(Sample(input: value, target: value * 10))
            }
            continuation.finish()
        }

        let transformed = source
            .map { Sample(input: $0.input + 1, target: $0.target + 1) }
            .filter { $0.input.isMultiple(of: 2) }
        let loader = try DataLoader(
            source: transformed,
            batchSize: 2,
            collate: sumSamples
        )

        let batches = try await collect(loader)
        #expect(batches.map(\.input) == [6, 14, 10])
        #expect(batches.map(\.target) == [42, 122, 91])
        #expect(batches.map(\.sampleCount) == [2, 2, 1])
    }

    @Test func indexedFullShuffleIsSeededAndPreservesEverySample() async throws {
        let samples = (0..<40).map { Sample(input: $0, target: -$0) }
        let firstLoader = try DataLoader(
            source: samples,
            batchSize: 1,
            shuffle: .full(seed: 713),
            collate: firstSample
        )
        let secondLoader = try DataLoader(
            source: samples,
            batchSize: 1,
            shuffle: .full(seed: 713),
            collate: firstSample
        )

        let firstOrder = try await collect(firstLoader).map(\.input)
        let secondOrder = try await collect(secondLoader).map(\.input)
        #expect(firstOrder == secondOrder)
        #expect(firstOrder.sorted() == Array(0..<40))
        #expect(firstOrder != Array(0..<40))
    }

    @Test func streamBufferedShuffleIsSeededBoundedAndLossless() async throws {
        let firstCounter = PullCounter()
        let firstLoader = try DataLoader(
            source: CountingSampleSequence(count: 100, counter: firstCounter),
            batchSize: 1,
            shuffle: .buffered(windowSize: 5, seed: 9),
            collate: firstSample
        )
        var firstIterator = firstLoader.makeAsyncIterator()
        let first = try await firstIterator.next()
        #expect(first != nil)
        #expect(firstCounter.value >= 5)
        #expect(firstCounter.value <= 6)
        #expect(firstCounter.value < 100)

        var firstOrder = [first!.input]
        while let batch = try await firstIterator.next() {
            firstOrder.append(batch.input)
        }

        let secondCounter = PullCounter()
        let secondLoader = try DataLoader(
            source: CountingSampleSequence(count: 100, counter: secondCounter),
            batchSize: 1,
            shuffle: .buffered(windowSize: 5, seed: 9),
            collate: firstSample
        )
        let secondOrder = try await collect(secondLoader).map(\.input)

        #expect(firstOrder == secondOrder)
        #expect(firstOrder.sorted() == Array(0..<100))
    }

    @Test func fullShuffleRejectsAStreamBeforeReadingIt() {
        let counter = PullCounter()
        #expect(throws: DataLoaderError.fullShuffleRequiresIndexedSource) {
            try DataLoader(
                source: CountingSampleSequence(count: 10, counter: counter),
                batchSize: 2,
                shuffle: .full(seed: 1),
                collate: firstSample
            )
        }
        #expect(counter.value == 0)
    }

    @Test func collatorCannotReportAnIncorrectSampleCount() async throws {
        let samples = (0..<3).map { Sample(input: $0, target: $0) }
        let loader = try DataLoader(
            source: samples,
            batchSize: 2,
            collate: { batch in
                Batch(input: batch[0].input, target: batch[0].target, sampleCount: 1)
            }
        )
        var iterator = loader.makeAsyncIterator()

        do {
            _ = try await iterator.next()
            Issue.record("Expected the false collated sample count to fail")
        } catch let error as DataLoaderError {
            #expect(error == .collatedSampleCountMismatch(expected: 2, actual: 1))
        }
    }

    @Test func batchingHandlesEmptyExactPartialAndDroppedRemainders() async throws {
        let empty: [Sample<Int, Int>] = []
        let emptyLoader = try DataLoader(source: empty, batchSize: 2, collate: sumSamples)
        #expect(try await collect(emptyLoader).isEmpty)

        let exactSamples = (0..<4).map { Sample(input: $0, target: $0) }
        let exactLoader = try DataLoader(source: exactSamples, batchSize: 2, collate: sumSamples)
        let exactBatches = try await collect(exactLoader)
        #expect(exactBatches.map(\.sampleCount) == [2, 2])

        let partialSamples = (0..<5).map { Sample(input: $0, target: $0) }
        let partialLoader = try DataLoader(source: partialSamples, batchSize: 2, collate: sumSamples)
        let partialBatches = try await collect(partialLoader)
        #expect(partialBatches.map(\.sampleCount) == [2, 2, 1])
        #expect(partialBatches.map(\.input) == [1, 5, 4])

        let dropLoader = try DataLoader(
            source: partialSamples,
            batchSize: 2,
            dropLast: true,
            collate: sumSamples
        )
        let droppedBatches = try await collect(dropLoader)
        #expect(droppedBatches.map(\.sampleCount) == [2, 2])
        #expect(droppedBatches.map(\.input) == [1, 5])
    }

    @Test func throwingTransformFailureReachesTheConsumer() async throws {
        let source = AsyncThrowingStream<Sample<Int, Int>, Error> { continuation in
            continuation.yield(Sample(input: 1, target: 1))
            continuation.yield(Sample(input: 2, target: 2))
            continuation.finish()
        }
        let transformed = source.map { sample throws -> Sample<Int, Int> in
            guard sample.input == 2 else { return sample }
            throw PipelineTestFailure.transform
        }
        let loader = try DataLoader(
            source: transformed,
            batchSize: 1,
            collate: firstSample
        )
        var iterator = loader.makeAsyncIterator()

        #expect(try await iterator.next()?.input == 1)
        do {
            _ = try await iterator.next()
            Issue.record("Expected the transform failure to propagate")
        } catch let error as PipelineTestFailure {
            #expect(error == .transform)
        }
    }

    @Test func throwingSourceFailureReachesTheConsumer() async throws {
        let source = AsyncThrowingStream<Sample<Int, Int>, Error> { continuation in
            continuation.yield(Sample(input: 7, target: 7))
            continuation.finish(throwing: PipelineTestFailure.source)
        }
        let loader = try DataLoader(source: source, batchSize: 2, collate: sumSamples)
        var iterator = loader.makeAsyncIterator()

        do {
            _ = try await iterator.next()
            Issue.record("Expected the source failure to propagate")
        } catch let error as PipelineTestFailure {
            #expect(error == .source)
        }
    }

    private func sumSamples(_ samples: [Sample<Int, Int>]) -> Batch<Int, Int> {
        Batch(
            input: samples.map(\.input).reduce(0, +),
            target: samples.map(\.target).reduce(0, +),
            sampleCount: samples.count
        )
    }

    private func firstSample(_ samples: [Sample<Int, Int>]) -> Batch<Int, Int> {
        Batch(
            input: samples[0].input,
            target: samples[0].target,
            sampleCount: samples.count
        )
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

private final class PullCounter {
    var value = 0
}

private struct CountingSampleSequence: AsyncSequence {
    typealias Element = Sample<Int, Int>
    typealias Failure = Never

    let count: Int
    let counter: PullCounter

    func makeAsyncIterator() -> Iterator {
        Iterator(count: count, counter: counter)
    }

    struct Iterator: AsyncIteratorProtocol {
        typealias Element = Sample<Int, Int>
        typealias Failure = Never

        let count: Int
        let counter: PullCounter
        var offset = 0

        mutating func next() async -> Element? {
            guard offset < count else { return nil }
            defer {
                offset += 1
                counter.value += 1
            }
            return Sample(input: offset, target: offset)
        }
    }
}
