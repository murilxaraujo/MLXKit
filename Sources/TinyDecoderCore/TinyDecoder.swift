import Foundation
import MLX
import MLXNN
import MLXKit

/// Multi-head causal self-attention using public MLX array operations.
///
/// Scores are `QKᵀ / sqrt(headWidth)`. The upper-triangular future positions
/// receive a large negative additive mask before softmax.
final class CausalSelfAttention: Module {
    @ModuleInfo private var query: Linear
    @ModuleInfo private var key: Linear
    @ModuleInfo private var value: Linear
    @ModuleInfo private var output: Linear
    private let headCount: Int
    private let headWidth: Int

    init(width: Int, headCount: Int) {
        precondition(width % headCount == 0)
        self.query = Linear(width, width, bias: false)
        self.key = Linear(width, width, bias: false)
        self.value = Linear(width, width, bias: false)
        self.output = Linear(width, width, bias: false)
        self.headCount = headCount
        self.headWidth = width / headCount
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        func splitHeads(_ values: MLXArray) -> MLXArray {
            unflatten(values, axis: -1, shape: [headCount, headWidth])
                .transposed(0, 2, 1, 3)
        }

        let queries = splitHeads(query(input))
        let keys = splitHeads(key(input))
        let values = splitHeads(value(input))
        let sequenceLength = input.dim(1)
        let scale = Float(1) / Float(Foundation.sqrt(Double(headWidth)))
        let maskValues = (0..<sequenceLength).flatMap { queryPosition in
            (0..<sequenceLength).map { keyPosition in
                keyPosition <= queryPosition ? Float(0) : Float(-1e9)
            }
        }
        let causalMask = MLXArray(maskValues).reshaped([
            1, 1, sequenceLength, sequenceLength,
        ])
        let scores = matmul(queries, keys.transposed(0, 1, 3, 2)) * scale + causalMask
        let attentionWeights = softmax(scores, axis: -1, precise: true)
        let attended = matmul(attentionWeights, values)
        let mergedHeads = attended.transposed(0, 2, 1, 3).flattened(start: -2, end: -1)
        return output(mergedHeads)
    }
}

final class DecoderBlock: Module {
    @ModuleInfo private var attentionNorm: LayerNorm
    @ModuleInfo private var attention: CausalSelfAttention
    @ModuleInfo private var mlpNorm: LayerNorm
    @ModuleInfo private var mlpInput: Linear
    @ModuleInfo private var mlpOutput: Linear

    init(width: Int, headCount: Int) {
        precondition(headCount > 0 && width % headCount == 0)
        self.attentionNorm = LayerNorm(dimensions: width)
        self.attention = CausalSelfAttention(width: width, headCount: headCount)
        self.mlpNorm = LayerNorm(dimensions: width)
        self.mlpInput = Linear(width, width * 4)
        self.mlpOutput = Linear(width * 4, width)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let attentionOutput = input + attention(attentionNorm(input))
        let mlpOutput = mlpOutput(gelu(mlpInput(mlpNorm(attentionOutput))))
        return attentionOutput + mlpOutput
    }
}

// Engine operations are awaited serially by the example. The caller does not
// access the module while an awaited operation is mutating its parameters.
public final class TinyDecoder: Module, TrainableModel, @unchecked Sendable {
    public typealias Input = MLXArray
    public typealias Output = MLXArray

    @ModuleInfo private var tokenEmbedding: Embedding
    @ModuleInfo private var positionEmbedding: Embedding
    @ModuleInfo private var block0: DecoderBlock
    @ModuleInfo private var block1: DecoderBlock
    @ModuleInfo private var finalNorm: LayerNorm
    private let maximumSequenceLength: Int

    public init(
        vocabularySize: Int,
        width: Int,
        headCount: Int,
        maximumSequenceLength: Int
    ) {
        precondition(vocabularySize > 1)
        precondition(headCount > 0 && width > 0 && width % headCount == 0)
        precondition(maximumSequenceLength > 0)
        self.tokenEmbedding = Embedding(embeddingCount: vocabularySize, dimensions: width)
        self.positionEmbedding = Embedding(embeddingCount: maximumSequenceLength, dimensions: width)
        self.block0 = DecoderBlock(width: width, headCount: headCount)
        self.block1 = DecoderBlock(width: width, headCount: headCount)
        self.finalNorm = LayerNorm(dimensions: width)
        self.maximumSequenceLength = maximumSequenceLength
        super.init()
    }

    public func callAsFunction(_ tokenIDs: MLXArray) -> MLXArray {
        let sequenceLength = tokenIDs.dim(1)
        precondition(sequenceLength <= maximumSequenceLength, "Input sequence exceeds configured length")

        let positions = MLXArray((0..<sequenceLength).map(Int32.init))
        let embeddings = tokenEmbedding(tokenIDs) + positionEmbedding(positions)
        let hidden = block1(block0(embeddings))
        return tokenEmbedding.asLinear(finalNorm(hidden))
    }
}

public struct NextTokenObjective: Objective {
    public typealias Output = MLXArray
    public typealias Target = MLXArray

    public init() {}

    public func callAsFunction(output: MLXArray, target: MLXArray) -> MLXArray {
        MLXNN.crossEntropy(logits: output, targets: target, reduction: .mean)
    }
}
