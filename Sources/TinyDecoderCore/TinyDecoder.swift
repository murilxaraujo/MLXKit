import Foundation
import MLX
import MLXNN
import MLXKit

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

        let attended = MLXFast.scaledDotProductAttention(
            queries: splitHeads(query(input)),
            keys: splitHeads(key(input)),
            values: splitHeads(value(input)),
            scale: Float(1) / Float(Foundation.sqrt(Double(headWidth))),
            mask: .causal
        )
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

public final class TinyDecoder: Module, TrainableModel {
    public typealias Input = MLXArray
    public typealias Output = MLXArray

    @ModuleInfo private var tokenEmbedding: Embedding
    @ModuleInfo private var positionEmbedding: Embedding
    @ModuleInfo private var block: DecoderBlock
    @ModuleInfo private var finalNorm: LayerNorm
    private let maximumSequenceLength: Int

    public init(vocabularySize: Int, width: Int, headCount: Int, maximumSequenceLength: Int) {
        precondition(vocabularySize > 1)
        precondition(width > 0 && width % headCount == 0)
        precondition(maximumSequenceLength > 0)
        self.tokenEmbedding = Embedding(embeddingCount: vocabularySize, dimensions: width)
        self.positionEmbedding = Embedding(embeddingCount: maximumSequenceLength, dimensions: width)
        self.block = DecoderBlock(width: width, headCount: headCount)
        self.finalNorm = LayerNorm(dimensions: width)
        self.maximumSequenceLength = maximumSequenceLength
        super.init()
    }

    public func callAsFunction(_ tokenIDs: MLXArray) -> MLXArray {
        let sequenceLength = tokenIDs.dim(1)
        precondition(sequenceLength <= maximumSequenceLength, "Input sequence exceeds configured length")

        let positions = MLXArray((0..<sequenceLength).map(Int32.init))
        let embeddings = tokenEmbedding(tokenIDs) + positionEmbedding(positions)
        let hidden = block(embeddings)
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
