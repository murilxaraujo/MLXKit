# MLXJev

MLXJev is an inference product alongside MLXKit's training APIs. It defines a
shared typed decision request and response, with independent scoring strategies
for System One/qev-style and JevMLX-style inference.

## Use the strategy API

```swift
import MLXJev

let model = try await MLXLanguageModelAdapter.load(
    modelID: "mlx-community/Qwen3.5-2B-4bit"
)
let request = DecisionRequest(
    state: .string("A customer reports a duplicate card charge."),
    questions: [
        DecisionQuestion(
            id: "team",
            instructions: .string("Which team should handle this?"),
            kind: .choice(options: [
                DecisionOption(id: "billing", description: .string("charges and refunds")),
                DecisionOption(id: "technical", description: .string("product bugs")),
            ])
        ),
    ]
)

let systemOne = try await DecisionEngine(model: model, strategy: .systemOne).decide(request)
let jevMLX = try await DecisionEngine(
    model: model,
    strategy: .jevMLX(scoring: .labels)
).decide(request)
```

`StrategyBenchmarkRunner.compare(_:model:)` executes both strategies
sequentially on the same model and request, retaining a failure for any
unsupported strategy. It reports inference telemetry, not quality or parity;
those need a labeled evaluation set and separately versioned reference
fixtures.

## Current scope

- `DecisionQuestionKind` exposes boolean (`noul`), choice, ordinal (`score`),
  and multi-select inputs. Option arrays preserve caller order.
- `.systemOne` uses letter answer slots, one model read per question, and the
  System One choice and ordinal confidence formulas. It rejects multi-select.
- `.jevMLX(scoring: .slots)` reads tokenizer-verified one-token labels. The
  `.labels` path scores each candidate continuation with autoregressive log
  probabilities. Multi-select is a set of independent yes/no decisions.
- `MLXLanguageModelAdapter.load(modelID:)` uses the VLM factory and processor.
  Image-backed one-shot slot scoring is validated with OneJev 1.6. Video and
  media-conditioned continuation scoring remain unsupported.
- `MLXLanguageModelAdapter.loadLanguageModel(modelID:)` uses the LLM factory
  for text-only models, including Qwen2.5 1.5B Instruct 4-bit.
- This correctness-first baseline does not share KV prefixes across questions.
  Cache branching, video processing, calibration loading, and Python golden-logit
  parity are follow-up work.
- `DecisionLimits` bounds questions, options, combined input bytes, media bytes,
  token-prefix length, and model reads. Callers should also enforce an external deadline for
  model loading and GPU operations. Label scoring fails closed if the tokenizer
  cannot preserve the rendered prompt at the continuation boundary.

Confidence is strategy-specific. System One's formulas are not JevMLX
probabilities or calibrated confidence, and the unified result records their
semantics rather than implying they are interchangeable.
