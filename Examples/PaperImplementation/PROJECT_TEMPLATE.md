# <Project name>: paper component implementation

> Copy this file to `Examples/<ProjectName>/README.md`. Replace every
> bracketed prompt. Remove only sections that do not apply, and explain why.

## Outcome and scope

- **Paper component/result in scope:** [one precise component or claim]
- **Not in scope:** [larger results or systems this project does not reproduce]
- **Example product/target:** [SwiftPM product and target, or why no executable
  is needed]
- **Capability boundary:** [what a successful run demonstrates and does not]

## Citation, version, and license

- **Citation:** [authors, title, venue, year, pages]
- **Version/source:** [arXiv or publisher URL, version/revision, access date]
- **Code/assets license notes:** [paper, reference implementation, and any
  reused assets; state when this project uses no third-party code/assets]
- **Implementation status:** [from-scratch component, adapted code, or API
  composition; link exact source files]

## Requirements and commands

- **Supported baseline:** [OS, hardware, Xcode/Swift, MLX version]
- **Network/data requirements:** [resolve dependencies separately; say whether
  any run-time data or downloads are required]
- **Build:** `swift build --product <ExampleProduct>`
- **Full/local run:** `swift run <ExampleProduct>`
- **Smoke:** `swift run <ExampleProduct> --smoke`
- **Tests:** `swift test --filter <RelevantTestName>`
- **Hardware-gated tests, if any:** [command and required hardware]

## Architecture and tensor shapes

Describe the paper component and its relationship to the complete example.
Record each tensor shape with named axes, for example:

| Value | Shape | Dtype | Notes |
|---|---|---|---|
| Input | `[batch, sequence, feature]` | [dtype] | [mask/padding convention] |
| Intermediate | [shape] | [dtype] | [operation] |
| Output | [shape] | [dtype] | [interpretation] |

- **Layers/operations and order:** [include residuals, normalization, masks,
  activation, ties, and any omitted paper components]
- **Initialization:** [paper initializer, explicit MLX initializer, or MLX
  module default; include seed and what it controls]
- **Custom MLX implementation:** [file and key operations; public MLX APIs used]
- **MLXKit boundary:** [model, objective, batching, engine, metrics, and
  callbacks used; identify any custom scalar `TrainingStep`]

## Objective and training configuration

Write the objective as a formula and explain labels/target alignment, masking,
and reduction. Then record:

| Setting | Value | Paper value | Deviation/rationale |
|---|---|---|---|
| Optimizer | [name and version] | [paper] | [why] |
| Learning rate/schedule | [value] | [paper] | [why] |
| Batch size/update count | [values] | [paper] | [why] |
| Seed/dtype | [values] | [paper] | [why] |

## Fixture and evaluation

- **Fixture provenance/generator:** [checked-in, generated, or optional external
  data; include exact generation parameters and license/source]
- **Train/validation/test split:** [fixed IDs, seeds, and split boundaries]
- **Evaluation metric:** [definition and reduction]
- **Correctness checks:** [finite values/gradients, shape, invariants, parameter
  update, learning trend, direct-MLX comparison, etc.]
- **Expected local result:** [fixture-specific range or invariant; avoid broad
  quality claims]
- **What this fixture does not test:** [paper-scale, external validity, and
  out-of-scope conditions]

## Inference artifact and reconstruction

- **Artifact kind and file format:** [inference weights, not training state]
- **Stored weights:** [which module parameters]
- **Caller-owned metadata:** [architecture/config version, vocabulary or label
  mapping, preprocessing, source/seed, and other required values]
- **Reconstruction:** [how code rebuilds the model and preprocessing before
  loading weights]
- **Round-trip test/tolerance:** [input, expected outputs, `rtol`/`atol`]
- **Optimizer state:** [whether supported; do not imply that a weights-only
  inference artifact resumes training]

## Reproduction steps and results

List exact commands from a clean checkout after SwiftPM dependencies have been
resolved. Record OS, hardware, Swift/Xcode/MLX versions, smoke/full mode, seed,
and observed results. Distinguish required results from informational logs.

| Check | Command | Result | Notes/skips |
|---|---|---|---|
| Build | [command] | [pass/fail] | [environment] |
| Smoke | [command] | [pass/fail] | [runtime or skip reason] |
| Tests | [command] | [pass/fail] | [hardware gate, if any] |

## Deviations from the paper

List every known architecture, initialization, objective, data, optimizer,
training-budget, evaluation, or artifact deviation. Explain why it exists and
which result it can affect. “None” is a substantive claim; use it only after a
specific comparison.

## Limitations and claims

- **Claims supported by this project:** [specific local observations]
- **Claims not supported:** [paper claims, broad quality, scaling, performance,
  or generalization not measured here]
- **Known limitations:** [small fixture, missing components, hardware/version
  sensitivity, reproducibility boundaries]
- **Next validation needed:** [larger/independent evidence, if relevant]
