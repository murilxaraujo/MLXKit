# MLXKit 2.0 Product Scope

## Product promise

MLXKit is a Swift-native framework that makes common machine-learning workflows
on MLX easier to express while preserving the ability to build models that the
framework authors did not anticipate. It provides useful higher-level
conventions; it does not replace MLX or restrict users to built-in model types.

Keras is a reference for an approachable workflow and clear defaults. It is not
a feature-parity target. MLXKit should use Swift and MLX idioms, and expose a
public path to MLX operations when a higher-level MLXKit facility is not enough.

## Users and representative workflows

### Swift developers prototyping a model

These users want to move from data and a model definition to training and
evaluation without writing the routine loop themselves.

**Representative workflow:** define a small classifier with MLX modules, load
or construct a dataset, configure an optimizer and objective, train with
validation, inspect metrics, run predictions, and save model weights. The
common path should be discoverable and concise, with actionable errors when
inputs or configuration are invalid.

### Engineers integrating model training into an application or product

These users need training behavior that is predictable, testable, and
observable, and persistence that states what can and cannot be restored. They
need to know which hardware, operating-system, Swift, and MLX configurations
are supported before they depend on them.

**Representative workflow:** use a documented platform configuration, train
and evaluate a model with explicit data and metric semantics, handle
cancellation or failures, persist the required model or training state, and
validate the workflow with tests and a runnable example. The exact support
contract is defined by the M0 platform matrix, not assumed here.

### Researchers and engineers building custom models

These users need to implement embedding models, language-model components, and
paper-specific architectures or objectives. They need flexible tensor shapes,
custom model and training logic, and access to MLX operations.

**Representative workflow:** implement a custom layer or Transformer component,
including behavior such as positional encoding, attention, or masking; define a
custom differentiable objective or training step; verify gradients and
parameter updates; then train and evaluate using MLXKit where useful. This work
must be possible through public MLXKit extension points and MLX APIs. Users must
not need to edit MLXKit's source or depend on its internal implementation.

## Required outcomes for 2.0

1. **A useful high-level path.** A user can express a conventional model
   workflow without implementing routine training, evaluation, metric, and
   persistence orchestration from scratch.
2. **An open custom-model path.** Public APIs can express custom models,
   objectives, metrics, data handling, and training steps as needed. MLX remains
   directly usable for lower-level operations and custom components.
3. **Correct, explicit behavior.** Training, data, metrics, callbacks, and
   persistence have documented semantics. Invalid or unsupported requests fail
   clearly rather than silently doing something else.
4. **Validated claims.** Every configuration called supported has a stated
   validation path. Tests, examples, and benchmark results identify their
   workload and environment; performance claims are not generalized beyond
   their evidence.
5. **A usable 2.0 release.** Public API decisions, breaking changes, supported
   configurations, migration guidance, examples, and known limitations are
   documented before the 2.0.0 release.

## Optional outcomes

Convenience layers, losses, metrics, callbacks, data transforms, and training
controls are valuable when they serve demonstrated workflows. They are not
required as exhaustive catalogs. They may be added when they preserve the
custom-model path, have clear semantics, and can be tested and documented.

## Non-goals

- Reproducing all Keras features or matching another framework's API surface.
- Reimplementing MLX tensors, automatic differentiation, or core numerical
  operations inside MLXKit.
- Requiring all users to use MLXKit abstractions when direct MLX is more
  appropriate.
- Promising support for operating systems, architectures, backends, or toolchain
  versions that are not validated in the M0 support matrix.
- Claiming large-scale LLM training capability based only on small example
  models.
- Providing hosted training, managed datasets, model serving, or a catalog of
  pretrained models as part of the 2.0 product contract.
- Making universal performance claims or treating a TensorFlow/Keras comparison
  as an isolated framework comparison when the backend or implementation differs.

## 2.0 success criteria and roadmap traceability

The 2.0 release is successful when the following outcomes are reviewed and
evidenced. These are acceptance gates, not claims about the current release.

| Outcome and review evidence | Roadmap |
|---|---|
| Public API design is reviewed against both a conventional classifier and a custom Transformer-style model; both can be expressed without editing MLXKit internals. | M1 |
| Deterministic integration tests show expected gradients, parameter updates, and documented loss/metric aggregation; invalid configuration and key failure paths are tested. | M2 |
| A custom architecture and objective exercise public extension points and MLX operations, with tests for finite gradients and parameter updates. The example is not presented as proof of large-scale LLM training. | M3 |
| In-memory and streaming data paths are demonstrated; tests cover documented ordering, partial or empty input, errors, cancellation, and bounded prefetch where provided. | M4 |
| Training controls have documented mathematical and ordering semantics, and tests demonstrate the documented results or a clear error for unsupported combinations. | M5 |
| Weight loading, inference artifacts, and resumable checkpoints are distinguished; tests verify supported restored state and useful failures for corrupt or incompatible files. | M6 |
| Classifier, embedding, and small language-model example projects run from a clean checkout following their instructions; CI smoke-tests the supported short paths. | M7 |
| Local benchmarks compare equivalent direct-MLX and MLXKit work, repeat measurements, record environment and quality results, and report TensorFlow/Keras backend differences. Any overhead target is set only after a baseline is collected. | M8 |
| The release candidate passes the M0 support matrix and required tests; documentation includes the 1.x-to-2.0 migration path, platform limits, examples, and known limitations. No known critical issue can hang training, silently skip requested behavior, or falsely report successful persistence. | M9 |

M0 establishes the product and platform contract. The outcomes above map each
subsequent milestone to that contract. Later milestones remain planning drafts
until the preceding milestone is complete; API and implementation details are
decided in their respective milestones.

## M0 contract documents

- <doc:MLXKitAndMLX> defines the MLXKit/MLX responsibility boundary and open
  M1 API decisions.
- <doc:PlatformSupport> defines the tested platform and backend baseline.
- <doc:CompatibilityPolicy> defines pre-2.0 and 2.0 compatibility promises.
- <doc:ReferenceWorkloads> defines the workloads and validation method for M8.

## M1 design work

- <doc:M1ModelContract> records the approved model-contract direction for M1.
  The integrated API design and migration map remain in progress; this is not
  the final 2.0 API promise.
- <doc:M1ObjectiveMetricContract> records the pair-approved objective/metric
  direction while implementation semantics remain assigned to M2.
