# MLXKit and MLX Responsibilities

MLXKit adds a higher-level training workflow to MLX Swift. It should coordinate
common tasks and provide useful extension points; it should not reproduce the
tensor, automatic differentiation, or hardware stack underneath it.

## Responsibility map

| Area | MLXKit owns | MLX and companion packages own |
|---|---|---|
| Model definition | A public contract for participating in MLXKit workflows and convenient composition points. | `MLXArray`, `MLXNN.Module`, built-in layers, parameter storage, and the primitives for implementing custom layers. |
| Numerical computation | Integration of model outputs with training objectives and evaluation. | Array operations, broadcasting, device dispatch, lazy evaluation, automatic differentiation, and gradient transforms. |
| Optimization | Training-loop coordination, schedule integration, and any higher-level controls MLXKit explicitly documents. | Optimizer implementations and their core parameter-update behavior in MLXOptimizers. |
| Training and evaluation | The optional fit/evaluate/predict workflow, lifecycle coordination, metric aggregation, callback behavior, and actionable validation of MLXKit configuration. | The forward pass, gradients, tensor evaluation, and device execution used by that workflow. |
| Data | MLXKit-facing sample, batching, loading, transform, and cancellation conventions. | Tensor representation and numerical operations applied to data. |
| Persistence | Higher-level distinctions, metadata, and user-facing error/reporting behavior that MLXKit documents. | Tensor weight encoding and underlying MLX serialization facilities where MLXKit delegates them. |

This table states the intended product boundary. It does not freeze API names,
protocol requirements, data shapes, or implementation details; M1 owns those
decisions.

## User workflows as boundary checks

### Conventional supervised workflow

A user defines a classifier with MLXNN modules, prepares data, and asks MLXKit
to coordinate training, validation, metrics, callbacks, and weight persistence.
MLXKit owns the orchestration and its documented semantics. MLX owns tensor
math, differentiation, and execution. A user can use MLX directly for the model
forward pass and can evaluate MLX arrays when needed.

### Custom research workflow

A user implements an embedding layer, attention operation, positional encoding,
or a paper-specific objective with public MLXKit extension points and MLX
operations. The user can use MLXKit's training loop where it fits or supply
custom training behavior when it does not. No step requires editing MLXKit
source, relying on internal symbols, or waiting for MLXKit to add every model
primitive.

## Direct MLX escape hatch

MLXKit is an optional orchestration layer, not a replacement for MLX. Users can
define custom `MLXNN.Module` types, use `MLXArray` operations in model and loss
code, and use MLX's differentiation APIs when they need a training path outside
the high-level engine. Custom behavior passed into MLXKit must be expressible
through public extension points; internal MLXKit implementation details are not
an extension contract.

## Decisions for M1

M1 must resolve these API-level questions without changing the responsibility
boundary above:

- How models express multiple or structured inputs and outputs while retaining
  MLX's dynamic tensor dimensions.
- Which operations are offered through MLXKit protocols versus ordinary MLX
  values and closures.
- How users replace or extend the standard training step for contrastive,
  token-level, or other objectives.
- Which data abstractions are required for in-memory and streaming sources.
- How optimizer and learning-rate schedule types compose without duplicating
  MLXOptimizers behavior.

The answer may be a small MLXKit surface plus direct MLX composition. It must
not require MLXKit to implement a parallel tensor or autodiff API.
