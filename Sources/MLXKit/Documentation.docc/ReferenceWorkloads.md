# Reference Workloads and Quality Goals

This plan selects small, repeatable workloads for validating MLXKit 2.0 and
estimating orchestration overhead against direct MLX. It defines the benchmark
questions; it does not implement a benchmark harness or claim results.

## Workloads

| Workload | Fixed input and model shape | MLXKit capability and correctness checks | Measurements |
|---|---|---|---|
| Dense supervised classifier | Deterministic synthetic or checked-in classification fixture; input `[32, 784]`, 10 output classes, two dense layers (128 hidden units). | Train/evaluate/predict, loss and accuracy aggregation, callbacks, and weight save/load. Verify finite loss and outputs, equal initial weights, and direct-MLX versus MLXKit loss/gradient/update agreement within a documented numerical tolerance. On the fixed fixture, verify the expected learning trend rather than a universal accuracy threshold. | Training steps/second and samples/second; steady-state inference latency; optional save/load time. |
| Contrastive embedding model | Paired feature vectors `[32, 2, 128]`; projection to 64 dimensions; in-batch contrastive objective with fixed seed and temperature. | Custom model and differentiable objective, paired/structured data, and custom training behavior where needed. Verify finite gradients and updates, normalized embedding behavior, and equivalent direct-MLX and MLXKit objective values and updates. | Training steps/second; pairs/second; embedding inference latency. |
| Small decoder-only language model | Token IDs `[4, 64]`; vocabulary 256; width 128; 4 attention heads; 2 decoder blocks; causal mask. | Variable sequence dimensions, custom positional or attention behavior, token-level loss, and short autoregressive inference. Verify causal masking, finite gradients, matching direct-MLX and MLXKit results within tolerance, and a learning trend on a fixed tiny fixture. This workload validates an implementation path, not large-scale LLM capability. | Training tokens/second; next-token inference latency; peak memory where reliably measurable. |

Fixtures should be deterministic, local, and small enough for a quick run on the
supported baseline. No network download is required for the correctness path.
If a later example uses a downloaded dataset, it must document a reproducible
setup separately.

## Correctness before performance

For each pairwise direct-MLX/MLXKit comparison:

1. Use the same MLX version, hardware, dtype, fixed seed, input tensors, model
   architecture, initial parameters, optimizer, and mathematical objective.
2. Compare forward outputs, loss, gradients, and one or more parameter updates
   within tolerances justified for the dtype and operation. Report tolerances;
   do not require bitwise equality across devices or versions.
3. Check finite values and a workload-specific quality signal before accepting
   timing results. Keep integration correctness gates separate from benchmark
   performance results.
4. Measure a complete MLXKit workflow against an equivalent direct-MLX loop.
   The difference estimates the cost of the MLXKit orchestration for that
   workload; it is not a universal overhead constant.

## Measurement method

- Record the model identifier, macOS build, architecture, memory, Xcode and
  Swift versions, MLX Swift version, MLXKit revision, backend, dtype, shapes,
  batch/sequence lengths, seed, optimizer, and benchmark mode.
- Warm up each workload for at least 20 steps. Then measure 5 independent
  repeats of 100 steady-state steps (or the nearest complete workload unit).
  Report the median and spread, not only the fastest run.
- Keep setup, first-use/lazy evaluation, and steady-state timing distinct.
  Ensure MLX work is evaluated before stopping the timer; state exactly which
  work the timer includes.
- Report throughput and latency with units. Record memory only when the same
  measurement method is available for both implementations.
- Avoid concurrent heavy workloads and report material machine state that can
  affect results. Provide a quick mode for development and a repeated mode for
  comparison.

These sample counts are a starting protocol. M8 may revise them after measuring
runtime and variance, but any change must be documented and applied equally to
both implementations.

## Framework comparisons and target setting

Direct MLX versus MLXKit is the primary comparison for estimating MLXKit
orchestration overhead. TensorFlow/Keras comparisons are optional and valid only
when an equivalent TensorFlow backend and workload are available. Record the
actual backend (for example, Metal or CPU) and implementation differences;
otherwise report the comparison as a comparison of complete stacks, not
framework-only overhead.

Do not set a universal numeric overhead target in M0. M8 must first collect a
baseline using this method, then propose per-workload targets with evidence and
measurement variance. Any target must include the workload, hardware/backend,
and acceptance method it applies to.
