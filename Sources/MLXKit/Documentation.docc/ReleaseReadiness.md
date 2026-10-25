# Release readiness and public API audit

This page records the repeatable pre-release checks and the first source-level
API disposition review for MLXKit 2.0. It is not a declaration that 2.0.0 has
been released or that its release-candidate gate has passed. See
<doc:CompatibilityPolicy>, <doc:PlatformSupport>, and
<doc:MigrationGuide> for the contracts behind this checklist.

## Release-readiness command

On the supported Apple-silicon/macOS environment, run from a clean checkout
(the target also clears SwiftPM build products first):

```sh
make release-check
```

The target runs these checks sequentially:

1. `swift build -c release`
2. `swift test`
3. `make docs`
4. `make smoke-examples`
5. `make test-gpu` (requires Apple silicon, Metal, and Xcode)

`make test-gpu` preserves the `xcodebuild` exit status when truncating its
displayed log. A green tail command alone must not hide an Xcode test failure.
The release-candidate run must also record the exact commit, OS, architecture,
Xcode, Swift, resolved MLX Swift version, and test outcomes. Run full example
commands and the M8 benchmark procedure from a clean checkout before the
release; smoke timings and fixture metrics are not benchmark or quality claims.

## Initial public API disposition

The source-level review compared the public declarations and current contracts
with the tagged 1.0.0 API, `API2Design`, M2–M6 semantics, and the M0
compatibility/support policy. The review is not an ABI audit; MLXKit does not
promise binary compatibility.

| Surface reviewed | 2.0 disposition | Migration/release note |
|---|---|---|
| `TrainableModel`, `TrainingConfiguration`, `TrainingEngine`, objectives, metrics, and typed errors | Keep as the redesigned public training contract. | The `forward`/`compile` model has intentional source-breaking changes; see <doc:MigrationGuide>. Review any addition/removal against `API2Design` before the 2.0 tag. |
| Generic `Sample` and `Batch` with array conveniences | Keep generic input/target/sample-count storage and the array-specific `features`, `labels`, and `size` convenience API. | New examples use the generic names; the conveniences do not preserve the old `DataLoader` initializer. |
| `Dataset` and `InMemoryDataset` | Keep the indexed array convenience while `DataLoader` also accepts typed async sample sources. | Do not imply every source is indexed or known-length. |
| `RunningLoss` | Keep as a public standalone accumulator, explicitly not a `Metric`. | It remains unsuitable for a metric array; engine loss reporting is trainer-owned. |
| Deprecated `Checkpointing.save(model:epoch:step:to:)` and `load(into:from:)` | Keep the deprecated wrappers with their documented progress-only guarantee. | Direct callers to `saveTrainingProgress` or an optimizer-aware `saveTrainingCheckpoint`; do not imply optimizer restoration. |
| `CheckpointableOptimizer` and typed checkpoint errors | Keep as the opt-in extension point and explicit failure boundary. | Only claim resume support for conformers with tested validation and restore behavior. |
| Public `@unchecked Sendable` reference types | Retain only where required by the existing async/callback boundary; this is not a promise that model operations or mutable instances are generally thread-safe. | Serialize training and access to a model/optimizer; any stronger concurrency guarantee requires separate review and tests. |

No removal is proposed solely from this initial audit. A release-candidate review
must check the final public surface after M7/M8 integration and record any
remaining provisional behavior. Source API stability begins at 2.0.0; future
2.x changes must follow the compatibility policy. No ABI stability or
universal cross-version artifact compatibility is promised.

### Existing regression evidence reviewed

- Training lifecycle, callback failures, cancellation, and throwing batch
  sequences: `TrainingEngineTests.swift` and `TrainingEngineGPUTests.swift`.
- Custom decoder causality, logits shape, gradients, and parameter updates:
  `TinyDecoderGPUTests.swift`; the end-to-end custom workflows run through the
  M7 example smoke commands.
- Artifact validation, atomic replacement, corrupt/unknown versions, optimizer
  restore, and interrupted/resumed training: `CheckpointGPUTests.swift`.
- Data-source, transform, prefetch failure, and cooperative cancellation:
  `DataPipelineTests.swift` and `PrefetchedSequenceTests.swift`.

The release candidate must rerun these suites and review their results on the
supported CI environment; this source review does not replace that gate.

## Release note and deferred-issue templates

Use these headings for the GitHub 2.0.0 release:

1. **Summary** — What 2.0.0 provides.
2. **Breaking changes** — User-visible source and behavior changes, with links
   to the matching migration steps.
3. **Migration** — Link to <doc:MigrationGuide> and note that compatibility
   shims exist only where they are explicitly listed.
4. **Supported environment** — Exact tested OS, architecture, Swift/Xcode,
   MLX Swift version, and backend, linked to <doc:PlatformSupport>.
5. **Validation** — Candidate revision and release-check results; link to the
   benchmark method without presenting one machine's timing as universal.
6. **Known limitations** — Unsupported platforms, optimizer/checkpoint limits,
   benchmark constraints, and any deferred work.

Each deferred issue should record:

| Field | Required detail |
|---|---|
| Issue | Link and owner |
| Impact | Affected users/workflows and severity |
| Workaround | Available workaround, or `none known` |
| Disposition | Why it does not block this release and when it will be revisited |

## Release-candidate gate

The 2.0.0 candidate is not ready until all of these are true:

- [ ] M0–M8 acceptance criteria are complete. M7 examples are merged and M8
  benchmark methodology is implemented and documented.
- [ ] Each configuration called supported has executable CI evidence. Anything
  else is clearly marked experimental or unsupported.
- [ ] The migration guide reflects actual source and behavior changes from
  1.0.0 and has been checked against the final public API.
- [ ] `make release-check` passes on the exact candidate commit and its complete
  environment is recorded. Full examples and M8 benchmark commands are also
  verified from a clean checkout.
- [ ] No unresolved release-blocker can cause training to hang, silently skip
  requested behavior, partially accept a failed restore, or report a failed
  save as successful. Deferred defects have a documented impact and owner.
- [ ] Release notes state the source-compatibility policy, tested support
  baseline, dependency baseline, migration guidance, and known limitations.
- [ ] The tagged package resolves and builds in a clean consumer checkout.

Do not create the `2.0.0` tag or GitHub release before this gate passes.

## Phase A validation record

Validated on 2026-10-25 with macOS 27.0.1, Apple silicon (`arm64`), Xcode
27.0, Swift 6.4, and the resolved MLX Swift 0.32.3 dependency:

| Check | Result |
|---|---|
| `swift build -c release` | Passed |
| `swift test` | Passed, 92 tests |
| `make docs` | Passed; both new pages are included in the DocC catalog |
| `make smoke-examples` | Passed for classifier, embedding, and decoder examples |
| `make test-gpu` | Passed, 189 tests; the `xcodebuild` failure status is preserved through log truncation |
| `git diff --check` | Passed |

The complete `make release-check` command passed. This is local Phase A evidence,
not release-candidate approval: M7 PR #40 remains unmerged and M8 benchmark
implementation and full release validation are still required before an RC or
2.0.0 tag. A final `make docs` rerun after moving the examples link into DocC's
`See Also` section completed without DocC content warnings. The MLX Swift build
emits C++17-extension warnings from its generated Metal sources; these are
dependency warnings, not MLXKit compile errors. The hosted CI result for these
M9 changes has not run yet.
