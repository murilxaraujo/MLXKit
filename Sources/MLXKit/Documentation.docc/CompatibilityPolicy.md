# Compatibility and Release Policy

## Before 2.0.0

MLXKit is in active pre-2.0 development. Source-breaking API changes are allowed
in any pre-2.0 release, including 1.x releases. Do not interpret a 1.x version
number as a promise that source code will compile unchanged with every later
pre-2.0 release. Release notes must identify breaking changes and give a
migration note when a change affects users.

The version number still communicates release scale, but pre-2.0 source
compatibility takes priority over preserving an API that blocks the 2.0 product
contract. Patch releases should normally contain fixes; if a breaking change is
necessary in a pre-2.0 patch, it must be called out explicitly rather than
hidden behind the patch number.

## Starting with 2.0.0

The public source API is stable within a 2.x major version:

- Patch releases contain compatible fixes.
- Minor releases add functionality without removing or changing existing
  documented behavior incompatibly. Deprecations must include a replacement or
  clear migration advice.
- A source-breaking change requires a new major version and release notes with
  migration guidance.

This is a source-package compatibility policy. MLXKit does not promise binary
framework or ABI compatibility. Applications must rebuild against the Swift,
Xcode, operating-system, and MLX Swift versions listed as supported in
<doc:PlatformSupport>.

## 1.x to 2.0 migration

The 2.0 release must include a migration guide with before-and-after examples
for changed model, data, training, callback, and persistence workflows. The
guide must call out removed APIs, changed semantics, and any required changes to
custom model or training code. Compatibility shims are not promised; the guide
must identify any shims that are actually provided rather than imply they
exist.

## Limits of compatibility promises

- Pre-2.0 releases have no source-compatibility guarantee across releases.
- MLXKit cannot promise compatibility with MLX Swift versions outside the
  validated platform matrix or override upstream MLX API and behavior changes.
- No binary/ABI stability is promised.
- Cross-version compatibility is limited to the artifact versions and legacy
  readers explicitly documented in <doc:M6PersistenceSemantics>. Weight files,
  inference artifacts, progress metadata, and optimizer-aware training
  checkpoints are distinct and must not be assumed interchangeable.
- The supported platform/toolchain matrix is the limit of tested claims; a
  package manifest's version range does not itself establish validation.
