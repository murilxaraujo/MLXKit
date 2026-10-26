# Platform and Backend Support

This matrix describes the MLXKit 2.0 support contract using the repository's
manifest, lockfile, CI workflow, and local validation. A package declaration or
successful compile alone is not proof that a configuration is supported.

## Supported baseline

| Dimension | Supported baseline | Evidence and limits |
|---|---|---|
| Operating system | macOS 27.x on Apple silicon | `Package.swift` declares macOS 27 as the deployment minimum. The current local validation host is macOS 27.0.1. CI selects Xcode 27.0 and now requires and records macOS 27.x. Later macOS major versions are not yet separately validated. |
| Architecture | Apple silicon (`arm64`) | The package requirements and MLX GPU test path target Apple silicon. CI records `uname -m`; a run is evidence for arm64 only if its log reports `arm64`. |
| Swift and Xcode | Swift 6.4 with Xcode 27.0 | `swift-tools-version` is 6.4, the local validation host uses Swift 6.4 / Xcode 27.0, and CI selects Xcode 27.0. Other compiler versions are not a tested compatibility promise. |
| MLX Swift | 0.32.3 for the validated dependency resolution | `Package.swift` requires `mlx-swift` from 0.32.3 and `Package.resolved` pins 0.32.3. Other versions permitted by the package's version range are not covered by this validation. |
| Backend | Metal on Apple silicon for GPU-dependent execution | CI runs GPU-dependent tests through Xcode with `MLX_GPU_TESTS`. The ordinary `swift test` path runs non-GPU-dependent tests; it is not evidence for a separate CPU compute backend. |

The CI workflow performs a library build, the non-GPU-dependent test suite,
example smoke runs, and both library and example GPU-dependent test suites on
its `xcode-27` runner. Its environment-reporting
step records the OS, architecture, Xcode, Swift, and resolved MLX Swift version.
Treat a CI run as validation of the exact environment printed in its log, not
of every environment covered by a broad version range.

## Experimental or unsupported configurations

- macOS 28 or later, newer Swift/Xcode versions, or MLX Swift versions other
  than the pinned 0.32.3 baseline are **experimental** until CI validates them.
- Intel Macs are **unsupported**; the published product requirement is Apple
  silicon and there is no Intel CI target.
- Linux, Windows, and non-Apple backends are **unsupported**; MLXKit has no
  package or CI target for them.
- A machine without Metal may run the non-GPU-dependent tests where its Swift
  toolchain can build the package, but it is not a supported MLXKit compute
  configuration.

## Validation commands

The CI workflow runs these checks on the supported baseline:

```bash
swift build
swift test
xcodebuild test \
  -scheme MLXKit \
  -destination 'platform=macOS' \
  -only-testing:MLXKitTests \
  OTHER_SWIFT_FLAGS='-DMLX_GPU_TESTS' \
  -parallel-testing-enabled NO
make smoke-examples
make test-examples-gpu
```

Run `make test-all` locally to include the GPU-dependent suites. GPU validation
requires Apple silicon and Metal; example validation also requires Tuist. A change to the declared support baseline
must update this matrix and add or revise an executable CI validation path.
