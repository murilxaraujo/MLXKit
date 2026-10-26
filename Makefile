.PHONY: build build-release test test-gpu test-examples-gpu test-all smoke-examples examples-install examples-generate release-check clean docs docs-preview run-example run-contrastive-embedding run-tiny-decoder run-data-pipeline

# Build the library
build:
	swift build

# Build optimized products as a release-candidate readiness check.
build-release:
	swift build -c release

# Run unit tests (no GPU required, works in CI and on any Mac)
test:
	swift test

# Run GPU tests only (requires Apple Silicon + Metal + Xcode)
test-gpu:
	set -o pipefail; xcodebuild test \
		-scheme MLXKit \
		-destination 'platform=macOS' \
		-only-testing:MLXKitTests \
		OTHER_SWIFT_FLAGS='-DMLX_GPU_TESTS' \
		-parallel-testing-enabled NO \
		2>&1 | tail -30

# Run all tests (SPM unit tests plus library and example GPU suites)
test-all: test test-gpu test-examples-gpu

# Full release-readiness gate. Requires Apple silicon, Metal, and Xcode for GPU tests.
# Keep the subcommands sequential so SwiftPM does not concurrently mutate .build.
release-check:
	$(MAKE) clean
	$(MAKE) build-release
	$(MAKE) test
	$(MAKE) docs
	$(MAKE) smoke-examples
	$(MAKE) test-gpu
	$(MAKE) test-examples-gpu

# Resolve the standalone Tuist project's dependencies and generate its Xcode project.
examples-install:
	tuist install --path Examples

examples-generate: examples-install
	tuist generate --path Examples --no-open

# Bounded, deterministic local-data smoke runs for the example portfolio.
smoke-examples: examples-generate
	tuist run --path Examples MNISTExample --smoke
	tuist run --path Examples ContrastiveEmbeddingExample --smoke
	tuist run --path Examples TinyDecoder --smoke

# GPU-dependent example implementation tests (requires Apple Silicon + Metal + Xcode).
test-examples-gpu: examples-generate
	set -o pipefail; xcodebuild test \
		-project Examples/MLXKitExamples.xcodeproj \
		-scheme TinyDecoder \
		-destination 'platform=macOS' \
		-parallel-testing-enabled NO \
		2>&1 | tail -30

# Clean build artifacts
clean:
	swift package clean

# Generate DocC documentation
docs:
	swift package generate-documentation --target MLXKit

# Preview DocC documentation in browser
docs-preview:
	swift package --disable-sandbox preview-documentation --target MLXKit

# Run the MNIST example
run-example: examples-generate
	tuist run --path Examples MNISTExample

# Run the deterministic paired-view contrastive embedding example
run-contrastive-embedding: examples-generate
	tuist run --path Examples ContrastiveEmbeddingExample

# Run the deterministic custom tiny-decoder example
run-tiny-decoder: examples-generate
	tuist run --path Examples TinyDecoder

# Run the local in-memory and streaming data-pipeline example
run-data-pipeline: examples-generate
	tuist run --path Examples DataPipelineExample
