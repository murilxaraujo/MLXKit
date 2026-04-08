.PHONY: build test test-gpu test-all clean docs docs-preview run-example

# Build the library
build:
	swift build

# Run unit tests (no GPU required, works in CI and on any Mac)
test:
	swift test

# Run GPU tests only (requires Apple Silicon + Metal + Xcode)
test-gpu:
	xcodebuild test \
		-scheme MLXKit-Package \
		-destination 'platform=macOS' \
		-only-testing:MLXKitTests \
		OTHER_SWIFT_FLAGS='-DMLX_GPU_TESTS' \
		-parallel-testing-enabled NO \
		2>&1 | tail -30

# Run all tests (unit via SPM + GPU via xcodebuild)
test-all: test test-gpu

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
run-example:
	swift run MNISTExample
