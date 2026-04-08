import Testing
import MLX
@testable import MLXKit

@Test func mlxKitModuleImports() async throws {
    // Verify that the MLXKit module links and DType constants are accessible.
    // Verify that the MLXKit module links and DType constants are accessible.
    let dtype: DType = .float32
    #expect(dtype == .float32)
}
