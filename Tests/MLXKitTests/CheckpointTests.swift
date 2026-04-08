import Testing
import Foundation
@testable import MLXKit

@Suite("CheckpointError")
struct CheckpointErrorTests {
    @Test func fileNotFoundError() {
        let url = URL(filePath: "/nonexistent/path/model.safetensors")
        let error = CheckpointError.fileNotFound(url)
        #expect(error.errorDescription?.contains("not found") == true)
    }
}
