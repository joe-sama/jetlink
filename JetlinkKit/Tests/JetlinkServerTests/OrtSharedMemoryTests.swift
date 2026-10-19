import Foundation
import Testing

@testable import JetlinkORT

@Suite("QNN shared memory", .serialized)
struct OrtSharedMemoryTests {
  @Test("A CPU session cannot silently substitute heap memory for QNN RPC memory")
  func cpuDoesNotPretendToShare() throws {
    let session = try OrtSession(model: Golden("tiny_stateful").model, provider: nil)
    #expect(throws: OrtError.self) {
      _ = try OrtSharedAllocator(session: session)
    }
  }
}
