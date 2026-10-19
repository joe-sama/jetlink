import Foundation
import JetlinkTestSupport
import Testing

@testable import JetlinkServer

@Suite("Hash-bound causal bundle", .serialized)
struct CausalPrefillBundleTests {
  @Test("The real manifest decodes, but an unvalidated source cannot be substituted")
  func sourceIdentity() throws {
    let temporary = try TemporaryDirectory()
    let bundle = temporary.url.appending(path: "bundle")
    let staged = temporary.url.appending(path: "staged")
    try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: staged, withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: Fixture.url("causal_prefill_manifest.json"), to: bundle.appending(path: "manifest.json"))
    let previous = ProcessInfo.processInfo.environment["JETLINK_PREFILL_BUNDLE"]
    setenv("JETLINK_PREFILL_BUNDLE", bundle.path, 1)
    defer {
      if let previous { setenv("JETLINK_PREFILL_BUNDLE", previous, 1) } else { unsetenv("JETLINK_PREFILL_BUNDLE") }
    }
    #expect(throws: HostError.self) {
      _ = try CausalPrefillBundle.prepare(model: Fixture.url("tiny_stateful.onnx"), into: staged)
    }
    #expect(try FileManager.default.contentsOfDirectory(atPath: staged.path).isEmpty)
    // A valid model cannot make an altered manifest acceptable either.
    try Data("{}".utf8).write(to: bundle.appending(path: "manifest.json"))
    #expect(throws: HostError.self) {
      _ = try CausalPrefillBundle.prepare(model: Fixture.url("tiny_stateful.onnx"), into: staged)
    }
  }
}
