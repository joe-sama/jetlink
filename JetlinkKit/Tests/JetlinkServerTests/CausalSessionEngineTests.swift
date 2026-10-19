import Foundation
import Testing

@testable import JetlinkORT
@testable import JetlinkServer

@Suite("Same-frame recurrent session DAG", .serialized)
struct CausalSessionEngineTests {
  @Test("Real ORT bindings follow dependencies and loop the same state in either schedule", arguments: [false, true])
  func statefulBindings(parallel: Bool) throws {
    // Deliberately not topological: graph I/O, rather than array position,
    // identifies host tensors and orders the completion barriers.
    let plans = ["join", "decode", "history", "vision"].map {
      SessionPlan(model: Fixture.url("parallel_\($0).onnx"), provider: nil, label: "CPU")
    }
    let engine = try OrtEngine(
      plans: plans, device: "test", keepAlive: false, keepCPUWarm: false,
      dataflow: true, parallel: parallel)
    defer { engine.close() }
    #expect(Set(engine.inputs.keys) == ["x", "state_h"])
    #expect(Set(engine.outputs.keys) == ["outputs", "next_state_h"])
    try engine.loopState([("state_h", "next_state_h")])
    let x = try #require(engine.hostInput("x"))
    let output = try #require(engine.output("outputs"))
    x.storeBytes(of: Float(2), as: Float.self)
    for frame in 0..<20 {
      try engine.run()
      #expect(output.load(as: Float.self) == Float(5 + frame))
    }
    engine.resetState()
    try engine.run()
    #expect(output.load(as: Float.self) == 5)
  }

  @Test("Unsupported hardware is rejected before model preparation or loading")
  func unsupportedHardware() throws {
    let backend = OrtBackend(profile: .htpParallel, preparer: ONNXPreparer(), chip: "SM8650")
    #expect(throws: HostError.self) {
      try backend.build(model: Fixture.url("tiny_stateful.onnx"), artifact: URL(fileURLWithPath: "/unused"), report: { _, _, _ in }, metaExtra: [:])
    }
    #expect(throws: HostError.self) {
      _ = try backend.load(artifact: URL(fileURLWithPath: "/unused"), report: { _, _, _ in })
    }
  }
}
