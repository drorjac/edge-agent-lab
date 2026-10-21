import Foundation
import CoreML

// Normalization is embedded in the exported model. Feed original sensor units.
let labels = ["walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying"]

enum DemoError: Error, CustomStringConvertible {
    case usage
    case invalidInput
    case missingOutput
    var description: String {
        switch self {
        case .usage: return "Usage: activity-demo MODEL.mlpackage demo_input.json"
        case .invalidInput: return "Expected sensors with shape [1, 9, 128] and finite values"
        case .missingOutput: return "Model did not return six logits"
        }
    }
}

func run() throws {
    guard CommandLine.arguments.count == 3 else { throw DemoError.usage }
    let source = URL(fileURLWithPath: CommandLine.arguments[1])
    let inputURL = URL(fileURLWithPath: CommandLine.arguments[2])
    let data = try Data(contentsOf: inputURL)
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let batch = object["sensors"] as? [[[Double]]], batch.count == 1,
          batch[0].count == 9,
          batch[0].allSatisfy({ $0.count == 128 && $0.allSatisfy(\.isFinite) }) else {
        throw DemoError.invalidInput
    }
    let input = try MLMultiArray(shape: [1, 9, 128], dataType: .float32)
    for channel in 0..<9 {
        for sample in 0..<128 {
            input[[0, NSNumber(value: channel), NSNumber(value: sample)]] =
                NSNumber(value: batch[0][channel][sample])
        }
    }
    let compiled = try MLModel.compileModel(at: source)
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .cpuOnly
    let model = try MLModel(contentsOf: compiled, configuration: configuration)
    let features = try MLDictionaryFeatureProvider(dictionary: ["sensors": MLFeatureValue(multiArray: input)])
    let prediction = try model.prediction(from: features)
    guard let output = prediction.featureValue(for: "logits")?.multiArrayValue,
          output.count == 6 else { throw DemoError.missingOutput }
    let logits = (0..<6).map { output[$0].doubleValue }
    let maximum = logits.max()!
    let expValues = logits.map { exp($0 - maximum) }
    let total = expValues.reduce(0, +)
    let probabilities = expValues.map { $0 / total }
    let best = probabilities.enumerated().max(by: { $0.element < $1.element })!.offset
    let result: [String: Any] = ["activity": labels[best], "probabilities": probabilities,
                               "compute_units": "CPU_ONLY", "source": source.lastPathComponent]
    let resultData = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
    print(String(decoding: resultData, as: UTF8.self))
}

do { try run() } catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
