import Foundation

public enum NemotronError: Error, LocalizedError, Sendable {
    case invalidPackage(String)
    case invalidRuntimeSupport(String)
    case modelAssetUnavailable(String)
    case modelLoadFailed(aot: String?, source: String)
    case missingFunction(String)
    case inference(String)
    case invalidFeatureChunk(String)
    case invalidAudio(String)
    case unsupportedScalarType(String)
    case concurrentOperation

    public var errorDescription: String? {
        switch self {
        case .invalidPackage(let message):
            "Invalid model package: \(message)"
        case .invalidRuntimeSupport(let message):
            "Invalid runtime support: \(message)"
        case .modelAssetUnavailable(let message):
            "Model asset unavailable: \(message)"
        case .modelLoadFailed(let aot, let source):
            if let aot {
                "AOT model load failed (\(aot)); source fallback also failed (\(source))"
            } else {
                "Source model load failed: \(source)"
            }
        case .missingFunction(let name):
            "The model does not contain required function '\(name)'"
        case .inference(let message):
            "CoreAI inference failed: \(message)"
        case .invalidFeatureChunk(let message):
            "Invalid feature chunk: \(message)"
        case .invalidAudio(let message):
            "Invalid audio: \(message)"
        case .unsupportedScalarType(let type):
            "Unsupported CoreAI scalar type: \(type)"
        case .concurrentOperation:
            "A session operation is already in progress; await it before submitting the next stream chunk"
        }
    }
}
