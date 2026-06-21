import CoreAI

public enum NemotronPlatform: String, Codable, Sendable, CaseIterable {
    case iOS = "ios"
    case macOS = "macos"
}

public enum NemotronComputePreference: String, Codable, Sendable, CaseIterable {
    /// Let CoreAI partition the graph across the compute units supported by
    /// the model and current device.
    case automatic

    /// Prefer GPU execution. This is the validated low-latency path for the
    /// current Nemotron FastConformer export.
    case gpu

    /// Prefer Neural Engine execution. A device-matched AOT asset is required
    /// because the portable GPU-oriented source graph is not ANE-safe.
    case neuralEngine = "neural-engine"
}

enum CoreAIPlatformPolicy {
    static var current: NemotronPlatform {
        #if os(iOS)
        .iOS
        #elseif os(macOS)
        .macOS
        #else
        #error("NemotronCoreAI supports only iOS and macOS")
        #endif
    }

    static func specializationOptions(
        for preference: NemotronComputePreference
    ) -> SpecializationOptions {
        switch preference {
        case .automatic:
            .default
        case .gpu:
            SpecializationOptions(preferredComputeUnitKind: .gpu)
        case .neuralEngine:
            SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        }
    }
}
