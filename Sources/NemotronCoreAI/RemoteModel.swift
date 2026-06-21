import CoreAI
import Foundation

/// The CoreAI platform and architecture used to select a compiled model.
public struct NemotronDeviceTarget: Codable, Hashable, Sendable {
    public let platform: NemotronPlatform
    public let architecture: String

    public init(platform: NemotronPlatform, architecture: String) {
        self.platform = platform
        self.architecture = architecture
    }

    /// The target reported by the CoreAI runtime on this device.
    public static var current: Self {
        Self(
            platform: CoreAIPlatformPolicy.current,
            architecture: AIModel.deviceArchitectureName
        )
    }
}

/// A Hugging Face model repository containing a Nemotron CoreAI package.
public struct NemotronRemoteModel: Hashable, Sendable {
    public let repositoryID: String
    public let revision: String
    public let endpoint: URL

    public init(
        repositoryID: String,
        revision: String = "main",
        endpoint: URL = URL(string: "https://huggingface.co")!
    ) {
        self.repositoryID = repositoryID
        self.revision = revision
        self.endpoint = endpoint
    }

    /// The validated public model package published for this SDK.
    public static let published = Self(
        repositoryID: "mweinbach/nemotron-3.5-asr-streaming-0.6b-coreai",
        revision: "v0.1.1"
    )
}

public enum NemotronModelCachePolicy: Sendable, Equatable {
    /// Reuse a complete local package without making a network request.
    case useCache
    /// Download and verify the selected package again before replacing the cache.
    case reloadIgnoringCache
}

public struct NemotronModelDownloadProgress: Sendable, Equatable {
    public enum Phase: String, Sendable, Equatable {
        case resolving
        case downloading
        case verifying
        case ready
    }

    public let phase: Phase
    public let currentFile: String?
    public let completedBytes: Int64
    public let totalBytes: Int64

    public var fractionCompleted: Double? {
        guard totalBytes > 0 else { return nil }
        return min(1, Double(completedBytes) / Double(totalBytes))
    }
}

public struct CachedNemotronModel: Sendable, Equatable {
    public let packageURL: URL
    public let repositoryID: String
    public let requestedRevision: String
    public let resolvedRevision: String
    public let target: NemotronDeviceTarget
    public let computePreference: NemotronComputePreference
    public let selectedKind: ResolvedModelAsset.Kind
    public let selectedAssetPath: String
    public let totalBytes: Int64
    public let cacheHit: Bool
}
