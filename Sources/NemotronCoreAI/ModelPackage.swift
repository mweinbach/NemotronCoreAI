import CoreAI
import Foundation

public struct PackageResource: Codable, Sendable, Equatable {
    public let path: String
    public let bytes: Int
    public let sha256: String
}

public struct SourceModelVariant: Codable, Sendable, Equatable {
    public let path: String
    public let asset: String
    public let bytes: Int?
    public let coreAIMainHash: String

    enum CodingKeys: String, CodingKey {
        case path
        case asset
        case bytes
        case coreAIMainHash = "coreai_main_hash"
    }
}

public struct AOTModelVariant: Codable, Sendable, Equatable {
    public let product: String
    public let path: String
    public let preferredCompute: String
    public let latencyMS: Int
    public let bytes: Int?
    public let coreAIMainHash: String

    enum CodingKeys: String, CodingKey {
        case product
        case path
        case preferredCompute = "preferred_compute"
        case latencyMS = "latency_ms"
        case bytes
        case coreAIMainHash = "coreai_main_hash"
    }
}

public struct NemotronPackageManifest: Codable, Sendable, Equatable {
    public let formatVersion: Int
    public let model: String
    public let sourceCheckpointSHA256: String?
    public let streamingSourceVariant: String
    public let pythonRuntimeDefault: String?
    public let native320msDefault: String?
    public let pythonPath: String?
    public let runtimeSupport: PackageResource?
    public let variants: [String: SourceModelVariant]
    /// Platform -> compute preference -> architecture -> artifact.
    public let aot320ms: [String: [String: [String: AOTModelVariant]]]?
    /// Version-1 compatibility for the original macOS GPU-only package.
    public let aot320msGPU: [String: AOTModelVariant]?
    public let deviceArchitectures: [String: String]

    enum CodingKeys: String, CodingKey {
        case formatVersion = "format_version"
        case model
        case sourceCheckpointSHA256 = "source_checkpoint_sha256"
        case streamingSourceVariant = "streaming_source_variant"
        case pythonRuntimeDefault = "python_runtime_default"
        case native320msDefault = "native_320ms_default"
        case pythonPath = "python_path"
        case runtimeSupport = "runtime_support"
        case variants
        case aot320ms = "aot_320ms"
        case aot320msGPU = "aot_320ms_gpu"
        case deviceArchitectures = "device_architectures"
    }
}

public struct ResolvedModelAsset: Sendable, Equatable {
    public enum Kind: String, Sendable, Equatable {
        case aheadOfTime = "aot-320ms-gpu"
        case source
    }

    public let url: URL
    public let kind: Kind
    public let sourceVariant: String
    public let architecture: String
    public let platform: NemotronPlatform
    public let computePreference: NemotronComputePreference
    public let latencyMS: Int
    public let expectedCoreAIMainHash: String

    public var isAheadOfTime: Bool { kind == .aheadOfTime }
}

public struct NemotronModelPackage: Sendable {
    public static let supportedAOTArchitectures: Set<String> = [
        "h15d", "h16g", "h17g", "h17s", "h17c", "h18p",
    ]

    public let rootURL: URL
    public let manifest: NemotronPackageManifest
    public let runtimeSupport: RuntimeSupport

    public static func load(from rootURL: URL) throws -> Self {
        let manifestURL = rootURL.appendingPathComponent("package-manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw NemotronError.invalidPackage("missing \(manifestURL.lastPathComponent)")
        }

        let manifest: NemotronPackageManifest
        do {
            manifest = try JSONDecoder().decode(
                NemotronPackageManifest.self,
                from: Data(contentsOf: manifestURL)
            )
        } catch {
            throw NemotronError.invalidPackage("could not decode package-manifest.json: \(error)")
        }
        guard (1...2).contains(manifest.formatVersion) else {
            throw NemotronError.invalidPackage("unsupported format_version \(manifest.formatVersion)")
        }
        guard !manifest.variants.isEmpty else {
            throw NemotronError.invalidPackage("variants is empty")
        }

        guard let supportResource = manifest.runtimeSupport else {
            throw NemotronError.invalidPackage("runtime_support metadata is required")
        }
        let supportURL = rootURL.appendingPathComponent(supportResource.path)
        let supportData: Data
        do {
            supportData = try Data(contentsOf: supportURL)
        } catch {
            throw NemotronError.invalidPackage("could not read \(supportURL.path): \(error)")
        }
        guard supportData.count == supportResource.bytes else {
            throw NemotronError.invalidPackage(
                "runtime_support byte count is \(supportData.count), expected \(supportResource.bytes)"
            )
        }
        let expectedSupportHash = try PackageIntegrity.normalizedHash(
            supportResource.sha256, field: "runtime_support.sha256")
        guard PackageIntegrity.sha256Hex(supportData) == expectedSupportHash else {
            throw NemotronError.invalidPackage("runtime_support.sha256 does not match \(supportResource.path)")
        }
        let runtimeSupport = try RuntimeSupport.decode(supportData, source: supportURL.path)
        guard runtimeSupport.model == manifest.model else {
            throw NemotronError.invalidPackage(
                "runtime-support model '\(runtimeSupport.model)' does not match manifest model '\(manifest.model)'"
            )
        }
        return Self(rootURL: rootURL, manifest: manifest, runtimeSupport: runtimeSupport)
    }

    public func sourceAsset(
        variant requestedVariant: String? = nil,
        latencyMS: Int,
        computePreference: NemotronComputePreference = .automatic,
        platform requestedPlatform: NemotronPlatform? = nil
    ) throws -> ResolvedModelAsset {
        let platform = requestedPlatform ?? CoreAIPlatformPolicy.current
        let variantName = requestedVariant ?? manifest.streamingSourceVariant
        guard let variant = manifest.variants[variantName] else {
            throw NemotronError.modelAssetUnavailable(
                "source variant '\(variantName)' is absent; available variants: \(manifest.variants.keys.sorted().joined(separator: ", "))"
            )
        }
        let url = rootURL.appendingPathComponent(variant.asset)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw NemotronError.modelAssetUnavailable("source asset does not exist at \(url.path)")
        }
        return ResolvedModelAsset(
            url: url,
            kind: .source,
            sourceVariant: variantName,
            architecture: AIModel.deviceArchitectureName,
            platform: platform,
            computePreference: computePreference,
            latencyMS: latencyMS,
            expectedCoreAIMainHash: variant.coreAIMainHash
        )
    }

    public func preferredAsset(
        variant requestedVariant: String? = nil,
        latencyMS: Int = 320,
        preferAOT: Bool = true,
        computePreference: NemotronComputePreference = .automatic,
        architecture: String = AIModel.deviceArchitectureName
    ) throws -> ResolvedModelAsset {
        try preferredAsset(
            variant: requestedVariant,
            latencyMS: latencyMS,
            preferAOT: preferAOT,
            computePreference: computePreference,
            architecture: architecture,
            platform: CoreAIPlatformPolicy.current
        )
    }

    func preferredAsset(
        variant requestedVariant: String?,
        latencyMS: Int,
        preferAOT: Bool,
        computePreference: NemotronComputePreference,
        architecture: String,
        platform: NemotronPlatform
    ) throws -> ResolvedModelAsset {
        let sourceVariant = requestedVariant ?? manifest.streamingSourceVariant
        let policy = runtimeSupport.deploymentPolicy.aot
        let isAOTRequest =
            preferAOT
            && sourceVariant == policy.eligibleVariant
            && latencyMS == policy.eligibleLatencyMS
        if isAOTRequest,
            Self.supportedAOTArchitectures.contains(architecture),
            let aot = aotVariant(
                platform: platform,
                computePreference: computePreference,
                architecture: architecture
            ),
            aot.latencyMS == policy.eligibleLatencyMS,
            aot.preferredCompute.lowercased() == computePreference.rawValue
        {
            let url = rootURL.appendingPathComponent(aot.path)
            if FileManager.default.fileExists(atPath: url.path) {
                return ResolvedModelAsset(
                    url: url,
                    kind: .aheadOfTime,
                    sourceVariant: sourceVariant,
                    architecture: architecture,
                    platform: platform,
                    computePreference: computePreference,
                    latencyMS: latencyMS,
                    expectedCoreAIMainHash: aot.coreAIMainHash
                )
            }
        }
        if computePreference == .neuralEngine {
            throw NemotronError.modelAssetUnavailable(
                "no validated Neural Engine AOT asset for \(platform.rawValue)/\(architecture)"
            )
        }
        if isAOTRequest, manifest.variants[policy.fallbackVariant] != nil {
            return try sourceAsset(
                variant: policy.fallbackVariant,
                latencyMS: latencyMS,
                computePreference: computePreference,
                platform: platform
            )
        }
        return try sourceAsset(
            variant: sourceVariant,
            latencyMS: latencyMS,
            computePreference: computePreference,
            platform: platform
        )
    }

    func candidateAssets(
        variant: String?,
        latencyMS: Int,
        preferAOT: Bool,
        computePreference: NemotronComputePreference,
        architecture: String = AIModel.deviceArchitectureName
    ) throws -> [ResolvedModelAsset] {
        let preferred = try preferredAsset(
            variant: variant,
            latencyMS: latencyMS,
            preferAOT: preferAOT,
            computePreference: computePreference,
            architecture: architecture
        )
        guard preferred.isAheadOfTime else { return [preferred] }
        let fallback = runtimeSupport.deploymentPolicy.aot.fallbackVariant
        guard
            let source = try? sourceAsset(
                variant: fallback,
                latencyMS: latencyMS,
                computePreference: computePreference
            )
        else {
            return [preferred]
        }
        return [preferred, source]
    }

    private func aotVariant(
        platform: NemotronPlatform,
        computePreference: NemotronComputePreference,
        architecture: String
    ) -> AOTModelVariant? {
        if let matrix = manifest.aot320ms,
            let compute = matrix[platform.rawValue]?[computePreference.rawValue]
        {
            return compute[architecture]
        }
        guard manifest.formatVersion == 1,
            platform == .macOS,
            computePreference == .gpu
        else {
            return nil
        }
        return manifest.aot320msGPU?[architecture]
    }

    func validateAssetIntegrity(_ asset: ResolvedModelAsset) throws {
        let expected = try PackageIntegrity.normalizedHash(
            asset.expectedCoreAIMainHash,
            field: "coreai_main_hash"
        )
        let hashURL = asset.url.appendingPathComponent("main.hash")
        let hashData: Data
        do {
            hashData = try Data(contentsOf: hashURL)
        } catch {
            throw NemotronError.invalidPackage("could not read \(hashURL.path): \(error)")
        }
        guard hashData.count == 32 else {
            throw NemotronError.invalidPackage("\(hashURL.path) must contain exactly 32 bytes")
        }
        guard PackageIntegrity.binaryHex(hashData) == expected else {
            throw NemotronError.invalidPackage("coreai_main_hash does not match \(hashURL.path)")
        }
    }
}
