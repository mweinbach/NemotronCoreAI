import CoreAI
import Foundation

public enum NemotronCoreAI {
    /// Detects this device, downloads and verifies only its required model,
    /// then returns a ready-to-stream session. The package is reused from the
    /// local cache on later launches.
    public static func loadSession(
        remoteModel: NemotronRemoteModel = .published,
        modelManager: NemotronModelManager = .shared,
        latencyMS: Int = 320,
        sourceVariant: String? = nil,
        preferAOT: Bool = true,
        computePreference: NemotronComputePreference = .gpu,
        targetLanguage: String = "auto",
        stripLanguageTag: Bool = false,
        cachePolicy: NemotronModelCachePolicy = .useCache,
        cacheDirectory: URL? = nil,
        authorizationToken: String? = nil,
        fallbackToSourceOnAOTFailure: Bool = true,
        downloadProgress: (@Sendable (NemotronModelDownloadProgress) -> Void)? = nil
    ) async throws -> NemotronASRSession {
        let target = NemotronDeviceTarget.current
        let cached = try await modelManager.prepareModel(
            remoteModel,
            target: target,
            latencyMS: latencyMS,
            sourceVariant: sourceVariant,
            preferAOT: preferAOT,
            computePreference: computePreference,
            cachePolicy: cachePolicy,
            cacheDirectory: cacheDirectory,
            authorizationToken: authorizationToken,
            progress: downloadProgress
        )

        do {
            return try await loadSession(
                packageURL: cached.packageURL,
                latencyMS: latencyMS,
                sourceVariant: sourceVariant,
                preferAOT: preferAOT,
                computePreference: computePreference,
                targetLanguage: targetLanguage,
                stripLanguageTag: stripLanguageTag
            )
        } catch {
            guard cached.selectedKind == .aheadOfTime,
                fallbackToSourceOnAOTFailure,
                computePreference != .neuralEngine
            else {
                throw error
            }

            let aotFailure = String(describing: error)
            do {
                let package = try NemotronModelPackage.load(from: cached.packageURL)
                let fallbackVariant = package.runtimeSupport.deploymentPolicy.aot.fallbackVariant
                let fallback = try await modelManager.prepareModel(
                    remoteModel,
                    target: target,
                    latencyMS: latencyMS,
                    sourceVariant: fallbackVariant,
                    preferAOT: false,
                    computePreference: computePreference,
                    cachePolicy: cachePolicy,
                    cacheDirectory: cacheDirectory,
                    authorizationToken: authorizationToken,
                    progress: downloadProgress
                )
                return try await loadSession(
                    packageURL: fallback.packageURL,
                    latencyMS: latencyMS,
                    sourceVariant: fallbackVariant,
                    preferAOT: false,
                    computePreference: computePreference,
                    targetLanguage: targetLanguage,
                    stripLanguageTag: stripLanguageTag
                )
            } catch {
                throw NemotronError.modelLoadFailed(
                    aot: aotFailure,
                    source: String(describing: error)
                )
            }
        }
    }

    /// Loads a platform, architecture, and compute-matched 320 ms AOT artifact.
    /// Automatic and GPU requests fall back to the portable source model.
    /// Neural Engine requests require a validated AOT artifact so an
    /// incompatible source graph can never crash the process during inference.
    public static func loadSession(
        packageURL: URL,
        latencyMS: Int = 320,
        sourceVariant: String? = nil,
        preferAOT: Bool = true,
        computePreference: NemotronComputePreference = .automatic,
        targetLanguage: String = "auto",
        stripLanguageTag: Bool = false
    ) async throws -> NemotronASRSession {
        let package = try NemotronModelPackage.load(from: packageURL)
        let mode = try package.runtimeSupport.streamingMode(latencyMS: latencyMS)
        let candidates = try package.candidateAssets(
            variant: sourceVariant,
            latencyMS: latencyMS,
            preferAOT: preferAOT,
            computePreference: computePreference
        )
        var aotFailure: String?
        for candidate in candidates {
            do {
                try package.validateAssetIntegrity(candidate)
                let options = CoreAIPlatformPolicy.specializationOptions(for: computePreference)
                let model = try await AIModel(contentsOf: candidate.url, options: options)
                return try await NemotronASRSession.make(
                    model: model,
                    package: package,
                    selectedAsset: candidate,
                    mode: mode,
                    targetLanguage: targetLanguage,
                    stripLanguageTag: stripLanguageTag
                )
            } catch {
                if candidate.isAheadOfTime {
                    aotFailure = String(describing: error)
                    continue
                }
                throw NemotronError.modelLoadFailed(
                    aot: aotFailure,
                    source: String(describing: error)
                )
            }
        }
        throw NemotronError.modelLoadFailed(
            aot: aotFailure,
            source: "no source fallback candidate was available"
        )
    }
}
