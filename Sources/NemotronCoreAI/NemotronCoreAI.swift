import CoreAI
import Foundation

public enum NemotronCoreAI {
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
