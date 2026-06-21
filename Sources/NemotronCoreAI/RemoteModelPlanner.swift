import Foundation

struct RemoteModelPlan: Sendable, Equatable {
    let assetPath: String
    let sourceVariant: String
    let selectedKind: ResolvedModelAsset.Kind
}

enum RemoteModelPlanner {
    static func plan(
        manifest: NemotronPackageManifest,
        runtimeSupport: RuntimeSupport,
        target: NemotronDeviceTarget,
        latencyMS: Int,
        sourceVariant requestedVariant: String?,
        preferAOT: Bool,
        computePreference: NemotronComputePreference
    ) throws -> RemoteModelPlan {
        let sourceVariant = requestedVariant ?? manifest.streamingSourceVariant
        let policy = runtimeSupport.deploymentPolicy.aot
        let isAOTRequest =
            preferAOT
            && sourceVariant == policy.eligibleVariant
            && latencyMS == policy.eligibleLatencyMS

        if isAOTRequest,
            NemotronModelPackage.supportedAOTArchitectures.contains(target.architecture),
            let aot = aotVariant(
                manifest: manifest,
                target: target,
                computePreference: computePreference
            ),
            aot.latencyMS == latencyMS,
            aot.preferredCompute.lowercased() == computePreference.rawValue
        {
            return RemoteModelPlan(
                assetPath: aot.path,
                sourceVariant: sourceVariant,
                selectedKind: .aheadOfTime
            )
        }

        if computePreference == .neuralEngine {
            throw NemotronError.modelAssetUnavailable(
                "no validated Neural Engine AOT asset for \(target.platform.rawValue)/\(target.architecture)"
            )
        }

        let selectedSource: String
        if isAOTRequest, manifest.variants[policy.fallbackVariant] != nil {
            selectedSource = policy.fallbackVariant
        } else {
            selectedSource = sourceVariant
        }
        guard let source = manifest.variants[selectedSource] else {
            throw NemotronError.modelAssetUnavailable(
                "source variant '\(selectedSource)' is absent from the remote package"
            )
        }
        return RemoteModelPlan(
            assetPath: source.asset,
            sourceVariant: selectedSource,
            selectedKind: .source
        )
    }

    private static func aotVariant(
        manifest: NemotronPackageManifest,
        target: NemotronDeviceTarget,
        computePreference: NemotronComputePreference
    ) -> AOTModelVariant? {
        if let matrix = manifest.aot320ms {
            return matrix[target.platform.rawValue]?[computePreference.rawValue]?[target.architecture]
        }
        guard manifest.formatVersion == 1,
            target.platform == .macOS,
            computePreference == .gpu
        else {
            return nil
        }
        return manifest.aot320msGPU?[target.architecture]
    }
}
