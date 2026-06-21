import XCTest

@testable import NemotronCoreAI

final class RemoteModelPlannerTests: XCTestCase {
    func testSelectsExactMacAndPhoneAOTArtifacts() throws {
        let manifest = try makeManifest()
        let support = makeRuntimeSupport()

        let mac = try RemoteModelPlanner.plan(
            manifest: manifest,
            runtimeSupport: support,
            target: NemotronDeviceTarget(platform: .macOS, architecture: "h15d"),
            latencyMS: 320,
            sourceVariant: nil,
            preferAOT: true,
            computePreference: .gpu
        )
        XCTAssertEqual(mac.selectedKind, .aheadOfTime)
        XCTAssertEqual(mac.assetPath, "aot/macos/gpu/model-h15d.aimodelc")

        let phone = try RemoteModelPlanner.plan(
            manifest: manifest,
            runtimeSupport: support,
            target: NemotronDeviceTarget(platform: .iOS, architecture: "h18p"),
            latencyMS: 320,
            sourceVariant: nil,
            preferAOT: true,
            computePreference: .gpu
        )
        XCTAssertEqual(phone.selectedKind, .aheadOfTime)
        XCTAssertEqual(phone.assetPath, "aot/ios/gpu/model-h18p.aimodelc")
    }

    func testUsesPortableSourceForAutomaticUnknownAndOtherLatencies() throws {
        let manifest = try makeManifest()
        let support = makeRuntimeSupport()
        let knownTarget = NemotronDeviceTarget(platform: .macOS, architecture: "h15d")

        let automatic = try RemoteModelPlanner.plan(
            manifest: manifest,
            runtimeSupport: support,
            target: knownTarget,
            latencyMS: 320,
            sourceVariant: nil,
            preferAOT: true,
            computePreference: .automatic
        )
        XCTAssertEqual(automatic.selectedKind, .source)
        XCTAssertEqual(automatic.sourceVariant, "fp16_320ms")

        let unknown = try RemoteModelPlanner.plan(
            manifest: manifest,
            runtimeSupport: support,
            target: NemotronDeviceTarget(platform: .macOS, architecture: "future-chip"),
            latencyMS: 320,
            sourceVariant: nil,
            preferAOT: true,
            computePreference: .gpu
        )
        XCTAssertEqual(unknown.selectedKind, .source)
        XCTAssertEqual(unknown.sourceVariant, "fp16_320ms")

        let lowLatency = try RemoteModelPlanner.plan(
            manifest: manifest,
            runtimeSupport: support,
            target: knownTarget,
            latencyMS: 160,
            sourceVariant: nil,
            preferAOT: true,
            computePreference: .gpu
        )
        XCTAssertEqual(lowLatency.selectedKind, .source)
        XCTAssertEqual(lowLatency.sourceVariant, "fp16")
    }

    func testNeuralEngineRequiresAValidatedArtifact() throws {
        XCTAssertThrowsError(
            try RemoteModelPlanner.plan(
                manifest: makeManifest(),
                runtimeSupport: makeRuntimeSupport(),
                target: NemotronDeviceTarget(platform: .macOS, architecture: "h15d"),
                latencyMS: 320,
                sourceVariant: nil,
                preferAOT: true,
                computePreference: .neuralEngine
            )
        )
    }

    private func makeManifest() throws -> NemotronPackageManifest {
        let supportData = try JSONEncoder().encode(makeRuntimeSupport())
        let hash = String(repeating: "0", count: 64)
        return NemotronPackageManifest(
            formatVersion: 2,
            model: "nvidia/nemotron-3.5-asr-streaming-0.6b",
            sourceCheckpointSHA256: nil,
            streamingSourceVariant: "fp16",
            pythonRuntimeDefault: "fp16",
            native320msDefault: "aot-320ms-gpu",
            pythonPath: "src",
            runtimeSupport: PackageResource(
                path: "runtime-support.json",
                bytes: supportData.count,
                sha256: PackageIntegrity.sha256Hex(supportData)
            ),
            variants: [
                "fp16": SourceModelVariant(
                    path: "reference-fp16",
                    asset: "reference-fp16/model.aimodel",
                    bytes: nil,
                    coreAIMainHash: hash
                ),
                "fp16_320ms": SourceModelVariant(
                    path: "source-fp16-320ms",
                    asset: "source-fp16-320ms/model.aimodel",
                    bytes: nil,
                    coreAIMainHash: hash
                ),
            ],
            aot320ms: [
                "macos": [
                    "gpu": [
                        "h15d": AOTModelVariant(
                            product: "M3 Ultra",
                            path: "aot/macos/gpu/model-h15d.aimodelc",
                            preferredCompute: "gpu",
                            latencyMS: 320,
                            bytes: nil,
                            coreAIMainHash: hash
                        )
                    ]
                ],
                "ios": [
                    "gpu": [
                        "h18p": AOTModelVariant(
                            product: "A19 family",
                            path: "aot/ios/gpu/model-h18p.aimodelc",
                            preferredCompute: "gpu",
                            latencyMS: 320,
                            bytes: nil,
                            coreAIMainHash: hash
                        )
                    ]
                ],
            ],
            aot320msGPU: nil,
            deviceArchitectures: ["M3 Ultra": "h15d", "A19 family": "h18p"]
        )
    }
}
