import Foundation
import XCTest

@testable import NemotronCoreAI

final class ModelPackageTests: XCTestCase {
    func testAOTSelectionIsStrictlyGatedAndCoversTargetArchitectures() throws {
        let root = try makePackageFixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let package = try NemotronModelPackage.load(from: root)

        for architecture in ["h15d", "h16g", "h17g", "h17s", "h17c"] {
            let selection = try package.preferredAsset(
                variant: "fp16", latencyMS: 320, preferAOT: true,
                computePreference: .gpu, architecture: architecture
            )
            XCTAssertTrue(selection.isAheadOfTime)
            XCTAssertEqual(selection.architecture, architecture)
            XCTAssertTrue(selection.url.lastPathComponent.contains(architecture))
        }

        let compressed = try package.preferredAsset(
            variant: "int8", latencyMS: 320, preferAOT: true,
            computePreference: .gpu, architecture: "h15d"
        )
        XCTAssertEqual(compressed.kind, .source)
        XCTAssertEqual(compressed.sourceVariant, "int8")

        let lowerLatency = try package.preferredAsset(
            variant: "fp16", latencyMS: 160, preferAOT: true,
            computePreference: .gpu, architecture: "h15d"
        )
        XCTAssertEqual(lowerLatency.kind, .source)
        XCTAssertEqual(lowerLatency.sourceVariant, "fp16")
    }

    func testUnknownOrMissingAOTFallsBackToDedicated320msSource() throws {
        let root = try makePackageFixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let package = try NemotronModelPackage.load(from: root)

        let unknown = try package.preferredAsset(
            variant: "fp16", latencyMS: 320, preferAOT: true,
            computePreference: .gpu, architecture: "future-chip"
        )
        XCTAssertEqual(unknown.kind, .source)
        XCTAssertEqual(unknown.sourceVariant, "fp16_320ms")

        try FileManager.default.removeItem(
            at: root.appendingPathComponent("aot/macos/gpu/model-h15d.aimodelc")
        )
        let missing = try package.preferredAsset(
            variant: "fp16", latencyMS: 320, preferAOT: true,
            computePreference: .gpu, architecture: "h15d"
        )
        XCTAssertEqual(missing.kind, .source)
        XCTAssertEqual(missing.sourceVariant, "fp16_320ms")
    }

    func testAOTSelectionIsPlatformSpecific() throws {
        let root = try makePackageFixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let package = try NemotronModelPackage.load(from: root)

        let selection = try package.preferredAsset(
            variant: "fp16",
            latencyMS: 320,
            preferAOT: true,
            computePreference: .gpu,
            architecture: "h16g",
            platform: .iOS
        )

        XCTAssertEqual(selection.kind, .source)
        XCTAssertEqual(selection.sourceVariant, "fp16_320ms")

        let phone = try package.preferredAsset(
            variant: "fp16",
            latencyMS: 320,
            preferAOT: true,
            computePreference: .gpu,
            architecture: "h18p",
            platform: .iOS
        )
        XCTAssertTrue(phone.isAheadOfTime)
        XCTAssertEqual(phone.platform, .iOS)
    }

    func testNeuralEngineRequiresValidatedAOT() throws {
        let root = try makePackageFixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let package = try NemotronModelPackage.load(from: root)

        XCTAssertThrowsError(
            try package.preferredAsset(
                variant: "fp16",
                latencyMS: 320,
                preferAOT: true,
                computePreference: .neuralEngine,
                architecture: "h15d"
            )
        )
    }

    func testRuntimeSupportAndSelectedAssetIntegrityAreValidated() throws {
        let root = try makePackageFixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let package = try NemotronModelPackage.load(from: root)
        let selection = try package.preferredAsset(
            variant: "fp16", latencyMS: 320, preferAOT: true,
            computePreference: .gpu, architecture: "h15d"
        )
        XCTAssertNoThrow(try package.validateAssetIntegrity(selection))

        try Data(repeating: 0xff, count: 32).write(to: selection.url.appendingPathComponent("main.hash"))
        XCTAssertThrowsError(try package.validateAssetIntegrity(selection))

        var supportData = try Data(contentsOf: root.appendingPathComponent("runtime-support.json"))
        supportData.append(0)
        try supportData.write(to: root.appendingPathComponent("runtime-support.json"))
        XCTAssertThrowsError(try NemotronModelPackage.load(from: root))
    }

    private func makePackageFixture() throws -> URL {
        let root = try temporaryDirectory()
        let mainHash = Data((0..<32).map(UInt8.init))
        let mainHashHex = PackageIntegrity.binaryHex(mainHash)
        let variants: [String: SourceModelVariant] = [
            "fp16": SourceModelVariant(
                path: "reference-fp16", asset: "reference-fp16/model.aimodel", bytes: nil,
                coreAIMainHash: mainHashHex),
            "fp16_320ms": SourceModelVariant(
                path: "source-fp16-320ms", asset: "source-fp16-320ms/model.aimodel", bytes: nil,
                coreAIMainHash: mainHashHex),
            "int8": SourceModelVariant(
                path: "deployment-int8", asset: "deployment-int8/model.aimodel", bytes: nil,
                coreAIMainHash: mainHashHex),
        ]
        for variant in variants.values {
            try FileManager.default.createDirectory(
                at: root.appendingPathComponent(variant.asset),
                withIntermediateDirectories: true
            )
            try mainHash.write(to: root.appendingPathComponent(variant.asset).appendingPathComponent("main.hash"))
        }
        var macGPU: [String: AOTModelVariant] = [:]
        for architecture in ["h15d", "h16g", "h17g", "h17s", "h17c"] {
            let path = "aot/macos/gpu/model-\(architecture).aimodelc"
            try FileManager.default.createDirectory(
                at: root.appendingPathComponent(path),
                withIntermediateDirectories: true
            )
            macGPU[architecture] = AOTModelVariant(
                product: architecture,
                path: path,
                preferredCompute: "gpu",
                latencyMS: 320,
                bytes: nil,
                coreAIMainHash: mainHashHex
            )
            try mainHash.write(to: root.appendingPathComponent(path).appendingPathComponent("main.hash"))
        }
        let phonePath = "aot/ios/gpu/model-h18p.aimodelc"
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent(phonePath),
            withIntermediateDirectories: true
        )
        let phoneGPU = AOTModelVariant(
            product: "A19 family",
            path: phonePath,
            preferredCompute: "gpu",
            latencyMS: 320,
            bytes: nil,
            coreAIMainHash: mainHashHex
        )
        try mainHash.write(
            to: root.appendingPathComponent(phonePath).appendingPathComponent("main.hash")
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let runtimeSupportData = try encoder.encode(makeRuntimeSupport())
        let manifest = NemotronPackageManifest(
            formatVersion: 2,
            model: "nvidia/nemotron-3.5-asr-streaming-0.6b",
            sourceCheckpointSHA256: nil,
            streamingSourceVariant: "fp16",
            pythonRuntimeDefault: "fp16",
            native320msDefault: "aot-320ms-gpu",
            pythonPath: "src",
            runtimeSupport: PackageResource(
                path: "runtime-support.json",
                bytes: runtimeSupportData.count,
                sha256: PackageIntegrity.sha256Hex(runtimeSupportData)
            ),
            variants: variants,
            aot320ms: [
                "macos": ["gpu": macGPU],
                "ios": ["gpu": ["h18p": phoneGPU]],
            ],
            aot320msGPU: nil,
            deviceArchitectures: [
                "M3 Ultra": "h15d", "M4": "h16g", "M5": "h17g",
                "M5 Pro": "h17s", "M5 Max": "h17c", "A19 family": "h18p",
            ]
        )
        try encoder.encode(manifest).write(to: root.appendingPathComponent("package-manifest.json"))
        try runtimeSupportData.write(to: root.appendingPathComponent("runtime-support.json"))
        return root
    }
}
