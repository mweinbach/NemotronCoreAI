import Foundation
import XCTest

@testable import NemotronCoreAI

final class NemotronModelManagerTests: XCTestCase {
    override func setUp() {
        super.setUp()
        ModelURLProtocol.registry.reset()
    }

    func testDownloadsOnlyTheSelectedArtifactThenUsesCacheOffline() async throws {
        let fixture = try RemotePackageFixture()
        fixture.install()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ModelURLProtocol.self]
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        let manager = NemotronModelManager(session: session)
        let cache = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: cache) }
        let progress = DownloadProgressRecorder()

        let downloaded = try await manager.prepareModel(
            fixture.remoteModel,
            target: fixture.target,
            computePreference: .gpu,
            cacheDirectory: cache,
            authorizationToken: "test-token",
            progress: { progress.record($0) }
        )

        XCTAssertFalse(downloaded.cacheHit)
        XCTAssertEqual(downloaded.resolvedRevision, fixture.commit)
        XCTAssertEqual(downloaded.selectedKind, .aheadOfTime)
        XCTAssertEqual(downloaded.selectedAssetPath, fixture.aotPath)
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: downloaded.packageURL
                    .appendingPathComponent(fixture.aotPath)
                    .appendingPathComponent("main.hash").path
            )
        )
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: downloaded.packageURL
                    .appendingPathComponent("source-fp16-320ms/model.aimodel/main.hash").path
            )
        )
        let firstRequests = ModelURLProtocol.registry.requests()
        XCTAssertTrue(firstRequests.allSatisfy { $0.authorization == "Bearer test-token" })
        XCTAssertTrue(firstRequests.contains { $0.path.hasSuffix("/package-manifest.json") })
        XCTAssertTrue(firstRequests.contains { $0.path.hasSuffix("/runtime-support.json") })
        XCTAssertTrue(firstRequests.contains { $0.path.hasSuffix("/compiled.bin") })
        XCTAssertFalse(firstRequests.contains { $0.path.contains("source-fp16-320ms") })
        XCTAssertEqual(progress.values().last?.phase, .ready)
        XCTAssertEqual(progress.values().last?.fractionCompleted, 1)

        ModelURLProtocol.registry.goOffline()
        let cached = try await manager.prepareModel(
            fixture.remoteModel,
            target: fixture.target,
            computePreference: .gpu,
            cacheDirectory: cache
        )

        XCTAssertTrue(cached.cacheHit)
        XCTAssertEqual(cached.packageURL, downloaded.packageURL)
        XCTAssertEqual(ModelURLProtocol.registry.requests().count, firstRequests.count)
    }

    func testRejectsAnLFSHashMismatch() async throws {
        let fixture = try RemotePackageFixture(corruptPayloadHash: true)
        fixture.install()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ModelURLProtocol.self]
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        let manager = NemotronModelManager(session: session)
        let cache = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: cache) }

        do {
            _ = try await manager.prepareModel(
                fixture.remoteModel,
                target: fixture.target,
                computePreference: .gpu,
                cacheDirectory: cache
            )
            XCTFail("A corrupt LFS object must not enter the cache")
        } catch {
            XCTAssertTrue(String(describing: error).contains("SHA-256 mismatch"))
        }
    }
}

private struct RemotePackageFixture {
    let commit = String(repeating: "a", count: 40)
    let endpoint = URL(string: "https://models.example.test")!
    let repositoryID = "owner/model"
    let revision = "v1"
    let target = NemotronDeviceTarget(platform: .macOS, architecture: "h15d")
    let aotPath = "aot/macos/gpu/model-h15d.aimodelc"
    let responses: [String: Data]

    var remoteModel: NemotronRemoteModel {
        NemotronRemoteModel(repositoryID: repositoryID, revision: revision, endpoint: endpoint)
    }

    init(corruptPayloadHash: Bool = false) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let supportData = try encoder.encode(makeRuntimeSupport())
        let mainHash = Data((0..<32).map(UInt8.init))
        let mainHashHex = PackageIntegrity.binaryHex(mainHash)
        let payload = Data("miniature compiled CoreAI payload".utf8)
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
                bytes: supportData.count,
                sha256: PackageIntegrity.sha256Hex(supportData)
            ),
            variants: [
                "fp16": SourceModelVariant(
                    path: "reference-fp16",
                    asset: "reference-fp16/model.aimodel",
                    bytes: nil,
                    coreAIMainHash: mainHashHex
                ),
                "fp16_320ms": SourceModelVariant(
                    path: "source-fp16-320ms",
                    asset: "source-fp16-320ms/model.aimodel",
                    bytes: nil,
                    coreAIMainHash: mainHashHex
                ),
            ],
            aot320ms: [
                "macos": [
                    "gpu": [
                        "h15d": AOTModelVariant(
                            product: "M3 Ultra",
                            path: aotPath,
                            preferredCompute: "gpu",
                            latencyMS: 320,
                            bytes: nil,
                            coreAIMainHash: mainHashHex
                        )
                    ]
                ]
            ],
            aot320msGPU: nil,
            deviceArchitectures: ["M3 Ultra": "h15d"]
        )
        let manifestData = try encoder.encode(manifest)
        let files: [String: Data] = [
            "package-manifest.json": manifestData,
            "runtime-support.json": supportData,
            "\(aotPath)/main.hash": mainHash,
            "\(aotPath)/compiled.bin": payload,
        ]
        let siblings: [[String: Any]] = files.map { path, data in
            let hash =
                path.hasSuffix("compiled.bin") && corruptPayloadHash
                ? String(repeating: "f", count: 64)
                : PackageIntegrity.sha256Hex(data)
            return [
                "rfilename": path,
                "size": data.count,
                "lfs": ["sha256": hash],
            ]
        }
        let modelInfo = try JSONSerialization.data(
            withJSONObject: ["sha": commit, "siblings": siblings],
            options: [.sortedKeys]
        )
        var responses = [
            "/api/models/owner/model/revision/v1": modelInfo
        ]
        for (path, data) in files {
            responses["/owner/model/resolve/\(commit)/\(path)"] = data
        }
        self.responses = responses
    }

    func install() {
        ModelURLProtocol.registry.install(responses)
    }
}

private final class ModelURLProtocol: URLProtocol, @unchecked Sendable {
    static let registry = ModelResponseRegistry()

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host == "models.example.test"
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard let url = request.url,
            let data = Self.registry.response(
                path: url.path,
                authorization: request.value(forHTTPHeaderField: "Authorization")
            ),
            let response = HTTPURLResponse(
                url: url,
                statusCode: 200,
                httpVersion: "HTTP/1.1",
                headerFields: ["Content-Length": String(data.count)]
            )
        else {
            client?.urlProtocol(
                self,
                didFailWithError: URLError(.notConnectedToInternet)
            )
            return
        }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

private final class ModelResponseRegistry: @unchecked Sendable {
    struct Request: Sendable {
        let path: String
        let authorization: String?
    }

    private let lock = NSLock()
    private var installedResponses: [String: Data] = [:]
    private var recordedRequests: [Request] = []
    private var offline = false

    func reset() {
        lock.lock()
        installedResponses = [:]
        recordedRequests = []
        offline = false
        lock.unlock()
    }

    func install(_ responses: [String: Data]) {
        lock.lock()
        installedResponses = responses
        offline = false
        lock.unlock()
    }

    func goOffline() {
        lock.lock()
        offline = true
        lock.unlock()
    }

    func response(path: String, authorization: String?) -> Data? {
        lock.lock()
        defer { lock.unlock() }
        guard !offline else { return nil }
        recordedRequests.append(Request(path: path, authorization: authorization))
        return installedResponses[path]
    }

    func requests() -> [Request] {
        lock.lock()
        defer { lock.unlock() }
        return recordedRequests
    }
}

private final class DownloadProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var progress: [NemotronModelDownloadProgress] = []

    func record(_ value: NemotronModelDownloadProgress) {
        lock.lock()
        progress.append(value)
        lock.unlock()
    }

    func values() -> [NemotronModelDownloadProgress] {
        lock.lock()
        defer { lock.unlock() }
        return progress
    }
}
