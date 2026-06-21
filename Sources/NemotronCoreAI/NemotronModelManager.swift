import Foundation

/// Resolves, downloads, verifies, and caches only the model needed by a device.
public actor NemotronModelManager {
    public static let shared = NemotronModelManager()

    private static let markerName = ".nemotron-cache.json"
    private static let metadataLimit = 8 * 1024 * 1024
    private static let storageReserve: Int64 = 256 * 1024 * 1024

    private let fileManager: FileManager
    private let session: URLSession

    public init() {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 120
        configuration.timeoutIntervalForResource = 24 * 60 * 60
        configuration.waitsForConnectivity = true
        self.session = URLSession(configuration: configuration)
        self.fileManager = .default
    }

    init(session: URLSession, fileManager: FileManager = .default) {
        self.session = session
        self.fileManager = fileManager
    }

    /// Returns the default purgeable cache used by the package.
    public nonisolated static func defaultCacheDirectory() throws -> URL {
        try FileManager.default.url(
            for: .cachesDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        .appendingPathComponent("NemotronCoreAI", isDirectory: true)
        .appendingPathComponent("Models", isDirectory: true)
    }

    /// Downloads the exact AOT model for `target`, or the portable source fallback.
    public func prepareModel(
        _ remoteModel: NemotronRemoteModel = .published,
        target: NemotronDeviceTarget = .current,
        latencyMS: Int = 320,
        sourceVariant: String? = nil,
        preferAOT: Bool = true,
        computePreference: NemotronComputePreference = .gpu,
        cachePolicy: NemotronModelCachePolicy = .useCache,
        cacheDirectory: URL? = nil,
        authorizationToken: String? = nil,
        progress: (@Sendable (NemotronModelDownloadProgress) -> Void)? = nil
    ) async throws -> CachedNemotronModel {
        try validate(remoteModel: remoteModel, target: target)
        let root = try cacheDirectory ?? Self.defaultCacheDirectory()
        try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
        let packageURL = cachePackageURL(
            root: root,
            remoteModel: remoteModel,
            target: target,
            latencyMS: latencyMS,
            sourceVariant: sourceVariant,
            preferAOT: preferAOT,
            computePreference: computePreference
        )

        if cachePolicy == .useCache,
            let cached = try? validatedCachedModel(
                at: packageURL,
                remoteModel: remoteModel,
                target: target,
                latencyMS: latencyMS,
                sourceVariant: sourceVariant,
                preferAOT: preferAOT,
                computePreference: computePreference
            )
        {
            progress?(
                NemotronModelDownloadProgress(
                    phase: .ready,
                    currentFile: nil,
                    completedBytes: cached.totalBytes,
                    totalBytes: cached.totalBytes
                )
            )
            return cached
        }

        progress?(
            NemotronModelDownloadProgress(
                phase: .resolving,
                currentFile: nil,
                completedBytes: 0,
                totalBytes: 0
            )
        )
        let modelInfoURL = try hubURL(
            remoteModel: remoteModel,
            pathSegments: ["api", "models"] + remoteModel.repositoryID.split(separator: "/").map(String.init)
                + ["revision", remoteModel.revision],
            queryItems: [URLQueryItem(name: "blobs", value: "true")]
        )
        let modelInfoData = try await fetchData(
            from: modelInfoURL,
            authorizationToken: authorizationToken,
            maximumBytes: Self.metadataLimit
        )
        let modelInfo: HubModelInfo
        do {
            modelInfo = try JSONDecoder().decode(HubModelInfo.self, from: modelInfoData)
        } catch {
            throw NemotronError.modelDownload("could not decode Hugging Face model metadata: \(error)")
        }
        try validateRevision(modelInfo.sha)

        let manifestURL = try resolveURL(
            remoteModel: remoteModel,
            revision: modelInfo.sha,
            relativePath: "package-manifest.json"
        )
        let manifestData = try await fetchData(
            from: manifestURL,
            authorizationToken: authorizationToken,
            maximumBytes: Self.metadataLimit
        )
        let manifest: NemotronPackageManifest
        do {
            manifest = try JSONDecoder().decode(NemotronPackageManifest.self, from: manifestData)
        } catch {
            throw NemotronError.modelDownload("could not decode remote package-manifest.json: \(error)")
        }
        guard (1...2).contains(manifest.formatVersion), !manifest.variants.isEmpty else {
            throw NemotronError.modelDownload("remote package manifest is unsupported or empty")
        }
        guard let supportResource = manifest.runtimeSupport else {
            throw NemotronError.modelDownload("remote package does not declare runtime_support")
        }
        let supportURL = try resolveURL(
            remoteModel: remoteModel,
            revision: modelInfo.sha,
            relativePath: supportResource.path
        )
        let supportData = try await fetchData(
            from: supportURL,
            authorizationToken: authorizationToken,
            maximumBytes: Self.metadataLimit
        )
        guard supportData.count == supportResource.bytes else {
            throw NemotronError.modelDownload(
                "runtime support is \(supportData.count) bytes, expected \(supportResource.bytes)"
            )
        }
        let supportHash = try PackageIntegrity.normalizedHash(
            supportResource.sha256,
            field: "runtime_support.sha256"
        )
        guard PackageIntegrity.sha256Hex(supportData) == supportHash else {
            throw NemotronError.modelDownload("runtime support SHA-256 does not match the package manifest")
        }
        let runtimeSupport = try RuntimeSupport.decode(supportData, source: supportResource.path)
        guard runtimeSupport.model == manifest.model else {
            throw NemotronError.modelDownload("runtime support and package manifest describe different models")
        }

        let plan = try RemoteModelPlanner.plan(
            manifest: manifest,
            runtimeSupport: runtimeSupport,
            target: target,
            latencyMS: latencyMS,
            sourceVariant: sourceVariant,
            preferAOT: preferAOT,
            computePreference: computePreference
        )
        _ = try safePathComponents(plan.assetPath)
        let prefix = plan.assetPath + "/"
        let assetFiles = modelInfo.siblings
            .filter { $0.filename.hasPrefix(prefix) }
            .sorted { $0.filename < $1.filename }
        guard !assetFiles.isEmpty,
            assetFiles.contains(where: { $0.filename == prefix + "main.hash" }),
            Set(assetFiles.map(\.filename)).count == assetFiles.count
        else {
            throw NemotronError.modelDownload("the selected remote asset is missing or incomplete")
        }
        let assetBytes = try assetFiles.reduce(Int64(0)) { total, file in
            guard file.size >= 0, total <= Int64.max - file.size else {
                throw NemotronError.modelDownload("remote asset byte count overflow")
            }
            return total + file.size
        }
        let metadataBytes = Int64(manifestData.count) + Int64(supportData.count)
        guard assetBytes <= Int64.max - metadataBytes else {
            throw NemotronError.modelDownload("remote package byte count overflow")
        }
        let totalBytes = assetBytes + metadataBytes
        try ensureAvailableStorage(totalBytes, at: root)

        let stagingURL = packageURL.deletingLastPathComponent().appendingPathComponent(
            ".\(packageURL.lastPathComponent).staging-\(UUID().uuidString)",
            isDirectory: true
        )
        try? fileManager.removeItem(at: stagingURL)
        try fileManager.createDirectory(at: stagingURL, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: stagingURL) }
        try manifestData.write(
            to: safeDestination(relativePath: "package-manifest.json", under: stagingURL),
            options: .atomic
        )
        let localSupportURL = try safeDestination(relativePath: supportResource.path, under: stagingURL)
        try fileManager.createDirectory(
            at: localSupportURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try supportData.write(to: localSupportURL, options: .atomic)

        var completedBytes = Int64(manifestData.count + supportData.count)
        for remoteFile in assetFiles {
            progress?(
                NemotronModelDownloadProgress(
                    phase: .downloading,
                    currentFile: remoteFile.filename,
                    completedBytes: completedBytes,
                    totalBytes: totalBytes
                )
            )
            let destination = try safeDestination(relativePath: remoteFile.filename, under: stagingURL)
            try await download(
                remoteFile,
                from: remoteModel,
                revision: modelInfo.sha,
                to: destination,
                authorizationToken: authorizationToken,
                completedBytes: completedBytes,
                totalBytes: totalBytes,
                progress: progress
            )
            if remoteFile.lfs != nil {
                progress?(
                    NemotronModelDownloadProgress(
                        phase: .verifying,
                        currentFile: remoteFile.filename,
                        completedBytes: completedBytes + remoteFile.size,
                        totalBytes: totalBytes
                    )
                )
                try await verifyLFSFile(remoteFile, at: destination)
            }
            completedBytes += remoteFile.size
        }

        let marker = CacheMarker(
            formatVersion: 1,
            repositoryID: remoteModel.repositoryID,
            requestedRevision: remoteModel.revision,
            resolvedRevision: modelInfo.sha,
            target: target,
            computePreference: computePreference,
            sourceVariant: plan.sourceVariant,
            selectedKind: plan.selectedKind.rawValue,
            selectedAssetPath: plan.assetPath,
            totalBytes: totalBytes
        )
        let markerData = try JSONEncoder().encode(marker)
        try markerData.write(
            to: stagingURL.appendingPathComponent(Self.markerName),
            options: .atomic
        )
        try validateStagedPackage(
            at: stagingURL,
            plan: plan,
            target: target,
            latencyMS: latencyMS,
            sourceVariant: sourceVariant,
            preferAOT: preferAOT,
            computePreference: computePreference
        )

        try fileManager.createDirectory(
            at: packageURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if fileManager.fileExists(atPath: packageURL.path) {
            try fileManager.removeItem(at: packageURL)
        }
        try fileManager.moveItem(at: stagingURL, to: packageURL)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var mutablePackageURL = packageURL
        try? mutablePackageURL.setResourceValues(values)

        let cached = CachedNemotronModel(
            packageURL: packageURL,
            repositoryID: remoteModel.repositoryID,
            requestedRevision: remoteModel.revision,
            resolvedRevision: modelInfo.sha,
            target: target,
            computePreference: computePreference,
            selectedKind: plan.selectedKind,
            selectedAssetPath: plan.assetPath,
            totalBytes: totalBytes,
            cacheHit: false
        )
        progress?(
            NemotronModelDownloadProgress(
                phase: .ready,
                currentFile: nil,
                completedBytes: totalBytes,
                totalBytes: totalBytes
            )
        )
        return cached
    }

    /// Removes every package managed under the selected cache root.
    public func removeAllCachedModels(cacheDirectory: URL? = nil) throws {
        let root = try cacheDirectory ?? Self.defaultCacheDirectory()
        if fileManager.fileExists(atPath: root.path) {
            try fileManager.removeItem(at: root)
        }
    }

    private func validatedCachedModel(
        at packageURL: URL,
        remoteModel: NemotronRemoteModel,
        target: NemotronDeviceTarget,
        latencyMS: Int,
        sourceVariant: String?,
        preferAOT: Bool,
        computePreference: NemotronComputePreference
    ) throws -> CachedNemotronModel {
        let markerData = try Data(contentsOf: packageURL.appendingPathComponent(Self.markerName))
        let marker = try JSONDecoder().decode(CacheMarker.self, from: markerData)
        guard marker.formatVersion == 1,
            marker.repositoryID == remoteModel.repositoryID,
            marker.requestedRevision == remoteModel.revision,
            marker.target == target,
            marker.computePreference == computePreference,
            marker.totalBytes >= 0,
            let kind = ResolvedModelAsset.Kind(rawValue: marker.selectedKind)
        else {
            throw NemotronError.modelDownload("cached model metadata does not match this request")
        }
        try validateRevision(marker.resolvedRevision)
        let package = try NemotronModelPackage.load(from: packageURL)
        let asset = try package.preferredAsset(
            variant: sourceVariant,
            latencyMS: latencyMS,
            preferAOT: preferAOT,
            computePreference: computePreference,
            architecture: target.architecture,
            platform: target.platform
        )
        try package.validateAssetIntegrity(asset)
        let relativePath = try relativePath(of: asset.url, under: packageURL)
        guard relativePath == marker.selectedAssetPath,
            asset.kind == kind,
            asset.sourceVariant == marker.sourceVariant
        else {
            throw NemotronError.modelDownload("cached model selection does not match its marker")
        }
        return CachedNemotronModel(
            packageURL: packageURL,
            repositoryID: marker.repositoryID,
            requestedRevision: marker.requestedRevision,
            resolvedRevision: marker.resolvedRevision,
            target: marker.target,
            computePreference: marker.computePreference,
            selectedKind: kind,
            selectedAssetPath: marker.selectedAssetPath,
            totalBytes: marker.totalBytes,
            cacheHit: true
        )
    }

    private func validateStagedPackage(
        at packageURL: URL,
        plan: RemoteModelPlan,
        target: NemotronDeviceTarget,
        latencyMS: Int,
        sourceVariant: String?,
        preferAOT: Bool,
        computePreference: NemotronComputePreference
    ) throws {
        let package = try NemotronModelPackage.load(from: packageURL)
        let asset = try package.preferredAsset(
            variant: sourceVariant,
            latencyMS: latencyMS,
            preferAOT: preferAOT,
            computePreference: computePreference,
            architecture: target.architecture,
            platform: target.platform
        )
        try package.validateAssetIntegrity(asset)
        guard try relativePath(of: asset.url, under: packageURL) == plan.assetPath,
            asset.kind == plan.selectedKind
        else {
            throw NemotronError.modelDownload("downloaded package selected an unexpected asset")
        }
    }

    private func download(
        _ remoteFile: HubSibling,
        from remoteModel: NemotronRemoteModel,
        revision: String,
        to destination: URL,
        authorizationToken: String?,
        completedBytes: Int64,
        totalBytes: Int64,
        progress: (@Sendable (NemotronModelDownloadProgress) -> Void)?
    ) async throws {
        let url = try resolveURL(
            remoteModel: remoteModel,
            revision: revision,
            relativePath: remoteFile.filename
        )
        let request = request(url: url, authorizationToken: authorizationToken)
        let temporaryURL: URL
        let response: URLResponse
        do {
            let delegate = ModelDownloadDelegate(
                filename: remoteFile.filename,
                fileBytes: remoteFile.size,
                completedBytes: completedBytes,
                totalBytes: totalBytes,
                progress: progress
            )
            (temporaryURL, response) = try await session.download(
                for: request,
                delegate: delegate
            )
        } catch {
            throw NemotronError.modelDownload("download failed for \(remoteFile.filename): \(error)")
        }
        try validate(response: response, url: url)
        try fileManager.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try? fileManager.removeItem(at: destination)
        do {
            try fileManager.moveItem(at: temporaryURL, to: destination)
        } catch {
            throw NemotronError.modelDownload("could not cache \(remoteFile.filename): \(error)")
        }
        let actualSize =
            try destination.resourceValues(forKeys: [.fileSizeKey]).fileSize.map(Int64.init) ?? -1
        guard actualSize == remoteFile.size else {
            try? fileManager.removeItem(at: destination)
            throw NemotronError.modelDownload(
                "\(remoteFile.filename) is \(actualSize) bytes, expected \(remoteFile.size)"
            )
        }
    }

    private func verifyLFSFile(_ remoteFile: HubSibling, at url: URL) async throws {
        guard let lfs = remoteFile.lfs else { return }
        let expected = try PackageIntegrity.normalizedHash(lfs.sha256, field: "lfs.sha256")
        let actual = try await Task.detached(priority: .utility) {
            try PackageIntegrity.sha256Hex(fileAt: url)
        }.value
        guard actual == expected else {
            try? fileManager.removeItem(at: url)
            throw NemotronError.modelDownload("SHA-256 mismatch for \(remoteFile.filename)")
        }
    }

    private func fetchData(
        from url: URL,
        authorizationToken: String?,
        maximumBytes: Int
    ) async throws -> Data {
        let request = request(url: url, authorizationToken: authorizationToken)
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw NemotronError.modelDownload("request failed for \(url.absoluteString): \(error)")
        }
        try validate(response: response, url: url)
        guard data.count <= maximumBytes else {
            throw NemotronError.modelDownload("metadata response exceeded \(maximumBytes) bytes")
        }
        return data
    }

    private func validate(response: URLResponse, url: URL) throws {
        guard let response = response as? HTTPURLResponse,
            (200..<300).contains(response.statusCode)
        else {
            let status = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw NemotronError.modelDownload("HTTP \(status) for \(url.absoluteString)")
        }
    }

    private func request(url: URL, authorizationToken: String?) -> URLRequest {
        var request = URLRequest(url: url)
        request.setValue("NemotronCoreAI/0.2", forHTTPHeaderField: "User-Agent")
        if let authorizationToken, !authorizationToken.isEmpty {
            request.setValue("Bearer \(authorizationToken)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    private func resolveURL(
        remoteModel: NemotronRemoteModel,
        revision: String,
        relativePath: String
    ) throws -> URL {
        let relativeComponents = try safePathComponents(relativePath)
        return try hubURL(
            remoteModel: remoteModel,
            pathSegments: remoteModel.repositoryID.split(separator: "/").map(String.init)
                + ["resolve", revision]
                + relativeComponents,
            queryItems: [URLQueryItem(name: "download", value: "true")]
        )
    }

    private func hubURL(
        remoteModel: NemotronRemoteModel,
        pathSegments: [String],
        queryItems: [URLQueryItem] = []
    ) throws -> URL {
        guard var components = URLComponents(url: remoteModel.endpoint, resolvingAgainstBaseURL: false),
            components.scheme == "https",
            components.host != nil
        else {
            throw NemotronError.modelDownload("Hugging Face endpoint must be an absolute HTTPS URL")
        }
        var allowed = CharacterSet.urlPathAllowed
        allowed.remove(charactersIn: "/%?#")
        let encoded = try pathSegments.map { segment -> String in
            guard !segment.isEmpty,
                segment != ".",
                segment != "..",
                let value = segment.addingPercentEncoding(withAllowedCharacters: allowed)
            else {
                throw NemotronError.modelDownload("invalid URL path segment")
            }
            return value
        }
        let basePath = components.percentEncodedPath.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        components.percentEncodedPath = "/" + ([basePath] + encoded).filter { !$0.isEmpty }.joined(separator: "/")
        components.queryItems = queryItems.isEmpty ? nil : queryItems
        guard let url = components.url else {
            throw NemotronError.modelDownload("could not construct Hugging Face URL")
        }
        return url
    }

    private func safeDestination(relativePath: String, under root: URL) throws -> URL {
        var destination = root
        for component in try safePathComponents(relativePath) {
            destination.appendPathComponent(component)
        }
        return destination
    }

    private func safePathComponents(_ relativePath: String) throws -> [String] {
        guard !relativePath.hasPrefix("/") else {
            throw NemotronError.modelDownload("remote path must be relative")
        }
        let components = relativePath.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        guard !components.isEmpty,
            components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." })
        else {
            throw NemotronError.modelDownload("remote path contains an unsafe component")
        }
        return components
    }

    private func relativePath(of url: URL, under root: URL) throws -> String {
        let rootPath = root.standardizedFileURL.path + "/"
        let path = url.standardizedFileURL.path
        guard path.hasPrefix(rootPath) else {
            throw NemotronError.modelDownload("cached asset escaped its package root")
        }
        return String(path.dropFirst(rootPath.count))
    }

    private func ensureAvailableStorage(_ requiredBytes: Int64, at root: URL) throws {
        let values = try root.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        guard let available = values.volumeAvailableCapacityForImportantUsage, available > 0 else {
            return
        }
        let requiredWithReserve =
            requiredBytes > Int64.max - Self.storageReserve
            ? Int64.max
            : requiredBytes + Self.storageReserve
        guard available >= requiredWithReserve else {
            throw NemotronError.insufficientStorage(required: requiredWithReserve, available: available)
        }
    }

    private func validate(remoteModel: NemotronRemoteModel, target: NemotronDeviceTarget) throws {
        guard remoteModel.repositoryID.split(separator: "/").count == 2,
            !remoteModel.revision.isEmpty,
            !target.architecture.isEmpty
        else {
            throw NemotronError.modelDownload("invalid repository, revision, or device architecture")
        }
    }

    private func validateRevision(_ revision: String) throws {
        guard (40...64).contains(revision.count),
            revision.utf8.allSatisfy({ byte in
                (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(byte)
                    || (UInt8(ascii: "a")...UInt8(ascii: "f")).contains(byte)
            })
        else {
            throw NemotronError.modelDownload("Hugging Face returned an invalid revision SHA")
        }
    }

    private func cachePackageURL(
        root: URL,
        remoteModel: NemotronRemoteModel,
        target: NemotronDeviceTarget,
        latencyMS: Int,
        sourceVariant: String?,
        preferAOT: Bool,
        computePreference: NemotronComputePreference
    ) -> URL {
        let repository = cacheKey(remoteModel.repositoryID)
        let revision = cacheKey(remoteModel.revision)
        let request = cacheKey(
            [
                target.platform.rawValue,
                target.architecture,
                computePreference.rawValue,
                String(latencyMS),
                sourceVariant ?? "default",
                preferAOT ? "aot" : "source",
            ].joined(separator: "-")
        )
        return
            root
            .appendingPathComponent(repository, isDirectory: true)
            .appendingPathComponent(revision, isDirectory: true)
            .appendingPathComponent(request, isDirectory: true)
    }

    private func cacheKey(_ value: String) -> String {
        let readable = value.map { character -> Character in
            character.isLetter || character.isNumber || character == "-" || character == "_"
                ? character
                : "-"
        }
        let prefix = String(readable.prefix(64))
        let digest = PackageIntegrity.sha256Hex(Data(value.utf8)).prefix(12)
        return "\(prefix)-\(digest)"
    }
}

private struct HubModelInfo: Decodable {
    let sha: String
    let siblings: [HubSibling]
}

private struct HubSibling: Decodable {
    let filename: String
    let size: Int64
    let lfs: HubLFS?

    enum CodingKeys: String, CodingKey {
        case filename = "rfilename"
        case size
        case lfs
    }
}

private struct HubLFS: Decodable {
    let sha256: String
}

private struct CacheMarker: Codable {
    let formatVersion: Int
    let repositoryID: String
    let requestedRevision: String
    let resolvedRevision: String
    let target: NemotronDeviceTarget
    let computePreference: NemotronComputePreference
    let sourceVariant: String
    let selectedKind: String
    let selectedAssetPath: String
    let totalBytes: Int64
}

private final class ModelDownloadDelegate: NSObject, URLSessionDownloadDelegate,
    @unchecked Sendable
{
    private let filename: String
    private let fileBytes: Int64
    private let completedBytes: Int64
    private let totalBytes: Int64
    private let progress: (@Sendable (NemotronModelDownloadProgress) -> Void)?

    init(
        filename: String,
        fileBytes: Int64,
        completedBytes: Int64,
        totalBytes: Int64,
        progress: (@Sendable (NemotronModelDownloadProgress) -> Void)?
    ) {
        self.filename = filename
        self.fileBytes = fileBytes
        self.completedBytes = completedBytes
        self.totalBytes = totalBytes
        self.progress = progress
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        let downloaded = min(max(0, totalBytesWritten), fileBytes)
        progress?(
            NemotronModelDownloadProgress(
                phase: .downloading,
                currentFile: filename,
                completedBytes: min(totalBytes, completedBytes + downloaded),
                totalBytes: totalBytes
            )
        )
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {}
}
