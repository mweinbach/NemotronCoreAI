import CryptoKit
import Foundation

enum PackageIntegrity {
    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func sha256Hex(fileAt url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 4 * 1024 * 1024), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func binaryHex(_ data: Data) -> String {
        data.map { String(format: "%02x", $0) }.joined()
    }

    static func normalizedHash(_ value: String, field: String) throws -> String {
        let normalized = value.lowercased()
        guard normalized.count == 64,
            normalized.utf8.allSatisfy({ byte in
                (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(byte)
                    || (UInt8(ascii: "a")...UInt8(ascii: "f")).contains(byte)
            })
        else {
            throw NemotronError.invalidPackage("\(field) must be a 64-character lowercase hexadecimal value")
        }
        return normalized
    }
}
