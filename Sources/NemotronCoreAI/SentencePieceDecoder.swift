import Foundation

public struct SentencePieceDecoder: Sendable {
    private let metadata: TokenizerMetadata

    public init(metadata: TokenizerMetadata) {
        self.metadata = metadata
    }

    public func decode(_ tokenIDs: [Int]) -> String {
        var result = ""
        var bytes: [UInt8] = []

        func flushBytes(into output: inout String, bytes: inout [UInt8]) {
            guard !bytes.isEmpty else { return }
            output += String(decoding: bytes, as: UTF8.self)
            bytes.removeAll(keepingCapacity: true)
        }

        for tokenID in tokenIDs {
            if tokenID == metadata.beginningOfSentenceID
                || tokenID == metadata.endOfSentenceID
                || tokenID == metadata.paddingID
            {
                continue
            }

            let piece: String
            if tokenID == metadata.unknownID {
                piece = metadata.unknownSurface
            } else if metadata.pieces.indices.contains(tokenID) {
                piece = metadata.pieces[tokenID]
            } else {
                continue
            }

            if let byte = Self.byteValue(piece) {
                bytes.append(byte)
            } else {
                flushBytes(into: &result, bytes: &bytes)
                result += piece
            }
        }
        flushBytes(into: &result, bytes: &bytes)
        result = result.replacingOccurrences(of: "▁", with: " ")
        if result.first == " " {
            result.removeFirst()
        }
        return result
    }

    private static func byteValue(_ piece: String) -> UInt8? {
        guard piece.count == 6,
            piece.hasPrefix("<0x"),
            piece.hasSuffix(">")
        else { return nil }
        let start = piece.index(piece.startIndex, offsetBy: 3)
        let end = piece.index(start, offsetBy: 2)
        return UInt8(piece[start..<end], radix: 16)
    }
}
