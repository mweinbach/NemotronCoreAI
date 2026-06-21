import XCTest

@testable import NemotronCoreAI

final class SentencePieceDecoderTests: XCTestCase {
    func testSentencePieceSpacingSpecialTokensAndByteFallback() {
        let metadata = TokenizerMetadata(
            type: "sentencepiece",
            pieces: [
                "<unk>", "<s>", "</s>", "▁hello", "▁world",
                "<0xF0>", "<0x9F>", "<0x98>", "<0x80>", "<en-US>", "<pad>",
            ],
            unknownID: 0,
            unknownSurface: " ⁇ ",
            beginningOfSentenceID: 1,
            endOfSentenceID: 2,
            paddingID: 10,
            assetRelativeToReferenceBundle: "tokenizer.model",
            assetSHA256: "tokenizer",
            pieceCount: 11
        )
        let decoder = SentencePieceDecoder(metadata: metadata)
        XCTAssertEqual(
            decoder.decode([1, 3, 4, 5, 6, 7, 8, 9, 2, 10]),
            "hello world😀<en-US>"
        )
        XCTAssertEqual(decoder.decode([0]), "⁇ ")
    }

    func testInvalidAndBlankIDsAreExcluded() {
        let metadata = TokenizerMetadata(
            type: "sentencepiece",
            pieces: ["<unk>", "<s>", "</s>", "▁ok", "<pad>"],
            unknownID: 0,
            unknownSurface: " ⁇ ",
            beginningOfSentenceID: 1,
            endOfSentenceID: 2,
            paddingID: 4,
            assetRelativeToReferenceBundle: "tokenizer.model",
            assetSHA256: "tokenizer",
            pieceCount: 5
        )
        let decoder = SentencePieceDecoder(metadata: metadata)
        XCTAssertEqual(decoder.decode([99, 3, 5]), "ok")
    }
}
