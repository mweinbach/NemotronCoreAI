import XCTest

@testable import NemotronCoreAI

final class LogMelFrontendTests: XCTestCase {
    func testCenteredSTFTMatchesAnalyticalImpulseSpectrum() throws {
        var filterbank = [Float](repeating: 0, count: 128 * 257)
        filterbank[0] = 1
        filterbank[257 + 128] = 1
        let support = makeRuntimeSupport(filterbankValues: filterbank)
        try support.validate()
        let frontend = LogMelFrontend(support: support)
        var pcm = [Float](repeating: 0, count: 160)
        pcm[0] = 1

        let features = try frontend.extract(pcm: pcm)
        XCTAssertEqual(features.frameCount, 2)
        XCTAssertEqual(features.validFrameCount, 1)
        let guardValue = support.featureExtractor.logGuard
        XCTAssertEqual(features.values[0], logf(0.0009 + guardValue), accuracy: 2e-4)
        XCTAssertEqual(features.values[2], logf(1.9409 + guardValue), accuracy: 2e-4)
        for bin in 0..<128 {
            XCTAssertEqual(features.values[bin * 2 + 1], 0)
        }
    }

    func testTerminalSentinelControlsTailThreshold() throws {
        let support = makeRuntimeSupport()
        let frontend = LogMelFrontend(support: support)
        let mode = try support.streamingMode(latencyMS: 320)

        let exactlyEightFrames = try frontend.extract(pcm: [Float](repeating: 0, count: 1_120))
        XCTAssertEqual(exactlyEightFrames.frameCount, 8)
        let emitted = try frontend.chunks(from: exactlyEightFrames, mode: mode)
        XCTAssertEqual(emitted.count, 1)
        XCTAssertEqual(emitted[0].validLength, 17)

        let onlySevenFrames = try frontend.extract(pcm: [Float](repeating: 0, count: 1_119))
        XCTAssertEqual(onlySevenFrames.frameCount, 7)
        XCTAssertTrue(try frontend.chunks(from: onlySevenFrames, mode: mode).isEmpty)

        let exactFullBoundary = try frontend.extract(pcm: [Float](repeating: 0, count: 5_120))
        XCTAssertEqual(exactFullBoundary.frameCount, 33)
        let fullChunks = try frontend.chunks(from: exactFullBoundary, mode: mode)
        XCTAssertEqual(fullChunks.count, 1, "the lone terminal sentinel must not create another chunk")
        XCTAssertEqual(fullChunks[0].validLength, 41)
    }

    func testChunkerUsesLeftOverlapAndBinMajorLayout() throws {
        let support = makeRuntimeSupport()
        let frontend = LogMelFrontend(support: support)
        let mode = try support.streamingMode(latencyMS: 320)
        let frameCount = 33
        var values = [Float](repeating: 0, count: 128 * frameCount)
        for bin in 0..<128 {
            for frame in 0..<frameCount {
                values[bin * frameCount + frame] = Float(bin * 1_000 + frame + 1)
            }
        }
        let features = LogMelFeatures(
            values: values, bins: 128, frameCount: frameCount, validFrameCount: 32
        )
        let chunks = try frontend.chunks(from: features, mode: mode)
        XCTAssertEqual(chunks.count, 1)
        for frame in 0..<9 {
            XCTAssertEqual(chunks[0].values[frame], 0)
        }
        XCTAssertEqual(Array(chunks[0].values[9..<41]), (1...32).map(Float.init))
    }
}
