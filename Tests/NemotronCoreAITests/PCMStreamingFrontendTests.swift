import Foundation
import XCTest

@testable import NemotronCoreAI

final class PCMStreamingFrontendTests: XCTestCase {
    func testSentinelTailBoundaryAcrossFinish() throws {
        let support = makeRuntimeSupport()

        var tooShort = try PCMStreamingFrontend(support: support, latencyMS: 320)
        XCTAssertTrue(try tooShort.push([Float](repeating: 0, count: 1_119)).isEmpty)
        XCTAssertTrue(try tooShort.finish().isEmpty)

        var sentinelQualified = try PCMStreamingFrontend(support: support, latencyMS: 320)
        XCTAssertTrue(try sentinelQualified.push([Float](repeating: 0, count: 1_120)).isEmpty)
        let chunks = try sentinelQualified.finish()
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0].validLength, 17)
    }

    func testPacketSplitsMatchWholeWaveformChunks() throws {
        let support = makeRuntimeSupport()
        let frontend = LogMelFrontend(support: support)
        let mode = try support.streamingMode(latencyMS: 320)
        let pcm = (0..<7_313).map { index in
            Float(sin(Double(index) * 0.031) * 0.4 + cos(Double(index) * 0.007) * 0.1)
        }
        let wholeFeatures = try frontend.extract(pcm: pcm)
        let expected = try frontend.chunks(from: wholeFeatures, mode: mode)

        var streaming = try PCMStreamingFrontend(support: support, latencyMS: 320)
        var actual: [FeatureChunk] = []
        let packetSizes = [1, 37, 159, 2, 641, 80, 997, 13, 2_048, 333, 1_007, 777]
        var offset = 0
        var packetIndex = 0
        while offset < pcm.count {
            let size = min(packetSizes[packetIndex % packetSizes.count], pcm.count - offset)
            actual.append(contentsOf: try streaming.push(Array(pcm[offset..<(offset + size)])))
            offset += size
            packetIndex += 1
        }
        actual.append(contentsOf: try streaming.finish())

        XCTAssertEqual(streaming.receivedSampleCount, pcm.count)
        XCTAssertLessThan(streaming.bufferedSampleCount, 600)
        XCTAssertEqual(actual.count, expected.count)
        for index in expected.indices {
            XCTAssertEqual(actual[index].featureWidth, expected[index].featureWidth)
            XCTAssertEqual(actual[index].validLength, expected[index].validLength)
            XCTAssertEqual(actual[index].values.count, expected[index].values.count)
            for valueIndex in expected[index].values.indices {
                XCTAssertEqual(actual[index].values[valueIndex], expected[index].values[valueIndex], accuracy: 1e-5)
            }
        }
    }

    func testResetAllowsReuseAfterFinish() throws {
        let support = makeRuntimeSupport()
        var streaming = try PCMStreamingFrontend(support: support, latencyMS: 80)
        _ = try streaming.push([Float](repeating: 0, count: 1_120))
        _ = try streaming.finish()
        XCTAssertTrue(streaming.isFinished)
        XCTAssertThrowsError(try streaming.push([0]))
        streaming.reset()
        XCTAssertFalse(streaming.isFinished)
        XCTAssertEqual(streaming.bufferedSampleCount, 0)
        XCTAssertNoThrow(try streaming.push([0]))
    }

    func testFinishFlushesAHighRatePacketBeforeAnySampleWasEmitted() throws {
        let support = makeRuntimeSupport()
        var streaming = try PCMStreamingFrontend(support: support, latencyMS: 320)

        XCTAssertTrue(try streaming.push([0.25], sampleRate: 48_000).isEmpty)
        XCTAssertNoThrow(try streaming.finish())
        XCTAssertEqual(streaming.receivedSampleCount, 1)
    }
}
