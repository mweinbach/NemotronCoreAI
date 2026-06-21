import XCTest

@testable import NemotronCoreAI

final class PCMStreamingResamplerTests: XCTestCase {
    func test48kHzDownsamplingIsPacketSplitInvariant() throws {
        let input = (0..<4_800).map(Float.init)

        var whole = PCMStreamingResampler()
        var expected = try whole.push(input, inputSampleRate: 48_000)
        expected += try whole.finish()

        var split = PCMStreamingResampler()
        var actual: [Float] = []
        let packetSizes = [1, 2, 7, 31, 113, 509]
        var offset = 0
        var packetIndex = 0
        while offset < input.count {
            let count = min(packetSizes[packetIndex % packetSizes.count], input.count - offset)
            actual += try split.push(
                Array(input[offset..<(offset + count)]),
                inputSampleRate: 48_000
            )
            offset += count
            packetIndex += 1
        }
        actual += try split.finish()

        XCTAssertEqual(expected.count, 1_600)
        XCTAssertEqual(actual, expected)
        XCTAssertEqual(actual, stride(from: 0, to: 4_800, by: 3).map(Float.init))
    }

    func test44100HzResamplingIsPacketSplitInvariant() throws {
        let input = (0..<4_410).map { sin(Float($0) * 0.013) }

        var whole = PCMStreamingResampler()
        var expected = try whole.push(input, inputSampleRate: 44_100)
        expected += try whole.finish()

        var split = PCMStreamingResampler()
        var actual: [Float] = []
        for packet in input.chunked(maximumCount: 137) {
            actual += try split.push(packet, inputSampleRate: 44_100)
        }
        actual += try split.finish()

        XCTAssertEqual(expected.count, 1_600)
        XCTAssertEqual(actual, expected)
    }

    func testInputSampleRateCannotChangeMidStream() throws {
        var resampler = PCMStreamingResampler()
        _ = try resampler.push([0, 1], inputSampleRate: 48_000)
        XCTAssertThrowsError(try resampler.push([2], inputSampleRate: 44_100))
    }
}

extension Array {
    fileprivate func chunked(maximumCount: Int) -> [[Element]] {
        stride(from: 0, to: count, by: maximumCount).map { start in
            Array(self[start..<Swift.min(start + maximumCount, count)])
        }
    }
}
