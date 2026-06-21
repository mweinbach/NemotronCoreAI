import AVFoundation
import XCTest

@testable import NemotronCoreAI

final class AudioFileLoaderTests: XCTestCase {
    func testLoads16kHzMonoPCM() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("mono.wav")
        let expected: [Float] = [0, 0.25, -0.5, 0.75]
        try writeWAV(url: url, channels: 1, samples: expected)
        let actual = try AudioFileLoader.load16kHzMono(from: url)
        XCTAssertEqual(actual.count, expected.count)
        for index in expected.indices {
            XCTAssertEqual(actual[index], expected[index], accuracy: 1e-6)
        }
    }

    func testRejectsNonMonoFile() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("stereo.wav")
        try writeWAV(url: url, channels: 2, samples: [0, 0])
        XCTAssertThrowsError(try AudioFileLoader.load16kHzMono(from: url))
    }

    private func writeWAV(url: URL, channels: AVAudioChannelCount, samples: [Float]) throws {
        guard
            let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: 16_000,
                channels: channels,
                interleaved: false
            ),
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: AVAudioFrameCount(samples.count)
            )
        else {
            XCTFail("could not allocate test audio")
            return
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        for channel in 0..<Int(channels) {
            guard let destination = buffer.floatChannelData?[channel] else {
                XCTFail("missing test channel")
                return
            }
            for index in samples.indices {
                destination[index] = samples[index]
            }
        }
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
    }
}
