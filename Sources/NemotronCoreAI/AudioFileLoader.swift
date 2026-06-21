import AVFoundation
import Foundation

public enum AudioFileLoader {
    /// Loads a decoded 16 kHz mono file. AVAudioFile handles WAV, AIFF, CAF,
    /// FLAC, and other formats supported by the host OS.
    public static func load16kHzMono(from url: URL) throws -> [Float] {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw NemotronError.invalidAudio("could not open \(url.path): \(error)")
        }

        let format = file.processingFormat
        guard abs(format.sampleRate - 16_000) < 0.5 else {
            throw NemotronError.invalidAudio("file sample rate is \(format.sampleRate) Hz; expected 16000 Hz")
        }
        guard format.channelCount == 1 else {
            throw NemotronError.invalidAudio("file has \(format.channelCount) channels; expected mono")
        }
        guard file.length > 0, file.length <= AVAudioFramePosition(UInt32.max) else {
            throw NemotronError.invalidAudio("file is empty or too large to decode in one buffer")
        }
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: AVAudioFrameCount(file.length)
            )
        else {
            throw NemotronError.invalidAudio("could not allocate an audio buffer")
        }
        do {
            try file.read(into: buffer)
        } catch {
            throw NemotronError.invalidAudio("could not decode \(url.path): \(error)")
        }
        let count = Int(buffer.frameLength)
        guard count > 0 else {
            throw NemotronError.invalidAudio("decoded file contains no samples")
        }

        switch format.commonFormat {
        case .pcmFormatFloat32:
            guard let pointer = buffer.floatChannelData?[0] else {
                throw NemotronError.invalidAudio("decoded float32 samples are unavailable")
            }
            return Array(UnsafeBufferPointer(start: pointer, count: count))
        case .pcmFormatFloat64:
            let buffers = UnsafeMutableAudioBufferListPointer(buffer.mutableAudioBufferList)
            guard let data = buffers.first?.mData else {
                throw NemotronError.invalidAudio("decoded float64 samples are unavailable")
            }
            let pointer = data.assumingMemoryBound(to: Double.self)
            return UnsafeBufferPointer(start: pointer, count: count).map(Float.init)
        case .pcmFormatInt16:
            guard let pointer = buffer.int16ChannelData?[0] else {
                throw NemotronError.invalidAudio("decoded int16 samples are unavailable")
            }
            return UnsafeBufferPointer(start: pointer, count: count).map { Float($0) / 32_768 }
        case .pcmFormatInt32:
            guard let pointer = buffer.int32ChannelData?[0] else {
                throw NemotronError.invalidAudio("decoded int32 samples are unavailable")
            }
            return UnsafeBufferPointer(start: pointer, count: count).map { Float($0) / 2_147_483_648 }
        case .otherFormat:
            throw NemotronError.invalidAudio("unsupported decoded PCM format")
        @unknown default:
            throw NemotronError.invalidAudio("unknown decoded PCM format")
        }
    }
}
