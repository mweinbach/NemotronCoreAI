import Accelerate
import Foundation

public struct LogMelFeatures: Sendable, Equatable {
    public let values: [Float]
    public let bins: Int
    public let frameCount: Int
    public let validFrameCount: Int

    public init(values: [Float], bins: Int, frameCount: Int, validFrameCount: Int) {
        self.values = values
        self.bins = bins
        self.frameCount = frameCount
        self.validFrameCount = validFrameCount
    }
}

public struct FeatureChunk: Sendable, Equatable {
    /// Row-major `[1, bins, featureWidth]` data, with frame as the innermost dimension.
    public let values: [Float]
    public let featureWidth: Int
    public let validLength: Int

    public init(values: [Float], featureWidth: Int, validLength: Int) {
        self.values = values
        self.featureWidth = featureWidth
        self.validLength = validLength
    }
}

public struct LogMelFrontend: Sendable {
    public let support: RuntimeSupport

    public init(support: RuntimeSupport) {
        self.support = support
    }

    /// Reproduces the NeMo export frontend: preemphasis, center-padded STFT,
    /// power spectrum, 128x257 filterbank projection, and guarded natural log.
    public func extract(pcm: [Float], sampleRate: Int = 16_000) throws -> LogMelFeatures {
        guard sampleRate == support.sampleRate else {
            throw NemotronError.invalidAudio("expected \(support.sampleRate) Hz, received \(sampleRate) Hz")
        }
        guard !pcm.isEmpty else {
            throw NemotronError.invalidAudio("PCM input is empty")
        }
        guard pcm.allSatisfy(\.isFinite) else {
            throw NemotronError.invalidAudio("PCM contains a non-finite sample")
        }

        let configuration = support.featureExtractor
        let frameCount = pcm.count / configuration.hopSamples + 1
        let validFrameCount = pcm.count / configuration.hopSamples
        var mel = try extractFrameRange(pcm: pcm, frames: 0..<frameCount)
        if validFrameCount < frameCount {
            for bin in 0..<support.filterbank.rows {
                for frame in validFrameCount..<frameCount {
                    mel[bin * frameCount + frame] = 0
                }
            }
        }
        return LogMelFeatures(
            values: mel,
            bins: support.filterbank.rows,
            frameCount: frameCount,
            validFrameCount: validFrameCount
        )
    }

    /// Computes only the requested absolute STFT frames. Streaming callers use
    /// this to avoid recomputing frames that have already become stable.
    func extractFrameRange(
        pcm: [Float],
        pcmStartSample: Int = 0,
        totalSampleCount: Int? = nil,
        frames: Range<Int>
    ) throws -> [Float] {
        guard !frames.isEmpty else { return [] }
        let configuration = support.featureExtractor
        let absoluteSampleCount = totalSampleCount ?? pcm.count
        let maximumFrameCount = absoluteSampleCount / configuration.hopSamples + 1
        let windowOffset = (configuration.fftSize - configuration.windowSamples) / 2
        let firstSignalSample = max(
            0,
            frames.lowerBound * configuration.hopSamples + windowOffset - configuration.centerPaddingSamples
        )
        let requiredBufferStart = max(0, firstSignalSample - 1)
        guard !pcm.isEmpty,
            pcmStartSample >= 0,
            pcmStartSample + pcm.count == absoluteSampleCount,
            pcmStartSample <= requiredBufferStart,
            frames.lowerBound >= 0,
            frames.upperBound <= maximumFrameCount
        else {
            throw NemotronError.invalidAudio("requested STFT frame range \(frames) is unavailable")
        }

        let fftSize = configuration.fftSize
        let frequencyBins = fftSize / 2 + 1
        let projectedFrameCount = frames.count
        guard
            let dft = vDSP_DFT_zop_CreateSetup(
                nil,
                vDSP_Length(fftSize),
                vDSP_DFT_Direction.FORWARD
            )
        else {
            throw NemotronError.inference("Accelerate could not create a \(fftSize)-point DFT setup")
        }
        defer { vDSP_DFT_DestroySetup(dft) }

        var realInput = [Float](repeating: 0, count: fftSize)
        let imaginaryInput = [Float](repeating: 0, count: fftSize)
        var realOutput = [Float](repeating: 0, count: fftSize)
        var imaginaryOutput = [Float](repeating: 0, count: fftSize)
        // Power is [frequencyBin, localFrame] so a single vDSP matrix
        // multiplication projects all newly stable frames.
        var power = [Float](repeating: 0, count: frequencyBins * projectedFrameCount)
        for (localFrame, absoluteFrame) in frames.enumerated() {
            for index in realInput.indices {
                realInput[index] = 0
            }
            let paddedFrameStart = absoluteFrame * configuration.hopSamples
            for windowIndex in support.window.values.indices {
                let paddedSignalIndex = paddedFrameStart + windowOffset + windowIndex
                let signalIndex = paddedSignalIndex - configuration.centerPaddingSamples
                guard signalIndex >= 0, signalIndex < absoluteSampleCount else { continue }
                let localSignalIndex = signalIndex - pcmStartSample
                guard pcm.indices.contains(localSignalIndex) else {
                    throw NemotronError.invalidAudio(
                        "PCM history required for STFT frame \(absoluteFrame) was discarded")
                }
                let emphasizedSample: Float
                if signalIndex == 0 {
                    emphasizedSample = pcm[localSignalIndex]
                } else {
                    emphasizedSample =
                        pcm[localSignalIndex] - configuration.preemphasis * pcm[localSignalIndex - 1]
                }
                realInput[windowOffset + windowIndex] = emphasizedSample * support.window.values[windowIndex]
            }

            realInput.withUnsafeBufferPointer { realPointer in
                imaginaryInput.withUnsafeBufferPointer { imaginaryPointer in
                    realOutput.withUnsafeMutableBufferPointer { realOutputPointer in
                        imaginaryOutput.withUnsafeMutableBufferPointer { imaginaryOutputPointer in
                            vDSP_DFT_Execute(
                                dft,
                                realPointer.baseAddress!,
                                imaginaryPointer.baseAddress!,
                                realOutputPointer.baseAddress!,
                                imaginaryOutputPointer.baseAddress!
                            )
                        }
                    }
                }
            }
            for bin in 0..<frequencyBins {
                let real = realOutput[bin]
                let imaginary = imaginaryOutput[bin]
                power[bin * projectedFrameCount + localFrame] = real * real + imaginary * imaginary
            }
        }

        var mel = [Float](repeating: 0, count: support.filterbank.rows * projectedFrameCount)
        support.filterbank.values.withUnsafeBufferPointer { filterbankPointer in
            power.withUnsafeBufferPointer { powerPointer in
                mel.withUnsafeMutableBufferPointer { melPointer in
                    vDSP_mmul(
                        filterbankPointer.baseAddress!, 1,
                        powerPointer.baseAddress!, 1,
                        melPointer.baseAddress!, 1,
                        vDSP_Length(support.filterbank.rows),
                        vDSP_Length(projectedFrameCount),
                        vDSP_Length(frequencyBins)
                    )
                }
            }
        }
        for index in mel.indices {
            mel[index] = logf(mel[index] + configuration.logGuard)
        }
        return mel
    }

    public func chunks(from features: LogMelFeatures, mode: StreamingMode) throws -> [FeatureChunk] {
        guard features.bins == support.featureExtractor.bins,
            features.values.count == features.bins * features.frameCount
        else {
            throw NemotronError.invalidFeatureChunk("feature matrix dimensions are inconsistent")
        }
        let overlap = support.featureExtractor.hostManagedOverlapFrames
        let newFramesPerChunk = mode.featureWidth - overlap
        let minimumNewFrames = support.streamingContract.tailMinimumFramesIncludingSentinel
        guard newFramesPerChunk >= minimumNewFrames else {
            throw NemotronError.invalidRuntimeSupport(
                "streaming mode \(mode.latencyMS) ms accepts fewer than 8 new frames")
        }

        var chunks: [FeatureChunk] = []
        var offset = 0
        while offset < features.frameCount {
            let remaining = features.frameCount - offset
            guard remaining >= minimumNewFrames else { break }
            let currentCount = min(newFramesPerChunk, remaining)
            chunks.append(try makeChunk(from: features, mode: mode, offset: offset, currentCount: currentCount))
            offset += newFramesPerChunk
        }
        return chunks
    }

    func makeChunk(
        from features: LogMelFeatures,
        mode: StreamingMode,
        offset: Int,
        currentCount: Int
    ) throws -> FeatureChunk {
        guard offset >= 0,
            currentCount > 0,
            offset + currentCount <= features.frameCount,
            features.values.count == features.bins * features.frameCount
        else {
            throw NemotronError.invalidFeatureChunk("chunk range is outside the feature matrix")
        }
        let overlap = support.streamingContract.featureOverlapFrames
        let previousCount = min(overlap, offset)
        let leftPadding = overlap - previousCount
        var values = [Float](repeating: 0, count: features.bins * mode.featureWidth)
        for bin in 0..<features.bins {
            let sourceRow = bin * features.frameCount
            let destinationRow = bin * mode.featureWidth
            if previousCount > 0 {
                for index in 0..<previousCount {
                    values[destinationRow + leftPadding + index] =
                        features.values[sourceRow + offset - previousCount + index]
                }
            }
            for index in 0..<currentCount {
                values[destinationRow + overlap + index] = features.values[sourceRow + offset + index]
            }
        }
        return FeatureChunk(
            values: values,
            featureWidth: mode.featureWidth,
            validLength: overlap + currentCount
        )
    }
}
