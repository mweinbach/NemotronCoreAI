import Foundation

/// Incremental PCM frontend for live streams.
///
/// `push` emits only full chunks whose centered STFT frames are stable. `finish`
/// right-pads the final valid frames after preemphasis, appends the required
/// zero sentinel, and applies the same eight-frame tail threshold as whole-file
/// transcription. The latency mode is fixed for the lifetime of the value.
public struct PCMStreamingFrontend: Sendable {
    public let latencyMS: Int
    public private(set) var isFinished = false
    public var bufferedSampleCount: Int { pcm.count }
    public private(set) var receivedSampleCount = 0

    private let support: RuntimeSupport
    private let mode: StreamingMode
    private let frontend: LogMelFrontend
    private var resampler: PCMStreamingResampler
    private var pcm: [Float] = []
    private var pcmStartSample = 0
    private var featureValues: [Float] = []
    private var featureBaseFrame = 0
    private var featureFrameCount = 0
    private var nextChunkOffset = 0

    public init(support: RuntimeSupport, latencyMS: Int = 320) throws {
        self.support = support
        self.mode = try support.streamingMode(latencyMS: latencyMS)
        self.frontend = LogMelFrontend(support: support)
        self.resampler = PCMStreamingResampler(outputSampleRate: Double(support.sampleRate))
        self.latencyMS = latencyMS
    }

    /// Adds a packet of mono float PCM and returns any newly stable full chunks.
    public mutating func push(_ samples: [Float], sampleRate: Int = 16_000) throws -> [FeatureChunk] {
        guard !isFinished else {
            throw NemotronError.invalidAudio("PCM stream is finished; reset before pushing more samples")
        }
        guard !samples.isEmpty else { return [] }
        let resampled = try resampler.push(samples, inputSampleRate: Double(sampleRate))
        guard !resampled.isEmpty else { return [] }
        pcm.append(contentsOf: resampled)
        receivedSampleCount += resampled.count

        let configuration = support.featureExtractor
        let windowOffset = (configuration.fftSize - configuration.windowSamples) / 2
        let futureSamples = windowOffset + configuration.windowSamples - configuration.centerPaddingSamples
        let stableFrameCount: Int
        if receivedSampleCount >= futureSamples {
            stableFrameCount = (receivedSampleCount - futureSamples) / configuration.hopSamples + 1
        } else {
            stableFrameCount = 0
        }
        try appendFrames(upTo: stableFrameCount)
        return try emitChunks(allowPartialTail: false)
    }

    /// Finalizes the stream and emits its sentinel-qualified tail, if any.
    public mutating func finish() throws -> [FeatureChunk] {
        guard !isFinished else {
            throw NemotronError.invalidAudio("PCM stream has already been finished")
        }
        let finalSamples = try resampler.finish()
        if !finalSamples.isEmpty {
            pcm.append(contentsOf: finalSamples)
            receivedSampleCount += finalSamples.count
        }
        let validFrames = receivedSampleCount / support.featureExtractor.hopSamples
        try appendFrames(upTo: validFrames)
        appendZeroFrames(count: support.streamingContract.terminalZeroSentinelFrames)
        let chunks = try emitChunks(allowPartialTail: true)
        isFinished = true
        return chunks
    }

    public mutating func reset() {
        pcm.removeAll(keepingCapacity: true)
        pcmStartSample = 0
        receivedSampleCount = 0
        featureValues.removeAll(keepingCapacity: true)
        featureBaseFrame = 0
        featureFrameCount = 0
        nextChunkOffset = 0
        isFinished = false
        resampler.reset()
    }

    private mutating func appendFrames(upTo upperBound: Int) throws {
        guard upperBound >= featureFrameCount else {
            throw NemotronError.invalidAudio("stable PCM frame count moved backwards")
        }
        guard upperBound > featureFrameCount else { return }
        let range = featureFrameCount..<upperBound
        let values = try frontend.extractFrameRange(
            pcm: pcm,
            pcmStartSample: pcmStartSample,
            totalSampleCount: receivedSampleCount,
            frames: range
        )
        appendFeatureColumns(values, count: range.count)
        prunePCMHistory()
    }

    private mutating func appendZeroFrames(count: Int) {
        guard count > 0 else { return }
        appendFeatureColumns(
            [Float](repeating: 0, count: support.featureExtractor.bins * count),
            count: count
        )
    }

    private mutating func appendFeatureColumns(_ newValues: [Float], count newCount: Int) {
        let bins = support.featureExtractor.bins
        precondition(newValues.count == bins * newCount)
        let oldCount = featureFrameCount - featureBaseFrame
        precondition(featureValues.count == bins * oldCount)
        var combined = [Float](repeating: 0, count: bins * (oldCount + newCount))
        for bin in 0..<bins {
            let oldRow = bin * oldCount
            let newRow = bin * newCount
            let combinedRow = bin * (oldCount + newCount)
            for frame in 0..<oldCount {
                combined[combinedRow + frame] = featureValues[oldRow + frame]
            }
            for frame in 0..<newCount {
                combined[combinedRow + oldCount + frame] = newValues[newRow + frame]
            }
        }
        featureValues = combined
        featureFrameCount += newCount
    }

    private mutating func emitChunks(allowPartialTail: Bool) throws -> [FeatureChunk] {
        let overlap = support.streamingContract.featureOverlapFrames
        let newFrames = mode.featureWidth - overlap
        let minimumTail = support.streamingContract.tailMinimumFramesIncludingSentinel
        var chunks: [FeatureChunk] = []
        while true {
            let remaining = featureFrameCount - nextChunkOffset
            if allowPartialTail {
                guard remaining >= minimumTail else { break }
            } else {
                guard remaining >= newFrames else { break }
            }
            let currentCount = min(newFrames, remaining)
            let storedCount = featureFrameCount - featureBaseFrame
            let storedFeatures = LogMelFeatures(
                values: featureValues,
                bins: support.featureExtractor.bins,
                frameCount: storedCount,
                validFrameCount: storedCount
            )
            chunks.append(
                try frontend.makeChunk(
                    from: storedFeatures,
                    mode: mode,
                    offset: nextChunkOffset - featureBaseFrame,
                    currentCount: currentCount
                )
            )
            nextChunkOffset += newFrames
        }
        pruneEmittedFeatures()
        return chunks
    }

    private mutating func pruneEmittedFeatures() {
        let overlap = support.streamingContract.featureOverlapFrames
        let retainFrom = min(featureFrameCount, max(0, nextChunkOffset - overlap))
        guard retainFrom > featureBaseFrame else { return }
        let oldCount = featureFrameCount - featureBaseFrame
        let droppedCount = retainFrom - featureBaseFrame
        let retainedCount = oldCount - droppedCount
        let bins = support.featureExtractor.bins
        var retained = [Float](repeating: 0, count: bins * retainedCount)
        for bin in 0..<bins {
            let oldRow = bin * oldCount
            let newRow = bin * retainedCount
            for frame in 0..<retainedCount {
                retained[newRow + frame] = featureValues[oldRow + droppedCount + frame]
            }
        }
        featureValues = retained
        featureBaseFrame = retainFrom
    }

    private mutating func prunePCMHistory() {
        let configuration = support.featureExtractor
        let windowOffset = (configuration.fftSize - configuration.windowSamples) / 2
        let nextSignalSample =
            featureFrameCount * configuration.hopSamples + windowOffset - configuration.centerPaddingSamples
        let retainFrom = max(0, nextSignalSample - 1)
        let dropCount = min(pcm.count, max(0, retainFrom - pcmStartSample))
        guard dropCount > 0 else { return }
        pcm.removeFirst(dropCount)
        pcmStartSample += dropCount
    }
}
