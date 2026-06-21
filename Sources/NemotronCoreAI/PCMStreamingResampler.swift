import Foundation

/// Packet-stable linear PCM resampling for live microphone streams.
///
/// The resampler retains interpolation state across calls, so changing packet
/// boundaries does not change the resulting 16 kHz samples.
public struct PCMStreamingResampler: Sendable {
    public let outputSampleRate: Double

    private var inputSampleRate: Double?
    private var bufferedSamples: [Float] = []
    private var bufferStartIndex: Int64 = 0
    private var nextOutputPosition: Double = 0
    private var receivedInputSamples: Int64 = 0
    private var emittedOutputSamples: Int64 = 0
    private var isFinished = false

    public init(outputSampleRate: Double = 16_000) {
        self.outputSampleRate = outputSampleRate
    }

    public mutating func push(
        _ samples: [Float],
        inputSampleRate: Double
    ) throws -> [Float] {
        guard !isFinished else {
            throw NemotronError.invalidAudio("PCM resampler is already finished")
        }
        try establish(inputSampleRate: inputSampleRate, samples: samples)
        guard !samples.isEmpty else { return [] }

        bufferedSamples.append(contentsOf: samples)
        receivedInputSamples += Int64(samples.count)
        return emitAvailable(allowTerminalHold: false, maximumOutputCount: nil)
    }

    /// Emits any final sample whose interpolation needs the terminal input.
    public mutating func finish() throws -> [Float] {
        guard !isFinished else {
            throw NemotronError.invalidAudio("PCM resampler is already finished")
        }
        guard let inputSampleRate, receivedInputSamples > 0 else {
            throw NemotronError.invalidAudio("PCM resampler is empty")
        }
        let expectedOutputCount = max(
            1,
            Int64(
                (Double(receivedInputSamples) * outputSampleRate / inputSampleRate)
                    .rounded()
            )
        )
        let output = emitAvailable(
            allowTerminalHold: true,
            maximumOutputCount: expectedOutputCount
        )
        isFinished = true
        return output
    }

    public mutating func reset() {
        inputSampleRate = nil
        bufferedSamples.removeAll(keepingCapacity: true)
        bufferStartIndex = 0
        nextOutputPosition = 0
        receivedInputSamples = 0
        emittedOutputSamples = 0
        isFinished = false
    }

    private mutating func establish(
        inputSampleRate: Double,
        samples: [Float]
    ) throws {
        guard inputSampleRate.isFinite, inputSampleRate > 0 else {
            throw NemotronError.invalidAudio("invalid input sample rate \(inputSampleRate)")
        }
        guard outputSampleRate.isFinite, outputSampleRate > 0 else {
            throw NemotronError.invalidAudio("invalid output sample rate \(outputSampleRate)")
        }
        guard samples.allSatisfy(\.isFinite) else {
            throw NemotronError.invalidAudio("PCM packet contains a non-finite sample")
        }
        if let establishedRate = self.inputSampleRate {
            guard abs(establishedRate - inputSampleRate) < 0.5 else {
                throw NemotronError.invalidAudio(
                    "input sample rate changed from \(establishedRate) Hz to \(inputSampleRate) Hz"
                )
            }
        } else {
            self.inputSampleRate = inputSampleRate
        }
    }

    private mutating func emitAvailable(
        allowTerminalHold: Bool,
        maximumOutputCount: Int64?
    ) -> [Float] {
        guard let inputSampleRate, !bufferedSamples.isEmpty else { return [] }
        let ratio = inputSampleRate / outputSampleRate
        let lastAvailableIndex = bufferStartIndex + Int64(bufferedSamples.count) - 1
        let terminalLimit = Double(receivedInputSamples)
        var output: [Float] = []

        while nextOutputPosition < terminalLimit
            && maximumOutputCount.map({ emittedOutputSamples < $0 }) != false
        {
            let lowerIndex = Int64(floor(nextOutputPosition))
            let requestedUpperIndex = Int64(ceil(nextOutputPosition))
            if requestedUpperIndex > lastAvailableIndex && !allowTerminalHold {
                break
            }
            let upperIndex = min(requestedUpperIndex, lastAvailableIndex)
            let lowerOffset = Int(lowerIndex - bufferStartIndex)
            let upperOffset = Int(upperIndex - bufferStartIndex)
            guard lowerOffset >= 0, upperOffset >= 0,
                lowerOffset < bufferedSamples.count,
                upperOffset < bufferedSamples.count
            else {
                break
            }
            let fraction = Float(nextOutputPosition - Double(lowerIndex))
            let lower = bufferedSamples[lowerOffset]
            let upper = bufferedSamples[upperOffset]
            output.append(lower + ((upper - lower) * fraction))
            emittedOutputSamples += 1
            nextOutputPosition += ratio
        }

        let firstNeededIndex = Int64(floor(nextOutputPosition))
        let removableCount = min(
            bufferedSamples.count,
            max(0, Int(firstNeededIndex - bufferStartIndex))
        )
        if removableCount > 0 {
            bufferedSamples.removeFirst(removableCount)
            bufferStartIndex += Int64(removableCount)
        }
        return output
    }
}
