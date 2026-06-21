import Foundation
import NemotronCoreAI

@main
struct NemotronCoreAICLI {
    struct Options {
        var packageURL: URL
        var audioURL: URL?
        var sourceVariant: String?
        var latencyMS = 320
        var preferAOT = true
        var computePreference = NemotronComputePreference.automatic
        var targetLanguage = "auto"
        var stripLanguageTag = false
        var packetSamples: Int?
    }

    static func main() async {
        do {
            try await run(arguments: Array(CommandLine.arguments.dropFirst()))
        } catch {
            writeError("error: \(error.localizedDescription)\n")
            writeError(usage)
            Foundation.exit(EXIT_FAILURE)
        }
    }

    static func run(arguments: [String]) async throws {
        guard let command = arguments.first else {
            print(usage, terminator: "")
            return
        }
        if command == "--help" || command == "-h" || command == "help" {
            print(usage, terminator: "")
            return
        }
        guard command == "inspect" || command == "transcribe" else {
            throw CLIError("unknown command '\(command)'")
        }
        let options = try parse(command: command, arguments: Array(arguments.dropFirst()))
        let session = try await NemotronCoreAI.loadSession(
            packageURL: options.packageURL,
            latencyMS: options.latencyMS,
            sourceVariant: options.sourceVariant,
            preferAOT: options.preferAOT,
            computePreference: options.computePreference,
            targetLanguage: options.targetLanguage,
            stripLanguageTag: options.stripLanguageTag
        )
        let information = await session.information()
        print("model: \(information.model)")
        print("architecture: \(information.architecture)")
        print("platform: \(information.platform.rawValue)")
        print("compute: \(information.computePreference.rawValue)")
        print("asset: \(information.modelURL.path)")
        print("asset-kind: \(information.modelKind.rawValue)")
        print("source-variant: \(information.sourceVariant)")
        print("streaming: \(information.latencyMS) ms via \(information.encoderFunction)")
        print("fused-predict-joint: \(information.usesFusedPredictorJoint ? "yes" : "no")")

        guard command == "transcribe" else { return }
        guard let audioURL = options.audioURL else {
            throw CLIError("transcribe requires an audio file")
        }
        if let packetSamples = options.packetSamples {
            let pcm = try AudioFileLoader.load16kHzMono(from: audioURL)
            try await session.beginPCMStream(
                targetLanguage: options.targetLanguage,
                stripLanguageTag: options.stripLanguageTag
            )
            var offset = 0
            var packets = 0
            while offset < pcm.count {
                let count = min(packetSamples, pcm.count - offset)
                _ = try await session.pushPCM(Array(pcm[offset..<(offset + count)]))
                offset += count
                packets += 1
            }
            let final = try await session.finishPCMStream()
            print("audio-seconds: \(String(format: "%.3f", Double(pcm.count) / 16_000))")
            print("packets: \(packets)")
            print("tokens: \(final.tokenIDs.count)")
            print(final.text)
            return
        }
        let result = try await session.transcribe(
            fileURL: audioURL,
            targetLanguage: options.targetLanguage,
            stripLanguageTag: options.stripLanguageTag
        )
        print("audio-seconds: \(String(format: "%.3f", result.audioSeconds))")
        print("chunks: \(result.processedChunks)")
        print("tokens: \(result.tokenIDs.count)")
        print(result.text)
    }

    static func parse(command: String, arguments: [String]) throws -> Options {
        guard let packagePath = arguments.first, !packagePath.hasPrefix("-") else {
            throw CLIError("\(command) requires a model package path")
        }
        var options = Options(
            packageURL: URL(fileURLWithPath: packagePath).standardizedFileURL
        )
        var index = 1
        if command == "transcribe" {
            guard arguments.indices.contains(index), !arguments[index].hasPrefix("-") else {
                throw CLIError("transcribe requires an audio file after the package path")
            }
            options.audioURL = URL(fileURLWithPath: arguments[index]).standardizedFileURL
            index += 1
        }

        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "--variant":
                index += 1
                guard arguments.indices.contains(index) else { throw CLIError("--variant requires a value") }
                options.sourceVariant = arguments[index]
            case "--latency":
                index += 1
                guard arguments.indices.contains(index), let latency = Int(arguments[index]) else {
                    throw CLIError("--latency requires an integer")
                }
                options.latencyMS = latency
            case "--language":
                index += 1
                guard arguments.indices.contains(index) else { throw CLIError("--language requires a value") }
                options.targetLanguage = arguments[index]
            case "--source-only":
                options.preferAOT = false
            case "--compute":
                index += 1
                guard arguments.indices.contains(index),
                    let preference = NemotronComputePreference(rawValue: arguments[index])
                else {
                    throw CLIError("--compute requires automatic, gpu, or neural-engine")
                }
                options.computePreference = preference
            case "--strip-language-tag":
                options.stripLanguageTag = true
            case "--packet-samples":
                index += 1
                guard arguments.indices.contains(index),
                    let samples = Int(arguments[index]),
                    samples > 0
                else {
                    throw CLIError("--packet-samples requires a positive integer")
                }
                options.packetSamples = samples
            default:
                throw CLIError("unknown option '\(argument)'")
            }
            index += 1
        }
        return options
    }

    static let usage = """
        Usage:
          nemotron-coreai inspect MODEL_PACKAGE [options]
          nemotron-coreai transcribe MODEL_PACKAGE AUDIO_FILE [options]

        Options:
          --variant NAME          Select a source package variant (default: fp16)
          --latency MS            Streaming encoder latency: 80, 160, 320, 560, 1120
          --language NAME         Prompt dictionary key (default: auto)
          --compute UNIT          automatic, gpu, or neural-engine
          --source-only           Disable device-matched 320 ms AOT selection
          --strip-language-tag    Remove a terminal SentencePiece language tag
          --packet-samples N      Exercise live PCM streaming with N samples per packet
        """ + "\n"

    static func writeError(_ message: String) {
        FileHandle.standardError.write(Data(message.utf8))
    }

    struct CLIError: LocalizedError {
        let message: String
        init(_ message: String) { self.message = message }
        var errorDescription: String? { message }
    }
}
