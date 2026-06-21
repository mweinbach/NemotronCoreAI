import Foundation

public struct FeatureExtractorConfiguration: Codable, Sendable, Equatable {
    public let bins: Int
    public let windowSamples: Int
    public let hopSamples: Int
    public let fftSize: Int
    public let preemphasis: Float
    public let hostManagedOverlapFrames: Int
    public let logGuard: Float
    public let centerPaddingSamples: Int
    public let center: Bool
    public let padMode: String
    public let power: Int
    public let preemphasisStateDomain: String
    public let terminalPaddingDomain: String

    enum CodingKeys: String, CodingKey {
        case bins
        case windowSamples = "window_samples"
        case hopSamples = "hop_samples"
        case fftSize = "fft_size"
        case preemphasis
        case hostManagedOverlapFrames = "host_managed_overlap_frames"
        case logGuard = "log_guard"
        case centerPaddingSamples = "center_padding_samples"
        case center
        case padMode = "pad_mode"
        case power
        case preemphasisStateDomain = "preemphasis_state_domain"
        case terminalPaddingDomain = "terminal_padding_domain"
    }
}

public struct RuntimeTensorData: Codable, Sendable, Equatable {
    public let dtype: String
    public let shape: [Int]
    public let sha256: String
    public let values: [Float]
}

public struct Filterbank: Codable, Sendable, Equatable {
    public let rows: Int
    public let columns: Int
    public let dtype: String
    public let shape: [Int]
    public let sha256: String
    public let values: [Float]
}

public struct AudioContract: Codable, Sendable, Equatable {
    public let sampleRate: Int
    public let channels: Int
    public let wireFormats: [String]

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case channels
        case wireFormats = "wire_formats"
    }
}

public struct StreamingContract: Codable, Sendable, Equatable {
    public let featureOverlapFrames: Int
    public let terminalZeroSentinelFrames: Int
    public let tailMinimumFramesIncludingSentinel: Int
    public let modeSwitchRequiresReset: Bool

    enum CodingKeys: String, CodingKey {
        case featureOverlapFrames = "feature_overlap_frames"
        case terminalZeroSentinelFrames = "terminal_zero_sentinel_frames"
        case tailMinimumFramesIncludingSentinel = "tail_minimum_frames_including_sentinel"
        case modeSwitchRequiresReset = "mode_switch_requires_reset"
    }
}

public struct DecoderContract: Codable, Sendable, Equatable {
    public let primeTokenID: Int
    public let blankAdvancesPredictor: Bool
    public let decodeCompletePrefixForPartials: Bool

    enum CodingKeys: String, CodingKey {
        case primeTokenID = "prime_token_id"
        case blankAdvancesPredictor = "blank_advances_predictor"
        case decodeCompletePrefixForPartials = "decode_complete_prefix_for_partials"
    }
}

public struct NamedIOContract: Codable, Sendable, Equatable {
    public let encoderInputs: [String]
    public let encoderOutputs: [String]
    public let predictorInputs: [String]
    public let predictorOutputs: [String]
    public let jointInputs: [String]
    public let jointOutputs: [String]

    enum CodingKeys: String, CodingKey {
        case encoderInputs = "encoder_inputs"
        case encoderOutputs = "encoder_outputs"
        case predictorInputs = "predictor_inputs"
        case predictorOutputs = "predictor_outputs"
        case jointInputs = "joint_inputs"
        case jointOutputs = "joint_outputs"
    }
}

public struct FunctionContract: Codable, Sendable, Equatable {
    public let encoderByLatencyMS: [String: String]
    public let predictor: String
    public let joint: String
    public let fusedPredictorJoint: String?
    public let namedIO: NamedIOContract

    enum CodingKeys: String, CodingKey {
        case encoderByLatencyMS = "encoder_by_latency_ms"
        case predictor
        case joint
        case fusedPredictorJoint = "fused_predictor_joint"
        case namedIO = "named_io"
    }
}

public struct AOTDeploymentPolicy: Codable, Sendable, Equatable {
    public let eligibleVariant: String
    public let eligibleLatencyMS: Int
    public let fallbackVariant: String
    public let preferredCompute: String

    enum CodingKeys: String, CodingKey {
        case eligibleVariant = "eligible_variant"
        case eligibleLatencyMS = "eligible_latency_ms"
        case fallbackVariant = "fallback_variant"
        case preferredCompute = "preferred_compute"
    }
}

public struct DeploymentPolicy: Codable, Sendable, Equatable {
    public let defaultVariant: String
    public let defaultLatencyMS: Int
    public let aot: AOTDeploymentPolicy

    enum CodingKeys: String, CodingKey {
        case defaultVariant = "default_variant"
        case defaultLatencyMS = "default_latency_ms"
        case aot
    }
}

public struct TokenizerMetadata: Codable, Sendable, Equatable {
    public let type: String
    public let pieces: [String]
    public let unknownID: Int
    public let unknownSurface: String
    public let beginningOfSentenceID: Int
    public let endOfSentenceID: Int
    public let paddingID: Int
    public let assetRelativeToReferenceBundle: String
    public let assetSHA256: String
    public let pieceCount: Int

    enum CodingKeys: String, CodingKey {
        case type
        case pieces
        case unknownID = "unknown_id"
        case unknownSurface = "unknown_surface"
        case beginningOfSentenceID = "beginning_of_sentence_id"
        case endOfSentenceID = "end_of_sentence_id"
        case paddingID = "padding_id"
        case assetRelativeToReferenceBundle = "asset_relative_to_reference_bundle"
        case assetSHA256 = "asset_sha256"
        case pieceCount = "piece_count"
    }
}

public struct DecodingConfiguration: Codable, Sendable, Equatable {
    public let blankTokenID: Int
    public let vocabularySizeWithBlank: Int
    public let maxSymbolsPerFrame: Int

    enum CodingKeys: String, CodingKey {
        case blankTokenID = "blank_token_id"
        case vocabularySizeWithBlank = "vocabulary_size_with_blank"
        case maxSymbolsPerFrame = "max_symbols_per_frame"
    }
}

public struct StreamingMode: Codable, Sendable, Equatable {
    public let entrypoint: String
    public let latencyMS: Int
    public let rightContextFrames: Int
    public let featureWidth: Int
    public let validOutputFrames: Int

    enum CodingKeys: String, CodingKey {
        case entrypoint
        case latencyMS = "latency_ms"
        case rightContextFrames = "right_context_frames"
        case featureWidth = "feature_width"
        case validOutputFrames = "valid_output_frames"
    }
}

public struct EncoderStateShape: Codable, Sendable, Equatable {
    public let channelCache: [Int]
    public let timeCache: [Int]
    public let cacheLengths: [Int]

    enum CodingKeys: String, CodingKey {
        case channelCache = "channel_cache"
        case timeCache = "time_cache"
        case cacheLengths = "cache_lengths"
    }
}

public struct PredictorStateShape: Codable, Sendable, Equatable {
    public let hiddenState: [Int]
    public let cellState: [Int]

    enum CodingKeys: String, CodingKey {
        case hiddenState = "hidden_state"
        case cellState = "cell_state"
    }
}

public struct RuntimeSupport: Codable, Sendable, Equatable {
    public let formatVersion: Int
    public let kind: String
    public let model: String
    public let sampleRate: Int
    public let audio: AudioContract
    public let featureExtractor: FeatureExtractorConfiguration
    public let window: RuntimeTensorData
    public let filterbank: Filterbank
    public let tokenizer: TokenizerMetadata
    public let decoding: DecodingConfiguration
    public let promptDictionary: [String: Int]
    public let streamingModes: [StreamingMode]
    public let encoderState: EncoderStateShape
    public let predictorState: PredictorStateShape
    public let defaultPrompt: String
    public let streamingContract: StreamingContract
    public let decoderContract: DecoderContract
    public let functions: FunctionContract
    public let deploymentPolicy: DeploymentPolicy

    enum CodingKeys: String, CodingKey {
        case formatVersion = "format_version"
        case kind
        case model
        case sampleRate = "sample_rate"
        case audio
        case featureExtractor = "feature_extractor"
        case window
        case filterbank
        case tokenizer
        case decoding
        case promptDictionary = "prompt_dictionary"
        case streamingModes = "streaming_modes"
        case encoderState = "encoder_state"
        case predictorState = "predictor_state"
        case defaultPrompt = "default_prompt"
        case streamingContract = "streaming_contract"
        case decoderContract = "decoder_contract"
        case functions
        case deploymentPolicy = "deployment_policy"
    }

    public static func load(from url: URL) throws -> Self {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw NemotronError.invalidRuntimeSupport("could not read \(url.path): \(error)")
        }
        return try decode(data, source: url.path)
    }

    static func decode(_ data: Data, source: String) throws -> Self {
        let support: Self
        do {
            support = try JSONDecoder().decode(Self.self, from: data)
        } catch {
            throw NemotronError.invalidRuntimeSupport("could not decode \(source): \(error)")
        }
        try support.validate()
        return support
    }

    public func streamingMode(latencyMS: Int) throws -> StreamingMode {
        guard let mode = streamingModes.first(where: { $0.latencyMS == latencyMS }) else {
            throw NemotronError.invalidRuntimeSupport(
                "unsupported latency \(latencyMS) ms; available modes: \(streamingModes.map(\.latencyMS).sorted())"
            )
        }
        return mode
    }

    public func promptIndex(for language: String) throws -> Int {
        guard let value = promptDictionary[language] else {
            throw NemotronError.invalidRuntimeSupport(
                "unknown target language '\(language)'; available values: \(promptDictionary.keys.sorted().joined(separator: ", "))"
            )
        }
        return value
    }

    public func validate() throws {
        guard formatVersion == 1 else {
            throw NemotronError.invalidRuntimeSupport("unsupported format_version \(formatVersion)")
        }
        guard sampleRate == 16_000 else {
            throw NemotronError.invalidRuntimeSupport("sample_rate must be 16000, found \(sampleRate)")
        }
        guard kind == "streaming_rnnt",
            audio.sampleRate == sampleRate,
            audio.channels == 1
        else {
            throw NemotronError.invalidRuntimeSupport("audio contract is not 16 kHz mono streaming RNNT")
        }
        let feature = featureExtractor
        guard feature.bins == 128,
            feature.windowSamples == 400,
            feature.hopSamples == 160,
            feature.fftSize == 512,
            feature.centerPaddingSamples == 256,
            feature.hostManagedOverlapFrames == 9,
            feature.center,
            feature.padMode == "constant",
            feature.power == 2,
            feature.preemphasisStateDomain == "raw_pcm",
            feature.terminalPaddingDomain == "post_preemphasis"
        else {
            throw NemotronError.invalidRuntimeSupport("feature extractor geometry does not match the exported model")
        }
        guard window.dtype == "float32",
            window.shape == [feature.windowSamples],
            window.values.count == feature.windowSamples
        else {
            throw NemotronError.invalidRuntimeSupport("window is not a 400-element float32 tensor")
        }
        guard filterbank.rows == feature.bins,
            filterbank.columns == feature.fftSize / 2 + 1,
            filterbank.dtype == "float32",
            filterbank.shape == [filterbank.rows, filterbank.columns],
            filterbank.values.count == filterbank.rows * filterbank.columns
        else {
            throw NemotronError.invalidRuntimeSupport("filterbank is not 128x257 row-major data")
        }
        guard tokenizer.type == "sentencepiece" else {
            throw NemotronError.invalidRuntimeSupport("tokenizer type must be sentencepiece")
        }
        guard tokenizer.assetRelativeToReferenceBundle == "tokenizer.model",
            tokenizer.pieceCount == tokenizer.pieces.count,
            tokenizer.pieces.count + 1 == decoding.vocabularySizeWithBlank,
            decoding.blankTokenID == tokenizer.pieces.count,
            decoding.maxSymbolsPerFrame > 0
        else {
            throw NemotronError.invalidRuntimeSupport("tokenizer and RNNT vocabulary metadata disagree")
        }
        guard !streamingModes.isEmpty,
            Set(streamingModes.map(\.latencyMS)).count == streamingModes.count
        else {
            throw NemotronError.invalidRuntimeSupport("streaming modes are empty or contain duplicate latencies")
        }
        guard streamingContract.featureOverlapFrames == feature.hostManagedOverlapFrames,
            streamingContract.terminalZeroSentinelFrames == 1,
            streamingContract.tailMinimumFramesIncludingSentinel == 8,
            streamingContract.modeSwitchRequiresReset
        else {
            throw NemotronError.invalidRuntimeSupport("streaming contract does not match the host chunker")
        }
        guard decoderContract.primeTokenID == decoding.blankTokenID,
            !decoderContract.blankAdvancesPredictor,
            decoderContract.decodeCompletePrefixForPartials
        else {
            throw NemotronError.invalidRuntimeSupport("decoder contract does not match greedy RNNT semantics")
        }
        guard promptDictionary[defaultPrompt] != nil else {
            throw NemotronError.invalidRuntimeSupport("default prompt '\(defaultPrompt)' is absent")
        }
        let namedIO = functions.namedIO
        guard
            namedIO.encoderInputs
                == ["features", "feature_lengths", "channel_cache", "time_cache", "cache_lengths", "prompt_index"],
            namedIO.encoderOutputs
                == ["encoded", "encoded_lengths", "next_channel_cache", "next_time_cache", "next_cache_lengths"],
            namedIO.predictorInputs == ["token", "hidden_state", "cell_state"],
            namedIO.predictorOutputs == ["prediction", "next_hidden_state", "next_cell_state"],
            namedIO.jointInputs == ["encoded", "prediction"],
            namedIO.jointOutputs == ["logits"]
        else {
            throw NemotronError.invalidRuntimeSupport("named-I/O contract is not supported by this runtime")
        }
        guard functions.encoderByLatencyMS.count == streamingModes.count,
            streamingModes.allSatisfy({ functions.encoderByLatencyMS[String($0.latencyMS)] == $0.entrypoint }),
            functions.predictor == "predict_step",
            functions.joint == "joint_step"
        else {
            throw NemotronError.invalidRuntimeSupport("function map disagrees with streaming modes")
        }
        guard deploymentPolicy.aot.eligibleVariant == "fp16",
            deploymentPolicy.aot.eligibleLatencyMS == 320,
            deploymentPolicy.aot.fallbackVariant == "fp16_320ms",
            deploymentPolicy.aot.preferredCompute == "gpu"
        else {
            throw NemotronError.invalidRuntimeSupport("deployment policy is not the supported GPU AOT policy")
        }
        for mode in streamingModes {
            guard mode.featureWidth > feature.hostManagedOverlapFrames,
                mode.validOutputFrames > 0,
                !mode.entrypoint.isEmpty
            else {
                throw NemotronError.invalidRuntimeSupport("invalid streaming mode for \(mode.latencyMS) ms")
            }
        }
        guard encoderState.channelCache == [1, 24, 56, 1024],
            encoderState.timeCache == [1, 24, 1024, 8],
            encoderState.cacheLengths == [1],
            predictorState.hiddenState == [2, 1, 640],
            predictorState.cellState == [2, 1, 640]
        else {
            throw NemotronError.invalidRuntimeSupport("state tensor shapes do not match Nemotron 0.6B")
        }
    }
}
