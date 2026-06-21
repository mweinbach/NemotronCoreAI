import Foundation

@testable import NemotronCoreAI

func makeRuntimeSupport(
    window: [Float] = [Float](repeating: 1, count: 400),
    filterbankValues: [Float]? = nil
) -> RuntimeSupport {
    var defaultFilterbank = [Float](repeating: 0, count: 128 * 257)
    for row in 0..<128 {
        defaultFilterbank[row * 257] = 1
    }
    return RuntimeSupport(
        formatVersion: 1,
        kind: "streaming_rnnt",
        model: "nvidia/nemotron-3.5-asr-streaming-0.6b",
        sampleRate: 16_000,
        audio: AudioContract(sampleRate: 16_000, channels: 1, wireFormats: ["pcm_s16le", "pcm_f32le"]),
        featureExtractor: FeatureExtractorConfiguration(
            bins: 128,
            windowSamples: 400,
            hopSamples: 160,
            fftSize: 512,
            preemphasis: 0.97,
            hostManagedOverlapFrames: 9,
            logGuard: powf(2, -24),
            centerPaddingSamples: 256,
            center: true,
            padMode: "constant",
            power: 2,
            preemphasisStateDomain: "raw_pcm",
            terminalPaddingDomain: "post_preemphasis"
        ),
        window: RuntimeTensorData(
            dtype: "float32", shape: [400], sha256: "window", values: window
        ),
        filterbank: Filterbank(
            rows: 128,
            columns: 257,
            dtype: "float32",
            shape: [128, 257],
            sha256: "filterbank",
            values: filterbankValues ?? defaultFilterbank
        ),
        tokenizer: TokenizerMetadata(
            type: "sentencepiece",
            pieces: (0..<13_087).map { "piece\($0)" },
            unknownID: 0,
            unknownSurface: " ⁇ ",
            beginningOfSentenceID: 1,
            endOfSentenceID: 2,
            paddingID: 3,
            assetRelativeToReferenceBundle: "tokenizer.model",
            assetSHA256: "tokenizer",
            pieceCount: 13_087
        ),
        decoding: DecodingConfiguration(
            blankTokenID: 13_087,
            vocabularySizeWithBlank: 13_088,
            maxSymbolsPerFrame: 10
        ),
        promptDictionary: ["auto": 0, "en": 1],
        streamingModes: [
            StreamingMode(
                entrypoint: "encode_80ms", latencyMS: 80, rightContextFrames: 0, featureWidth: 17, validOutputFrames: 1),
            StreamingMode(
                entrypoint: "encode_160ms", latencyMS: 160, rightContextFrames: 0, featureWidth: 25,
                validOutputFrames: 2),
            StreamingMode(
                entrypoint: "encode_320ms", latencyMS: 320, rightContextFrames: 0, featureWidth: 41,
                validOutputFrames: 4),
            StreamingMode(
                entrypoint: "encode_560ms", latencyMS: 560, rightContextFrames: 0, featureWidth: 65,
                validOutputFrames: 7),
            StreamingMode(
                entrypoint: "encode_1120ms", latencyMS: 1120, rightContextFrames: 0, featureWidth: 121,
                validOutputFrames: 14),
        ],
        encoderState: EncoderStateShape(
            channelCache: [1, 24, 56, 1024],
            timeCache: [1, 24, 1024, 8],
            cacheLengths: [1]
        ),
        predictorState: PredictorStateShape(hiddenState: [2, 1, 640], cellState: [2, 1, 640]),
        defaultPrompt: "auto",
        streamingContract: StreamingContract(
            featureOverlapFrames: 9,
            terminalZeroSentinelFrames: 1,
            tailMinimumFramesIncludingSentinel: 8,
            modeSwitchRequiresReset: true
        ),
        decoderContract: DecoderContract(
            primeTokenID: 13_087,
            blankAdvancesPredictor: false,
            decodeCompletePrefixForPartials: true
        ),
        functions: FunctionContract(
            encoderByLatencyMS: [
                "80": "encode_80ms",
                "160": "encode_160ms",
                "320": "encode_320ms",
                "560": "encode_560ms",
                "1120": "encode_1120ms",
            ],
            predictor: "predict_step",
            joint: "joint_step",
            fusedPredictorJoint: "predict_joint_step",
            namedIO: NamedIOContract(
                encoderInputs: [
                    "features", "feature_lengths", "channel_cache", "time_cache", "cache_lengths", "prompt_index",
                ],
                encoderOutputs: [
                    "encoded", "encoded_lengths", "next_channel_cache", "next_time_cache", "next_cache_lengths",
                ],
                predictorInputs: ["token", "hidden_state", "cell_state"],
                predictorOutputs: ["prediction", "next_hidden_state", "next_cell_state"],
                jointInputs: ["encoded", "prediction"],
                jointOutputs: ["logits"]
            )
        ),
        deploymentPolicy: DeploymentPolicy(
            defaultVariant: "fp16",
            defaultLatencyMS: 320,
            aot: AOTDeploymentPolicy(
                eligibleVariant: "fp16",
                eligibleLatencyMS: 320,
                fallbackVariant: "fp16_320ms",
                preferredCompute: "gpu"
            )
        )
    )
}

func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("NemotronCoreAITests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
