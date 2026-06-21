import CoreAI
import Foundation

public struct SessionInformation: Sendable, Equatable {
    public let model: String
    public let modelURL: URL
    public let modelKind: ResolvedModelAsset.Kind
    public let sourceVariant: String
    public let architecture: String
    public let platform: NemotronPlatform
    public let computePreference: NemotronComputePreference
    public let latencyMS: Int
    public let encoderFunction: String
    public let usesFusedPredictorJoint: Bool
}

public struct StreamingUpdate: Sendable, Equatable {
    public let text: String
    public let tokenIDs: [Int]
    public let newTokenIDs: [Int]
}

public struct TranscriptionResult: Sendable, Equatable {
    public let text: String
    public let tokenIDs: [Int]
    public let audioSeconds: Double
    public let processedChunks: Int
    public let predictorCalls: Int
    public let jointCalls: Int
    public let fusedPredictorJointCalls: Int
}

public actor NemotronASRSession {
    private let model: AIModel
    private let package: NemotronModelPackage
    private let selectedAsset: ResolvedModelAsset
    private let mode: StreamingMode
    private let encoder: InferenceFunction
    private let predictor: InferenceFunction
    private let joint: InferenceFunction
    private let fusedPredictorJoint: InferenceFunction?
    private let tokenDecoder: SentencePieceDecoder

    private var channelCache: NDArray?
    private var timeCache: NDArray?
    private var cacheLengths: NDArray?
    private var promptIndex: NDArray?
    private var prediction: NDArray?
    private var hiddenState: NDArray?
    private var cellState: NDArray?
    private var tokenIDs: [Int] = []
    private var stripLanguageTag = false
    private var processedChunks = 0
    private var predictorCalls = 0
    private var jointCalls = 0
    private var fusedPredictorJointCalls = 0
    private var operationInFlight = false
    private var pcmStreamingFrontend: PCMStreamingFrontend?

    init(
        model: AIModel,
        package: NemotronModelPackage,
        selectedAsset: ResolvedModelAsset,
        mode: StreamingMode
    ) throws {
        self.model = model
        self.package = package
        self.selectedAsset = selectedAsset
        self.mode = mode
        self.tokenDecoder = SentencePieceDecoder(metadata: package.runtimeSupport.tokenizer)

        let functions = package.runtimeSupport.functions
        guard let encoder = try model.loadFunction(named: mode.entrypoint) else {
            throw NemotronError.missingFunction(mode.entrypoint)
        }
        guard let predictor = try model.loadFunction(named: functions.predictor) else {
            throw NemotronError.missingFunction(functions.predictor)
        }
        guard let joint = try model.loadFunction(named: functions.joint) else {
            throw NemotronError.missingFunction(functions.joint)
        }
        self.encoder = encoder
        self.predictor = predictor
        self.joint = joint

        if let fusedName = functions.fusedPredictorJoint,
            model.functionNames.contains(fusedName)
        {
            guard let fused = try model.loadFunction(named: fusedName) else {
                throw NemotronError.missingFunction(fusedName)
            }
            self.fusedPredictorJoint = fused
        } else {
            self.fusedPredictorJoint = nil
        }
        try Self.validateNamedIO(
            package.runtimeSupport.functions.namedIO,
            encoder: encoder,
            predictor: predictor,
            joint: joint,
            fusedPredictorJoint: self.fusedPredictorJoint
        )
    }

    static func make(
        model: AIModel,
        package: NemotronModelPackage,
        selectedAsset: ResolvedModelAsset,
        mode: StreamingMode,
        targetLanguage: String,
        stripLanguageTag: Bool
    ) async throws -> NemotronASRSession {
        let session = try NemotronASRSession(
            model: model,
            package: package,
            selectedAsset: selectedAsset,
            mode: mode
        )
        try await session.reset(targetLanguage: targetLanguage, stripLanguageTag: stripLanguageTag)
        return session
    }

    public func information() -> SessionInformation {
        SessionInformation(
            model: package.manifest.model,
            modelURL: selectedAsset.url,
            modelKind: selectedAsset.kind,
            sourceVariant: selectedAsset.sourceVariant,
            architecture: selectedAsset.architecture,
            platform: selectedAsset.platform,
            computePreference: selectedAsset.computePreference,
            latencyMS: mode.latencyMS,
            encoderFunction: mode.entrypoint,
            usesFusedPredictorJoint: fusedPredictorJoint != nil
        )
    }

    /// Discards every encoder and predictor state tensor. A session's latency
    /// mode is immutable; create a new session to switch streaming modes.
    public func reset(
        targetLanguage: String = "auto",
        stripLanguageTag: Bool = false
    ) async throws {
        try beginOperation()
        defer { operationInFlight = false }
        try await resetImplementation(
            targetLanguage: targetLanguage,
            stripLanguageTag: stripLanguageTag
        )
    }

    private func resetImplementation(
        targetLanguage: String,
        stripLanguageTag: Bool
    ) async throws {
        let support = package.runtimeSupport
        let prompt = try support.promptIndex(for: targetLanguage)

        let nextChannelCache = try CoreAITensorIO.zeroFloatInput(
            shape: support.encoderState.channelCache,
            function: encoder,
            name: "channel_cache"
        )
        let nextTimeCache = try CoreAITensorIO.zeroFloatInput(
            shape: support.encoderState.timeCache,
            function: encoder,
            name: "time_cache"
        )
        let nextCacheLengths = try CoreAITensorIO.int32Input(
            [0], shape: support.encoderState.cacheLengths, function: encoder, name: "cache_lengths"
        )
        let nextPromptIndex = try CoreAITensorIO.int32Input(
            [Int32(prompt)], shape: [1], function: encoder, name: "prompt_index"
        )

        let initialHidden = try CoreAITensorIO.zeroFloatInput(
            shape: support.predictorState.hiddenState,
            function: predictor,
            name: "hidden_state"
        )
        let initialCell = try CoreAITensorIO.zeroFloatInput(
            shape: support.predictorState.cellState,
            function: predictor,
            name: "cell_state"
        )
        let primeToken = try CoreAITensorIO.int32Input(
            [Int32(support.decoderContract.primeTokenID)],
            shape: [1, 1],
            function: predictor,
            name: "token"
        )
        var primeOutputs: InferenceFunction.Outputs
        do {
            primeOutputs = try await predictor.run(inputs: [
                "token": primeToken,
                "hidden_state": initialHidden,
                "cell_state": initialCell,
            ])
        } catch {
            throw NemotronError.inference("predictor priming failed: \(error)")
        }
        let nextPrediction = try CoreAITensorIO.take(
            &primeOutputs, named: "prediction", function: predictor.descriptor.name)
        let nextHidden = try CoreAITensorIO.take(
            &primeOutputs, named: "next_hidden_state", function: predictor.descriptor.name)
        let nextCell = try CoreAITensorIO.take(
            &primeOutputs, named: "next_cell_state", function: predictor.descriptor.name)

        channelCache = nextChannelCache
        timeCache = nextTimeCache
        cacheLengths = nextCacheLengths
        promptIndex = nextPromptIndex
        prediction = nextPrediction
        hiddenState = nextHidden
        cellState = nextCell
        tokenIDs.removeAll(keepingCapacity: true)
        self.stripLanguageTag = stripLanguageTag
        processedChunks = 0
        predictorCalls = 1
        jointCalls = 0
        fusedPredictorJointCalls = 0
        pcmStreamingFrontend = nil
    }

    /// Starts a live mono PCM stream using this session's fixed latency mode.
    public func beginPCMStream(
        targetLanguage: String = "auto",
        stripLanguageTag: Bool = false
    ) async throws {
        try beginOperation()
        defer { operationInFlight = false }
        try await resetImplementation(
            targetLanguage: targetLanguage,
            stripLanguageTag: stripLanguageTag
        )
        pcmStreamingFrontend = try PCMStreamingFrontend(
            support: package.runtimeSupport,
            latencyMS: mode.latencyMS
        )
    }

    /// Adds live PCM, resamples it to 16 kHz, and advances CoreAI for every
    /// newly stable feature chunk.
    /// The returned text is decoded from the complete token prefix.
    public func pushPCM(_ samples: [Float], sampleRate: Int = 16_000) async throws -> StreamingUpdate {
        try beginOperation()
        defer { operationInFlight = false }
        guard var frontend = pcmStreamingFrontend else {
            throw NemotronError.invalidAudio("call beginPCMStream() before pushPCM()")
        }
        let firstNewToken = tokenIDs.count
        let chunks = try frontend.push(samples, sampleRate: sampleRate)
        pcmStreamingFrontend = frontend
        do {
            for chunk in chunks {
                _ = try await processFeatureChunkImplementation(chunk)
            }
        } catch {
            // The frontend may already have advanced beyond the failed chunk;
            // require an explicit restart instead of risking state divergence.
            pcmStreamingFrontend = nil
            throw error
        }
        return StreamingUpdate(
            text: decodedText(),
            tokenIDs: tokenIDs,
            newTokenIDs: Array(tokenIDs[firstNewToken...])
        )
    }

    /// Finalizes a live PCM stream, including right padding and the terminal
    /// zero sentinel, and returns the final complete-prefix transcription.
    public func finishPCMStream() async throws -> StreamingUpdate {
        try beginOperation()
        defer { operationInFlight = false }
        guard var frontend = pcmStreamingFrontend else {
            throw NemotronError.invalidAudio("call beginPCMStream() before finishPCMStream()")
        }
        let firstNewToken = tokenIDs.count
        let chunks = try frontend.finish()
        pcmStreamingFrontend = frontend
        do {
            for chunk in chunks {
                _ = try await processFeatureChunkImplementation(chunk)
            }
        } catch {
            pcmStreamingFrontend = nil
            throw error
        }
        pcmStreamingFrontend = nil
        return StreamingUpdate(
            text: decodedText(),
            tokenIDs: tokenIDs,
            newTokenIDs: Array(tokenIDs[firstNewToken...])
        )
    }

    /// Advances the retained encoder caches and RNNT state by one fixed-width
    /// feature chunk. Call `reset` before starting a logically new stream.
    public func processFeatureChunk(_ chunk: FeatureChunk) async throws -> StreamingUpdate {
        try beginOperation()
        defer { operationInFlight = false }
        guard pcmStreamingFrontend == nil else {
            throw NemotronError.invalidFeatureChunk(
                "finish or reset the active PCM stream before submitting feature chunks directly"
            )
        }
        return try await processFeatureChunkImplementation(chunk)
    }

    private func processFeatureChunkImplementation(_ chunk: FeatureChunk) async throws -> StreamingUpdate {
        let support = package.runtimeSupport
        let overlap = support.streamingContract.featureOverlapFrames
        let minimum = support.streamingContract.tailMinimumFramesIncludingSentinel
        guard chunk.featureWidth == mode.featureWidth else {
            throw NemotronError.invalidFeatureChunk(
                "session expects width \(mode.featureWidth), received \(chunk.featureWidth)"
            )
        }
        guard chunk.values.count == support.featureExtractor.bins * mode.featureWidth else {
            throw NemotronError.invalidFeatureChunk("values must contain 128x\(mode.featureWidth) elements")
        }
        guard chunk.values.allSatisfy(\.isFinite) else {
            throw NemotronError.invalidFeatureChunk("values contain a non-finite feature")
        }
        guard (overlap + minimum)...mode.featureWidth ~= chunk.validLength else {
            throw NemotronError.invalidFeatureChunk(
                "validLength must be in \(overlap + minimum)...\(mode.featureWidth), received \(chunk.validLength)"
            )
        }
        guard let currentChannelCache = channelCache,
            let currentTimeCache = timeCache,
            let currentCacheLengths = cacheLengths,
            let currentPromptIndex = promptIndex,
            let currentPrediction = prediction,
            let currentHiddenState = hiddenState,
            let currentCellState = cellState
        else {
            throw NemotronError.inference("session has not been initialized; call reset()")
        }

        let features = try CoreAITensorIO.floatInput(
            chunk.values,
            shape: [1, support.featureExtractor.bins, mode.featureWidth],
            function: encoder,
            name: "features"
        )
        let featureLengths = try CoreAITensorIO.int32Input(
            [Int32(chunk.validLength)], shape: [1], function: encoder, name: "feature_lengths"
        )
        var encoderOutputs: InferenceFunction.Outputs
        do {
            encoderOutputs = try await encoder.run(inputs: [
                "features": features,
                "feature_lengths": featureLengths,
                "channel_cache": currentChannelCache,
                "time_cache": currentTimeCache,
                "cache_lengths": currentCacheLengths,
                "prompt_index": currentPromptIndex,
            ])
        } catch {
            throw NemotronError.inference("\(mode.entrypoint) failed: \(error)")
        }

        let encodedArray = try CoreAITensorIO.take(&encoderOutputs, named: "encoded", function: encoder.descriptor.name)
        let encodedLengths = try CoreAITensorIO.take(
            &encoderOutputs, named: "encoded_lengths", function: encoder.descriptor.name)
        let nextChannelCache = try CoreAITensorIO.take(
            &encoderOutputs, named: "next_channel_cache", function: encoder.descriptor.name)
        let nextTimeCache = try CoreAITensorIO.take(
            &encoderOutputs, named: "next_time_cache", function: encoder.descriptor.name)
        let nextCacheLengths = try CoreAITensorIO.take(
            &encoderOutputs, named: "next_cache_lengths", function: encoder.descriptor.name)

        guard encodedArray.shape.count == 3,
            encodedArray.shape[0] == 1,
            encodedArray.shape[1] == 1024
        else {
            throw NemotronError.inference("encoder output shape is \(encodedArray.shape), expected [1, 1024, T]")
        }
        let outputFrameCapacity = encodedArray.shape[2]
        let encodedLength = try CoreAITensorIO.firstInteger(encodedLengths)
        guard (0...outputFrameCapacity).contains(encodedLength) else {
            throw NemotronError.inference(
                "encoder reported \(encodedLength) frames for capacity \(outputFrameCapacity)"
            )
        }
        let encodedValues = try CoreAITensorIO.flattenFloat(encodedArray)

        var nextPrediction = currentPrediction
        var nextHiddenState = currentHiddenState
        var nextCellState = currentCellState
        var nextTokenIDs = tokenIDs
        let firstNewToken = nextTokenIDs.count
        var nextPredictorCalls = predictorCalls
        var nextJointCalls = jointCalls
        var nextFusedCalls = fusedPredictorJointCalls

        for frameIndex in 0..<encodedLength {
            var frameValues = [Float](repeating: 0, count: 1024)
            for channel in 0..<1024 {
                frameValues[channel] = encodedValues[channel * outputFrameCapacity + frameIndex]
            }
            let jointFrame = try CoreAITensorIO.floatInput(
                frameValues, shape: [1, 1024, 1], function: joint, name: "encoded"
            )
            var jointOutputs: InferenceFunction.Outputs
            do {
                jointOutputs = try await joint.run(inputs: [
                    "encoded": jointFrame,
                    "prediction": nextPrediction,
                ])
            } catch {
                throw NemotronError.inference("joint_step failed: \(error)")
            }
            nextJointCalls += 1
            let initialLogits = try CoreAITensorIO.take(&jointOutputs, named: "logits", function: joint.descriptor.name)
            var token = try CoreAITensorIO.argmax(initialLogits)

            for symbolIndex in 0..<support.decoding.maxSymbolsPerFrame {
                if token == support.decoding.blankTokenID { break }
                guard token >= 0, token < support.decoding.blankTokenID else {
                    throw NemotronError.inference("RNNT emitted out-of-vocabulary token \(token)")
                }
                nextTokenIDs.append(token)
                let isLastSymbol = symbolIndex == support.decoding.maxSymbolsPerFrame - 1

                if let fusedPredictorJoint, !isLastSymbol {
                    let fusedToken = try CoreAITensorIO.int32Input(
                        [Int32(token)], shape: [1, 1], function: fusedPredictorJoint, name: "token"
                    )
                    let fusedFrame = try CoreAITensorIO.floatInput(
                        frameValues, shape: [1, 1024, 1], function: fusedPredictorJoint, name: "encoded"
                    )
                    var fusedOutputs: InferenceFunction.Outputs
                    do {
                        fusedOutputs = try await fusedPredictorJoint.run(inputs: [
                            "token": fusedToken,
                            "hidden_state": nextHiddenState,
                            "cell_state": nextCellState,
                            "encoded": fusedFrame,
                        ])
                    } catch {
                        throw NemotronError.inference("predict_joint_step failed: \(error)")
                    }
                    nextFusedCalls += 1
                    let logits = try CoreAITensorIO.take(
                        &fusedOutputs, named: "logits", function: fusedPredictorJoint.descriptor.name)
                    nextPrediction = try CoreAITensorIO.take(
                        &fusedOutputs, named: "prediction", function: fusedPredictorJoint.descriptor.name)
                    nextHiddenState = try CoreAITensorIO.take(
                        &fusedOutputs, named: "next_hidden_state", function: fusedPredictorJoint.descriptor.name)
                    nextCellState = try CoreAITensorIO.take(
                        &fusedOutputs, named: "next_cell_state", function: fusedPredictorJoint.descriptor.name)
                    token = try CoreAITensorIO.argmax(logits)
                    continue
                }

                let predictorToken = try CoreAITensorIO.int32Input(
                    [Int32(token)], shape: [1, 1], function: predictor, name: "token"
                )
                var predictorOutputs: InferenceFunction.Outputs
                do {
                    predictorOutputs = try await predictor.run(inputs: [
                        "token": predictorToken,
                        "hidden_state": nextHiddenState,
                        "cell_state": nextCellState,
                    ])
                } catch {
                    throw NemotronError.inference("predict_step failed: \(error)")
                }
                nextPredictorCalls += 1
                nextPrediction = try CoreAITensorIO.take(
                    &predictorOutputs, named: "prediction", function: predictor.descriptor.name)
                nextHiddenState = try CoreAITensorIO.take(
                    &predictorOutputs, named: "next_hidden_state", function: predictor.descriptor.name)
                nextCellState = try CoreAITensorIO.take(
                    &predictorOutputs, named: "next_cell_state", function: predictor.descriptor.name)

                if !isLastSymbol {
                    var repeatJointOutputs: InferenceFunction.Outputs
                    do {
                        repeatJointOutputs = try await joint.run(inputs: [
                            "encoded": jointFrame,
                            "prediction": nextPrediction,
                        ])
                    } catch {
                        throw NemotronError.inference("joint_step failed: \(error)")
                    }
                    nextJointCalls += 1
                    let logits = try CoreAITensorIO.take(
                        &repeatJointOutputs, named: "logits", function: joint.descriptor.name
                    )
                    token = try CoreAITensorIO.argmax(logits)
                }
            }
        }

        // Commit the encoder and RNNT state atomically only after the complete
        // chunk has decoded successfully.
        channelCache = nextChannelCache
        timeCache = nextTimeCache
        cacheLengths = nextCacheLengths
        prediction = nextPrediction
        hiddenState = nextHiddenState
        cellState = nextCellState
        tokenIDs = nextTokenIDs
        predictorCalls = nextPredictorCalls
        jointCalls = nextJointCalls
        fusedPredictorJointCalls = nextFusedCalls
        processedChunks += 1

        return StreamingUpdate(
            text: decodedText(),
            tokenIDs: tokenIDs,
            newTokenIDs: Array(tokenIDs[firstNewToken...])
        )
    }

    public func transcribe(
        pcm: [Float],
        sampleRate: Int = 16_000,
        targetLanguage: String = "auto",
        stripLanguageTag: Bool = false
    ) async throws -> TranscriptionResult {
        try beginOperation()
        defer { operationInFlight = false }
        return try await transcribeImplementation(
            pcm: pcm,
            sampleRate: sampleRate,
            targetLanguage: targetLanguage,
            stripLanguageTag: stripLanguageTag
        )
    }

    private func transcribeImplementation(
        pcm: [Float],
        sampleRate: Int,
        targetLanguage: String,
        stripLanguageTag: Bool
    ) async throws -> TranscriptionResult {
        try await resetImplementation(targetLanguage: targetLanguage, stripLanguageTag: stripLanguageTag)
        let features = try LogMelFrontend(support: package.runtimeSupport).extract(
            pcm: pcm,
            sampleRate: sampleRate
        )
        let chunks = try LogMelFrontend(support: package.runtimeSupport).chunks(from: features, mode: mode)
        for chunk in chunks {
            _ = try await processFeatureChunkImplementation(chunk)
        }
        return TranscriptionResult(
            text: decodedText(),
            tokenIDs: tokenIDs,
            audioSeconds: Double(pcm.count) / Double(sampleRate),
            processedChunks: processedChunks,
            predictorCalls: predictorCalls,
            jointCalls: jointCalls,
            fusedPredictorJointCalls: fusedPredictorJointCalls
        )
    }

    public func transcribe(
        fileURL: URL,
        targetLanguage: String = "auto",
        stripLanguageTag: Bool = false
    ) async throws -> TranscriptionResult {
        try beginOperation()
        defer { operationInFlight = false }
        let pcm = try AudioFileLoader.load16kHzMono(from: fileURL)
        return try await transcribeImplementation(
            pcm: pcm,
            sampleRate: package.runtimeSupport.sampleRate,
            targetLanguage: targetLanguage,
            stripLanguageTag: stripLanguageTag
        )
    }

    private func decodedText() -> String {
        // SentencePiece decode is not token-wise compositional. Decode the
        // complete prefix for every partial result to preserve exact spacing.
        let text = tokenDecoder.decode(tokenIDs)
        guard stripLanguageTag else { return text }
        return text.replacingOccurrences(
            of: #"\s*<[a-z]{2}(?:-[A-Z]{2})?>\s*$"#,
            with: "",
            options: .regularExpression
        )
    }

    private static func validateNamedIO(
        _ namedIO: NamedIOContract,
        encoder: InferenceFunction,
        predictor: InferenceFunction,
        joint: InferenceFunction,
        fusedPredictorJoint: InferenceFunction?
    ) throws {
        let contracts: [(String, [String], [String], InferenceFunction)] = [
            ("encoder", namedIO.encoderInputs, namedIO.encoderOutputs, encoder),
            ("predictor", namedIO.predictorInputs, namedIO.predictorOutputs, predictor),
            ("joint", namedIO.jointInputs, namedIO.jointOutputs, joint),
        ]
        for (label, inputs, outputs, function) in contracts {
            guard Set(function.descriptor.inputNames) == Set(inputs),
                Set(function.descriptor.outputNames) == Set(outputs)
            else {
                throw NemotronError.inference(
                    "\(label) named-I/O contract does not match \(function.descriptor.name)"
                )
            }
        }
        if let fusedPredictorJoint {
            guard
                Set(fusedPredictorJoint.descriptor.inputNames)
                    == Set(["token", "hidden_state", "cell_state", "encoded"]),
                Set(fusedPredictorJoint.descriptor.outputNames)
                    == Set(["logits", "prediction", "next_hidden_state", "next_cell_state"])
            else {
                throw NemotronError.inference(
                    "fused predictor/joint named-I/O contract does not match \(fusedPredictorJoint.descriptor.name)"
                )
            }
        }
    }

    private func beginOperation() throws {
        guard !operationInFlight else { throw NemotronError.concurrentOperation }
        operationInFlight = true
    }
}
