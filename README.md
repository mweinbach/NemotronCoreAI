# NemotronCoreAI

`NemotronCoreAI` is a Swift 6 package for real-time, on-device transcription
with NVIDIA's `nemotron-3.5-asr-streaming-0.6b` converted to CoreAI. It supports
iOS 27+ and macOS 27+.

The runtime implements the complete streaming path: incremental 16 kHz PCM,
NeMo-compatible log-mel features, cache-aware FastConformer state, greedy RNNT
decoding, predictor hidden/cell state, and SentencePiece detokenization. Stream
state lives inside an actor and is retained across packets until `finish` or
`reset`.

Model artifacts are hosted separately at
[`mweinbach/nemotron-3.5-asr-streaming-0.6b-coreai`](https://huggingface.co/mweinbach/nemotron-3.5-asr-streaming-0.6b-coreai)
under the model's OpenMDW-1.1 license.

## Add the package

```swift
dependencies: [
    .package(
        url: "https://github.com/mweinbach/NemotronCoreAI.git",
        from: "0.1.0"
    )
]
```

Add `NemotronCoreAI` to the application target, then provide a local model
package directory containing `package-manifest.json`, `runtime-support.json`,
and either a matching `.aimodelc` or source `.aimodel`.

## Live streaming

```swift
import NemotronCoreAI

let session = try await NemotronCoreAI.loadSession(
    packageURL: modelPackageURL,
    latencyMS: 320,
    computePreference: .gpu
)

try await session.beginPCMStream(targetLanguage: "auto")

// Push arbitrary packet sizes. Audio must be mono Float PCM; pass the actual
// input sample rate and the runtime performs packet-stable resampling to 16 kHz.
let partial = try await session.pushPCM(samples, sampleRate: 48_000)
print(partial.text)

let final = try await session.finishPCMStream()
print(final.text)
```

For file transcription:

```swift
let result = try await session.transcribe(fileURL: audioURL)
print(result.text)
```

Supported streaming encoder modes are 80, 160, 320, 560, and 1120 ms. The
published AOT artifacts specialize the balanced 320 ms path; source assets
retain all five modes.

## Compute selection

```swift
// CoreAI chooses among the compute units supported by the source graph.
computePreference: .automatic

// Validated low-latency path and the published AOT matrix.
computePreference: .gpu

// Reserved for a validated ANE-authored artifact.
computePreference: .neuralEngine
```

The current FastConformer source and GPU AOT artifacts are validated with
`.automatic` and `.gpu`. Pure Neural Engine specialization is deliberately
guarded: the Xcode 27 beta ANE compiler rejects the current graph's rank-6
transpose generated around the cache-aware encoder. `neuralEngine` therefore
requires a future device-matched ANE artifact and throws a recoverable error
when none exists, instead of allowing CoreAI to abort the process. The public
API and manifest already reserve that compute lane.

## Published AOT targets

| Platform | CoreAI architecture | Hardware |
| --- | --- | --- |
| macOS | `h15d` | M3 Ultra |
| macOS | `h16g` | M4 |
| macOS | `h17g` | M5 |
| macOS | `h17s` | M5 Pro |
| macOS | `h17c` | M5 Max |
| iOS | `h18p` | A19 / A19 Pro family |

Artifacts are platform-specific even when an architecture code appears on
multiple Apple platforms. Selection uses `AIModel.deviceArchitectureName`, the
current OS, and the requested compute preference. Missing or unknown GPU AOT
assets fall back to a portable source model when one is present.

For an AOT-only download on this M3 Ultra:

```sh
hf download mweinbach/nemotron-3.5-asr-streaming-0.6b-coreai \
  --local-dir ./NemotronModel \
  --include package-manifest.json \
  --include runtime-support.json \
  --include 'aot/macos/gpu/*h15d.aimodelc/**'
```

Use `h18p` and `aot/ios/gpu/` for iPhone 17, iPhone Air, and iPhone 17 Pro.
Applications shipping large artifacts should use Background Assets or their
own download layer rather than embedding every architecture in the app.

## CLI

```sh
swift run nemotron-coreai inspect /path/to/model-package --compute gpu
swift run nemotron-coreai transcribe /path/to/model-package audio.flac \
  --compute gpu --packet-samples 1024
```

## Verification

```sh
xcrun swift-format lint --strict --recursive Sources Tests
swift test
xcodebuild build \
  -scheme NemotronCoreAI \
  -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO
```

GitHub Actions runs package and formatting checks on `macos-26`. Full tests and
the generic iOS build run automatically as soon as the hosted image exposes the
Xcode 27 CoreAI SDK; local Xcode 27 validation remains the release gate until
then.

The Swift source is MIT-licensed. Model weights and compiled artifacts remain
subject to OpenMDW-1.1; see the model repository for the retained license and
NVIDIA notices.
