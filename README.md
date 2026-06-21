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
        from: "0.2.0"
    )
]
```

Add `NemotronCoreAI` to the application target. The package can detect the
current CoreAI device architecture, fetch only its matching model from Hugging
Face, verify it, and reuse it from the local cache.

## Managed model loading

The common path is one call:

```swift
let session = try await NemotronCoreAI.loadSession(
    computePreference: .gpu,
    downloadProgress: { progress in
        print(progress.phase, progress.fractionCompleted ?? 0)
    }
)
```

The default published model is pinned to a validated Hugging Face revision.
`AIModel.deviceArchitectureName` and the current OS select the exact AOT
artifact. Only `package-manifest.json`, `runtime-support.json`, and that one
`.aimodelc` are downloaded. If no matching AOT artifact exists, the manager
downloads the portable 320 ms source model instead. A runtime AOT load failure
also triggers a verified source fallback by default.

Models are cached under the app's purgeable Caches directory. Prefetch or clear
that cache explicitly with:

```swift
let model = try await NemotronModelManager.shared.prepareModel()
print(model.packageURL, model.cacheHit)

try await NemotronModelManager.shared.removeAllCachedModels()
```

Pass `cacheDirectory` to either API if the app should use an Application
Support location instead. Use `cachePolicy: .reloadIgnoringCache` to refresh a
mutable revision, and `authorizationToken` for a private Hugging Face model.
The downloader reports byte-level progress and validates advertised file sizes,
LFS SHA-256 hashes, runtime metadata, and the CoreAI `main.hash` before making a
staged package visible to the app.

## Live streaming

```swift
import NemotronCoreAI

let session = try await NemotronCoreAI.loadSession(
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
own prefetch lifecycle when they need downloads to continue while suspended.
The managed loader intentionally stores one device-specific artifact rather
than embedding every architecture in the app.

For a fully offline or pre-bundled deployment, pass a package directory
directly:

```swift
let session = try await NemotronCoreAI.loadSession(
    packageURL: modelPackageURL,
    computePreference: .gpu
)
```

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
