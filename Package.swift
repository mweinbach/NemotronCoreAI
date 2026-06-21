// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "NemotronCoreAI",
    platforms: [
        .iOS("27.0"),
        .macOS("27.0"),
    ],
    products: [
        .library(name: "NemotronCoreAI", targets: ["NemotronCoreAI"]),
        .executable(name: "nemotron-coreai", targets: ["NemotronCoreAICLI"]),
    ],
    targets: [
        .target(
            name: "NemotronCoreAI",
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("CoreAI"),
            ]
        ),
        .executableTarget(
            name: "NemotronCoreAICLI",
            dependencies: ["NemotronCoreAI"]
        ),
        .testTarget(
            name: "NemotronCoreAITests",
            dependencies: ["NemotronCoreAI"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
