import CoreAI
import XCTest

@testable import NemotronCoreAI

final class CoreAIPlatformPolicyTests: XCTestCase {
    func testAutomaticUsesDefaultSpecialization() {
        XCTAssertEqual(
            CoreAIPlatformPolicy.specializationOptions(for: .automatic),
            SpecializationOptions.default
        )
    }

    func testExplicitComputePreferencesMapToCoreAI() {
        XCTAssertEqual(
            CoreAIPlatformPolicy.specializationOptions(for: .gpu).preferredComputeUnitKind,
            .gpu
        )
        XCTAssertEqual(
            CoreAIPlatformPolicy.specializationOptions(for: .neuralEngine).preferredComputeUnitKind,
            .neuralEngine
        )
    }

    func testCurrentDeviceTargetUsesCoreAIRuntimeIdentity() {
        let target = NemotronDeviceTarget.current
        #if os(macOS)
        XCTAssertEqual(target.platform, .macOS)
        #else
        XCTAssertEqual(target.platform, .iOS)
        #endif
        XCTAssertEqual(target.architecture, AIModel.deviceArchitectureName)
        XCTAssertFalse(target.architecture.isEmpty)
    }
}
