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
}
