import XCTest
@testable import Raphael

final class ServerConfigurationTests: XCTestCase {
    func testHTTPSOriginAndMatchingBoundHostAreRequired() {
        XCTAssertNotNil(ServerConfiguration(urlString: "https://raphael.example.org", boundHost: "raphael.example.org"))
        for value in ["http://raphael.example.org", "https://user:password@raphael.example.org", "https://raphael.example.org/private", "https://raphael.example.org?token=synthetic", "https://raphael.example.org#section", "https://raphael.example.org:8443", "https://your-authorized-host.invalid", "https://localhost"] {
            XCTAssertNil(ServerConfiguration(urlString: value, boundHost: URL(string: value)?.host ?? ""), value)
        }
        XCTAssertNil(ServerConfiguration(urlString: "https://raphael.example.org", boundHost: "other.example.org"))
    }
    func testEveryNativeSectionUsesTheSameServiceAndExistingWebPaths() throws {
        let configuration = try XCTUnwrap(ServerConfiguration(urlString: "https://raphael.example.org", boundHost: "raphael.example.org"))
        for section in PatientSection.allCases {
            XCTAssertEqual(configuration.url(for: section).host, "raphael.example.org")
            XCTAssertEqual(configuration.url(for: section).scheme, "https")
        }
        XCTAssertEqual(PatientSection.intelligence.path, "static/patient-ai.html")
        XCTAssertEqual(PatientSection.account.path, "static/account.html")
    }
}
