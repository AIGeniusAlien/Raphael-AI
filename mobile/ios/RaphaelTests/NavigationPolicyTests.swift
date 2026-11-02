import XCTest
@testable import Raphael

final class NavigationPolicyTests: XCTestCase {
    private let policy = NavigationPolicy(origin: URL(string: "https://raphael.example.org")!)
    func testOriginCannotBeChangedByRedirectOrCredentialURL() {
        XCTAssertEqual(policy.decision(for: URL(string: "https://raphael.example.org/static/care.html")!), .sameOrigin)
        for value in ["https://raphael.example.org.attacker.invalid", "http://raphael.example.org", "https://raphael.example.org:8443", "https://user:password@raphael.example.org", "file:///private/record", "javascript:alert(1)", "https://example.org"] {
            XCTAssertEqual(policy.decision(for: URL(string: value)!), .blocked, value)
        }
    }
    func testPublicSourceLinksAreExplicitlySeparatedFromThePatientSession() {
        XCTAssertEqual(policy.decision(for: URL(string: "https://www.cdc.gov/antibiotic-use/about/index.html")!), .externalCDC)
        XCTAssertEqual(policy.decision(for: URL(string: "https://www.cdc.gov/other?patient=synthetic")!), .blocked)
        XCTAssertEqual(policy.decision(for: URL(string: "https://www.cdc.gov.attacker.invalid/")!), .blocked)
    }
}
