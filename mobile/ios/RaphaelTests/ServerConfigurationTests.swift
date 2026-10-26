import XCTest
import WebKit
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
    func testReadinessRejectsLegacyMissingAndMalformedServiceResponses() {
        XCTAssertTrue(ReadinessPayload.isSyntheticReady(Data(#"{"status":"ready","scope":"synthetic-development"}"#.utf8)))
        for text in [#"{"status":"ready","scope":"legacy-development"}"#, #"{"status":"not_ready","scope":"synthetic-development"}"#, #"{"status":"ready"}"#, "not JSON"] {
            XCTAssertFalse(ReadinessPayload.isSyntheticReady(Data(text.utf8)))
        }
    }

    @MainActor
    func testPublicWebKitBridgeResolvesPromiseWithoutOverlayLibrary() async throws {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        let web = WKWebView(frame: .zero, configuration: configuration)
        let loaded = expectation(description: "Synthetic local document loaded")
        let navigation = BridgeTestNavigation(loaded: loaded)
        web.navigationDelegate = navigation
        defer { web.navigationDelegate = nil; _ = navigation }
        web.loadHTMLString("<html><body>Synthetic bridge test;no patient data.</body></html>", baseURL: nil)
        await fulfillment(of: [loaded], timeout: 10)
        let value = try await WebKitScriptBridge.call("return await Promise.resolve(true)", in: web)
        XCTAssertEqual(value as? Bool, true)
    }

}


@MainActor
private final class BridgeTestNavigation: NSObject, WKNavigationDelegate {
    let loaded: XCTestExpectation
    init(loaded: XCTestExpectation) { self.loaded = loaded }
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) { loaded.fulfill() }
}
