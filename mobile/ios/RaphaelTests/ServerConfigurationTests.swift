import XCTest
import SwiftUI
import UIKit
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

    func testNativeSessionCookieScopeAndExpiryAreStrict() throws {
        let origin = try XCTUnwrap(URL(string: "https://raphael.example.org"))
        let now = Date()
        let parsed = HTTPCookie.cookies(withResponseHeaderFields: ["Set-Cookie":"raphael_dev_session=synthetic-parser-fixture; HttpOnly; Max-Age=28800; Path=/api/dev; SameSite=strict; Secure"], for: origin)
        XCTAssertEqual(parsed.count,1)
        XCTAssertTrue(NativeIdentity.accepts(try XCTUnwrap(parsed.first), origin:origin, now:now))
        let properties: [HTTPCookiePropertyKey: Any] = [
            .name: "raphael_dev_session", .value: "synthetic-token", .domain: "raphael.example.org",
            .path: "/api/dev", .secure: "TRUE", .expires: now.addingTimeInterval(600),
            HTTPCookiePropertyKey("HttpOnly"): "TRUE"
        ]
        let valid = try XCTUnwrap(HTTPCookie(properties: properties))
        XCTAssertTrue(NativeIdentity.accepts(valid, origin: origin, now: now))
        for (key, value) in [(HTTPCookiePropertyKey.domain, "other.example.org" as Any),
                              (.path, "/" as Any), (.name, "other" as Any),
                              (.expires, now.addingTimeInterval(-1) as Any)] {
            var changed = properties; changed[key] = value
            XCTAssertFalse(NativeIdentity.accepts(try XCTUnwrap(HTTPCookie(properties: changed)), origin: origin, now: now))
        }
        var insecure = properties; insecure.removeValue(forKey: .secure)
        XCTAssertFalse(NativeIdentity.accepts(try XCTUnwrap(HTTPCookie(properties: insecure)), origin: origin, now: now))
        var readable = properties; readable.removeValue(forKey: HTTPCookiePropertyKey("HttpOnly"))
        XCTAssertFalse(NativeIdentity.accepts(try XCTUnwrap(HTTPCookie(properties: readable)), origin: origin, now: now))
    }

    @MainActor
    func testNativeSecureEntryPreservesExactBindingAndClearsWithoutKeyboardTransformations() throws {
        var value = ""
        let entry = NativeSecretEntry(text: Binding(get: { value }, set: { value = $0 }), title: "Password")
        let coordinator = entry.makeCoordinator()
        let input = entry.makeInput(coordinator: coordinator)
        let exact = "Synthetic-Exact_Entry9!"
        input.text = exact
        input.sendActions(for: .editingChanged)
        XCTAssertEqual(value, exact)
        XCTAssertTrue(input.isSecureTextEntry)
        XCTAssertEqual(input.autocapitalizationType, .none)
        XCTAssertEqual(input.autocorrectionType, .no)
        XCTAssertEqual(input.smartDashesType, .no)
        let captured = try JSONEncoder().encode(NativeCredentials(tenant: "synthetic", username: "synthetic", password: value, otp: ""))
        value = ""
        entry.updateInput(input, coordinator: coordinator)
        XCTAssertEqual(input.text, "")
        let sent = try XCTUnwrap(JSONSerialization.jsonObject(with: captured) as? [String:String])
        XCTAssertEqual(sent["password"],exact)
        var otp = ""
        let reused = NativeSecretEntry(text: Binding(get: { otp }, set: { otp = $0 }), title: "Authenticator", oneTimeCode: true)
        reused.updateInput(input, coordinator: coordinator)
        XCTAssertEqual(input.accessibilityIdentifier,"native-otp")
        XCTAssertEqual(input.textContentType,.oneTimeCode)
        input.text = "123456"
        input.sendActions(for:.editingChanged)
        XCTAssertEqual(otp,"123456")
        XCTAssertEqual(value,"")
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
        do {
            _ = try await WebKitScriptBridge.call("throw new Error('Synthetic bridge rejection')", in: web)
            XCTFail("Rejected JavaScript must not be accepted as successful server revocation")
        } catch {
            // Expected:the native sign-out caller takes its warning/local-clear path.
        }
    }

}


@MainActor
private final class BridgeTestNavigation: NSObject, WKNavigationDelegate {
    let loaded: XCTestExpectation
    init(loaded: XCTestExpectation) { self.loaded = loaded }
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) { loaded.fulfill() }
}
