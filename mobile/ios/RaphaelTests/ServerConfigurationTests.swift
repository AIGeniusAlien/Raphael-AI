import XCTest
import SwiftUI
import Combine
import UIKit
import WebKit
import UniformTypeIdentifiers
@testable import Raphael

final class ServerConfigurationTests: XCTestCase {
    func testReleasePolicyValidatesActualManifestContractAndLimitsCandidateNavigation() throws {
        func manifest(scope: String = "consumer_candidate", version: String = "1", hash: String = String(repeating: "a", count: 64), capabilities: [String: Bool]? = nil) throws -> Data {
            let candidate: [String: Bool] = ["own_linked_records": true, "own_approved_notes": true, "own_document_download": true, "own_account_preferences": true, "own_record_export": true, "own_feedback": true, "registration": false, "external_password_recovery": false, "clinical_assessment": false, "clinical_generation": false, "radiology": false, "ambient_recording": false, "clinical_tools": false, "external_clinical_communications": false, "administration": false, "complete_account_erasure": false]
            return try JSONSerialization.data(withJSONObject: ["scope": scope, "api_version": version, "policy_sha256": hash, "operator_paused": false, "capabilities": scope == "development" ? NSNull() as Any : (capabilities ?? candidate) as Any, "public_distribution_authorized": false, "clinical_production_authorized": false])
        }
        let candidate = try NativeReleasePolicy.validated(manifest())
        XCTAssertEqual(candidate.sections(role: "patient"), [.dashboard])
        XCTAssertEqual(candidate.sections(role: "clinician"), [])
        XCTAssertFalse(candidate.allowsMFA)
        XCTAssertTrue(candidate.allowsExport)
        var pausedObject = try XCTUnwrap(JSONSerialization.jsonObject(with: manifest()) as? [String: Any])
        pausedObject["operator_paused"] = true
        let paused = try NativeReleasePolicy.validated(JSONSerialization.data(withJSONObject: pausedObject))
        XCTAssertFalse(paused.allowsExport)
        pausedObject["clinical_production_authorized"] = true
        XCTAssertThrowsError(try NativeReleasePolicy.validated(JSONSerialization.data(withJSONObject: pausedObject)))
        XCTAssertTrue(candidate.allowsNavigation(URL(string: "https://raphael.example.org/")!))
        XCTAssertTrue(candidate.allowsNavigation(URL(string: "https://raphael.example.org/static/release-preview.html")!))
        for path in ["/static/chart.html", "/static/account.html", "/static/intelligence.html", "/?patient_id=1"] {
            XCTAssertFalse(candidate.allowsNavigation(URL(string: "https://raphael.example.org" + path)!))
        }
        let development = try NativeReleasePolicy.validated(manifest(scope: "development"))
        XCTAssertEqual(development.sections(role: "patient"), PatientSection.allCases)
        XCTAssertTrue(development.allowsMFA)
        for data in [try manifest(scope: "unknown"), try manifest(version: "2"), try manifest(hash: "bad"), try manifest(capabilities: ["clinical_generation": true]), Data("{}".utf8)] {
            XCTAssertThrowsError(try NativeReleasePolicy.validated(data))
        }
    }

    @MainActor
    func testMFAEditorsSurviveRepeatedReconciliationAndExplicitlyEraseBothFields() {
        let credentials=NativeMFACredentials()
        var passwordFlags=[Bool](),factorFlags=[Bool]()
        let passwordChanges=credentials.$hasPassword.dropFirst().sink {passwordFlags.append($0)}
        let factorChanges=credentials.$hasFactor.dropFirst().sink {factorFlags.append($0)}
        defer {passwordChanges.cancel();factorChanges.cancel()}
        // Use one shared private store exactly as the real screen does.
        let actual=NativeMFACredentialFields(credentials:credentials),coordinator=actual.makeCoordinator()
        let inputs=actual.makeInputs(coordinator)
        let password="Synthetic-Exact_Entry9!",factor="012345"
        for character in password {
            inputs.password.text=(inputs.password.text ?? "")+String(character)
            inputs.password.sendActions(for:.editingChanged)
            actual.updateInputs(inputs,coordinator:coordinator)
        }
        for character in factor {
            inputs.factor.text=(inputs.factor.text ?? "")+String(character)
            inputs.factor.sendActions(for:.editingChanged)
            actual.updateInputs(inputs,coordinator:coordinator)
        }
        XCTAssertEqual(passwordFlags,[true]);XCTAssertEqual(factorFlags,[true])
        XCTAssertEqual(inputs.password.text,password);XCTAssertEqual(inputs.factor.text,factor)
        XCTAssertTrue(credentials.hasPassword);XCTAssertTrue(credentials.hasFactor)
        XCTAssertTrue(inputs.password.isSecureTextEntry);XCTAssertTrue(inputs.factor.isSecureTextEntry)
        let captured=credentials.take();actual.updateInputs(inputs,coordinator:coordinator)
        XCTAssertEqual(captured.password,password);XCTAssertEqual(captured.factor,factor)
        XCTAssertEqual(inputs.password.text,"");XCTAssertEqual(inputs.factor.text,"")
        XCTAssertFalse(credentials.hasPassword);XCTAssertFalse(credentials.hasFactor)
        inputs.password.sendActions(for:.editingDidEnd);inputs.factor.sendActions(for:.editingDidEnd)
        XCTAssertEqual(credentials.snapshot().password,"");XCTAssertEqual(credentials.snapshot().factor,"")
    }
    func testNativeMFARejectsMalformedSecretRecoveryAndStatusResponses() throws {
        func reply(_ value:[String:Any]) throws -> NativeMFAReply { try JSONDecoder().decode(NativeMFAReply.self,from:JSONSerialization.data(withJSONObject:value)) }
        XCTAssertTrue(try reply(["enabled":false,"enrollment_available":true,"recovery_codes_remaining":0]).valid(for:.status))
        XCTAssertTrue(try reply(["enabled":false,"secret":String(repeating:"A",count:32),"expires_in_seconds":600]).valid(for:.enroll))
        XCTAssertFalse(try reply(["enabled":true,"secret":"not-a-key","expires_in_seconds":600]).valid(for:.enroll))
        XCTAssertFalse(try reply(["enabled":false,"enrollment_available":true,"recovery_codes_remaining":11]).valid(for:.status))
        let codes=(0..<10).map{String(format:"%020x",$0)}
        XCTAssertTrue(try reply(["enabled":true,"recovery_codes":codes]).valid(for:.confirm))
        XCTAssertFalse(try reply(["enabled":true,"recovery_codes":Array(repeating:codes[0],count:10)]).valid(for:.confirm))
        XCTAssertFalse(try reply(["enabled":true,"recovery_codes":codes,"secret":"unexpected"]).valid(for:.confirm))
        XCTAssertTrue(try reply(["enabled":false,"other_sessions_revoked":true]).valid(for:.disable))
        XCTAssertFalse(try reply(["enabled":false]).valid(for:.disable))
        XCTAssertFalse(try reply(["enabled":true]).valid(for:.disable))
    }
    @MainActor func testNativeMFAManagementRequiresAuthenticatedIdentity() async throws {
        let configuration=try XCTUnwrap(ServerConfiguration(urlString:"https://raphael.example.org",boundHost:"raphael.example.org"))
        let session=PatientWebSession(configuration:configuration)
        for operation in [NativeMFAOperation.status,.enroll,.confirm,.disable] {
            do { _ = try await session.manageMFA(operation,password:"synthetic-only",code:"000000");XCTFail("Unauthenticated management must fail before networking.") }
            catch NativeMFAError.session {} catch { XCTFail("Unexpected unauthenticated management failure type.") }
        }
        XCTAssertNil(session.identity)
    }
    @MainActor
    func testDiscardedWorkspaceGetsFreshEphemeralStoreWithoutWebKitCallback() throws {
        let configuration = try XCTUnwrap(ServerConfiguration(urlString: "https://raphael.example.org", boundHost: "raphael.example.org"))
        let session = PatientWebSession(configuration: configuration)
        let previous = session.webView
        let previousStore = previous.configuration.websiteDataStore
        session.discardLocalWorkspace()
        XCTAssertFalse(session.webView === previous)
        XCTAssertFalse(session.webView.configuration.websiteDataStore === previousStore)
        XCTAssertFalse(session.webView.configuration.websiteDataStore.isPersistent)
        XCTAssertNil(previous.navigationDelegate)
        XCTAssertNil(previous.uiDelegate)
        XCTAssertNil(session.webView.url)
        XCTAssertTrue(session.webView.navigationDelegate === session)
    }
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
        for section in [PatientSection.record, .appointments] {
            let url = configuration.url(for:section,patientID:7)
            XCTAssertEqual(url.host,"raphael.example.org")
            XCTAssertEqual(URLComponents(url:url,resolvingAgainstBaseURL:false)?.queryItems,[URLQueryItem(name:"patient_id",value:"7")])
        }
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
        // This bounds fixture process startup,not bridge execution or auth.
        // A measured cold simulator launch took22.6s;10s incorrectly failed
        // before either Promise behavior assertion could begin.
        await fulfillment(of: [loaded], timeout: 30)
        let value = try await WebKitScriptBridge.call("return await Promise.resolve(true)", in: web)
        XCTAssertEqual(value as? Bool, true)
        do {
            _ = try await WebKitScriptBridge.call("throw new Error('Synthetic bridge rejection')", in: web)
            XCTFail("Rejected JavaScript must not be accepted as successful server revocation")
        } catch {
            // Expected:the native sign-out caller takes its warning/local-clear path.
        }
    }

    private func exportIdentity(_ role:String = "patient",id:Int = 7) throws -> NativeIdentity {
        try JSONDecoder().decode(NativeIdentity.self,from:JSONSerialization.data(withJSONObject:["id":id,"tenant":"synthetic","username":"synthetic-export","role":role]))
    }
    private func exportFixture() -> [String:Any] {
        ["format":"raphael-synthetic-patient-export-v1","generated_at":"2026-01-01T00:00:00+00:00",
         "account":["id":7,"tenant":"synthetic","username":"synthetic-export","role":"patient"],
         "patients":[["id":9,"name":"Synthetic own patient"]],
         "entries":[["id":1,"status":"patient_reported"]],
         "approved_notes":[["id":1,"content":"Synthetic approved note","approved_by":8,"approved_at":"2026-01-01"]],
         "appointments":[],"documents":[],"shared_messages":[],"patient_reports":[],"intakes":[],"approved_summaries":[]]
    }
    func testNativePatientExportValidatesOwnAccountAndScope() throws {
        let data = try JSONSerialization.data(withJSONObject:exportFixture())
        let summary = try NativePatientExport.inspect(data,expected:exportIdentity())
        XCTAssertEqual(summary.records,1);XCTAssertEqual(summary.approvedNotes,1)
        XCTAssertEqual(summary.recordNames,["Synthetic own patient"]);XCTAssertEqual(summary.byteCount,data.count)
        XCTAssertThrowsError(try NativePatientExport.inspect(data,expected:exportIdentity(id:8)))
        XCTAssertThrowsError(try NativePatientExport.inspect(data,expected:exportIdentity("clinician")))
    }
    func testNativePatientExportRejectsCredentialDraftMalformedAndOversizedPayloads() throws {
        let user = try exportIdentity()
        for key in ["password_hash","token_hash","encrypted_secret","recovery_codes","access_token"] {
            var payload=exportFixture();payload["unexpected"]=[key:"SYNTHETIC-MUST-NOT-EXPORT"]
            XCTAssertThrowsError(try NativePatientExport.inspect(JSONSerialization.data(withJSONObject:payload),expected:user))
        }
        var draft=exportFixture();draft["entries"]=[["status":"draft"]]
        XCTAssertThrowsError(try NativePatientExport.inspect(JSONSerialization.data(withJSONObject:draft),expected:user))
        var unapproved=exportFixture();unapproved["approved_notes"]=[["id":1,"approved_by":NSNull()]]
        XCTAssertThrowsError(try NativePatientExport.inspect(JSONSerialization.data(withJSONObject:unapproved),expected:user))
        var many=exportFixture();many["patients"]=Array(repeating:["id":9,"name":"Synthetic"] as [String:Any],count:501)
        XCTAssertThrowsError(try NativePatientExport.inspect(JSONSerialization.data(withJSONObject:many),expected:user))
        var missing=exportFixture();missing.removeValue(forKey:"documents")
        XCTAssertThrowsError(try NativePatientExport.inspect(JSONSerialization.data(withJSONObject:missing),expected:user))
        XCTAssertThrowsError(try NativePatientExport.inspect(Data("not JSON".utf8),expected:user))
        XCTAssertThrowsError(try NativePatientExport.inspect(Data(repeating:32,count:NativePatientExport.maximumBytes+1),expected:user))
    }
    @MainActor
    func testCancelledNativeExportProviderCannotLoadUnauthenticatedData() async throws {
        let configuration=try XCTUnwrap(ServerConfiguration(urlString:"https://raphael.example.org",boundHost:"raphael.example.org"))
        let session=PatientWebSession(configuration:configuration)
        let ticket=NativeExportTicket(identity:try exportIdentity(),epoch:0)
        let provider=NativeExportActivity.provider(session:session,ticket:ticket)
        ticket.invalidate()
        let finished=expectation(description:"Cancelled provider rejects data")
        provider.loadDataRepresentation(forTypeIdentifier:UTType.json.identifier) { data,error in
            XCTAssertNil(data);XCTAssertNotNil(error);finished.fulfill()
        }
        await fulfillment(of:[finished],timeout:5)
        XCTAssertNil(session.nativeExportSummary);XCTAssertNil(session.nativeExportTicket)
    }

}


@MainActor
private final class BridgeTestNavigation: NSObject, WKNavigationDelegate {
    let loaded: XCTestExpectation
    init(loaded: XCTestExpectation) { self.loaded = loaded }
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) { loaded.fulfill() }
}
