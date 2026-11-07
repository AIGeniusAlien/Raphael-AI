import XCTest
import UIKit
import Vision
import CryptoKit

fileprivate let nativeFixturePassword = "__CI_SYNTHETIC_PASSWORD__"

final class RaphaelUITests: XCTestCase {
    private let fixturePassword = nativeFixturePassword
    private let fixtureControl = "__CI_CONTROL_TOKEN__"
    private let recoveryFirst = "__CI_RECOVERY_CODE__"
    private let recoverySecond = "__CI_RECOVERY_SECOND__"
    private var privateAuthenticatorTest=false
    override func tearDownWithError() throws {
        if !fixturePassword.hasPrefix("__CI_") { UIPasteboard.general.items = [] }
        let app = XCUIApplication()
        if app.state != .notRunning && !privateAuthenticatorTest {
            let finalState = XCTAttachment(screenshot: app.screenshot())
            finalState.name = "Final synthetic native UI state;workflow result is separate"
            finalState.lifetime = .keepAlways
            add(finalState)
        }
        if app.state != .notRunning { app.terminate() }
        try super.tearDownWithError()
    }

    func testAuthenticatedPatientAndClinicianUseActualHTTPSBackend() throws {
        try actualJourney(paste: false)
    }
    func testPasswordManagerPasteUsesActualAuthenticationAndProtectedWorkspace() throws {
        try actualJourney(paste: true)
    }
    private func actualJourney(paste: Bool) throws {
        continueAfterFailure = false
        let password = fixturePassword
        guard !password.hasPrefix("__CI_") else {
            throw XCTSkip("Requires ephemeral authenticated_ci.py backend;no authenticated native success implied.")
        }
        let app = XCUIApplication()
        app.launch()
        func tapNativeHeader(_ element: XCUIElement) {
            XCTAssertTrue(element.waitForExistence(timeout: 10))
            XCTAssertTrue(!element.frame.isEmpty && app.frame.intersects(element.frame))
            // The actual visible native header has no scroll container. Avoid
            // XCTest's failing AXScrollToVisible action;send the real screen tap.
            element.coordinate(withNormalizedOffset: CGVector(dx:0.5,dy:0.5)).tap()
        }
        func waitForNativeSessionClear(confirmedLogout: Bool) {
            XCTAssertTrue(app.staticTexts["Welcome to Raphael"].waitForExistence(timeout: 30))
            let idle = expectation(for: NSPredicate(format: "exists == false"), evaluatedWith: app.staticTexts["native-authentication-status"])
            wait(for: [idle], timeout: 20)
            if confirmedLogout { XCTAssertFalse(app.staticTexts["native-authentication-problem"].exists) }
        }
        func login(_ username: String, workspace: String) {
            let organization = app.textFields["Development organization"]
            XCTAssertTrue(organization.waitForExistence(timeout: 30))
            organization.tap(); organization.typeText("native-ci")
            XCTAssertTrue(organization.value as? String == "native-ci", "Organization must retain the exact synthetic input.")
            let user = app.textFields["Username"]
            user.tap(); user.typeText(username)
            XCTAssertTrue(user.value as? String == username, "Username must retain the exact synthetic input.")
            let secure = app.secureTextFields["native-password"]
            secure.tap()
            XCTAssertTrue(app.keyboards.firstMatch.waitForExistence(timeout: 10), "Actual software keyboard required for secure typing.")
            if paste {
                // Exercise a real password-manager-style paste into the native
                // secure control;no JS,token injection or API-login shortcut.
                UIPasteboard.general.string = password
                secure.press(forDuration: 1.1)
                let menu = app.menuItems["Paste"]
                let button = app.buttons["Paste"]
                if menu.waitForExistence(timeout: 3) { menu.tap() }
                else { XCTAssertTrue(button.waitForExistence(timeout: 5)); button.tap() }
                let allow = app.alerts.buttons["Allow Paste"]
                if allow.waitForExistence(timeout: 2) { allow.tap() }
                let ready = NSPredicate(format: "enabled == true")
                let entered = expectation(for: ready, evaluatedWith: app.buttons["Sign in"])
                wait(for: [entered], timeout: 10)
                UIPasteboard.general.items = []
            } else { secure.typeText(password) }
            app.buttons["Sign in"].tap()
            let reached = app.staticTexts[username + " · " + workspace + " · native-ci"].waitForExistence(timeout: 45)
            if !reached {
                let problem = app.staticTexts["native-authentication-problem"]
                let progress = app.staticTexts["native-authentication-status"]
                XCTFail(problem.exists ? problem.label : "Authenticated web identity did not appear. Native status: " + (progress.exists ? progress.label : "No in-flight request"))
            }
            if username == "native-clinician" {
                // This role has two authorized records. The portal deliberately
                // requires a real patient selection before displaying notes.
                // aria-pressed selectors may have a different native AX role.
                // Match the exact accessible name, preserving pressed semantics.
                // Any nested text match belongs to this same uniquely named
                // fixture control;the subsequent protected note proves its tap.
                let record = app.descendants(matching: .any).matching(NSPredicate(format: "label == %@", "Synthetic native patient")).firstMatch
                for _ in 0..<5 {
                    if record.waitForExistence(timeout: 2) && record.isHittable { break }
                    app.swipeUp()
                }
                XCTAssertTrue(record.exists)
                print("Synthetic record selector native accessibility type: \(record.elementType.rawValue)")
                record.tap()
            }
            XCTAssertTrue(app.staticTexts["Synthetic native approved note. No real clinical findings."].waitForExistence(timeout: 15))
        }
        login("native-patient", workspace: "Patient workspace")
        let patient = XCTAttachment(screenshot: app.screenshot())
        patient.name = "Authenticated synthetic patient over trusted HTTPS"
        patient.lifetime = .keepAlways; add(patient)
        tapNativeHeader(app.buttons["Patient sections"])
        app.buttons["Medical record"].tap()
        if !app.staticTexts["Synthetic native patient · Synthetic chart"].waitForExistence(timeout: 20) {
            let issue = app.staticTexts["native-workspace-problem"]
            XCTFail(issue.exists ? issue.label : "Protected chart heading did not appear after authorized navigation.")
        }
        tapNativeHeader(app.buttons["Patient sections"])
        app.buttons["Dashboard"].tap()
        XCTAssertTrue(app.buttons["Sign out"].waitForExistence(timeout: 15))
        tapNativeHeader(app.buttons["Privacy and session settings"])
        let readiness = app.buttons["Check backend connection"]
        XCTAssertTrue(readiness.waitForExistence(timeout: 10))
        readiness.tap()
        let ready = NSPredicate(format: "label BEGINSWITH %@", "Synthetic backend database ready.")
        expectation(for: ready, evaluatedWith: app.staticTexts["native-readiness-status"])
        waitForExpectations(timeout: 15)
        app.buttons["Sign out and clear local session"].tap()
        let confirmLogout = app.buttons["native-confirm-sign-out"]
        XCTAssertTrue(confirmLogout.waitForExistence(timeout: 10))
        confirmLogout.tap()
        waitForNativeSessionClear(confirmedLogout: true)
        app.terminate(); app.launch()
        login("native-clinician", workspace: "Physician workspace")
        let clinician = XCTAttachment(screenshot: app.screenshot())
        clinician.name = "Authenticated synthetic clinician over trusted HTTPS"
        clinician.lifetime = .keepAlways; add(clinician)
        // The native confirmation is dismissed; only the actual portal button
        // has this label here. Do not require a WebView accessibility ancestor.
        app.buttons["Sign out"].tap()
        // Native welcome appears only after the model independently observes
        // the revoked web session and clears its private cookie/workspace.
        waitForNativeSessionClear(confirmedLogout: false)
        app.terminate(); app.launch()
        XCTAssertTrue(app.textFields["Username"].waitForExistence(timeout: 30))
    }

    func testActualHTTPSAuthenticationContractIndependentOfWebKit() async throws {
        guard !fixturePassword.hasPrefix("__CI_") else { throw XCTSkip("Private actual-HTTPS CI fixture required.") }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpCookieStorage = nil
        configuration.httpShouldSetCookies = false
        configuration.urlCache = nil
        configuration.timeoutIntervalForRequest = 15
        configuration.timeoutIntervalForResource = 20
        let client = URLSession(configuration: configuration, delegate: ContractRedirectPolicy(), delegateQueue: nil)
        defer { client.invalidateAndCancel() }
        let origin = URL(string: "https://raphael-ci.test")!
        func call(_ path: String, cookie: HTTPCookie? = nil, body: [String:String]? = nil) async throws -> (Data, HTTPURLResponse) {
            var request = URLRequest(url: origin.appendingPathComponent(path))
            request.setValue(origin.absoluteString, forHTTPHeaderField: "Origin")
            if let cookie { for (key,value) in HTTPCookie.requestHeaderFields(with: [cookie]) { request.setValue(value, forHTTPHeaderField: key) } }
            if let body {
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONSerialization.data(withJSONObject: body)
            }
            let (data,response) = try await client.data(for: request)
            return (data, try XCTUnwrap(response as? HTTPURLResponse))
        }
        do {
            for username in ["native-patient", "native-clinician"] {
            let (_,login) = try await call("api/dev/login", body: ["tenant":"native-ci","username":username,"password":fixturePassword])
            XCTAssertEqual(login.statusCode, 200)
            let headers = login.allHeaderFields.reduce(into: [String:String]()) { output,pair in
                if let key = pair.key as? String, let value = pair.value as? String { output[key] = value }
            }
            let cookie = try XCTUnwrap(HTTPCookie.cookies(withResponseHeaderFields: headers, for: origin).first { $0.name == "raphael_dev_session" })
            XCTAssertTrue(cookie.isSecure && cookie.isHTTPOnly)
            let (sessionData,session) = try await call("api/dev/session", cookie:cookie)
            XCTAssertEqual(session.statusCode,200)
            let identity = try XCTUnwrap(JSONSerialization.jsonObject(with:sessionData) as? [String:Any])
            XCTAssertEqual(identity["role"] as? String, username == "native-patient" ? "patient" : "clinician")
            let (records,chart) = try await call("api/dev/patients", cookie:cookie)
            XCTAssertEqual(chart.statusCode,200)
            let patients = try XCTUnwrap(JSONSerialization.jsonObject(with: records) as? [[String:Any]])
            let patient = try XCTUnwrap(patients.first { $0["name"] as? String == "Synthetic native patient" })
            let id = try XCTUnwrap(patient["id"] as? Int)
            let (_,otherChart) = try await call("api/dev/patients/\(id+1)", cookie:cookie)
            XCTAssertEqual(otherChart.statusCode, username == "native-patient" ? 404 : 200)
            let (notes,response) = try await call("api/dev/patients/\(id)/notes", cookie:cookie)
            XCTAssertEqual(response.statusCode,200)
            XCTAssertTrue(String(decoding:notes,as:UTF8.self).contains("Synthetic native approved note. No real clinical findings."))
            let (_,forbidden) = try await call("api/dev/admin/users", cookie:cookie)
            XCTAssertEqual(forbidden.statusCode,403)
            let (_,logout) = try await call("api/dev/logout", cookie:cookie, body:[:])
            XCTAssertEqual(logout.statusCode,200)
            let (_,revoked) = try await call("api/dev/session", cookie:cookie)
            XCTAssertEqual(revoked.statusCode,401)
            }
        } catch {
            XCTFail("Independent native HTTPS contract failed (error code \((error as NSError).code)); no credential values logged.")
        }
    }

    func testNativeSignInControlsDoNotRequireRemoteWebKitAccessibility() throws {
        let app = XCUIApplication()
        app.launch()
        if app.staticTexts["configuration-required"].exists { throw XCTSkip("Configured synthetic HTTPS build required.") }
        XCTAssertTrue(app.staticTexts["Welcome to Raphael"].waitForExistence(timeout: 15))
        let organization = app.textFields["Development organization"]
        XCTAssertTrue(organization.exists)
        organization.tap(); organization.typeText("synthetic-accessibility-check")
        XCTAssertEqual(organization.value as? String, "synthetic-accessibility-check")
        XCTAssertTrue(app.secureTextFields["Password"].exists)
        XCTAssertFalse(app.webViews.firstMatch.exists)
    }

    func testUnconfiguredBuildFailsSafelyWithoutConnecting() {
        let app = XCUIApplication()
        app.launchArguments = ["--uitest-unconfigured"]
        app.launch()
        XCTAssertTrue(app.staticTexts["configuration-required"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts["Server configuration required"].exists)
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.name = "Unconfigured development app"
        screenshot.lifetime = .keepAlways
        add(screenshot)
    }

    private func configuredFixture() throws {
        if fixturePassword.hasPrefix("__CI_") || fixtureControl.hasPrefix("__CI_") { throw XCTSkip("Private actual HTTPS fixture required;no extended acceptance implied.") }
    }

    private func foregroundFailureState(_ app: XCUIApplication) -> String {
        let screenshot = app.screenshot()
        let attachment = XCTAttachment(screenshot:screenshot)
        attachment.name = "Synthetic foreground failure diagnostic"
        attachment.lifetime = .keepAlways;add(attachment)
        var flags:[String:Any] = ["application_state":app.state.rawValue,
            "native_privacy_cover_ax":app.staticTexts["Private workspace hidden"].exists,
            "native_sections_ax":app.buttons["Patient sections"].exists,
            "native_notice_ax":app.staticTexts["synthetic-notice"].exists,
            "sign_in_button_ax":app.buttons["Sign in"].exists,
            "ocr_available":false]
        if let image=screenshot.image.cgImage {
            let request=VNRecognizeTextRequest();request.recognitionLevel = .accurate;request.usesLanguageCorrection=false
            do {
                try VNImageRequestHandler(cgImage:image,options:[:]).perform([request])
                let text=(request.results ?? []).compactMap{$0.topCandidates(1).first?.string}.joined(separator:" ").lowercased()
                // Never publish recognized text: only fixed synthetic UI flags.
                flags["ocr_available"]=true
                flags["pixels_privacy_cover"]=text.contains("private workspace hidden")
                flags["pixels_patient_workspace"]=text.contains("patient workspace")
                flags["pixels_sign_in"]=text.contains("sign in")
                flags["pixels_synthetic_notice"]=text.contains("synthetic development")
            } catch { flags["ocr_available"]=false }
        }
        let data=(try? JSONSerialization.data(withJSONObject:flags,options:[.sortedKeys])) ?? Data()
        return String(data:data,encoding:.utf8) ?? "bounded diagnostic unavailable"
    }
    private enum FixtureFailure: Error { case requiredUIState }
    private func requireUI(_ condition: Bool, _ message: String, file: StaticString = #filePath, line: UInt = #line) throws {
        guard condition else {
            // Async XCTest must finish through Swift error propagation. With
            // continueAfterFailure=false XCTest can advance while its suspended
            // async body keeps operating on the next test's application.
            XCTFail(message, file:file, line:line)
            throw FixtureFailure.requiredUIState
        }
    }
    private func signIn(_ app: XCUIApplication, username: String, factor: String? = nil) throws {
        let organization = app.textFields["Development organization"]
        try requireUI(organization.waitForExistence(timeout:30),"Native organization input must appear.")
        organization.tap(); organization.typeText("native-ci")
        try requireUI(app.textFields["Username"].waitForExistence(timeout:10),"Native username input must appear.")
        app.textFields["Username"].tap(); app.textFields["Username"].typeText(username)
        let password = app.secureTextFields["native-password"]
        try requireUI(password.waitForExistence(timeout:10),"Native secure password input must appear.")
        password.tap(); password.typeText(fixturePassword)
        if let factor {
            app.buttons["Use an authenticator or recovery code"].tap()
            let field = app.secureTextFields["native-otp"]
            try requireUI(field.waitForExistence(timeout:10),"Native second-factor input must appear.");field.tap();field.typeText(factor)
        }
        app.buttons["Sign in"].tap()
        try requireUI(app.staticTexts[username+" · Patient workspace · native-ci"].waitForExistence(timeout:45),"Actual authenticated patient web workspace must appear.")
    }

    private func tapHeader(_ element: XCUIElement) throws {
        try requireUI(element.waitForExistence(timeout:10),"Visible native action must exist.")
        try requireUI(!element.frame.isEmpty,"Visible native action must have a real target frame.")
        element.coordinate(withNormalizedOffset:CGVector(dx:0.5,dy:0.5)).tap()
    }

    @MainActor
    func testMFARecoveryBackgroundForegroundAndOwnerSessionRevocation() async throws {
        try configuredFixture();continueAfterFailure=true
        let http=FixtureHTTPS();defer {http.close()}
        let (_,denied)=try await http.call("api/dev/login",body:["tenant":"native-ci","username":"native-mfa-patient","password":fixturePassword])
        XCTAssertEqual(denied.statusCode,401)
        let app=XCUIApplication();app.launch();try signIn(app,username:"native-mfa-patient",factor:recoveryFirst)
        XCUIDevice.shared.press(.home)
        try requireUI(app.wait(for:.runningBackground,timeout:10) || app.state == .runningBackgroundSuspended,"App must reach the background before activation.")
        app.activate()
        try requireUI(app.wait(for:.runningForeground,timeout:20),"App must return to foreground.")
        let identity = app.staticTexts["native-mfa-patient · Patient workspace · native-ci"]
        var resumed = identity.waitForExistence(timeout:20)
        let problem = app.staticTexts["native-workspace-problem"]
        // WebKit may legitimately terminate its content process in background.
        // Accept only its explicit existing recovery UI,then prove the actual
        // authenticated web workspace after one real user retry. Never bypass
        // sign-in or accept a native-only identity as web-session proof.
        if !resumed && problem.exists && problem.label.hasPrefix("The secure web session stopped.") {
            try tapHeader(app.buttons["Retry this section"])
            resumed = identity.waitForExistence(timeout:30)
            print("::notice title=Raphael native background recovery::Explicit user retry after reported WebKit content-process termination;authenticated web identity remains required.")
        }
        let stopped = problem.exists && problem.label.hasPrefix("The secure web session stopped.")
        let welcome = app.staticTexts["Welcome to Raphael"].exists
        let observed = resumed ? "authenticated web identity visible" : foregroundFailureState(app)
        try requireUI(resumed,"Actual authenticated web workspace must survive or visibly recover after background;reportedWebTermination=\(stopped),nativeSignIn=\(welcome),observed=\(observed).")
        let owner=try await http.login("native-mfa-patient",password:fixturePassword,factor:recoverySecond)
        let (sessionData,status)=try await http.call("api/dev/account/sessions",cookie:owner)
        XCTAssertEqual(status.statusCode,200)
        let sessions=try XCTUnwrap(JSONSerialization.jsonObject(with:sessionData) as? [[String:Any]])
        let appSessions=sessions.filter { ($0["current"] as? Bool)==false }
        XCTAssertEqual(appSessions.count,1)
        for session in appSessions {
            let id=try XCTUnwrap(session["id"] as? String)
            let (_,revoked)=try await http.call("api/dev/account/sessions/\(id)/revoke",cookie:owner,body:[:])
            XCTAssertEqual(revoked.statusCode,200)
        }
        try requireUI(app.staticTexts["Welcome to Raphael"].waitForExistence(timeout:30),"Owner revocation must return native sign-in.")
        let (_,reused)=try await http.call("api/dev/login",body:["tenant":"native-ci","username":"native-mfa-patient","password":fixturePassword,"otp":recoveryFirst])
        XCTAssertEqual(reused.statusCode,401)
        let logoutStatus=try await http.logout(owner);XCTAssertEqual(logoutStatus,200)
        app.terminate();app.launch()
        try requireUI(app.textFields["Username"].waitForExistence(timeout:20),"Relaunch must require fresh native sign-in.")
    }

    @MainActor
    func testReadinessFailureSlowTimeoutRetryAndAuthoritativeSessionExpiry() async throws {
        try configuredFixture();continueAfterFailure=true
        let http=FixtureHTTPS();defer {http.close()}
        let app=XCUIApplication();app.launch();try signIn(app,username:"native-expiry-patient")
        try tapHeader(app.buttons["Privacy and session settings"])
        try await http.control("unavailable",token:fixtureControl)
        app.buttons["Check backend connection"].tap()
        let status=app.staticTexts["native-readiness-status"]
        let unavailable=expectation(for:NSPredicate(format:"label BEGINSWITH %@","Backend unavailable"),evaluatedWith:status)
        await fulfillment(of:[unavailable],timeout:15)
        try await http.control("slow",token:fixtureControl)
        app.buttons["Check backend connection"].tap()
        let timeout=expectation(for:NSPredicate(format:"label BEGINSWITH %@","Connection unavailable"),evaluatedWith:status)
        await fulfillment(of:[timeout],timeout:20)
        try await http.control("healthy",token:fixtureControl)
        app.buttons["Check backend connection"].tap()
        let ready=expectation(for:NSPredicate(format:"label BEGINSWITH %@","Synthetic backend database ready."),evaluatedWith:status)
        await fulfillment(of:[ready],timeout:15)
        app.buttons["Done"].tap()
        try await http.control("unavailable",token:fixtureControl)
        let verificationProblem=app.staticTexts["native-workspace-problem"]
        XCTAssertTrue(verificationProblem.waitForExistence(timeout:30))
        XCTAssertTrue(verificationProblem.label.hasPrefix("Session verification unavailable."))
        try await http.control("healthy",token:fixtureControl)
        let recovered=expectation(for:NSPredicate(format:"exists == false"),evaluatedWith:verificationProblem)
        await fulfillment(of:[recovered],timeout:30)
        try await http.control("expire_fixture_session",token:fixtureControl)
        XCTAssertTrue(app.staticTexts["Welcome to Raphael"].waitForExistence(timeout:30))
        XCTAssertFalse(app.staticTexts["native-expiry-patient · Patient workspace · native-ci"].exists)
    }

    func testSignInLargestDynamicTypeHasScrollableAccessibleControls() throws {
        try configuredFixture();continueAfterFailure=false
        let app=XCUIApplication();app.launch()
        let heading=app.staticTexts["Welcome to Raphael"]
        XCTAssertTrue(heading.waitForExistence(timeout:20));let normalHeight=heading.frame.height
        app.terminate();app.launchArguments=["-UIPreferredContentSizeCategoryName","UICTContentSizeCategoryAccessibilityXXXL"];app.launch()
        XCTAssertTrue(heading.waitForExistence(timeout:20));XCTAssertGreaterThan(heading.frame.height,normalHeight)
        let password=app.secureTextFields["native-password"]
        for _ in 0..<6 { if password.exists && password.isHittable {break};app.swipeUp() }
        XCTAssertTrue(password.exists);XCTAssertTrue(password.isHittable)
        let screenshot=XCTAttachment(screenshot:app.screenshot());screenshot.name="Actual largest Dynamic Type synthetic sign-in";screenshot.lifetime = .keepAlways;add(screenshot)
    }

    private func visibleExportAction(_ app:XCUIApplication,_ identifier:String) throws -> XCUIElement {
        let button=app.buttons[identifier]
        for upward in [true,false] {
            for _ in 0..<7 {
                if button.exists && button.isHittable { return button }
                if upward { app.swipeUp() } else { app.swipeDown() }
            }
        }
        try requireUI(button.exists && button.isHittable,"Native export action must be reachable.")
        return button
    }

    private func enterPrivateMFAValue(_ app:XCUIApplication,_ identifier:String,_ value:String) throws {
        let field=app.secureTextFields[identifier]
        try requireUI(field.waitForExistence(timeout:10),"Private authenticator input must exist.")
        // A partially visible Form row can be hittable while its center is
        // covered by the keyboard. Scroll the actual row above that region.
        for _ in 0..<5 {
            if field.isHittable && field.frame.maxY<app.frame.height*0.55 {break}
            app.swipeUp()
        }
        try requireUI(field.isHittable,"Private authenticator input must be reachable.")
        field.tap()
        try requireUI(app.keyboards.firstMatch.waitForExistence(timeout:10),"Actual software keyboard is required.")
        field.typeText(value)
        // Secure accessibility values are masked representations,not a
        // credential-length oracle. Real endpoint acceptance below verifies
        // the exact password/factor;never inspect or log the private value.
    }

    // An independent authenticator implementation in the UI TEST target only.
    // It uses the actual setup key delivered to the enrolled user,never a
    // pre-enabled factor or an authentication bypass.
    private func authenticatorCode(_ secret:String,step:Int) throws -> String {
        let alphabet=Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"),characters=Array(secret)
        var bytes=[UInt8](),buffer:UInt32=0,bits=0
        for character in characters {
            guard let value=alphabet.firstIndex(of:character) else { throw FixtureFailure.requiredUIState }
            buffer=(buffer<<5)|UInt32(value);bits+=5
            if bits>=8 {bits-=8;bytes.append(UInt8((buffer>>bits)&255));buffer &= (1<<bits)-1}
        }
        var counter=UInt64(step).bigEndian
        let input=withUnsafeBytes(of:&counter){Data($0)}
        let digest=Array(HMAC<Insecure.SHA1>.authenticationCode(for:input,using:SymmetricKey(data:Data(bytes))))
        let offset=Int(digest.last! & 15)
        let number=(UInt32(digest[offset]&127)<<24)|(UInt32(digest[offset+1])<<16)|(UInt32(digest[offset+2])<<8)|UInt32(digest[offset+3])
        return String(format:"%06u",number%1000000)
    }
    @MainActor
    func testNativeTOTPEnrollmentVerificationRecoveryAndDisable() async throws {
        try configuredFixture();continueAfterFailure=true;privateAuthenticatorTest=true
        XCTAssertEqual(try authenticatorCode("GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ",step:1),"287082")
        let app=XCUIApplication();app.launch();try signIn(app,username:"native-totp-patient")
        try tapHeader(app.buttons["Privacy and session settings"])
        try tapHeader(visibleExportAction(app,"native-mfa-manage"))
        try requireUI(app.buttons["native-mfa-enroll"].waitForExistence(timeout:20),"Native authenticator management must be accessible.")
        let ready=expectation(for:NSPredicate(format:"label == %@","Authenticator status verified."),evaluatedWith:app.staticTexts["native-mfa-status"])
        await fulfillment(of:[ready],timeout:20)
        try enterPrivateMFAValue(app,"native-mfa-password",fixturePassword)
        try tapHeader(visibleExportAction(app,"native-mfa-enroll"))
        let key=app.staticTexts["native-mfa-setup-key"]
        try requireUI(key.waitForExistence(timeout:20),"Actual private enrollment key must be delivered.")
        let secret=key.label;try requireUI(secret.count==32,"Private setup key must have the expected bounded format.")
        let enrolledStep=Int(Date().timeIntervalSince1970)/30
        try enterPrivateMFAValue(app,"native-mfa-password",fixturePassword)
        try enterPrivateMFAValue(app,"native-mfa-code",try authenticatorCode(secret,step:enrolledStep))
        try tapHeader(visibleExportAction(app,"native-mfa-confirm"))
        let privateCodes=app.descendants(matching:.any).matching(identifier:"native-mfa-recovery-codes").firstMatch
        var disclosureScrolls=0
        _ = privateCodes.waitForExistence(timeout:5)
        for upward in [true,false] {
            for _ in 0..<7 {
                if privateCodes.exists && !privateCodes.frame.isEmpty && app.frame.intersects(privateCodes.frame) { break }
                if upward {app.swipeUp()} else {app.swipeDown()};disclosureScrolls+=1
            }
            if privateCodes.exists && !privateCodes.frame.isEmpty && app.frame.intersects(privateCodes.frame) {break}
        }
        let disclosureVisible=privateCodes.exists && !privateCodes.frame.isEmpty && app.frame.intersects(privateCodes.frame)
        let nativeStatus=app.staticTexts["native-mfa-status"]
        try requireUI(disclosureVisible,"Actual recovery disclosure must be visible. Native status: "+(nativeStatus.exists ? nativeStatus.label : "Not in current viewport"))
        print("Native recovery disclosure reached after \(disclosureScrolls) scrolls;accessibility type \(privateCodes.elementType.rawValue). No private value logged.")
        let codes=privateCodes.label.split(separator:"\n").map(String.init)
        try requireUI(codes.count==10,"All ten private recovery codes must be available.")
        try tapHeader(visibleExportAction(app,"native-mfa-clear"))
        XCTAssertFalse(key.exists);XCTAssertFalse(privateCodes.exists)
        try tapHeader(app.buttons["native-mfa-done"]);app.buttons["Done"].tap()
        app.buttons["Sign out"].tap()
        try requireUI(app.staticTexts["Welcome to Raphael"].waitForExistence(timeout:30),"Enrollment must preserve normal logout.")
        let http=FixtureHTTPS();defer{http.close()}
        let (_,passwordOnly)=try await http.call("api/dev/login",body:["tenant":"native-ci","username":"native-totp-patient","password":fixturePassword])
        XCTAssertEqual(passwordOnly.statusCode,401)
        // Enrollment consumes its step. Wait at most one bounded30s period;
        // never reset server replay state or inject a synthetic session.
        let next=TimeInterval((enrolledStep+1)*30+1)-Date().timeIntervalSince1970
        if next>0 {try await Task.sleep(nanoseconds:UInt64(next*1_000_000_000))}
        let loginCode=try authenticatorCode(secret,step:Int(Date().timeIntervalSince1970)/30)
        try signIn(app,username:"native-totp-patient",factor:loginCode)
        let (_,replayed)=try await http.call("api/dev/login",body:["tenant":"native-ci","username":"native-totp-patient","password":fixturePassword,"otp":loginCode])
        XCTAssertEqual(replayed.statusCode,401)
        let other=try await http.login("native-totp-patient",password:fixturePassword,factor:codes[1])
        try tapHeader(app.buttons["Privacy and session settings"])
        try tapHeader(visibleExportAction(app,"native-mfa-manage"))
        try requireUI(app.buttons["native-mfa-disable"].waitForExistence(timeout:20),"Enabled authenticator must offer protected management.")
        try enterPrivateMFAValue(app,"native-mfa-password",fixturePassword)
        try enterPrivateMFAValue(app,"native-mfa-code",codes[0])
        try tapHeader(visibleExportAction(app,"native-mfa-disable"))
        try tapHeader(app.buttons["native-mfa-confirm-disable"])
        let disabled=expectation(for:NSPredicate(format:"label == %@","Authenticator disabled. Other sessions revoked."),evaluatedWith:app.staticTexts["native-mfa-status"])
        await fulfillment(of:[disabled],timeout:20)
        let (_,revoked)=try await http.call("api/dev/session",cookie:other);XCTAssertEqual(revoked.statusCode,401)
        try tapHeader(app.buttons["native-mfa-done"]);app.buttons["Done"].tap();app.buttons["Sign out"].tap()
        try requireUI(app.staticTexts["Welcome to Raphael"].waitForExistence(timeout:30),"MFA management must preserve final session invalidation.")
    }

    @MainActor
    func testNativePatientExportCopyAndSystemShareCancellation() async throws {
        try configuredFixture();continueAfterFailure=true
        let app=XCUIApplication();app.launch();try signIn(app,username:"native-patient")
        try tapHeader(app.buttons["Privacy and session settings"])
        try tapHeader(visibleExportAction(app,"native-export-prepare"))
        try requireUI(app.staticTexts["native-export-summary"].waitForExistence(timeout:25),"Authorized export summary must appear.")
        let copyAction = try visibleExportAction(app,"native-export-copy")
        let review = XCTAttachment(screenshot:app.screenshot());review.name="Synthetic native export review before disclosure";review.lifetime = .keepAlways;add(review)
        try tapHeader(copyAction)
        try tapHeader(app.buttons["native-export-confirm-copy"])
        let copied=expectation(for:NSPredicate(format:"label BEGINSWITH %@","Sensitive JSON copied locally"),evaluatedWith:app.staticTexts["native-export-status"])
        await fulfillment(of:[copied],timeout:25)
        print("Native copy status verified;actual user paste after logout starts.")
        app.buttons["Done"].tap()
        try requireUI(app.buttons["Sign out"].waitForExistence(timeout:15),"Patient web sign-out must remain available after copy.")
        app.buttons["Sign out"].tap()
        try requireUI(app.staticTexts["Welcome to Raphael"].waitForExistence(timeout:30),"Deliberate export must not prevent logout.")
        let pasteTarget = app.textFields["Username"]
        try requireUI(pasteTarget.waitForExistence(timeout:10),"Native paste target must be available after logout.")
        pasteTarget.tap();pasteTarget.press(forDuration:1.1)
        let pasteMenu=app.menuItems["Paste"],pasteButton=app.buttons["Paste"]
        if pasteMenu.waitForExistence(timeout:3) { pasteMenu.tap() }
        else { try requireUI(pasteButton.waitForExistence(timeout:5),"Explicit native Paste action must be available.");pasteButton.tap() }
        let allowPaste=app.alerts.buttons["Allow Paste"]
        if allowPaste.waitForExistence(timeout:2) { allowPaste.tap() }
        let pasted=expectation(for:NSPredicate(format:"value BEGINSWITH %@","{"),evaluatedWith:pasteTarget)
        await fulfillment(of:[pasted],timeout:10)
        let text=try XCTUnwrap(pasteTarget.value as? String,"Real user paste must deliver the exported JSON.")
        let payload=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(text.utf8)) as? [String:Any])
        XCTAssertEqual(payload["format"] as? String,"raphael-synthetic-patient-export-v1")
        let account=try XCTUnwrap(payload["account"] as? [String:Any])
        XCTAssertEqual(account["username"] as? String,"native-patient");XCTAssertEqual(account["tenant"] as? String,"native-ci");XCTAssertEqual(account["role"] as? String,"patient")
        let records=try XCTUnwrap(payload["patients"] as? [[String:Any]])
        XCTAssertEqual(records.count,1);XCTAssertEqual(records.first?["name"] as? String,"Synthetic native patient")
        XCTAssertFalse(text.contains("Synthetic native unlinked record"));XCTAssertFalse(text.contains("password_hash"));XCTAssertFalse(text.contains(fixturePassword))
        UIPasteboard.general.items=[]
        // No sign-in is submitted with the pasted synthetic JSON. Relaunch
        // clears the unsent field before the separate real share workflow.
        app.terminate();app.launch();try signIn(app,username:"native-patient")
        try tapHeader(app.buttons["Privacy and session settings"])
        try tapHeader(visibleExportAction(app,"native-export-prepare"))
        try requireUI(app.staticTexts["native-export-summary"].waitForExistence(timeout:25),"A new export must be prepared before another delivery.")
        try tapHeader(visibleExportAction(app,"native-export-share"))
        try tapHeader(app.buttons["native-export-confirm-share"])
        try requireUI(app.buttons["native-export-share-cancel"].waitForExistence(timeout:15),"Real system share container must be presented.")
        let sharing = XCTAttachment(screenshot:app.screenshot());sharing.name="Actual system JSON share presentation;no destination selected";sharing.lifetime = .keepAlways;add(sharing)
        app.buttons["native-export-share-cancel"].tap()
        try requireUI(app.buttons["Done"].waitForExistence(timeout:15),"Cancelling share must return settings.")
        app.buttons["Done"].tap()
        // Use the real existing web sign-out,not a synthetic cleanup mutation.
        try requireUI(app.buttons["Sign out"].waitForExistence(timeout:15),"Patient portal sign-out must remain available.")
        app.buttons["Sign out"].tap()
        try requireUI(app.staticTexts["Welcome to Raphael"].waitForExistence(timeout:30),"Export must preserve normal session invalidation.")
    }

    @MainActor
    func testNativeExportFailureCancellationAndOwnerRevocation() async throws {
        try configuredFixture();continueAfterFailure=true
        UIPasteboard.general.string="Synthetic export cancellation sentinel"
        let before=UIPasteboard.general.changeCount
        let http=FixtureHTTPS();defer{http.close()}
        let app=XCUIApplication();app.launch();try signIn(app,username:"native-export-patient")
        try tapHeader(app.buttons["Privacy and session settings"])
        try await http.control("unavailable",token:fixtureControl)
        try tapHeader(visibleExportAction(app,"native-export-prepare"))
        let failed=expectation(for:NSPredicate(format:"label BEGINSWITH %@","Export unavailable."),evaluatedWith:app.staticTexts["native-export-status"])
        await fulfillment(of:[failed],timeout:25)
        XCTAssertFalse(app.buttons["native-export-copy"].exists)
        XCTAssertEqual(UIPasteboard.general.changeCount,before)
        try await http.control("healthy",token:fixtureControl)
        try await http.control("slow",token:fixtureControl)
        try tapHeader(visibleExportAction(app,"native-export-prepare"))
        try tapHeader(visibleExportAction(app,"native-export-cancel"))
        XCTAssertFalse(app.staticTexts["native-export-summary"].exists)
        try await http.control("healthy",token:fixtureControl)
        try tapHeader(visibleExportAction(app,"native-export-prepare"))
        try requireUI(app.staticTexts["native-export-summary"].waitForExistence(timeout:25),"A cancelled export can be prepared again.")
        let owner=try await http.login("native-export-patient",password:fixturePassword)
        let (bytes,_)=try await http.call("api/dev/account/sessions",cookie:owner)
        let sessions=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [[String:Any]])
        let others=sessions.filter{($0["current"] as? Bool)==false};XCTAssertEqual(others.count,1)
        for row in others {
            let id=try XCTUnwrap(row["id"] as? String)
            let (_,response)=try await http.call("api/dev/account/sessions/\(id)/revoke",cookie:owner,body:[:]);XCTAssertEqual(response.statusCode,200)
        }
        try requireUI(app.staticTexts["Welcome to Raphael"].waitForExistence(timeout:40),"Owner revocation must dismiss export/settings and clear prepared data.")
        XCTAssertFalse(app.buttons["native-export-copy"].exists)
        XCTAssertEqual(UIPasteboard.general.changeCount,before)
        let loggedOut=try await http.logout(owner);XCTAssertEqual(loggedOut,200)
    }

    func testProtectedDocumentConsentAndFullChartRevocationOverActualHTTPS() async throws {
        try configuredFixture()
        let http=FixtureHTTPS();defer {http.close()}
        let patient=try await http.login("native-consent-patient",password:fixturePassword)
        let consultant=try await http.login("native-consultant",password:fixturePassword)
        let (records,_)=try await http.call("api/dev/patients",cookie:patient)
        let rows=try XCTUnwrap(JSONSerialization.jsonObject(with:records) as? [[String:Any]])
        let id=try XCTUnwrap(rows.first?["id"] as? Int)
        let (grant,granted)=try await http.call("api/dev/patients/\(id)/chart-access",cookie:patient,body:["clinician_username":"native-consultant","expires_in_minutes":60,"purpose":"Synthetic native consent test","authorize_full_chart":true])
        XCTAssertEqual(granted.statusCode,200)
        let grantValue=try XCTUnwrap(JSONSerialization.jsonObject(with:grant) as? [String:Any])
        let clinician=try XCTUnwrap(grantValue["clinician_id"] as? Int)
        let doc=try await http.upload(patient: id,cookie:patient)
        let (_,privateDenied)=try await http.call("api/dev/documents/\(doc)/content",cookie:consultant);XCTAssertEqual(privateDenied.statusCode,404)
        let (_,allowed)=try await http.call("api/dev/patients/\(id)/document-consents",cookie:patient,body:["clinician_username":"native-consultant","granted":true]);XCTAssertEqual(allowed.statusCode,200)
        let (bytes,visible)=try await http.call("api/dev/documents/\(doc)/content",cookie:consultant);XCTAssertEqual(visible.statusCode,200);XCTAssertEqual(String(decoding:bytes,as:UTF8.self),"Synthetic protected native document.")
        let (_,revoked)=try await http.call("api/dev/patients/\(id)/document-consents",cookie:patient,body:["clinician_username":"native-consultant","granted":false]);XCTAssertEqual(revoked.statusCode,200)
        let (_,hidden)=try await http.call("api/dev/documents/\(doc)/content",cookie:consultant);XCTAssertEqual(hidden.statusCode,404)
        let (_,chartRevoked)=try await http.call("api/dev/patients/\(id)/chart-access/\(clinician)/revoke",cookie:patient,body:[:]);XCTAssertEqual(chartRevoked.statusCode,200)
        let (_,chartHidden)=try await http.call("api/dev/patients/\(id)/chart",cookie:consultant);XCTAssertEqual(chartHidden.statusCode,404)
        let patientLogout=try await http.logout(patient),consultantLogout=try await http.logout(consultant)
        XCTAssertEqual(patientLogout,200);XCTAssertEqual(consultantLogout,200)
    }
}

private final class FixtureHTTPS {
    private let origin=URL(string:"https://raphael-ci.test")!
    private let client:URLSession
    init() {let c=URLSessionConfiguration.ephemeral;c.httpCookieStorage=nil;c.httpShouldSetCookies=false;c.urlCache=nil;c.timeoutIntervalForRequest=15;c.timeoutIntervalForResource=20;client=URLSession(configuration:c,delegate:ContractRedirectPolicy(),delegateQueue:nil)}
    func close(){client.invalidateAndCancel()}
    func call(_ path:String,cookie:HTTPCookie?=nil,body:[String:Any]?=nil,control:String?=nil,multipart:Data?=nil,boundary:String?=nil) async throws -> (Data,HTTPURLResponse) {
        var request=URLRequest(url:origin.appendingPathComponent(path));request.setValue(origin.absoluteString,forHTTPHeaderField:"Origin")
        if let cookie {for (k,v) in HTTPCookie.requestHeaderFields(with:[cookie]){request.setValue(v,forHTTPHeaderField:k)}}
        if let control {request.setValue(control,forHTTPHeaderField:"X-Raphael-CI-Control")}
        if let body {request.httpMethod="POST";request.setValue("application/json",forHTTPHeaderField:"Content-Type");request.httpBody=try JSONSerialization.data(withJSONObject:body)}
        if let multipart,let boundary {request.httpMethod="POST";request.setValue("multipart/form-data; boundary="+boundary,forHTTPHeaderField:"Content-Type");request.httpBody=multipart}
        let (data,response)=try await client.data(for:request);return(data,try XCTUnwrap(response as? HTTPURLResponse))
    }
    func login(_ username:String,password:String,factor:String="") async throws -> HTTPCookie {
        let (_,response)=try await call("api/dev/login",body:["tenant":"native-ci","username":username,"password":password,"otp":factor]);XCTAssertEqual(response.statusCode,200)
        let headers=response.allHeaderFields.reduce(into:[String:String]()){out,pair in if let key=pair.key as? String,let value=pair.value as? String {out[key]=value}}
        return try XCTUnwrap(HTTPCookie.cookies(withResponseHeaderFields:headers,for:origin).first{$0.name=="raphael_dev_session"})
    }
    func logout(_ cookie:HTTPCookie) async throws -> Int {let (_,r)=try await call("api/dev/logout",cookie:cookie,body:[:]);return r.statusCode}
    func control(_ operation:String,token:String) async throws {let (_,r)=try await call("_ci/native-control",body:["operation":operation],control:token);XCTAssertEqual(r.statusCode,200)}
    func upload(patient:Int,cookie:HTTPCookie) async throws -> Int {
        let boundary="RaphaelSyntheticBoundary"
        let text="--\(boundary)\r\nContent-Disposition: form-data; name=\"title\"\r\n\r\nSynthetic native document\r\n--\(boundary)\r\nContent-Disposition: form-data; name=\"visibility\"\r\n\r\nprivate\r\n--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"synthetic.txt\"\r\nContent-Type: text/plain\r\n\r\nSynthetic protected native document.\r\n--\(boundary)--\r\n"
        let (data,r)=try await call("api/dev/patients/\(patient)/documents",cookie:cookie,multipart:Data(text.utf8),boundary:boundary);XCTAssertEqual(r.statusCode,200)
        return try XCTUnwrap((JSONSerialization.jsonObject(with:data) as? [String:Any])?["id"] as? Int)
    }
}

private final class ContractRedirectPolicy: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}


/// Selected only against the separately restarted actual candidate HTTPS server.
/// No mocked manifest, alternate app code path or production test endpoint.
final class ConsumerCandidateUITests: XCTestCase {
    func testActualCandidatePatientPreviewHidesClinicalControlsAndSignsOut() throws {
        continueAfterFailure = false
        XCTAssertFalse(nativeFixturePassword.hasPrefix("__CI_"), "Actual ephemeral HTTPS fixture credentials must be injected.")
        let app = XCUIApplication()
        app.launch()
        defer { app.terminate() }
        let organization = app.textFields["Development organization"]
        XCTAssertTrue(organization.waitForExistence(timeout:30))
        organization.tap(); organization.typeText("native-ci")
        let username = app.textFields["Username"]
        XCTAssertTrue(username.waitForExistence(timeout:10))
        username.tap(); username.typeText("native-patient")
        let password = app.secureTextFields["native-password"]
        XCTAssertTrue(password.waitForExistence(timeout:10))
        password.tap(); password.typeText(nativeFixturePassword)
        app.buttons["Sign in"].tap()
        XCTAssertTrue(app.staticTexts["Personal records preview"].waitForExistence(timeout:45))
        XCTAssertTrue(app.staticTexts["native-patient · native-ci"].waitForExistence(timeout:30))
        XCTAssertTrue(app.staticTexts["My linked records"].exists)
        XCTAssertTrue(app.staticTexts["Synthetic native approved note. No real clinical findings."].waitForExistence(timeout:20))
        let sections = app.buttons["Patient sections"]
        XCTAssertTrue(sections.waitForExistence(timeout:10))
        sections.coordinate(withNormalizedOffset:CGVector(dx:0.5,dy:0.5)).tap()
        XCTAssertTrue(app.buttons["Dashboard"].waitForExistence(timeout:10))
        for title in ["Medical record", "Emergency information", "Prepare a visit", "AI education", "Care messages", "Appointments", "Account and privacy"] {
            XCTAssertFalse(app.buttons[title].exists, "Unsupported native section must be absent: " + title)
        }
        app.buttons["Dashboard"].tap()
        let settings = app.buttons["Privacy and session settings"]
        XCTAssertTrue(settings.waitForExistence(timeout:10))
        settings.coordinate(withNormalizedOffset:CGVector(dx:0.5,dy:0.5)).tap()
        XCTAssertTrue(app.buttons["Account, records and feedback preview"].waitForExistence(timeout:10))
        XCTAssertFalse(app.buttons["native-mfa-manage"].exists)
        XCTAssertFalse(app.buttons["Account, export and deletion review"].exists)
        let logout = app.buttons["Sign out and clear local session"]
        XCTAssertTrue(logout.waitForExistence(timeout:10))
        logout.tap()
        let confirmation = app.buttons["native-confirm-sign-out"]
        XCTAssertTrue(confirmation.waitForExistence(timeout:10))
        confirmation.tap()
        XCTAssertTrue(app.staticTexts["Welcome to Raphael"].waitForExistence(timeout:30))
        let idle = expectation(for:NSPredicate(format:"exists == false"),evaluatedWith:app.staticTexts["native-authentication-status"])
        wait(for:[idle],timeout:20)
        XCTAssertFalse(app.staticTexts["native-authentication-problem"].exists)
        // The fixture cleanup independently queries the actual DB and fails
        // the job unless this native sign-out left zero active sessions.
    }
}
