import XCTest

final class RaphaelUITests: XCTestCase {
    private let fixturePassword = "__CI_SYNTHETIC_PASSWORD__"
    override func tearDownWithError() throws {
        let app = XCUIApplication()
        if app.state != .notRunning {
            let finalState = XCTAttachment(screenshot: app.screenshot())
            finalState.name = "Final synthetic native UI state;workflow result is separate"
            finalState.lifetime = .keepAlways
            add(finalState)
        }
        try super.tearDownWithError()
    }

    func testAuthenticatedPatientAndClinicianUseActualHTTPSBackend() throws {
        continueAfterFailure = false
        let password = fixturePassword
        guard !password.hasPrefix("__CI_") else {
            throw XCTSkip("Requires ephemeral authenticated_ci.py backend;no authenticated native success implied.")
        }
        let app = XCUIApplication()
        app.launch()
        func login(_ username: String, workspace: String) {
            let organization = app.textFields["Development organization"]
            XCTAssertTrue(organization.waitForExistence(timeout: 30))
            organization.tap(); organization.typeText("native-ci")
            XCTAssertTrue(organization.value as? String == "native-ci", "Organization must retain the exact synthetic input.")
            let user = app.textFields["Username"]
            user.tap(); user.typeText(username)
            XCTAssertTrue(user.value as? String == username, "Username must retain the exact synthetic input.")
            let secure = app.secureTextFields["Password"]
            secure.tap(); secure.typeText(password)
            app.buttons["Sign in"].tap()
            let reached = app.staticTexts[username + " · " + workspace + " · native-ci"].waitForExistence(timeout: 45)
            if !reached {
                let problem = app.staticTexts["native-authentication-problem"]
                let progress = app.staticTexts["native-authentication-status"]
                XCTFail(problem.exists ? problem.label : "Authenticated web identity did not appear. Native status: " + (progress.exists ? progress.label : "No in-flight request"))
            }
            XCTAssertTrue(app.staticTexts["Synthetic native approved note. No real clinical findings."].waitForExistence(timeout: 15))
        }
        login("native-patient", workspace: "Patient workspace")
        let patient = XCTAttachment(screenshot: app.screenshot())
        patient.name = "Authenticated synthetic patient over trusted HTTPS"
        patient.lifetime = .keepAlways; add(patient)
        app.buttons["Patient sections"].tap()
        app.buttons["Medical record"].tap()
        XCTAssertTrue(app.staticTexts["Synthetic native patient"].waitForExistence(timeout: 15))
        app.buttons["Patient sections"].tap()
        app.buttons["Dashboard"].tap()
        XCTAssertTrue(app.buttons["Sign out"].waitForExistence(timeout: 15))
        app.buttons["Privacy and session settings"].tap()
        let readiness = app.buttons["Check backend connection"]
        XCTAssertTrue(readiness.waitForExistence(timeout: 10))
        readiness.tap()
        let ready = NSPredicate(format: "label BEGINSWITH %@", "Synthetic backend database ready.")
        expectation(for: ready, evaluatedWith: app.staticTexts["native-readiness-status"])
        waitForExpectations(timeout: 15)
        app.buttons["Sign out and clear local session"].tap()
        app.buttons["Sign out"].tap()
        XCTAssertTrue(app.textFields["Username"].waitForExistence(timeout: 15))
        app.terminate(); app.launch()
        login("native-clinician", workspace: "Physician workspace")
        let clinician = XCTAttachment(screenshot: app.screenshot())
        clinician.name = "Authenticated synthetic clinician over trusted HTTPS"
        clinician.lifetime = .keepAlways; add(clinician)
        app.buttons["Sign out"].tap()
        XCTAssertTrue(app.textFields["Username"].waitForExistence(timeout: 15))
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
}

private final class ContractRedirectPolicy: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}
