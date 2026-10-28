import XCTest

final class RaphaelUITests: XCTestCase {
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
        let password = "__CI_SYNTHETIC_PASSWORD__"
        guard !password.hasPrefix("__CI_") else {
            throw XCTSkip("Requires ephemeral authenticated_ci.py backend;no authenticated native success implied.")
        }
        let app = XCUIApplication()
        app.launch()
        let web = app.webViews.firstMatch
        func scrollForm() {
            web.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.7))
                .press(forDuration: 0.1, thenDragTo: web.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.2)))
        }
        func login(_ username: String, workspace: String) {
            let organization = web.textFields["Development organization"]
            XCTAssertTrue(organization.waitForExistence(timeout: 30))
            let privacyCover = app.staticTexts["Private workspace hidden"].exists
            print("NATIVE_UI_GEOMETRY web=\(web.frame) organization=\(organization.frame) field_hittable=\(organization.isHittable) privacy_cover=\(privacyCover)")
            // Scroll the real WKWebView before focusing controls. WebKit's
            // accessibility activation point can be invalid even when the
            // field frame is on screen;coordinate taps are actual UI gestures.
            scrollForm()
            organization.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
            organization.typeText("native-ci")
            let user = web.textFields["Username"]
            if !user.isHittable { scrollForm() }
            user.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
            user.typeText(username)
            let secure = web.secureTextFields["Password"]
            if !secure.isHittable { scrollForm() }
            secure.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
            secure.typeText(password)
            web.buttons["Sign in"].tap()
            XCTAssertTrue(web.staticTexts[username + " · " + workspace + " · native-ci"].waitForExistence(timeout: 20))
            XCTAssertTrue(web.staticTexts["Synthetic native approved note. No real clinical findings."].waitForExistence(timeout: 15))
        }
        login("native-patient", workspace: "Patient workspace")
        let patient = XCTAttachment(screenshot: app.screenshot())
        patient.name = "Authenticated synthetic patient over trusted HTTPS"
        patient.lifetime = .keepAlways; add(patient)
        app.buttons["Patient sections"].tap()
        app.buttons["Medical record"].tap()
        XCTAssertTrue(web.staticTexts["Synthetic native patient"].waitForExistence(timeout: 15))
        app.buttons["Patient sections"].tap()
        app.buttons["Dashboard"].tap()
        XCTAssertTrue(web.buttons["Sign out"].waitForExistence(timeout: 15))
        app.buttons["Privacy and session settings"].tap()
        let readiness = app.buttons["Check backend connection"]
        XCTAssertTrue(readiness.waitForExistence(timeout: 10))
        readiness.tap()
        let ready = NSPredicate(format: "label BEGINSWITH %@", "Synthetic backend database ready.")
        expectation(for: ready, evaluatedWith: app.staticTexts["native-readiness-status"])
        waitForExpectations(timeout: 15)
        app.buttons["Sign out and clear local session"].tap()
        app.buttons["Sign out"].tap()
        XCTAssertTrue(web.textFields["Username"].waitForExistence(timeout: 15))
        app.terminate(); app.launch()
        login("native-clinician", workspace: "Physician workspace")
        let clinician = XCTAttachment(screenshot: app.screenshot())
        clinician.name = "Authenticated synthetic clinician over trusted HTTPS"
        clinician.lifetime = .keepAlways; add(clinician)
        web.buttons["Sign out"].tap()
        XCTAssertTrue(web.textFields["Username"].waitForExistence(timeout: 15))
        app.terminate(); app.launch()
        XCTAssertTrue(app.webViews.firstMatch.textFields["Username"].waitForExistence(timeout: 30))
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
