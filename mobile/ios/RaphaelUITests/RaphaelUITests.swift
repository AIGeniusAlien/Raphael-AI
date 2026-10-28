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
        func login(_ username: String, workspace: String) {
            let organization = app.textFields["Development organization"]
            XCTAssertTrue(organization.waitForExistence(timeout: 30))
            organization.tap(); organization.typeText("native-ci")
            let user = app.textFields["Username"]
            user.tap(); user.typeText(username)
            let secure = app.secureTextFields["Password"]
            secure.tap(); secure.typeText(password)
            app.buttons["Sign in"].tap()
            XCTAssertTrue(app.staticTexts[username + " · " + workspace + " · native-ci"].waitForExistence(timeout: 20))
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
