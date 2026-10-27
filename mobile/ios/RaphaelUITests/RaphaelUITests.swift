import XCTest

final class RaphaelUITests: XCTestCase {
    func testAuthenticatedPatientAndClinicianUseActualHTTPSBackend() throws {
        let password = "__CI_SYNTHETIC_PASSWORD__"
        guard !password.hasPrefix("__CI_") else {
            throw XCTSkip("Requires ephemeral authenticated_ci.py backend;no authenticated native success implied.")
        }
        let app = XCUIApplication()
        app.launch()
        let web = app.webViews.firstMatch
        func login(_ username: String, workspace: String) {
            let organization = web.textFields["Development organization"]
            XCTAssertTrue(organization.waitForExistence(timeout: 30))
            organization.tap(); organization.typeText("native-ci")
            let user = web.textFields["Username"]
            user.tap(); user.typeText(username)
            let secure = web.secureTextFields["Password"]
            secure.tap(); secure.typeText(password)
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
        web.buttons["Sign out"].tap()
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
