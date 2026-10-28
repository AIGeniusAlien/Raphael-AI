import Foundation
import WebKit
import Combine

@MainActor
final class PatientWebSession: NSObject, ObservableObject, WKNavigationDelegate, WKUIDelegate {
    let configuration: ServerConfiguration
    let webView: WKWebView
    @Published private(set) var identity: NativeIdentity?
    @Published private(set) var authenticating = false
    @Published private(set) var authenticationStatus = "Ready to sign in"
    @Published var authenticationProblem: String?
    private var authenticationEpoch = 0
    private var nativeCookie: HTTPCookie?
    @Published private(set) var loading = false
    @Published var problem: String?
    @Published var externalSource: URL?
    @Published private(set) var currentSection: PatientSection = .dashboard

    init(configuration: ServerConfiguration) {
        self.configuration = configuration
        let preferences = WKWebViewConfiguration()
        preferences.websiteDataStore = .nonPersistent()
        preferences.limitsNavigationsToAppBoundDomains = true
        preferences.allowsInlineMediaPlayback = false
        preferences.mediaTypesRequiringUserActionForPlayback = .all
        self.webView = WKWebView(frame: .zero, configuration: preferences)
        super.init()
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.allowsBackForwardNavigationGestures = true
        webView.isOpaque = false
        webView.scrollView.contentInsetAdjustmentBehavior = .automatic
        #if DEBUG
        webView.isInspectable = false
        #endif
    }

    func open(_ section: PatientSection) {
        finishConfirmation(false)
        currentSection = section
        problem = nil
        let destination = section == .intelligence && identity?.role == "clinician" ? configuration.origin.appendingPathComponent("static/intelligence.html") : configuration.url(for: section)
        let request = URLRequest(url: destination, cachePolicy: .reloadIgnoringLocalAndRemoteCacheData, timeoutInterval: 30)
        webView.load(request)
    }

    func retry() { open(currentSection) }

    // One memory-only cookie store is shared explicitly by native HTTP and WebKit.
    // No password/token is evaluated as JavaScript or written to UserDefaults/disk.
    private func request(_ path: String, body: Data? = nil) async throws -> (Data, HTTPURLResponse) {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpShouldSetCookies = false
        configuration.httpCookieStorage = nil
        configuration.urlCache = nil
        configuration.timeoutIntervalForRequest = 15
        configuration.timeoutIntervalForResource = 20
        let client = URLSession(configuration: configuration, delegate: ReadinessRedirectPolicy(), delegateQueue: nil)
        defer { client.invalidateAndCancel() }
        var request = URLRequest(url: self.configuration.origin.appendingPathComponent(path))
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue(self.configuration.origin.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/")), forHTTPHeaderField: "Origin")
        // Native authentication must not await a not-yet-mounted WebKit process.
        // This private cookie is memory-only and scoped identically to WebKit.
        let scoped = nativeCookie.map { NativeIdentity.accepts($0, origin: self.configuration.origin) ? [$0] : [] } ?? []
        for (key, value) in HTTPCookie.requestHeaderFields(with: scoped) { request.setValue(value, forHTTPHeaderField: key) }
        if let body {
            request.httpMethod = "POST"
            request.httpBody = body
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        let (data, response) = try await client.data(for: request)
        guard let http = response as? HTTPURLResponse,
              http.url?.host == self.configuration.origin.host,
              data.count <= 65536 else { throw URLError(.badServerResponse) }
        return (data, http)
    }

    func signIn(tenant: String, username: String, password: String, otp: String = "") async {
        guard !authenticating else { return }
        authenticating = true
        authenticationStatus = "Connecting securely"
        authenticationProblem = nil
        authenticationEpoch += 1
        let epoch = authenticationEpoch
        defer { authenticating = false }
        do {
            let body = try JSONSerialization.data(withJSONObject: ["tenant": tenant, "username": username, "password": password, "otp": otp])
            let (data, response) = try await request("api/dev/login", body: body)
            guard epoch == authenticationEpoch else { return }
            guard response.statusCode == 200 else {
                authenticationProblem = response.statusCode == 429 ? "Too many sign-in attempts. Wait before trying again." : "Sign-in was not accepted (HTTP \(response.statusCode)). Check your organization and credentials."
                return
            }
            let user = try JSONDecoder().decode(NativeIdentity.self, from: data)
            guard ["patient", "clinician", "org_admin"].contains(user.role) else { throw URLError(.badServerResponse) }
            let headers = response.allHeaderFields.reduce(into: [String:String]()) { result, pair in
                if let key = pair.key as? String, let value = pair.value as? String { result[key] = value }
            }
            let cookies = HTTPCookie.cookies(withResponseHeaderFields: headers, for: configuration.origin)
                .filter { NativeIdentity.accepts($0, origin: configuration.origin) }
            guard cookies.count == 1 else { throw URLError(.badServerResponse) }
            authenticationStatus = "Verifying your session"
            nativeCookie = cookies[0]
            let (sessionData, sessionResponse) = try await request("api/dev/session")
            guard sessionResponse.statusCode == 200,
                  try JSONDecoder().decode(NativeIdentity.self, from: sessionData) == user else { throw URLError(.userAuthenticationRequired) }
            guard epoch == authenticationEpoch else { await clearAuthentication(); return }
            authenticationStatus = "Opening workspace"
            await webView.configuration.websiteDataStore.httpCookieStore.setCookie(cookies[0])
            guard epoch == authenticationEpoch else { await clearAuthentication(); return }
            identity = user
            open(.dashboard)
        } catch {
            let code = (error as NSError).code
            authenticationProblem = "The secure service could not complete sign-in (error \(code)). Check your connection and service configuration, then try again."
        }
    }

    func checkAuthentication() async {
        guard identity != nil, !authenticating else { return }
        let epoch = authenticationEpoch
        do {
            let (_, response) = try await request("api/dev/session")
            guard epoch == authenticationEpoch else { return }
            if response.statusCode == 401 {
                await clearAuthentication()
                authenticationProblem = "Your session ended. Sign in again."
            }
        } catch {
            // A transient outage is not a successful logout or proof of expiry.
            problem = "Session verification unavailable. Check your connection and retry."
        }
    }

    private func clearAuthentication() async {
        authenticationEpoch += 1
        identity = nil
        nativeCookie = nil
        finishConfirmation(false)
        webView.stopLoading()
        await webView.configuration.websiteDataStore.removeData(ofTypes: WKWebsiteDataStore.allWebsiteDataTypes(), modifiedSince: .distantPast)
        webView.loadHTMLString("<html><body></body></html>", baseURL: nil)
    }

    func signOut() {
        Task { @MainActor [weak self] in
            guard let self else { return }
            let confirmed: Bool
            do {
                let (_, response) = try await self.request("api/dev/logout", body: Data("{}".utf8))
                confirmed = response.statusCode == 200 || response.statusCode == 401
            } catch { confirmed = false }
            await self.clearAuthentication()
            self.authenticationProblem = confirmed ? nil : "Local session cleared. Server revocation could not be confirmed. Review active sessions after signing in again."
        }
    }

    func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) { loading = true; problem = nil }
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) { loading = false; Task { await checkAuthentication() } }
    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) { failed(error) }
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) { failed(error) }
    private func failed(_ error: Error) {
        loading = false
        if (error as NSError).code == NSURLErrorCancelled { return }
        // Do not reflect failing URLs,chart contents or headers into native logs/error UI.
        problem = "This section could not load. Check your connection and try again. No records are available offline."
    }
    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        loading = false
        problem = "The secure web session stopped. Retry and sign in again if needed. Unsaved changes may have been lost."
    }

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard let url = action.request.url else { decisionHandler(.cancel); return }
        // about:blank is permitted only for WebKit's initial empty document.
        if url.absoluteString == "about:blank", webView.url == nil { decisionHandler(.allow); return }
        switch NavigationPolicy(origin: configuration.origin).decision(for: url) {
        case .sameOrigin:
            if action.shouldPerformDownload { problem = "Downloads are unavailable in the native development app. Use the protected web application for an explicit export."; decisionHandler(.cancel) }
            else { decisionHandler(.allow) }
        case .externalCDC:
            externalSource = url
            decisionHandler(.cancel)
        case .blocked:
            problem = "This destination is outside the configured Raphael service. Navigation was blocked."
            decisionHandler(.cancel)
        }
    }
    func webView(_ webView: WKWebView, decidePolicyFor response: WKNavigationResponse, decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void) {
        if let http = response.response as? HTTPURLResponse, http.statusCode >= 400 {
            loading = false
            problem = http.statusCode == 404 ? "This section is not available on the configured server version." : "The server could not complete this request. Sign in again or retry."
            decisionHandler(.cancel); return
        }
        guard response.canShowMIMEType else {
            problem = "This file cannot be displayed safely in the development app. No file was saved."
            decisionHandler(.cancel); return
        }
        decisionHandler(.allow)
    }
    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration, for action: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let url = action.request.url {
            switch NavigationPolicy(origin: self.configuration.origin).decision(for: url) {
            case .sameOrigin: webView.load(action.request)
            case .externalCDC: externalSource = url
            case .blocked: problem = "External navigation blocked."
            }
        }
        return nil
    }
    func webView(_ webView: WKWebView, requestMediaCapturePermissionFor origin: WKSecurityOrigin, initiatedByFrame frame: WKFrameInfo, type: WKMediaCaptureType, decisionHandler: @escaping (WKPermissionDecision) -> Void) {
        // Device audio is a release gate,not an unreviewed capability enabled by this wrapper.
        decisionHandler(.deny)
        problem = "Microphone capture is disabled in this native development build. Manual synthetic entry remains available on supported pages."
    }
    func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping () -> Void) {
        pendingAlertCompletion = completionHandler
        confirmationIsAlert = true
        confirmationMessage = message
    }
    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
        // Confirmation requests are bridged by the SwiftUI owner,so clinical signoff/consent is never auto-accepted.
        confirmationIsAlert = false
        pendingConfirmation = (message, completionHandler)
        confirmationMessage = message
    }
    @Published var confirmationMessage: String?
    @Published var confirmationIsAlert = false
    private var pendingAlertCompletion: (() -> Void)?
    private var pendingConfirmation: (String, (Bool) -> Void)?
    func finishConfirmation(_ accepted: Bool) {
        let callback = pendingConfirmation?.1
        let alertCallback = pendingAlertCompletion
        pendingAlertCompletion = nil
        pendingConfirmation = nil
        confirmationMessage = nil
        callback?(accepted)
        alertCallback?()
    }
}


/// The public Objective-C selector handles promises without the optional Swift
/// overlay absent from the observed Xcode16.4/iOS18.5 simulator runtime.
/// __callAsyncJavaScript is the Swift importer name for the PUBLIC refined
/// callAsyncJavaScript:arguments:inFrame:inContentWorld:completionHandler: API.
@MainActor
enum WebKitScriptBridge {
    static func call(_ body: String, in view: WKWebView) async throws -> Any? {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Any?, Error>) in
            view.__callAsyncJavaScript(body, arguments: [:], inFrame: nil, in: .page) { value, error in
                if let error { continuation.resume(throwing: error) }
                else { continuation.resume(returning: value) }
            }
        }
    }
}

struct NativeIdentity: Decodable, Equatable {
    let id: Int
    let tenant: String
    let username: String
    let role: String
    static func accepts(_ cookie: HTTPCookie, origin: URL, now: Date = Date()) -> Bool {
        cookie.name == "raphael_dev_session" && cookie.domain.lowercased() == origin.host?.lowercased()
            && cookie.path == "/api/dev" && cookie.isSecure && cookie.isHTTPOnly
            && (cookie.expiresDate.map { $0 > now } ?? false)
    }
}
