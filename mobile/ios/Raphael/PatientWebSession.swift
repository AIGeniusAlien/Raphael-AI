import Foundation
import WebKit
import Combine
import UniformTypeIdentifiers
import UIKit

@MainActor
final class PatientWebSession: NSObject, ObservableObject, WKNavigationDelegate, WKUIDelegate {
    let configuration: ServerConfiguration
    @Published private(set) var webView: WKWebView
    @Published private(set) var identity: NativeIdentity?
    @Published private(set) var authenticating = false
    @Published private(set) var authenticationStatus = "Ready to sign in"
    @Published var authenticationProblem: String?
    @Published private(set) var recordChoices: [NativeRecordChoice] = []
    @Published private(set) var pendingRecordSection: PatientSection?
    private var navigationEpoch = 0
    private var recordAuthenticationEpoch = 0
    private var authenticationEpoch = 0
    private var foregroundProbeEpoch = 0
    private var workspaceNeedsRecovery = false
    private var recoveringWorkspace = false
    private static let recoveryProblem = "The secure web session stopped. Retry replaces this private workspace; unsaved changes may be lost."
    private var nativeCookie: HTTPCookie?
    @Published private(set) var nativeExportSummary: NativeExportSummary?
    @Published private(set) var nativeExportTicket: NativeExportTicket?
    @Published private(set) var nativeExportBusy = false
    @Published private(set) var nativeExportStatus = "No export prepared."
    private var nativeExportEpoch = 0
    private var nativeExportTask: Task<Void,Never>?
    @Published private(set) var loading = false
    @Published var problem: String?
    @Published var externalSource: URL?
    @Published private(set) var currentSection: PatientSection = .dashboard

    init(configuration: ServerConfiguration) {
        self.configuration = configuration
        self.webView = Self.makeWebView()
        super.init()
        configureWebView()
    }

    private static func makeWebView() -> WKWebView {
        let preferences = WKWebViewConfiguration()
        preferences.websiteDataStore = .nonPersistent()
        preferences.limitsNavigationsToAppBoundDomains = true
        preferences.allowsInlineMediaPlayback = false
        preferences.mediaTypesRequiringUserActionForPlayback = .all
        return WKWebView(frame: .zero, configuration: preferences)
    }

    private func configureWebView() {
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.allowsBackForwardNavigationGestures = true
        webView.isOpaque = false
        webView.scrollView.contentInsetAdjustmentBehavior = .automatic
        #if DEBUG
        webView.isInspectable = false
        #endif
    }

    /// Discard the private browsing context without awaiting an unmounted
    /// WebKit process. Server revocation is handled separately by signOut.
    func discardLocalWorkspace() {
        foregroundProbeEpoch += 1
        navigationEpoch += 1
        cancelRecordSelection()
        workspaceNeedsRecovery = false
        recoveringWorkspace = false
        webView.navigationDelegate = nil
        webView.uiDelegate = nil
        webView.stopLoading()
        webView = Self.makeWebView()
        configureWebView()
        loading = false
        problem = nil
        externalSource = nil
        currentSection = .dashboard
    }

    func open(_ section: PatientSection) {
        guard identity != nil else { return }
        finishConfirmation(false)
        navigationEpoch += 1
        cancelRecordSelection()
        currentSection = section
        if workspaceNeedsRecovery || recoveringWorkspace {
            if !recoveringWorkspace { problem = Self.recoveryProblem }
            return
        }
        problem = nil
        if section.requiresPatient {
            let navigation = navigationEpoch
            let authentication = authenticationEpoch
            Task { await resolveRecord(section, navigation: navigation, authentication: authentication) }
        } else { loadSection(section) }
    }
    private func loadSection(_ section: PatientSection, patientID: Int? = nil) {
        let destination = section == .intelligence && identity?.role == "clinician" ? configuration.origin.appendingPathComponent("static/intelligence.html") : configuration.url(for: section, patientID: patientID)
        webView.load(URLRequest(url: destination, cachePolicy: .reloadIgnoringLocalAndRemoteCacheData, timeoutInterval: 30))
    }
    private func resolveRecord(_ section: PatientSection, navigation: Int, authentication: Int) async {
        do {
            let (data, response) = try await request("api/dev/patients")
            guard navigation == navigationEpoch, authentication == authenticationEpoch, identity != nil else { return }
            if response.statusCode == 401 { await clearAuthentication(); authenticationProblem = "Your session ended. Sign in again."; return }
            guard response.statusCode == 200 else { throw URLError(.badServerResponse) }
            let records = try JSONDecoder().decode([NativeRecordChoice].self, from: data)
            guard records.count <= 100, records.allSatisfy({ $0.id > 0 }), Set(records.map(\.id)).count == records.count else { throw URLError(.badServerResponse) }
            if records.count == 1 { loadSection(section, patientID: records[0].id) }
            else if records.isEmpty { problem = "No authorized patient record is linked to this account." }
            else {
                recordChoices = records
                recordAuthenticationEpoch = authentication
                pendingRecordSection = section
            }
        } catch {
            guard navigation == navigationEpoch, authentication == authenticationEpoch else { return }
            problem = "Authorized records could not load. Retry or select a record through the dashboard."
        }
    }
    func chooseRecord(_ record: NativeRecordChoice) {
        guard let section = pendingRecordSection, recordAuthenticationEpoch == authenticationEpoch,
              recordChoices.contains(where: { $0.id == record.id }), identity != nil else { return }
        cancelRecordSelection()
        loadSection(section, patientID: record.id)
    }
    func cancelRecordSelection() {
        guard pendingRecordSection != nil || !recordChoices.isEmpty else { return }
        navigationEpoch += 1
        recordChoices = []
        pendingRecordSection = nil
    }

    func retry() {
        if workspaceNeedsRecovery { Task { await recoverWorkspace() } }
        else { open(currentSection) }
    }

    func suspendWorkspaceProbe() {
        cancelNativeExport()
        foregroundProbeEpoch += 1
        if recoveringWorkspace {
            recoveringWorkspace = false
            workspaceNeedsRecovery = true
            problem = Self.recoveryProblem
        }
    }

    /// Probe only after a foreground HTTP session check succeeded. JavaScript
    /// returns a boolean only;it reads no credential or patient text into native
    /// state and never changes the document. Do not await an unresponsive WK
    /// callback:the independent deadline always makes recovery visible.
    private func probeForegroundWorkspace() {
        guard identity != nil, !authenticating, !recoveringWorkspace else { return }
        foregroundProbeEpoch += 1
        let probe = foregroundProbeEpoch
        let auth = authenticationEpoch
        let navigation = navigationEpoch
        let view = webView
        let expression = "Boolean(document.body && document.body.childElementCount > 0 && document.body.innerText.trim().length > 0 && document.location.protocol === 'https:' && document.visibilityState === 'visible')"
        view.evaluateJavaScript(expression) { [weak self, weak view] value, error in
            Task { @MainActor in
                guard let self, let view, self.webView === view,
                      auth == self.authenticationEpoch, navigation == self.navigationEpoch,
                      probe == self.foregroundProbeEpoch, self.identity != nil else { return }
                if error == nil, (value as? Bool) == true {
                    self.foregroundProbeEpoch += 1
                    if self.problem == Self.recoveryProblem { self.workspaceNeedsRecovery = false; self.problem = nil }
                } else { self.markWorkspaceStopped() }
            }
        }
        Task { @MainActor [weak self, weak view] in
            do { try await Task.sleep(for:.seconds(8)) } catch { return }
            guard let self, let view, self.webView === view,
                  auth == self.authenticationEpoch, navigation == self.navigationEpoch,
                  probe == self.foregroundProbeEpoch, self.identity != nil else { return }
            self.markWorkspaceStopped()
        }
    }

    private func markWorkspaceStopped() {
        foregroundProbeEpoch += 1
        recoveringWorkspace = false
        workspaceNeedsRecovery = true
        loading = false
        problem = Self.recoveryProblem
    }

    /// The user explicitly chooses replacement after the visible lost-edits
    /// warning. Revalidate the same identity with the actual server first;then
    /// mount ONE fresh private WK context and restore its scoped HttpOnly cookie.
    /// A late callback cannot revive a revoked/changed/discarded session.
    private func recoverWorkspace() async {
        guard !recoveringWorkspace, !authenticating, let expected = identity else { return }
        foregroundProbeEpoch += 1
        let attempt = foregroundProbeEpoch
        recoveringWorkspace = true
        let auth = authenticationEpoch
        do {
            let (data, response) = try await request("api/dev/session")
            guard auth == authenticationEpoch, identity == expected,
                  attempt == foregroundProbeEpoch, recoveringWorkspace else { return }
            if response.statusCode == 401 { recoveringWorkspace = false; await clearAuthentication(); authenticationProblem = "Your session ended. Sign in again."; return }
            guard response.statusCode == 200 else { throw URLError(.cannotConnectToHost) }
            guard (try? JSONDecoder().decode(NativeIdentity.self,from:data)) == expected else {
                recoveringWorkspace = false; await clearAuthentication(); authenticationProblem = "Your session changed. Sign in again."; return
            }
            guard let cookie = nativeCookie, NativeIdentity.accepts(cookie,origin:configuration.origin) else {
                recoveringWorkspace = false; await clearAuthentication(); authenticationProblem = "Your session ended. Sign in again."; return
            }
            let section = currentSection
            discardLocalWorkspace()
            currentSection = section
            recoveringWorkspace = true
            problem = "Restoring the private workspace. Unsaved changes from the stopped view may be lost."
            let view = webView
            let probe = foregroundProbeEpoch
            view.configuration.websiteDataStore.httpCookieStore.setCookie(cookie) { [weak self, weak view] in
                Task { @MainActor in
                    guard let self, let view, self.webView === view,
                          self.authenticationEpoch == auth, self.identity == expected,
                          self.foregroundProbeEpoch == probe, self.recoveringWorkspace else { return }
                    self.recoveringWorkspace = false
                    self.problem = nil
                    self.open(self.currentSection)
                }
            }
            Task { @MainActor [weak self, weak view] in
                do { try await Task.sleep(for:.seconds(8)) } catch { return }
                guard let self, let view, self.webView === view,
                      self.authenticationEpoch == auth, self.foregroundProbeEpoch == probe,
                      self.recoveringWorkspace else { return }
                self.recoveringWorkspace = false
                self.markWorkspaceStopped()
            }
        } catch {
            guard auth == authenticationEpoch, identity == expected,
                  attempt == foregroundProbeEpoch, recoveringWorkspace else { return }
            recoveringWorkspace = false
            workspaceNeedsRecovery = true
            problem = "The secure web session stopped. Server verification is unavailable; retry when connected. Unsaved changes may be lost."
        }
    }

    func cancelNativeExport() {
        nativeExportEpoch += 1
        nativeExportTask?.cancel();nativeExportTask = nil
        nativeExportTicket?.invalidate();nativeExportTicket = nil
        nativeExportSummary = nil
        nativeExportBusy = false
        nativeExportStatus = "Prepared export cleared. Already copied or shared data is outside Raphael's control."
    }

    private func exportContext(_ expected: NativeIdentity, _ epoch: Int) throws {
        guard !Task.isCancelled, nativeExportEpoch == epoch, identity == expected,
              expected.role == "patient", !authenticating else { throw URLError(.cancelled) }
    }

    /// Every transfer fetches a new own-record snapshot and rechecks the same
    /// server session AFTER the bounded response. No stale preview is exported.
    private func authorizedNativeExport(_ expected: NativeIdentity, _ epoch: Int) async throws -> Data {
        try exportContext(expected,epoch)
        let (data,response) = try await request("api/dev/account/export",maximumBytes:NativePatientExport.maximumBytes)
        try exportContext(expected,epoch)
        if response.statusCode == 401 { await clearAuthentication();throw URLError(.userAuthenticationRequired) }
        guard response.statusCode == 200, response.mimeType == "application/json" else { throw URLError(.badServerResponse) }
        _ = try NativePatientExport.inspect(data,expected:expected)
        let (sessionData,sessionResponse) = try await request("api/dev/session")
        try exportContext(expected,epoch)
        guard sessionResponse.statusCode == 200,
              (try? JSONDecoder().decode(NativeIdentity.self,from:sessionData)) == expected else {
            if sessionResponse.statusCode == 401 || sessionResponse.statusCode == 200 { await clearAuthentication() }
            throw URLError(.userAuthenticationRequired)
        }
        return data
    }

    func prepareNativeExport() {
        guard let expected = identity, expected.role == "patient", !nativeExportBusy else { return }
        cancelNativeExport()
        let epoch = nativeExportEpoch
        nativeExportBusy = true;nativeExportStatus = "Preparing an authorized, bounded record snapshot…"
        nativeExportTask = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                let data = try await self.authorizedNativeExport(expected,epoch)
                let summary = try NativePatientExport.inspect(data,expected:expected)
                try self.exportContext(expected,epoch)
                self.nativeExportSummary = summary
                self.nativeExportBusy = false
                self.nativeExportStatus = "Review the scope before copying or sharing. Data is fetched again at transfer time."
            } catch {
                guard self.nativeExportEpoch == epoch else { return }
                self.nativeExportBusy = false;self.nativeExportSummary = nil
                self.nativeExportStatus = "Export unavailable. Nothing was copied or shared. Check your session and connection,then prepare again."
            }
        }
    }

    func copyNativeExport() {
        guard let expected = identity, expected.role == "patient", nativeExportSummary != nil, !nativeExportBusy else { return }
        let epoch = nativeExportEpoch
        nativeExportBusy = true;nativeExportStatus = "Rechecking authorization before copying…"
        nativeExportTask = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                let data = try await self.authorizedNativeExport(expected,epoch)
                try self.exportContext(expected,epoch)
                // An intentional export is a disclosure boundary. Do not erase
                // a user's later clipboard on logout. Ask the OS for local-only
                // delivery and60s expiry;copies made elsewhere cannot be recalled.
                UIPasteboard.general.setItems([[UTType.json.identifier:data,
                    UTType.utf8PlainText.identifier:String(decoding:data,as:UTF8.self)]],
                    options:[.localOnly:true,.expirationDate:Date().addingTimeInterval(60)])
                self.nativeExportSummary = nil;self.nativeExportBusy = false
                self.nativeExportStatus = "Sensitive JSON copied locally; clipboard expiry requested in 60 seconds. Copies made elsewhere cannot be recalled."
            } catch {
                guard self.nativeExportEpoch == epoch else { return }
                self.nativeExportSummary = nil;self.nativeExportBusy = false
                self.nativeExportStatus = "Export unavailable. Nothing was copied. Prepare again after checking your session."
            }
        }
    }

    func beginNativeExportSharing() {
        guard let expected = identity, expected.role == "patient", nativeExportSummary != nil, !nativeExportBusy else { return }
        nativeExportTicket?.invalidate()
        nativeExportTicket = NativeExportTicket(identity:expected,epoch:nativeExportEpoch)
        nativeExportStatus = "Choose a destination deliberately. Backgrounding or session loss cancels pending export access."
    }

    func dataForNativeExportSharing(_ ticket: NativeExportTicket) async throws -> Data {
        guard ticket.isValid else { throw URLError(.cancelled) }
        let data = try await authorizedNativeExport(ticket.identity,ticket.epoch)
        try exportContext(ticket.identity,ticket.epoch)
        guard ticket.isValid, nativeExportTicket?.id == ticket.id else { throw URLError(.cancelled) }
        return data
    }

    // One memory-only cookie store is shared explicitly by native HTTP and WebKit.
    // No password/token is evaluated as JavaScript or written to UserDefaults/disk.
    private func request(_ path: String, body: Data? = nil, maximumBytes: Int = 65536) async throws -> (Data, HTTPURLResponse) {
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
        let (bytes, response) = try await client.bytes(for: request)
        guard let http = response as? HTTPURLResponse,
              http.url?.host == self.configuration.origin.host else { throw URLError(.badServerResponse) }
        var data = Data()
        for try await byte in bytes {
            guard data.count < maximumBytes else { throw URLError(.dataLengthExceedsMaximum) }
            data.append(byte)
            if data.count % 16384 == 0 { try Task.checkCancellation();await Task.yield() }
        }
        return (data, http)
    }

    func signIn(credentials: Data) async {
        guard !authenticating else { return }
        authenticating = true
        authenticationStatus = "Connecting securely"
        authenticationProblem = nil
        authenticationEpoch += 1
        let epoch = authenticationEpoch
        defer { authenticating = false }
        do {
            let (data, response) = try await request("api/dev/login", body: credentials)
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
            // The native server session is already verified. Mount the single
            // workspace now so WebKit can complete its cookie-process handoff.
            identity = user
            authenticationStatus = "Opening workspace"
            await webView.configuration.websiteDataStore.httpCookieStore.setCookie(cookies[0])
            guard epoch == authenticationEpoch else { await clearAuthentication(); return }
            open(.dashboard)
        } catch {
            let code = (error as NSError).code
            authenticationProblem = "The secure service could not complete sign-in (error \(code)). Check your connection and service configuration, then try again."
        }
    }

    func checkAuthentication(restoringWorkspace: Bool = false) async {
        guard identity != nil, !authenticating else { return }
        // Enforce local expiry even when the network cannot confirm it.
        guard let cookie = nativeCookie, NativeIdentity.accepts(cookie, origin: configuration.origin) else {
            await clearAuthentication()
            authenticationProblem = "Your session ended. Sign in again."
            return
        }
        let epoch = authenticationEpoch
        do {
            let (data, response) = try await request("api/dev/session")
            guard epoch == authenticationEpoch, !authenticating else { return }
            if response.statusCode == 401 {
                await clearAuthentication()
                authenticationProblem = "Your session ended. Sign in again."
            } else if response.statusCode != 200 {
                problem = "Session verification unavailable. Check your connection and retry."
            } else {
                guard (try? JSONDecoder().decode(NativeIdentity.self,from:data)) == identity else {
                    await clearAuthentication(); authenticationProblem = "Your session changed. Sign in again."; return
                }
                if problem == "Session verification unavailable. Check your connection and retry." { problem = nil }
                if restoringWorkspace { probeForegroundWorkspace() }
            }
        } catch {
            guard !Task.isCancelled, epoch == authenticationEpoch, !authenticating else { return }
            // A transient outage is not a successful logout or proof of expiry.
            problem = "Session verification unavailable. Check your connection and retry."
        }
    }

    private func clearAuthentication() async {
        cancelNativeExport()
        authenticationEpoch += 1
        let epoch = authenticationEpoch
        let alreadyAuthenticating = authenticating
        authenticating = true
        authenticationStatus = "Clearing local session"
        defer { if epoch == authenticationEpoch { authenticating = alreadyAuthenticating } }
        identity = nil
        cancelRecordSelection()
        nativeCookie = nil
        finishConfirmation(false)
        discardLocalWorkspace()
    }

    func signOut() {
        guard !authenticating else { return }
        cancelNativeExport()
        authenticationEpoch += 1
        authenticating = true
        authenticationStatus = "Signing out securely"
        cancelRecordSelection()
        identity = nil // Hide records before awaiting potentially unavailable networking.
        Task { @MainActor [weak self] in
            guard let self else { return }
            defer { self.authenticating = false }
            let confirmed: Bool
            do {
                let (_, response) = try await self.request("api/dev/logout", body: Data("{}".utf8))
                confirmed = response.statusCode == 200 || response.statusCode == 401
            } catch { confirmed = false }
            await self.clearAuthentication()
            self.authenticationProblem = confirmed ? nil : "Local session cleared. Server revocation could not be confirmed. Review active sessions after signing in again."
        }
    }

    func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
        guard webView === self.webView else { return }
        loading = true
        if !workspaceNeedsRecovery && !recoveringWorkspace { problem = nil }
    }
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        guard webView === self.webView else { return }
        loading = false
        Task { await checkAuthentication(restoringWorkspace:workspaceNeedsRecovery) }
    }
    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        guard webView === self.webView else { return }; failed(error)
    }
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        guard webView === self.webView else { return }; failed(error)
    }
    private func failed(_ error: Error) {
        loading = false
        if (error as NSError).code == NSURLErrorCancelled { return }
        // Do not reflect failing URLs,chart contents or headers into native logs/error UI.
        problem = "This section could not load. Check your connection and try again. No records are available offline."
    }
    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        guard webView === self.webView else { return }
        // This is the already-stopped old context while the user's recovery
        // request is being verified. Its late notification must not cancel the
        // new attempt. A death during the fresh cookie handoff still fails closed.
        if recoveringWorkspace && workspaceNeedsRecovery { return }
        loading = false
        markWorkspaceStopped()
    }

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard let url = action.request.url else { decisionHandler(.cancel); return }
        // about:blank is permitted only for WebKit's initial empty document.
        if url.absoluteString == "about:blank", webView.url == nil { decisionHandler(.allow); return }
        switch NavigationPolicy(origin: configuration.origin).decision(for: url) {
        case .sameOrigin:
            if action.shouldPerformDownload { problem = "Arbitrary downloads are unavailable. Use Protected record export in native settings for an explicit own-record JSON export;individual file downloads remain disabled."; decisionHandler(.cancel) }
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

struct NativeCredentials: Encodable {
    let tenant: String
    let username: String
    let password: String
    let otp: String
}

struct NativeRecordChoice: Decodable, Identifiable {
    let id: Int
    let name: String
}


struct NativeExportSummary {
    let records: Int
    let approvedNotes: Int
    let byteCount: Int
    let recordNames: [String]
    let generatedAt: String
}

enum NativePatientExport {
    static let maximumBytes = 5 * 1024 * 1024
    static func inspect(_ data: Data, expected: NativeIdentity) throws -> NativeExportSummary {
        guard expected.role == "patient", !data.isEmpty, data.count <= maximumBytes,
              let root = try JSONSerialization.jsonObject(with:data) as? [String:Any],
              root["format"] as? String == "raphael-synthetic-patient-export-v1",
              let account = root["account"] as? [String:Any],
              let generated = root["generated_at"] as? String, !generated.isEmpty,
              let patients = root["patients"] as? [[String:Any]], patients.count <= 500 else { throw URLError(.cannotParseResponse) }
        let identity = try JSONDecoder().decode(NativeIdentity.self,from:JSONSerialization.data(withJSONObject:account))
        guard identity == expected else { throw URLError(.userAuthenticationRequired) }
        let collections = ["entries","approved_notes","appointments","documents","shared_messages","patient_reports","intakes","approved_summaries"]
        for key in collections {
            guard let rows = root[key] as? [[String:Any]], rows.count <= 500 else { throw URLError(.cannotParseResponse) }
        }
        let names = try patients.map { row -> String in
            guard let id = row["id"] as? Int, id > 0, let name = row["name"] as? String, !name.isEmpty else { throw URLError(.cannotParseResponse) }
            return name
        }
        // Defense against accidentally returning credential tables or drafts.
        func credentials(_ value:Any,depth:Int = 0) -> Bool {
            if depth > 32 { return true }
            if let object = value as? [String:Any] {
                let forbidden:Set<String> = ["password","password_hash","token_hash","encrypted_secret","mfa_secret","recovery_codes","access_token","refresh_token","session_token","api_key"]
                return object.contains { forbidden.contains($0.key.lowercased()) || credentials($0.value,depth:depth+1) }
            }
            if let array = value as? [Any] { return array.contains { credentials($0,depth:depth+1) } }
            return false
        }
        guard let entries = root["entries"] as? [[String:Any]], let notes = root["approved_notes"] as? [[String:Any]],
              !credentials(root), !entries.contains(where:{$0["status"] as? String == "draft"}) else { throw URLError(.cannotParseResponse) }
        for note in notes {
            guard let approved = note["approved_by"] as? Int, approved > 0,
                  let at = note["approved_at"] as? String, !at.isEmpty else { throw URLError(.cannotParseResponse) }
        }
        return NativeExportSummary(records:patients.count,approvedNotes:notes.count,byteCount:data.count,recordNames:Array(names.prefix(10)),generatedAt:generated)
    }
}

/// A short-lived revocable capability,not cached patient bytes. Item-provider
/// requests fetch fresh server-authorized data;invalidation prevents late delivery.
final class NativeExportTicket: Identifiable, @unchecked Sendable {
    let id = UUID()
    let identity: NativeIdentity
    let epoch: Int
    private let expires = Date().addingTimeInterval(60)
    private let lock = NSLock()
    private var valid = true
    init(identity:NativeIdentity,epoch:Int) { self.identity = identity;self.epoch = epoch }
    var isValid:Bool { lock.lock();defer{lock.unlock()};return valid && Date() < expires }
    func invalidate() { lock.lock();valid = false;lock.unlock() }
}
