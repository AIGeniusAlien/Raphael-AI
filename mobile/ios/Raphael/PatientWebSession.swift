import Foundation
import WebKit
import Combine

@MainActor
final class PatientWebSession: NSObject, ObservableObject, WKNavigationDelegate, WKUIDelegate {
    let configuration: ServerConfiguration
    let webView: WKWebView
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
        let request = URLRequest(url: configuration.url(for: section), cachePolicy: .reloadIgnoringLocalAndRemoteCacheData, timeoutInterval: 30)
        webView.load(request)
    }

    func retry() { open(currentSection) }

    func signOut() {
        finishConfirmation(false)
        // JavaScript uses HttpOnly cookies through same-origin fetch;native code never reads tokens.
        let script = "const r = await fetch('/api/dev/logout', {method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:'{}'}); return r.ok;"
        Task { @MainActor [weak self] in
            guard let self else { return }
            let remotelyRevoked: Bool
            do {
                let value = try await WebKitScriptBridge.call(script, in: self.webView)
                remotelyRevoked = value as? Bool == true
            } catch {
                remotelyRevoked = false
            }
            self.webView.stopLoading()
            self.webView.configuration.websiteDataStore.removeData(ofTypes: WKWebsiteDataStore.allWebsiteDataTypes(), modifiedSince: .distantPast) {
                Task { @MainActor in
                    self.open(.dashboard)
                    if !remotelyRevoked {
                        self.problem = "Local session cleared. The server could not confirm revocation. Review active sessions after signing in again."
                    }
                }
            }
        }
    }

    func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) { loading = true; problem = nil }
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) { loading = false }
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
