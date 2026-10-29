import SwiftUI
import UIKit
import WebKit
import SafariServices

@main
struct RaphaelApp: App {
    var body: some Scene {
        WindowGroup {
            if let configuration = ServerConfiguration.bundled {
                PatientApplication(configuration: configuration)
            } else {
                ConfigurationRequiredView()
            }
        }
    }
}

struct ConfigurationRequiredView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "cross.case.fill").font(.largeTitle).accessibilityHidden(true)
            Text("Raphael").font(.largeTitle.bold())
            Text("Server configuration required").font(.title2).accessibilityIdentifier("configuration-required")
            Text("This development app needs an operator-configured HTTPS Raphael service. No connection has been made. Set the deployment URL and bound hostname in the Xcode configuration, then rebuild.")
                .multilineTextAlignment(.center)
            Text("Synthetic development only. Not an emergency service.").font(.footnote)
        }
        .padding(28)
    }
}

struct PatientApplication: View {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var session: PatientWebSession
    @State private var settings = false
    @State private var sourceToOpen: URL?
    @State private var sourceSheet = false

    init(configuration: ServerConfiguration) {
        _session = StateObject(wrappedValue: PatientWebSession(configuration: configuration))
    }

    var body: some View {
        ZStack {
            if session.identity == nil {
                NativeSignInView(session: session).privacySensitive().accessibilityHidden(scenePhase != .active)
            } else {
            VStack(spacing: 0) {
                // Sections replace the one shared WKWebView;there is no pushed
                // view hierarchy. A stable native header avoids duplicate UIKit
                // navigation-bar ownership during configured launch on iOS18.5.
                HStack {
                    Menu {
                        ForEach(session.identity?.role == "clinician" ? [.dashboard, .record, .intelligence, .messages, .appointments, .account] : PatientSection.allCases) { section in
                            Button(section.title, systemImage: section.symbol) { session.open(section) }
                        }
                    } label: { Label("Sections", systemImage: "line.3.horizontal") }
                    .accessibilityLabel(session.identity?.role == "clinician" ? "Clinician sections" : "Patient sections")
                    Spacer()
                    Text(session.currentSection.title).font(.headline).accessibilityAddTraits(.isHeader)
                    Spacer()
                    Button("Settings", systemImage: "gearshape") { settings = true }
                        .labelStyle(.iconOnly)
                        .accessibilityLabel("Privacy and session settings")
                }
                .padding(.horizontal).padding(.vertical, 8)
                Divider()
                Text("Synthetic development • Not an emergency channel")
                    .font(.footnote).padding(8).frame(maxWidth: .infinity)
                    .background(Color.yellow.opacity(0.18))
                    .accessibilityIdentifier("synthetic-notice")
                if session.authenticating {
                    HStack { ProgressView(); Text(session.authenticationStatus).accessibilityIdentifier("native-authentication-status") }.padding(8)
                }
                if session.loading {
                    ProgressView("Loading secure workspace").padding(8)
                }
                if let problem = session.problem {
                    VStack(spacing: 8) {
                        Text(problem).font(.callout).accessibilityAddTraits(.updatesFrequently)
                        Button("Retry this section") { session.retry() }
                    }
                    .padding().frame(maxWidth: .infinity).background(Color.orange.opacity(0.15))
                }
                PatientWebView(webView: session.webView)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .accessibilityHidden(session.authenticating)
                    .accessibilityIdentifier("patient-web-workspace")
            }
            .privacySensitive()
            .accessibilityHidden(scenePhase != .active)
            }
            if scenePhase != .active {
                Color(.systemBackground).ignoresSafeArea()
                VStack(spacing: 12) {
                    Image(systemName: "lock.shield").font(.largeTitle)
                    Text("Raphael").font(.title.bold())
                    Text("Private workspace hidden").font(.callout)
                }
                .accessibilityElement(children: .combine)
            }
        }
        .onAppear { if session.identity != nil && session.webView.url == nil { session.open(.dashboard) } }
        .task {
            while !Task.isCancelled {
                if scenePhase == .active { await session.checkAuthentication() }
                do { try await Task.sleep(for: .seconds(15)) } catch { break }
            }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active { session.finishConfirmation(false) }
            else { Task { await session.checkAuthentication() } }
        }
        .sheet(isPresented: $settings) {
            SettingsView(session: session) { settings = false }
        }
        .confirmationDialog("Open this CDC source in the system browser?", isPresented: Binding(
            get: { session.externalSource != nil },
            set: { if !$0 { session.externalSource = nil } }
        ), titleVisibility: .visible) {
            Button("Open CDC source") {
                sourceToOpen = session.externalSource
                session.externalSource = nil
                sourceSheet = true
            }
            Button("Cancel", role: .cancel) { session.externalSource = nil }
        } message: {
            Text("Only the public source link is opened. Your Raphael session is not shared with the source site.")
        }
        .sheet(isPresented: $sourceSheet) {
            if let url = sourceToOpen { PublicSourceBrowser(url: url).ignoresSafeArea() }
        }
        .alert(session.confirmationIsAlert ? "Raphael notice" : "Confirm Raphael action", isPresented: Binding(
            get: { session.confirmationMessage != nil },
            set: { if !$0 { session.finishConfirmation(false) } }
        )) {
            if session.confirmationIsAlert {
                Button("OK") { session.finishConfirmation(true) }
            } else {
                Button("Confirm") { session.finishConfirmation(true) }
                Button("Cancel", role: .cancel) { session.finishConfirmation(false) }
            }
        } message: { Text(session.confirmationMessage ?? "") }
    }
}

struct PatientWebView: UIViewRepresentable {
    let webView: WKWebView
    func makeUIView(context: Context) -> WKWebView { webView }
    func updateUIView(_ uiView: WKWebView, context: Context) {}
}

struct PublicSourceBrowser: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> SFSafariViewController { SFSafariViewController(url: url) }
    func updateUIViewController(_ uiViewController: SFSafariViewController, context: Context) {}
}

struct NativeSignInView: View {
    @ObservedObject var session: PatientWebSession
    @State private var tenant = ""
    @State private var username = ""
    @State private var password = ""
    @State private var otp = ""
    @State private var showFactor = false
    var body: some View {
        Form {
                Section {
                    Image("BrandMark").resizable().scaledToFit().frame(height: 64).accessibilityHidden(true)
                    Text("Welcome to Raphael").font(.title.bold())
                    Text("Sign in to your authorized synthetic workspace. Not an emergency service.").font(.callout)
                }
                Section("Secure sign-in") {
                    TextField("Development organization", text: $tenant).textContentType(.organizationName)
                        .textInputAutocapitalization(.never).autocorrectionDisabled()
                    TextField("Username", text: $username).textContentType(.username)
                        .textInputAutocapitalization(.never).autocorrectionDisabled()
                    NativeSecretEntry(text: $password, title: "Password").id("password-entry").frame(minHeight: 44)
                    Button(showFactor ? "Hide second-factor entry" : "Use an authenticator or recovery code") {
                        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                        showFactor.toggle()
                        if !showFactor { otp = "" }
                    }.font(.callout)
                    if showFactor {
                        NativeSecretEntry(text: $otp, title: "Authenticator or recovery code", oneTimeCode: true).id("otp-entry").frame(minHeight: 44)
                    }
                    Button(session.authenticating ? "Signing in" : "Sign in") {
                        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                        guard let captured = try? JSONEncoder().encode(NativeCredentials(tenant: tenant, username: username, password: password, otp: showFactor ? otp : "")) else {
                            session.authenticationProblem = "Sign-in could not be prepared. Check the entered fields."
                            return
                        }
                        // Own the complete request bytes BEFORE clearing any UI binding.
                        otp = ""
                        password = ""
                        Task { await session.signIn(credentials: captured) }
                    }.disabled(session.authenticating || tenant.isEmpty || username.isEmpty || password.isEmpty)
                    if session.authenticating { HStack { ProgressView(); Text(session.authenticationStatus).accessibilityIdentifier("native-authentication-status") } }
                    if let problem = session.authenticationProblem { Text(problem).accessibilityAddTraits(.updatesFrequently).accessibilityIdentifier("native-authentication-problem") }
                }
                Section("Privacy") {
                    Text("Credentials are sent only to the configured HTTPS Raphael service. Sessions are memory-only and end locally when the app process closes. Use your organization's recovery process if you cannot sign in.").font(.footnote)
                }
        }
    }
}


/// A stable UIKit secure field avoids SwiftUI secure-entry reconciliation while
/// the sign-in form changes enabled state. Input reaches its binding synchronously.
struct NativeSecretEntry: UIViewRepresentable {
    @Binding var text: String
    let title: String
    var oneTimeCode = false
    func makeCoordinator() -> Coordinator { Coordinator(text: $text) }
    func makeInput(coordinator: Coordinator) -> UITextField {
        let input = UITextField()
        input.isSecureTextEntry = true
        input.placeholder = title
        input.accessibilityLabel = title
        input.accessibilityIdentifier = oneTimeCode ? "native-otp" : "native-password"
        input.textContentType = oneTimeCode ? .oneTimeCode : .password
        input.autocapitalizationType = .none
        input.autocorrectionType = .no
        input.spellCheckingType = .no
        input.smartDashesType = .no
        input.smartQuotesType = .no
        input.returnKeyType = .done
        input.font = .preferredFont(forTextStyle: .body)
        input.adjustsFontForContentSizeCategory = true
        input.delegate = coordinator
        input.addTarget(coordinator, action: #selector(Coordinator.changed(_:)), for: [.editingChanged, .editingDidEnd])
        return input
    }
    func makeUIView(context: Context) -> UITextField { makeInput(coordinator: context.coordinator) }
    func updateInput(_ input: UITextField, coordinator: Coordinator) {
        // Form rows may be reused:refresh both the binding and its semantic identity.
        input.placeholder = title
        input.accessibilityLabel = title
        input.accessibilityIdentifier = oneTimeCode ? "native-otp" : "native-password"
        let kind: UITextContentType = oneTimeCode ? .oneTimeCode : .password
        if input.textContentType != kind { input.textContentType = kind }
        coordinator.text = $text
        if input.text != text { input.text = text }
    }
    func updateUIView(_ input: UITextField, context: Context) { updateInput(input, coordinator: context.coordinator) }
    final class Coordinator: NSObject, UITextFieldDelegate {
        var text: Binding<String>
        init(text: Binding<String>) { self.text = text }
        @objc func changed(_ input: UITextField) { text.wrappedValue = input.text ?? "" }
        func textFieldShouldReturn(_ textField: UITextField) -> Bool { textField.resignFirstResponder(); return true }
    }
}
