import SwiftUI
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
            NavigationStack {
                VStack(spacing: 0) {
                    Text("Synthetic development • Not an emergency channel")
                        .font(.footnote).padding(8).frame(maxWidth: .infinity)
                        .background(Color.yellow.opacity(0.18))
                        .accessibilityIdentifier("synthetic-notice")
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
                        .accessibilityIdentifier("patient-web-workspace")
                }
                .navigationTitle(session.currentSection.title)
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        Menu {
                            ForEach(PatientSection.allCases) { section in
                                Button(section.title, systemImage: section.symbol) { session.open(section) }
                            }
                        } label: { Label("Sections", systemImage: "line.3.horizontal") }
                        .accessibilityLabel("Patient sections")
                    }
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Settings", systemImage: "gearshape") { settings = true }
                            .accessibilityLabel("Privacy and session settings")
                    }
                }
            }
            .privacySensitive()
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
        .onAppear { if session.webView.url == nil { session.open(.dashboard) } }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active { session.finishConfirmation(false) }
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
