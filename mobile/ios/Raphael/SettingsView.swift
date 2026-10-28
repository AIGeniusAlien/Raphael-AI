import SwiftUI
import Foundation

struct SettingsView: View {
    @ObservedObject var session: PatientWebSession
    let dismiss: () -> Void
    @State private var confirmSignOut = false
    @State private var serviceStatus = "Not checked"
    @State private var checkingService = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Development service") {
                    Text(session.configuration.origin.host ?? "Configured service")
                        .font(.callout).textSelection(.enabled)
                    Text("The service address is fixed by the signed build. This app cannot connect to an arbitrary server.")
                        .font(.footnote)
                }
                Section("Backend readiness") {
                    Text(serviceStatus).accessibilityIdentifier("native-readiness-status")
                    Button(checkingService ? "Checking backend" : "Check backend connection") {
                        Task { await checkService() }
                    }.disabled(checkingService)
                    Text("Checks only the configured synthetic database service. No patient data, cookies or provider keys are sent. This does not check AI availability or clinical safety.").font(.footnote)
                }
                Section("Account and privacy") {
                    Button("Account, export and deletion review") { dismiss(); session.open(.account) }
                    Text("The server's account page owns profile, consent, sessions and data export. Requesting deletion review is not completed medical-record erasure; retention and identity review remain required.")
                        .font(.footnote)
                    Button("Sign out and clear local session", role: .destructive) { confirmSignOut = true }
                }
                Section("On-device protections") {
                    Label("Memory-only web session", systemImage: "lock")
                    Text("Native sign-in and the web workspace share the server's HttpOnly HTTPS cookie through an ephemeral WebKit store. No token or password is saved to disk. You must sign in after the app process restarts.")
                    Text("No offline medical record, service-worker cache, analytics SDK or push registration is added by this native client. Downloads and microphone capture are disabled. Native device behavior still requires verification.")
                    Text("Protect your device. The app cannot prevent deliberate screenshots or guarantee the operating system never retains temporary memory.")
                }
                Section("Clinical and emergency limits") {
                    Text("AI text is general education or a provisional draft, not a diagnosis, prescription or signed care plan. Clinician-approved information is labeled by the server.")
                    Text("No facility monitors this app. Do not delay local emergency services or urgent medical care to complete intake or messaging.")
                    Text("Changing sections reloads the shared web workspace. Save edits before navigating.")
                }
                if let email = session.configuration.supportEmail,
                   let url = URL(string: "mailto:" + email) {
                    Section("Support") {
                        Link("Contact configured support", destination: url)
                    }
                } else {
                    Section("Support") {
                        Text("A support address must be configured by the operator before distribution.")
                    }
                }
                Section("Build") {
                    Text("Raphael development • " + (Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "Unknown version"))
                    Text("No App Store submission, signing or clinical approval is represented by this build.")
                        .font(.footnote)
                }
            }
            .navigationTitle("Privacy and settings")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done", action: dismiss) } }
            .confirmationDialog("Sign out of this Raphael session?", isPresented: $confirmSignOut, titleVisibility: .visible) {
                Button("Sign out", role: .destructive) { dismiss(); session.signOut() }
                Button("Cancel", role: .cancel) {}
            } message: { Text("Unsaved edits will be lost. The app attempts server revocation and clears its local web session even if offline.") }
        }
    }
    @MainActor private func checkService() async {
        checkingService = true
        defer { checkingService = false }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpCookieStorage = nil
        configuration.urlCache = nil
        configuration.timeoutIntervalForRequest = 10
        configuration.timeoutIntervalForResource = 10
        let client = URLSession(configuration: configuration, delegate: ReadinessRedirectPolicy(), delegateQueue: nil)
        defer { client.invalidateAndCancel() }
        var request = URLRequest(url: session.configuration.origin.appendingPathComponent("readyz"))
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        do {
            let (bytes, response) = try await client.bytes(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                serviceStatus = "Backend unavailable or not ready. Retry later."; return
            }
            var data = Data()
            for try await byte in bytes {
                guard data.count < 65536 else { serviceStatus = "Unexpected backend response."; return }
                data.append(byte)
            }
            serviceStatus = ReadinessPayload.isSyntheticReady(data) ? "Synthetic backend database ready. AI and clinical safety are not verified." : "Backend is not an authorized synthetic-ready service."
        } catch {
            serviceStatus = "Connection unavailable. Check connectivity and retry. No patient request was sent."
        }
    }

}


struct ReadinessPayload: Decodable {
    let status: String
    let scope: String
    static func isSyntheticReady(_ data: Data) -> Bool {
        guard let payload = try? JSONDecoder().decode(ReadinessPayload.self, from: data) else { return false }
        return payload.status == "ready" && payload.scope == "synthetic-development"
    }
}

final class ReadinessRedirectPolicy: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}
