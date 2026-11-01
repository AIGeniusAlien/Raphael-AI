import SwiftUI
import Foundation
import UniformTypeIdentifiers

struct SettingsView: View {
    @ObservedObject var session: PatientWebSession
    let dismiss: () -> Void
    @State private var confirmSignOut = false
    @State private var exportAction: String?
    @Environment(\.scenePhase) private var scenePhase
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
                    Text("The account page provides profile, consent, sessions and web export. Requesting deletion review is not completed medical-record erasure; retention and identity review remain required.")
                        .font(.footnote)
                    Button("Sign out and clear local session", role: .destructive) { confirmSignOut = true }.disabled(session.authenticating)
                }
                if session.identity?.role == "patient" {
                    Section("Protected record export") {
                        Text("Own linked records only, up to 500 rows per section and 5 MiB. File bytes and unsigned clinician drafts are excluded. The snapshot is refreshed before copying or sharing.").font(.footnote)
                        Text(session.nativeExportStatus).accessibilityIdentifier("native-export-status")
                        Button(session.nativeExportBusy ? "Preparing export" : "Prepare record export") { session.prepareNativeExport() }
                            .disabled(session.nativeExportBusy).accessibilityIdentifier("native-export-prepare")
                        if let summary = session.nativeExportSummary {
                            Text("\(summary.records) \(summary.records == 1 ? "linked record" : "linked records") · \(summary.approvedNotes) \(summary.approvedNotes == 1 ? "approved note" : "approved notes") · \(summary.byteCount) bytes")
                                .accessibilityIdentifier("native-export-summary")
                            Text("Prepared: " + summary.generatedAt).font(.caption)
                            ForEach(Array(summary.recordNames.enumerated()),id:\.offset) { _,name in Text(name).font(.caption) }
                            if summary.records > summary.recordNames.count { Text("Only the first ten names are shown;the bounded export contains all listed records.").font(.footnote) }
                            Button("Copy JSON to local clipboard") { exportAction = "copy" }
                                .disabled(session.nativeExportBusy).accessibilityIdentifier("native-export-copy")
                            Button("Share or save JSON") { exportAction = "share" }
                                .disabled(session.nativeExportBusy).accessibilityIdentifier("native-export-share")
                        }
                        Button("Cancel and clear export") { session.cancelNativeExport() }
                            .accessibilityIdentifier("native-export-cancel")
                        Text("A copy you deliberately export is outside Raphael's control. Logout or consent revocation cannot erase it. Clipboard copy requests local-only delivery and expiry after 60 seconds. System destinations may retain their own copies. Use only protected destinations.").font(.footnote)
                    }
                }
                Section("On-device protections") {
                    Label("Memory-only web session", systemImage: "lock")
                    Text("Native sign-in and the web workspace share the server's HttpOnly HTTPS cookie through an ephemeral WebKit store. No token or password is saved to disk. You must sign in after the app process restarts.")
                    Text("No automatic offline record cache, service-worker cache, analytics SDK or push registration is added. A deliberate export may create copies outside Raphael. A bounded,explicit record JSON export is available in these settings. Arbitrary web downloads and microphone capture remain disabled. Physical-device destination behavior still requires verification.")
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
            .privacySensitive()
            .accessibilityHidden(scenePhase != .active)
            .overlay {
                if scenePhase != .active {
                    ZStack { Color(.systemBackground).ignoresSafeArea();Text("Private export hidden") }
                }
            }
            .navigationTitle("Privacy and settings")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done", action: dismiss) } }
            .confirmationDialog(confirmSignOut ? "Sign out of this Raphael session?" : "Export sensitive record data?",
                isPresented:Binding(get:{confirmSignOut || exportAction != nil},set:{if !$0 {confirmSignOut = false;exportAction = nil}}),titleVisibility:.visible) {
                if confirmSignOut {
                    Button("Sign out",role:.destructive) { confirmSignOut = false;dismiss();session.signOut() }
                        .accessibilityIdentifier("native-confirm-sign-out")
                } else if exportAction == "copy" {
                    Button("Copy JSON locally") { exportAction = nil;session.copyNativeExport() }
                        .accessibilityIdentifier("native-export-confirm-copy")
                } else if exportAction == "share" {
                    Button("Open system share sheet") { exportAction = nil;session.beginNativeExportSharing() }
                        .accessibilityIdentifier("native-export-confirm-share")
                }
                Button("Cancel",role:.cancel) { confirmSignOut = false;exportAction = nil }
            } message: {
                Text(confirmSignOut ? "Unsaved edits will be lost. The app attempts server revocation and clears its local web session even if offline." : "This hands sensitive records to the device or a destination you choose. Copies cannot be recalled by logout or consent revocation. File bytes and unsigned drafts are excluded. Continue only on a protected device.")
            }
            .sheet(item:Binding(get:{session.nativeExportTicket},set:{if $0 == nil {session.cancelNativeExport()}})) { ticket in
                VStack(spacing:12) {
                    Text("Sensitive record export").font(.headline)
                    Text("Choose a protected destination. A completed transfer cannot be recalled.").font(.footnote)
                    Button("Cancel export sharing") { session.cancelNativeExport() }
                        .accessibilityIdentifier("native-export-share-cancel")
                    NativeExportActivity(session:session,ticket:ticket)
                }.padding().privacySensitive()
            }
            .onDisappear { session.cancelNativeExport() }
            .onChange(of:scenePhase) { _,phase in
                if phase != .active { exportAction = nil;session.cancelNativeExport() }
            }
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


/// The app creates no export file. The OS or a user-chosen destination may
/// materialize its own copy after explicit confirmation. Every data-provider
/// request requires a live,matching session and a fresh bounded server export.
struct NativeExportActivity: UIViewControllerRepresentable {
    let session: PatientWebSession
    let ticket: NativeExportTicket
    func makeUIViewController(context:Context) -> UIActivityViewController {
        let provider = Self.provider(session:session,ticket:ticket)
        let items = UIActivityItemsConfiguration(itemProviders:[provider])
        items.metadataProvider = { key in key == .title ? "Sensitive synthetic patient records" : nil }
        let controller = UIActivityViewController(activityItemsConfiguration:items)
        controller.completionWithItemsHandler = { _,_,_,_ in Task { @MainActor in session.cancelNativeExport() } }
        return controller
    }
    @MainActor static func provider(session:PatientWebSession,ticket:NativeExportTicket) -> NSItemProvider {
        let provider = NSItemProvider()
        provider.suggestedName = "raphael-synthetic-patient-export.json"
        provider.registerDataRepresentation(forTypeIdentifier:UTType.json.identifier,visibility:.all) { [weak session,weak ticket] completion in
            let progress = Progress(totalUnitCount:1)
            let task = Task { @MainActor in
                guard let session,let ticket else { completion(nil,URLError(.cancelled));return }
                do {
                    let data = try await session.dataForNativeExportSharing(ticket)
                    guard !Task.isCancelled else { throw URLError(.cancelled) }
                    completion(data,nil);progress.completedUnitCount = 1
                } catch { completion(nil,URLError(.userAuthenticationRequired)) }
            }
            progress.cancellationHandler = { task.cancel() }
            return progress
        }
        return provider
    }
    func updateUIViewController(_ controller:UIActivityViewController,context:Context) {}
}
