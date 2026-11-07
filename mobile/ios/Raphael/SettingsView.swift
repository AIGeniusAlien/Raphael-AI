import SwiftUI
import Combine
import UIKit
import Foundation
import UniformTypeIdentifiers

struct SettingsView: View {
    @ObservedObject var session: PatientWebSession
    let dismiss: () -> Void
    @State private var confirmSignOut = false
    @State private var exportAction: String?
    @State private var showMFA = false
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
                    if session.releasePolicy?.isCandidate == true {
                        Button("Account, records and feedback preview") { dismiss(); session.open(.dashboard) }
                        Text("Synthetic candidate preview. Profile, session and deletion-review controls are provided by the server. Complete account erasure is unavailable.").font(.footnote)
                    } else {
                    Button("Account, export and deletion review") { dismiss(); session.open(.account) }
                    Text("The account page provides profile, consent, sessions and web export. Requesting deletion review is not completed medical-record erasure; retention and identity review remain required.")
                        .font(.footnote)
                    }
                    Button("Sign out and clear local session", role: .destructive) { confirmSignOut = true }.disabled(session.authenticating)
                }
                if session.releasePolicy?.allowsMFA == true {
                Section("Authenticator protection") {
                    Button("Manage authenticator") { showMFA = true }.accessibilityIdentifier("native-mfa-manage")
                    Text("Enroll an authenticator,confirm a code and save one-time recovery codes privately. This verifies an account factor,not clinical or legal identity.").font(.footnote)
                }
                }
                if session.identity?.role == "patient", session.releasePolicy?.allowsExport == true {
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
            .sheet(isPresented:$showMFA) { NativeMFAView(session:session,dismiss:{showMFA = false}) }
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


/// Only emptiness and explicit clearing revisions are published. Per-character
/// private values do not drive SwiftUI Form reconciliation or cell reuse.
final class NativeMFACredentials: ObservableObject {
    @Published private(set) var hasPassword=false
    @Published private(set) var hasFactor=false
    @Published private(set) var clearRevision=0
    private var password=""
    private var factor=""
    func updatePassword(_ value:String) { password=value;if hasPassword != !value.isEmpty {hasPassword = !value.isEmpty} }
    func updateFactor(_ value:String) { factor=value;if hasFactor != !value.isEmpty {hasFactor = !value.isEmpty} }
    func snapshot()->(password:String,factor:String) {(password,factor)}
    func take()->(password:String,factor:String) {let captured=snapshot();clear();return captured}
    func clear() {password="";factor="";hasPassword=false;hasFactor=false;clearRevision += 1}
}

/// One UIKit-owned pair remains stable when SwiftUI updates status/buttons.
/// Reconciliation never writes an in-flight editor;explicit clear erases both.
struct NativeMFACredentialFields: UIViewRepresentable {
    @ObservedObject var credentials:NativeMFACredentials
    final class Inputs:UIStackView {
        let password=UITextField()
        let factor=UITextField()
        var clearRevision=0
        override init(frame:CGRect) {
            super.init(frame:frame);axis = .vertical;spacing=12;alignment = .fill
            for (input,title,identifier,kind) in [
                (password,"Current password","native-mfa-password",UITextContentType.password),
                (factor,"Authenticator or unused recovery code","native-mfa-code",UITextContentType.oneTimeCode)] {
                input.isSecureTextEntry=true;input.placeholder=title;input.accessibilityLabel=title;input.accessibilityIdentifier=identifier
                input.textContentType=kind;input.autocapitalizationType = .none;input.autocorrectionType = .no;input.spellCheckingType = .no
                input.smartDashesType = .no;input.smartQuotesType = .no;input.returnKeyType = .done
                input.font = .preferredFont(forTextStyle:.body);input.adjustsFontForContentSizeCategory=true
                input.borderStyle = .roundedRect;input.heightAnchor.constraint(greaterThanOrEqualToConstant:44).isActive=true
                addArrangedSubview(input)
            }
        }
        required init(coder:NSCoder) {fatalError("Programmatic credential inputs only")}
    }
    final class Coordinator:NSObject,UITextFieldDelegate {
        var credentials:NativeMFACredentials
        init(_ credentials:NativeMFACredentials) {self.credentials=credentials}
        @objc func passwordChanged(_ field:UITextField) {credentials.updatePassword(field.text ?? "")}
        @objc func factorChanged(_ field:UITextField) {credentials.updateFactor(field.text ?? "")}
        func textFieldShouldReturn(_ field:UITextField)->Bool {field.resignFirstResponder();return true}
    }
    func makeCoordinator()->Coordinator {Coordinator(credentials)}
    func makeInputs(_ coordinator:Coordinator)->Inputs {
        let inputs=Inputs(frame:.zero),values=credentials.snapshot();inputs.password.text=values.password;inputs.factor.text=values.factor;inputs.clearRevision=credentials.clearRevision
        inputs.password.delegate=coordinator;inputs.factor.delegate=coordinator
        inputs.password.addTarget(coordinator,action:#selector(Coordinator.passwordChanged(_:)),for:.editingChanged)
        inputs.factor.addTarget(coordinator,action:#selector(Coordinator.factorChanged(_:)),for:.editingChanged)
        return inputs
    }
    func updateInputs(_ inputs:Inputs,coordinator:Coordinator) {
        coordinator.credentials=credentials
        guard inputs.clearRevision != credentials.clearRevision else {return}
        inputs.clearRevision=credentials.clearRevision
        // No editingDidEnd callback republishes the old value during erasure.
        inputs.endEditing(true);inputs.password.text="";inputs.factor.text=""
    }
    func makeUIView(context:Context)->Inputs {makeInputs(context.coordinator)}
    func updateUIView(_ inputs:Inputs,context:Context) {updateInputs(inputs,coordinator:context.coordinator)}
    func sizeThatFits(_ proposal:ProposedViewSize,uiView:Inputs,context:Context)->CGSize? {
        uiView.systemLayoutSizeFitting(CGSize(width:proposal.width ?? 320,height:UIView.layoutFittingCompressedSize.height),withHorizontalFittingPriority:.required,verticalFittingPriority:.fittingSizeLevel)
    }
}

struct NativeMFAView: View {
    @ObservedObject var session:PatientWebSession
    let dismiss:()->Void
    @Environment(\.scenePhase) private var scenePhase
    @State private var enabled:Bool?
    @State private var available = false
    @State private var remaining = 0
    @StateObject private var credentials = NativeMFACredentials()
    @State private var setupKey:String?
    @State private var recovery:[String] = []
    @State private var status = "Checking authenticator status…"
    @State private var busy = false
    @State private var confirmDisable = false
    @State private var epoch = 0
    @State private var operationTask:Task<Void,Never>?
    var body: some View {
        NavigationStack {
            Form {
                Section("Account protection") {
                    Text(status).accessibilityIdentifier("native-mfa-status")
                    if let enabled { Text(enabled ? "Authenticator enabled · \(remaining) recovery codes remaining" : "Authenticator not enabled") }
                    Button("Refresh authenticator status") { perform(.status) }.disabled(busy).accessibilityIdentifier("native-mfa-refresh")
                    Text("Keep your authenticator and recovery codes on a protected device. Raphael cannot recover a lost setup key. These controls do not verify professional credentials or patient identity.").font(.footnote)
                }
                Section("Verify credentials") {
                    NativeMFACredentialFields(credentials:credentials).id("mfa-credential-fields").frame(minHeight:100)
                    Button("Start authenticator enrollment") { perform(.enroll) }
                        .disabled(busy || enabled != false || !available || !credentials.hasPassword).accessibilityIdentifier("native-mfa-enroll")
                    if setupKey != nil {
                        Button("Confirm authenticator enrollment") { perform(.confirm) }
                            .disabled(busy || !credentials.hasPassword || !credentials.hasFactor).accessibilityIdentifier("native-mfa-confirm")
                    }
                    if enabled == true {
                        Button("Disable authenticator",role:.destructive) { confirmDisable = true }
                            .disabled(busy || !credentials.hasPassword || !credentials.hasFactor).accessibilityIdentifier("native-mfa-disable")
                    }
                }
                if let key = setupKey {
                    Section("Private setup key") {
                        Text("Enter this key in your own authenticator. It expires after ten minutes. Then reenter your password and a generated code. Do not share or screenshot the key.").font(.footnote)
                        Text(key).font(.system(.body,design:.monospaced)).accessibilityIdentifier("native-mfa-setup-key")
                    }
                }
                if !recovery.isEmpty {
                    Section("One-time recovery codes") {
                        Text("Record these privately now. Each works once with your password. Other sessions have been revoked. Wait for the next authenticator code before a new sign-in.").font(.footnote)
                        Text(recovery.joined(separator:"\n")).font(.system(.body,design:.monospaced)).accessibilityIdentifier("native-mfa-recovery-codes")
                    }
                }
                if setupKey != nil || !recovery.isEmpty {
                    Button("Clear private authenticator data") { clearPrivate() }.accessibilityIdentifier("native-mfa-clear")
                }
                Section("Interruption and privacy") {
                    Text("Closing or backgrounding clears displayed secrets and cancels pending access. A server change may already have completed;refresh status after interruption. No key or recovery code is copied to the clipboard or saved by this screen. Protect against screenshots and shoulder surfing.").font(.footnote)
                }
            }
            .privacySensitive().accessibilityHidden(scenePhase != .active)
            .overlay { if scenePhase != .active { ZStack {Color(.systemBackground).ignoresSafeArea();Text("Private authenticator data hidden")} } }
            .navigationTitle("Authenticator")
            .toolbar { ToolbarItem(placement:.confirmationAction) { Button("Done") { clearPrivate();dismiss() }.accessibilityIdentifier("native-mfa-done") } }
            .confirmationDialog("Disable authenticator protection?",isPresented:$confirmDisable,titleVisibility:.visible) {
                Button("Disable and revoke other sessions",role:.destructive) { perform(.disable) }.accessibilityIdentifier("native-mfa-confirm-disable")
                Button("Cancel",role:.cancel) { credentials.clear() }
            } message: { Text("This reduces account protection. Your password and an unused factor are required. Other signed-in sessions will be revoked.") }
            .task { perform(.status) }
            .onDisappear { clearPrivate() }
            .onChange(of:scenePhase) { _,phase in if phase != .active { clearPrivate();status = "Private data cleared. Refresh status before continuing.";enabled = nil } }
            .onChange(of:session.identity) { _,_ in clearPrivate();dismiss() }
        }
    }
    private func clearPrivate() {
        epoch += 1;operationTask?.cancel();operationTask = nil
        credentials.clear();setupKey = nil;recovery = [];busy = false;confirmDisable = false
    }
    private func perform(_ operation:NativeMFAOperation) {
        guard !busy else { return }
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),to:nil,from:nil,for:nil)
        let captured = credentials.take()
        if operation == .status { setupKey = nil;recovery = [] }
        epoch += 1;let attempt = epoch
        busy = true;status = "Verifying with the configured secure service…"
        operationTask = Task { @MainActor in
            do {
                let reply = try await session.manageMFA(operation,password:captured.password,code:captured.factor)
                guard attempt == epoch,!Task.isCancelled else { return }
                switch operation {
                case .status:enabled = reply.enabled;available = reply.enrollment_available == true;remaining = reply.recovery_codes_remaining ?? 0;status = available ? "Authenticator status verified." : "Enrollment unavailable. Operator key configuration is required."
                case .enroll:setupKey = reply.secret;recovery = [];status = "Enrollment pending. Confirm a code before protection is enabled."
                case .confirm:setupKey = nil;recovery = reply.recovery_codes ?? [];enabled = true;remaining = recovery.count;status = "Authenticator enabled. Save recovery codes privately now."
                case .disable:setupKey = nil;recovery = [];enabled = false;remaining = 0;status = "Authenticator disabled. Other sessions revoked."
                }
                busy = false
            } catch {
                guard attempt == epoch else { return }
                setupKey = nil;recovery = [];busy = false;enabled = nil
                status = (error as? NativeMFAError)?.localizedDescription ?? "Connection or response unavailable. Refresh status before retrying;the server may already have completed a change."
            }
        }
    }
}
