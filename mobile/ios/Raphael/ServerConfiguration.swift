import Foundation

struct ServerConfiguration: Equatable {
    let origin: URL
    let supportEmail: String?

    init?(urlString: String, boundHost: String, supportEmail: String? = nil) {
        guard let components = URLComponents(string: urlString),
              components.scheme?.lowercased() == "https",
              let host = components.host?.lowercased(), !host.isEmpty,
              host == boundHost.lowercased(), !host.contains("$"), host.contains("."), !host.contains(":"),
              !host.split(separator: ".").allSatisfy({ Int($0) != nil }),
              host != "localhost", !host.hasSuffix(".localhost"),
              !host.hasSuffix(".local"), !host.hasSuffix(".invalid"),
              components.port == nil || components.port == 443,
              components.user == nil, components.password == nil,
              components.query == nil, components.fragment == nil,
              components.path.isEmpty || components.path == "/",
              let origin = components.url else { return nil }
        self.origin = origin
        self.supportEmail = supportEmail.flatMap { value in
            value.contains("@") && !value.contains("$") ? value : nil
        }
    }

    static var bundled: ServerConfiguration? {
        #if DEBUG
        if ProcessInfo.processInfo.arguments.contains("--uitest-unconfigured") { return nil }
        #endif
        return ServerConfiguration(
            urlString: Bundle.main.object(forInfoDictionaryKey: "RaphaelServerURL") as? String ?? "",
            boundHost: Bundle.main.object(forInfoDictionaryKey: "RaphaelServerHost") as? String ?? "",
            supportEmail: Bundle.main.object(forInfoDictionaryKey: "RaphaelSupportEmail") as? String
        )
    }

    func url(for section: PatientSection, patientID: Int? = nil) -> URL {
        let base = origin.appendingPathComponent(section.path)
        guard section.requiresPatient, let patientID, patientID > 0,
              var components = URLComponents(url: base, resolvingAgainstBaseURL: false) else { return base }
        components.queryItems = [URLQueryItem(name: "patient_id", value: String(patientID))]
        return components.url ?? base
    }
}

enum PatientSection: String, CaseIterable, Identifiable {
    case dashboard, record, emergency, intake, intelligence, messages, appointments, account
    var id: String { rawValue }
    var requiresPatient: Bool { self == .record || self == .appointments }
    var title: String {
        switch self {
        case .dashboard: "Dashboard"
        case .record: "Medical record"
        case .emergency: "Emergency information"
        case .intake: "Prepare a visit"
        case .intelligence: "AI education"
        case .messages: "Care messages"
        case .appointments: "Appointments"
        case .account: "Account and privacy"
        }
    }
    var symbol: String {
        switch self {
        case .dashboard: "house"
        case .record: "heart.text.square"
        case .emergency: "cross.case"
        case .intake: "list.clipboard"
        case .intelligence: "text.bubble"
        case .messages: "envelope"
        case .appointments: "calendar"
        case .account: "person.crop.circle"
        }
    }
    var path: String {
        switch self {
        case .dashboard: ""
        case .record, .appointments: "static/chart.html"
        case .emergency, .intake: "static/care.html"
        case .intelligence: "static/patient-ai.html"
        case .messages: "static/inbox.html"
        case .account: "static/account.html"
        }
    }
}

enum NavigationDecision: Equatable { case sameOrigin, externalCDC, blocked }

struct NavigationPolicy {
    let origin: URL
    func decision(for destination: URL) -> NavigationDecision {
        guard let target = URLComponents(url: destination, resolvingAgainstBaseURL: false),
              target.scheme?.lowercased() == "https", target.user == nil, target.password == nil,
              let host = target.host?.lowercased(), target.port == nil || target.port == 443 else { return .blocked }
        if host == origin.host?.lowercased() { return .sameOrigin }
        // Source inspection is explicit and isolated in system Safari,never in the patient web session.
        if host == "www.cdc.gov", target.query == nil, target.fragment == nil { return .externalCDC }
        return .blocked
    }
}

/// Display policy only. Every request remains subject to server authorization.
struct NativeReleasePolicy: Decodable, Equatable {
    let scope: String
    let api_version: String
    let policy_sha256: String
    let operator_paused: Bool
    let capabilities: [String: Bool]?
    let public_distribution_authorized: Bool
    let clinical_production_authorized: Bool

    static func validated(_ data: Data) throws -> NativeReleasePolicy {
        let value = try JSONDecoder().decode(Self.self, from: data)
        guard value.api_version == "1", ["development", "consumer_candidate"].contains(value.scope),
              value.policy_sha256.count == 64,
              value.policy_sha256.allSatisfy({ "0123456789abcdef".contains($0) }),
              !value.public_distribution_authorized, !value.clinical_production_authorized else { throw URLError(.badServerResponse) }
        if value.scope == "consumer_candidate" {
            let allowed: Set<String> = ["own_linked_records", "own_approved_notes", "own_document_download", "own_account_preferences", "own_record_export", "own_feedback"]
            let denied: Set<String> = ["registration", "external_password_recovery", "clinical_assessment", "clinical_generation", "radiology", "ambient_recording", "clinical_tools", "external_clinical_communications", "administration", "complete_account_erasure"]
            guard let capabilities = value.capabilities,
                  Set(capabilities.keys) == allowed.union(denied),
                  denied.allSatisfy({ capabilities[$0] == false }) else { throw URLError(.badServerResponse) }
        } else if value.capabilities != nil { throw URLError(.badServerResponse) }
        return value
    }
    var isCandidate: Bool { scope == "consumer_candidate" }
    var allowsMFA: Bool { !isCandidate }
    var allowsExport: Bool { !isCandidate || (!operator_paused && capabilities?["own_record_export"] == true) }
    func sections(role: String) -> [PatientSection] {
        if isCandidate { return role == "patient" ? [.dashboard] : [] }
        return role == "clinician" ? [.dashboard, .record, .intelligence, .messages, .appointments, .account] : PatientSection.allCases
    }
    func allowsNavigation(_ url: URL) -> Bool {
        guard isCandidate else { return true }
        return ["", "/", "/static/release-preview.html"].contains(url.path) && url.query == nil
    }
}
