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

    func url(for section: PatientSection) -> URL {
        origin.appendingPathComponent(section.path)
    }
}

enum PatientSection: String, CaseIterable, Identifiable {
    case dashboard, record, emergency, intake, intelligence, messages, appointments, account
    var id: String { rawValue }
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
