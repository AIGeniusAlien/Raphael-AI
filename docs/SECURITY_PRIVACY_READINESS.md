# Technical security and HIPAA readiness evidence

Raphael is synthetic-development-only. This matrix supports qualified assessment;it is not a finding of HIPAA compliance,legal advice,completed risk analysis or permission to process PHI. No BAA,workforce program,production infrastructure or clinical authorization is established by source code. Applicability and implementation specifications must be evaluated by the owner's qualified privacy/security/legal officers.

| Rule/control | Technical evidence | State | Operational dependency / accountable role |
| --- | --- | --- | --- |
| 45 CFR164.308(a)(1),risk analysis/management | Requirements ledger,source/dependency scans,negative tests | PARTIALLY IMPLEMENTED | Security officer:formal system-wide threat/risk analysis,remediation ownership and risk acceptance |
| 164.308(a)(3)-(5),workforce/access/training | Named local accounts,roles,patient chart grants;no public self-registration | PARTIALLY IMPLEMENTED | Organization:identity proofing,workforce termination,training,sanctions,credential recovery and MFA |
| 164.308(a)(6),security incidents | Sanitized provider failures and local logs;no full response program | NOT IMPLEMENTED | Security officer:incident detection,on-call,response and breach assessment exercises |
| 164.308(a)(7),contingency | Non-destructive SQLite backup/integrity/restart tests | PARTIALLY IMPLEMENTED | Operator:encrypted off-site backup,restore drills,RPO/RTO,availability planning |
| 164.308(b),business associates | External AI/Qdrant paths explicit;development uses synthetic data | BLOCKED BY EXTERNAL DEPENDENCY | Owner/legal:vendor BAAs,subprocessors,permitted data regions/uses and contracts before PHI |
| 164.310,physical safeguards | No source-code claim | BLOCKED BY EXTERNAL DEPENDENCY | Hosting/operator:facility/device/media controls and disposal |
| 164.312(a),access control | Role/tenant/patient/thread scope;expiry,revoke,explicit chart/intake/AI consent;per-token checks | PARTIALLY IMPLEMENTED | Identity officer:MFA,recovery,guardian/proxy verification,emergency access policy;production review |
| 164.312(a)(2)(iv),encryption/decryption | iOS ephemeral session,no embedded keys;SQLite currently not encrypted by app | NOT SAFE FOR RELEASE | Operator:at-rest encryption/KMS,device/disk/backup controls and documented key lifecycle |
| 164.312(b),audit controls | Local audit events for mutations/access workflows;not comprehensive immutable audit storage | PARTIALLY IMPLEMENTED | Operator:tamper-evident centralized retention,access monitoring and review without PHI overcollection |
| 164.312(c),integrity | Additive migrations,trigger transaction tests,immutable approved note versions,source hashes | PARTIALLY IMPLEMENTED | Clinical/security owner:production integrity validation,signing policies and longitudinal correction procedures |
| 164.312(d),authentication | scrypt passwords,opaque HttpOnly sessions,login/request budgets;local identity only | PARTIALLY IMPLEMENTED | Owner:verified enrollment,recovery/SSO/MFA,distributed abuse prevention and penetration test |
| 164.312(e),transmission security | HTTPS provider/origin guards,HTTPS cookies/HSTS,CSRF fetch-metadata/origin+JSON checks,no-referrer/CSP | PARTIALLY IMPLEMENTED | Operator:authorized HTTPS deployment,trusted-proxy/host configuration,TLS lifecycle and transport review |
| 164.316,policies/documentation | Engineering ledger,technical reports and retained source history | PARTIALLY IMPLEMENTED | Compliance officer:approved policies,revision governance and required retention |
| Privacy164.502/504/506,permitted use/disclosure/minimum necessary | Patient-scoped reads,selected approved AI context,separate scoped snapshot consent | PARTIALLY IMPLEMENTED | Privacy officer:legal authority,per-purpose permissions,vendor terms and minimum-necessary assessment |
| Privacy164.508,authorizations | Revocable application permissions;not a legally sufficient authorization instrument | PARTIALLY IMPLEMENTED | Legal/privacy owner:valid forms,identity,guardian/deceased/minor rules and revocation obligations |
| Privacy164.520,notice | Draft privacy documents only | BLOCKED BY EXTERNAL DEPENDENCY | Owner/legal:actual entity identity,contacts,service/data inventory and published approved notice |
| Privacy164.524/526/528,access/amendment/accounting | Bounded own export and approved note provenance;not full legal rights fulfillment | PARTIALLY IMPLEMENTED | Privacy/records officer:identity/deadlines/exceptions,complete export/amendment/accounting processes |
| Retention/deletion and App Store account deletion | Request/cancel/admin acknowledgment;NO erasure or deactivation | NOT SAFE FOR RELEASE | Owner/legal:jurisdictional retention,legal hold,identity and policy-backed deletion implementation |

## Current bounded technical safeguards

Portal mode blocks legacy unauthenticated APIs. Secure headers now cover denial responses too. Same-origin JSON mutation checks reject malformed origins;Fetch Metadata rejects explicitly cross-site requests. Requests are budgeted per direct peer IP:300requests/60seconds and12expensive-generation requests/60seconds,maximum10,000active client keys. The limiter is bounded,thread-safe and ignores spoofed forwarded headers. It is intentionally single-process;NAT users share a budget,and proxies/workers require a separately reviewed distributed gateway. Login's existing persistent failed-attempt budget remains. This is not a production DDoS solution.

Secrets are supplied through existing scoped environment bindings;the local verification/start default removes provider bindings and inherited database settings. Tests use temporary synthetic databases. Logs redact configured credential values and avoid raw provider errors;this is not a guarantee that arbitrary logs are free of sensitive data. No medical data is sent to analytics/crash reporting. Native readiness sends no authentication cookies or patient fields and rejects redirects;device execution remains unverified.

Fresh dependency advisory and high-severity/high-confidence Bandit results are recorded with this increment. Narrow source scanning is not a full audit;lower-severity findings,manual review,malware/file scanning and penetration tests remain. Existing file type/size/parser checks do not establish antivirus protection.
