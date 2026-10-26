# Raphael native iOS development client

This is an actual SwiftUI/WebKit Xcode source project over the existing Raphael backend. It adds native section navigation, server-origin enforcement, privacy/session settings, an ephemeral shared authenticated web session, a background privacy cover, controlled source inspection, explicit JavaScript consent/confirmation bridging, loading/retry states and an account/deletion-review entry point. It does not duplicate the backend, embed provider keys, create a local medical database or install a fake AI provider.

**Native compile, simulator, physical-device, signing and App Store behavior have not run in this Linux environment.** Structural validation is not an iOS build result. The project has app, unit-test and UI-test targets and a shared scheme; the included macOS check command must be run before considering it build verified.

## Project and actual feature boundary

- `ios/Raphael.xcodeproj`: real project with Debug/Release, Swift sources, privacy manifest, launch configuration, opaque app icons, test targets and shared scheme.
- `PatientWebSession`: one shared `WKWebView` using `WKWebsiteDataStore.nonPersistent()`. The existing backend owns authentication, HttpOnly HTTPS cookies, authorization, consent, AI and persistence. No session cookie is read into native storage; no Keychain token is needed because the client deliberately does not persist credentials across process restarts.
- Native navigation opens existing patient dashboard, record/appointments, emergency/intake, AI education, inbox and account pages on the **same fixed configured HTTPS origin**. The backend version must implement those routes; missing routes display a useful unavailable state. This source project does not independently establish those server features.
- Native settings opens the real account page for profile/export/session/deletion review. A deletion-review request is **not completed medical-record erasure**. App Store account-deletion acceptance remains blocked pending retention/legal policy and full backend implementation.
- Native sign-out attempts same-origin server revocation and clears the ephemeral store even if offline. A failed remote revocation produces an explicit warning; clearing local cookies alone is not presented as server revocation.
- Only same-origin navigation is allowed in the patient web session. Public CDC links require an explicit action and open separately in system Safari. Other external links, non-HTTPS schemes, URL credentials and custom ports are denied. This is not an OAuth/SSO implementation.
- Web JavaScript confirmations are presented natively and never auto-accepted. Notices are also bridged. Changing sections or backgrounding cancels pending confirmations. No note is auto-signed.
- Microphone capture and downloads are explicitly disabled in this initial native build. Browser speech behavior must not be called tested iOS audio. App export can be used in a protected supported web client; there is no native download/export-completion claim.
- No offline PHI store, push registration, tracking/analytics SDK or native API-provider connection exists. System behavior, memory/disk lifecycle, background cover timing, VoiceOver and Dynamic Type still require native/device testing.

## Feasible checks on Linux

From the repository root using the existing Python environment:

```bash
/workspace/.raphael/venv/bin/python mobile/scripts/project_audit.py
/workspace/.raphael/venv/bin/python -m unittest discover -s mobile/tests -v
bash -n mobile/scripts/native_checks.sh
bash -n mobile/scripts/screenshot_booted_simulator.sh
```

The audit parses the OpenStep project, checks references/file coverage, all three target configurations, shared test scheme, resources/plists/privacy declarations, transport boundaries and every icon's dimensions/RGB format. The five Python tests include deliberate broken-reference/transport failures and release configuration validation. They do **not** type-check Swift or execute WebKit.

## Build and test on macOS

Use Xcode15 or newer with the iOS17 SDK; current Xcode16+ is recommended. No Apple account is needed for an unsigned simulator build. A device/archive/distribution requires the owner's real bundle registration, signing team, provisioning and review.

1. Open `mobile/ios/Raphael.xcodeproj` and select the shared `Raphael` scheme.
2. Copy `mobile/ios/Configuration/Local.xcconfig.example` to ignored `Local.xcconfig`.
3. Configure the **existing authorized, access-controlled synthetic HTTPS service** and matching hostname. Do not point the app at production data. `https:/$()/host` in xcconfig expands to `https://host`; the separator avoids an xcconfig comment. Never enter OpenAI/Qdrant keys in this file; it is bundled public configuration.
4. Configure owner bundle ID, support contact and, for devices, the real development team. The example values are intentionally unusable; no owner identity or account is invented. A Debug app with no valid service opens a configuration-required screen and makes no connection.
5. Provision synthetic accounts privately using the existing backend bootstrap instructions. There are no built-in review/demo credentials. Ensure the deployed backend preserves the current authorization and patient AI/account routes.
6. List installed simulators with `xcrun simctl list devices available`, then run:

```bash
export RAPHAEL_SIMULATOR_DESTINATION='platform=iOS Simulator,id=YOUR-INSTALLED-SIMULATOR-UDID'
bash mobile/scripts/native_checks.sh
```

This runs Xcode app compilation plus four native XCTest policy/configuration cases and one UI launch test. The UI test deliberately exercises safe unconfigured launch; an authenticated full native journey requires additional synthetic staging/device testing. This command has **not run successfully here** because Xcode is absent.

For just a simulator build:

```bash
xcodebuild -project mobile/ios/Raphael.xcodeproj -scheme Raphael -configuration Debug \
  -sdk iphonesimulator -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath mobile/artifacts/DerivedData CODE_SIGNING_ALLOWED=NO build
```

Release configurations execute a validation phase that rejects missing/unusable HTTPS origin, mismatched host, placeholder bundle identifier/support contact and missing device-signing team. Passing that syntax/configuration gate is not proof the bundle is registered, legal review complete or the service appropriately restricted.

## Submission and Android

`submission/` contains a privacy/data-flow draft, terms/safety draft, metadata/review notes, build/signing/screenshot/TestFlight instructions and a machine-readable readiness ledger. These are review inputs, not published legal policies or an accepted submission. The geometric R icon is an original source asset; brand/trademark approval is an owner gate.

Android native source/build is **not implemented in this increment**. Java is present, but Gradle, adb and Android SDK variables were absent. The shared backend/mobile web continues to work on supported browsers; it is not an APK result. A future Android client should reuse these same origin/session/consent contracts rather than duplicate the server. Android packaging did not delay the iOS source deliverable.
