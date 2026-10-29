> Release audit,2026-10-28: [Current source-scoped readiness audit](../../docs/RC_RELEASE_AUDIT.md). Sections below are historical checkpoints;older "current" or "final" wording is not a release approval.

## Current engineering assessment

Read docs/MASTER_ENGINEERING_REPORT.md first. The new configured native test found and fixed a main NavigationStack launch crash;actual authenticated readiness/logout test verification is in progress. The source-only statements below are historical. The current CI harness uses actual main:app,ephemeral synthetic records,simulator-trusted HTTPS and no AI provider credentials. It does not produce a signed physical-device or TestFlight build. Reserved `.test` origins are rejected by Release validation.

No TestFlight installation is available yet. After remaining engineering,privacy/account-erasure,clinical intended-use and independent security gates pass,the owner-only distribution steps are:

1. Supply an authorized synthetic HTTPS staging service,registered bundle identifier,Apple Developer team,valid support/privacy URLs and owner-approved icon. No staging deployment is performed by this task.
2. On the owner's Mac,copy `mobile/ios/Configuration/Local.xcconfig.example` to the ignored `Local.xcconfig` and enter the public staging origin/matching host,bundle,team and support address. Preserve the `https:/$()/hostname` xcconfig syntax. Never enter OpenAI/Qdrant keys in Xcode;the backend owns those secrets.
3. Open `mobile/ios/Raphael.xcodeproj`,select the shared Raphael scheme and owner signing team. Run the simulator suite and the required authenticated/physical-device checks against synthetic accounts. Local default builds skip the CI-only authenticated fixture case;that skip is not native authentication proof.
4. After those gates and explicit upload authorization,select Any iOS Device and Product > Archive with Release configuration. Release validation must pass;do not disable signing/TLS/origin gates. Validate the archive in Xcode Organizer,review privacy/export-compliance metadata,and distribute to the owner's App Store Connect app.
5. In App Store Connect,wait for build processing,complete required export/privacy information,then authorize internal TestFlight testers. External testing requires Apple's beta-review process. The tester installs Apple's TestFlight app,accepts the invitation and installs that specific processed build. No invitation,build number,upload or Apple approval is invented here.

Account deletion review is still not actual erasure. Native export/download,identity/MFA/guardian controls,device/accessibility evaluation,operations and legal obligations are not solved by a green simulator test. Do not upload or claim clinical release until those gates are actually satisfied.

Current evidence supersedes older unverified-build statements below:unsigned remote Xcode16.4/iOS18.5 simulator compilation and native unit/UI tests passed in run40421429685. No authenticated native/device/signing/submission success is claimed.

# Owner review package — source only, not submitted

The Linux source audit and five project/configuration tests pass. Swift compilation, simulator execution, physical-device accessibility, signing and App Store/TestFlight upload have NOT run. No owner Apple account, certificate, registered bundle, staging hostname or review credentials are invented.

Use mobile/README.md for the exact Xcode simulator build/test commands. Before a device archive the owner must provide an authorized synthetic HTTPS backend with the current branch, a registered bundle identifier, real support/privacy URLs, development team and signing/provisioning. Keep private review accounts out of Git. No public deployment has been authorized or performed.

Draft category: Health & Fitness, subject to owner/legal review. Draft description: “View your connected synthetic development record,manage sharing and explore source-linked health education with clinician review.” This is not final store copy or a medical-device claim. Confirm minimum functionality and WebKit wrapper acceptance with Apple;native navigation alone does not guarantee acceptance.

Screenshots must be captured from an actual simulator/device running the authenticated synthetic service using mobile/scripts/screenshot_booted_simulator.sh. Chromium screenshots are browser evidence only. Screenshots, App Preview, translations, age rating, support contacts, privacy nutrition labels and final metadata remain owner/reviewer work.

After passing native tests,complete privacy/legal/clinical/security review and implement policy-backed account deletion. Current account controls submit/cancel/acknowledge a deletion REVIEW;they do not erase accounts or satisfy Apple's deletion requirement. Complete protected native export/download before claiming mobile export. The microphone and downloads are deliberately denied in the current source.

Only then archive a signed Release build,validate in Xcode Organizer and obtain explicit owner authorization before uploading to App Store Connect/TestFlight. No submission or review approval is claimed. Android native source/build remains absent.
