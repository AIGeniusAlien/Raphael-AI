#!/usr/bin/env bash
set -euo pipefail
mobile_root="$(cd "$(dirname "$0")/.." && pwd)"
if ! command -v xcodebuild >/dev/null || ! command -v xcrun >/dev/null; then
  echo 'BLOCKED: macOS with Xcode/iOS SDK is required. No native build ran.' >&2
  exit 2
fi
: "${RAPHAEL_SIMULATOR_DESTINATION:?Set an installed iOS Simulator destination,for example platform=iOS Simulator,id=YOUR-SIMULATOR-UDID}"
python3 "$mobile_root/scripts/project_audit.py"
xcodebuild -project "$mobile_root/ios/Raphael.xcodeproj" -scheme Raphael -configuration Debug -sdk iphonesimulator -destination "$RAPHAEL_SIMULATOR_DESTINATION" -derivedDataPath "$mobile_root/artifacts/DerivedData" -resultBundlePath "$mobile_root/artifacts/NativeTests.xcresult" CODE_SIGNING_ALLOWED=NO test
