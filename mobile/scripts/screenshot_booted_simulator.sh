#!/usr/bin/env bash
set -euo pipefail
if ! command -v xcrun >/dev/null; then
  echo 'BLOCKED: screenshots require a booted iOS Simulator and xcrun. No screenshots were generated.' >&2
  exit 2
fi
: "${RAPHAEL_BUILT_APP:?Set the path to the built simulator Raphael.app}"
: "${RAPHAEL_APP_BUNDLE_ID:?Set the bundle ID used in that build}"
mobile_root="$(cd "$(dirname "$0")/.." && pwd)"
artifact_dir="$mobile_root/artifacts/screenshots"
mkdir -p "$artifact_dir"
xcrun simctl install booted "$RAPHAEL_BUILT_APP"
xcrun simctl launch booted "$RAPHAEL_APP_BUNDLE_ID"
echo 'Use only an authorized synthetic account. Navigate to the desired screen,then press Return.'
read -r _
xcrun simctl io booted screenshot "$artifact_dir/raphael-synthetic-screen.png"
