# One-command development and verification

From the repository root with Python3.12:

```bash
python -m dev_start --setup --provision
```

This command works through Python rather than shell-specific activation scripts. It creates/reuses .venv,installs pinned dependencies,checks compatibility,creates a NEW synthetic SQLite path,privately prompts for clinician and patient passwords,migrates additively and starts loopback8000. Existing user records are not overwritten. Keep the printed database path. To restart with installed dependencies:

```bash
.venv/bin/python -m dev_start --database /absolute/path/to/printed/portal.db
```

On Windows,use `.venv\Scripts\python.exe` instead of `.venv/bin/python`. On macOS,use `python3.12` for the initial command if `python` refers to another version. Windows/macOS installation has not run in this Linux task;platform-specific wheels are resolved under pinned constraints and pip check fails on incompatibilities. It is a cross-platform implementation,not a verified Windows/macOS build claim. Linux launcher startup/restart is tested over actual HTTP.

For the existing cloud environment:

```bash
/workspace/.raphael/venv/bin/python -m dev_start --provision
```

No provider keys or inherited DATABASE_URL are used by default. Without configured providers,AI fails safely rather than inventing responses. `--live-services` explicitly uses the existing five secure bindings and hosted384/Qdrant/claim-screen flags. Never put values into source/commands/chat. Use only the authorized synthetic development services;no PHI or production database.

Open the loopback service in a browser on the same machine or through an owner-configured PRIVATE forwarding mechanism;no hosted preview is created. Sign in to tenant synthetic with the privately provisioned account. Clinician creates a synthetic chart linked to username patient. Separate patient browser session can inspect approved notes,control chart grants,use consented AI,send internal messages and manage account settings. Deletion remains review-only.

## Automated checks

```bash
python -m dev_verify --all --require-browser --output verification-results
```

Activate installed dependencies first,or use the virtual environment's Python path. Install Node24. For browser tooling: `python -m playwright install chromium`;set CHROMIUM_EXECUTABLE to the installed executable when not discoverable. Linux cloud already has /usr/bin/chromium. The runner runs Python,frontend,actual Chromium,mobile structural/tests,traceability and deterministic safety protocol checks. It writes per-check logs and results.json,records exit codes/timeouts/skips and exits nonzero on critical regressions. A missing browser fails with --require-browser;otherwise it is explicitly skipped. No Xcode success is fabricated on Linux. Actual native execution requires macOS/Xcode and RAPHAEL_SIMULATOR_DESTINATION;see mobile/README.md.

Offline regressions receive an isolated environment with inherited provider credentials,backend flags and database paths removed. They use temporary databases and labeled provider fixtures. `--all` does not enable live requests or count skipped checks as passed.

Optional explicit actual remote verification:

```bash
python -m dev_verify --all --live --require-browser --output verification-results-live
```

Uses the existing secure development bindings,temporary synthetic records and real OpenAI/Qdrant/Chromium;results distinguish completed,blocked and failed outputs. It does not ingest or delete collections. No credential values are printed. Local advisory auditing is a separate network-dependent action;the runner marks it skipped rather than implying a fresh scan. GitHub Actions runs pip-audit and high-severity/high-confidence source scanning separately,uploads artifacts and has read-only repository permission.

The iOS CI job on macos-15 selects an actually installed iPhone simulator and runs xcodebuild tests without signing. A configured workflow is not a successful CI run;inspect the PR checks. No production deploy,Apple upload or clinical approval job exists.

## Safety evaluation boundary

`python -m eval.safety_eval --output verification-results/safety.json` exercises actual deterministic notice/citation/population rules using labeled synthetic fixtures. It requires100%protocol pass rate and records individual failures. It does not measure model diagnosis,medical correctness,rare-disease competence,pregnancy/dose safety or validated emergency recognition. Ten medical evaluation categories are explicitly pending expert reference criteria. No passing protocol score clears clinical release.

## Synthetic operator recovery

There is no active email/SMS reset service. A trusted local operator with database access can run:

```bash
RAPHAEL_DEV_SYNTHETIC_ONLY=1 python -m auth.dev_recover --database-url sqlite:////absolute/path/synthetic.db --tenant synthetic --username patient
```

On Windows PowerShell,set `$env:RAPHAEL_DEV_SYNTHETIC_ONLY='1'` first and run the Python command without the Unix prefix. The CLI privately prompts twice,replaces only an active matching account credential,atomically revokes all its sessions and records an operator event without impersonating the patient. It does not reactivate disabled accounts,change medical records or send an email. This is a synthetic operator recovery procedure,not production identity verification or a public recovery endpoint.
