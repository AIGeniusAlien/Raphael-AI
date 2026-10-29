# Security dependency remediation

Security-sensitive dependency upgrades are pinned in requirements.txt, aligned in pyproject.toml and captured transitively in requirements.lock. Install torch==2.6.0+cpu from https://download.pytorch.org/whl/cpu first, then install the lock with --no-deps; use requirements-dev.txt for optional browser tooling. Version upgrades were resolved against the verified package registry without disabling verification.

Replaced python-jose/ecdsa with PyJWT cryptography support. Legacy JWT signing now refuses missing/historical default keys and decoding requires expiry/subject/tenant/role with HS256 only. Existing portal cookie sessions and database records were not changed.

Validation:97 full Python tests passed in64.830s;2 dedicated JWT regressions also passed separately;2 real Chromium scenarios passed in32.542s; dependency compatibility passed for86 installed distributions. The prior attempt during an incomplete model-library upgrade failed import because PyTorch2.5 was unsupported; CPU PyTorch2.6 corrected that and the complete suite was rerun successfully.

Known-advisory audit is a package-database snapshot,not independent security certification,malware detection or clinical validation. Historical dependency audit failures remain in earlier reports for traceability; this document records the newer remediated build. No public application deployment or live clinical-service validation was performed.

Final pip-audit run after the CPU runtime upgrade: no known vulnerabilities found. Raw dependency/advisory snapshot is retained in docs/dependency-audit.json.
