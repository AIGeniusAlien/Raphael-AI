"""Audit installed local-version builds against public PyPI advisory metadata.

pip-audit may skip versions such as torch==2.15.0+cpu because PyPI only
publishes 2.15.0. This supplementary check audits their public base version;
it does not independently attest the CPU binary or replace pip-audit.
Network/missing metadata failures fail closed, with a machine-readable report.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import urllib.parse
import urllib.request
from packaging.version import Version


def audit_distribution(name, installed_version, fetch=None):
    public_version = Version(installed_version).public
    url = f'https://pypi.org/pypi/{urllib.parse.quote(name, safe="")}/{public_version}/json'
    if fetch is None:
        def fetch(url):
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.load(response)
    result = {'name': name, 'installed_version': installed_version,
              'public_version': public_version, 'source': url}
    try:
        metadata = fetch(url)
        # Missing advisory data is an error, never a clean result.
        advisories = metadata['vulnerabilities']
        if not isinstance(advisories, list):
            raise ValueError('invalid vulnerability metadata')
        result['vulnerabilities'] = [v for v in advisories if not v.get('withdrawn')]
        result['status'] = 'vulnerable' if result['vulnerabilities'] else 'no_public_advisories'
    except Exception as exc:
        result.update(status='error', error=type(exc).__name__)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    distributions = sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()
                           if Version(d.version).local is not None)
    results = [audit_distribution(name, version) for name, version in distributions]
    report = {'scope': 'Installed local-version builds; supplement to pip-audit. Public metadata does not independently attest binary contents.',
              'results': results}
    encoded = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end='')
    return int(any(r['status'] != 'no_public_advisories' for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
