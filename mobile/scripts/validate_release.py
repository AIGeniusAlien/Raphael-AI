"""Fail release configuration before packaging;does not sign,upload or validate legal approval."""
import argparse
import json
import os
import re
from urllib.parse import urlsplit


def validate(values):
    issues=[]
    raw=values.get('RAPHAEL_SERVER_URL','');host=values.get('RAPHAEL_SERVER_HOST','').lower()
    try:
        url=urlsplit(raw)
        valid=(url.scheme=='https' and bool(url.hostname) and url.hostname.lower()==host and url.username is None and url.password is None and not url.query and not url.fragment and url.path in ('','/') and url.port in (None,443) and '.' in host and ':' not in host and not all(x.isdigit() for x in host.split('.')) and not host.endswith(('.invalid','.localhost','.local','.test')) and '$' not in host)
    except ValueError:valid=False
    if not valid:issues.append('authorized_https_origin_and_matching_bound_hostname_required')
    bundle=values.get('RAPHAEL_BUNDLE_IDENTIFIER','')
    if not re.fullmatch(r'[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+){2,}',bundle) or bundle.startswith(('org.example.','com.example.','owner.')):issues.append('owner_registered_bundle_identifier_required')
    if not re.fullmatch(r'[A-Za-z0-9._+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',values.get('RAPHAEL_SUPPORT_EMAIL','')):issues.append('real_support_contact_required')
    if values.get('CODE_SIGNING_ALLOWED','YES')!='NO' and values.get('PLATFORM_NAME','iphoneos')=='iphoneos' and not re.fullmatch(r'[A-Z0-9]{10}',values.get('DEVELOPMENT_TEAM','')):issues.append('owner_development_team_required_for_device_signing')
    return issues


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build-environment',action='store_true',required=True);parser.parse_args()
    issues=validate(os.environ)
    print(json.dumps({'release_configuration':'blocked' if issues else 'validated','issues':issues,'signing_or_submission_performed':False}))
    return 1 if issues else 0


if __name__=='__main__':raise SystemExit(main())
