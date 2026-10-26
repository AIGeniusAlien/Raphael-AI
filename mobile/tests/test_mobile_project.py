"""Build-source and packaging regressions;these do not execute native Swift code."""
import importlib.util
from pathlib import Path
import plistlib
import shutil
import tempfile
import unittest

MOBILE=Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,MOBILE/'scripts'/f'{name}.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
project=load('project_audit');release=load('validate_release')


class MobileSourceProjectTests(unittest.TestCase):
    def test_actual_xcode_targets_sources_resources_icons_and_privacy_are_consistent(self):
        result=project.audit(MOBILE/'ios');self.assertEqual(result['native_targets'],3);self.assertEqual(result['xcode_compile'],'NOT_RUN')

    def test_dangling_project_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            copy=Path(directory)/'mobile';shutil.copytree(MOBILE,copy,ignore=shutil.ignore_patterns('__pycache__'))
            path=copy/'ios/Raphael.xcodeproj/project.pbxproj';text=path.read_text();key=project.parse_project(text)['rootObject'];path.write_text(text.replace('"rootObject" = "'+key+'"','"rootObject" = "FFFFFFFFFFFFFFFFFFFFFFFF"'))
            with self.assertRaises((KeyError,ValueError)):project.audit(copy/'ios')

    def test_arbitrary_http_transport_exception_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            copy=Path(directory)/'mobile';shutil.copytree(MOBILE,copy,ignore=shutil.ignore_patterns('__pycache__'))
            path=copy/'ios/Raphael/Info.plist';info=plistlib.loads(path.read_bytes());info['NSAppTransportSecurity']['NSAllowsArbitraryLoads']=True;path.write_bytes(plistlib.dumps(info))
            with self.assertRaises(ValueError):project.audit(copy/'ios')

    def test_release_requires_actual_origin_bundle_and_support_instead_of_placeholders(self):
        self.assertGreaterEqual(len(release.validate({})),3)
        values={'RAPHAEL_SERVER_URL':'https://raphael.example.org','RAPHAEL_SERVER_HOST':'raphael.example.org','RAPHAEL_BUNDLE_IDENTIFIER':'org.raphael.synthetic','RAPHAEL_SUPPORT_EMAIL':'support@example.org','CODE_SIGNING_ALLOWED':'NO'}
        self.assertEqual(release.validate(values),[])
        for invalid in ('http://raphael.example.org','https://user:password@raphael.example.org','https://raphael.example.org?token=synthetic','https://raphael.example.org/private','https://raphael.example.org:8443'):
            with self.subTest(url=invalid):self.assertIn('authorized_https_origin_and_matching_bound_hostname_required',release.validate({**values,'RAPHAEL_SERVER_URL':invalid}))

    def test_device_signing_gate_does_not_invent_an_owner_team(self):
        values={'RAPHAEL_SERVER_URL':'https://raphael.example.org','RAPHAEL_SERVER_HOST':'raphael.example.org','RAPHAEL_BUNDLE_IDENTIFIER':'org.raphael.synthetic','RAPHAEL_SUPPORT_EMAIL':'support@example.org','CODE_SIGNING_ALLOWED':'YES','PLATFORM_NAME':'iphoneos'}
        self.assertIn('owner_development_team_required_for_device_signing',release.validate(values))
