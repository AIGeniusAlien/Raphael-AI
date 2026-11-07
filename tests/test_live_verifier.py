"""Verifier acceptance contract;no network fixtures constitute live proof."""
import unittest
from unittest.mock import patch
from scripts.verify_live_medical import completed_supported_workflows,live_environment


class LiveVerifierTests(unittest.TestCase):
    def test_both_roles_must_complete_and_pass_support_screen(self):
        result={'physician_status':'complete','claim_screen':{'status':'passed'},'patient_workflow':{'status':'complete','claim_screen':{'status':'passed'}}}
        self.assertTrue(completed_supported_workflows(result))
        for status in ('blocked','failed','generating'):
            self.assertFalse(completed_supported_workflows({**result,'physician_status':status}))
            self.assertFalse(completed_supported_workflows({**result,'patient_workflow':{**result['patient_workflow'],'status':status}}))
        for screen in (None,{}, {'status':'blocked'}):
            self.assertFalse(completed_supported_workflows({**result,'claim_screen':screen}))
            self.assertFalse(completed_supported_workflows({**result,'patient_workflow':{**result['patient_workflow'],'claim_screen':screen}}))

    def test_live_verification_cannot_disable_claim_screening_via_inherited_flags(self):
        with patch.dict('os.environ',{'RAPHAEL_VERIFY_CLAIMS':'0','RAPHAEL_STREAM_EVIDENCE_BACKEND':'sqlite'}):
            env=live_environment('sqlite:///temporary-synthetic.sqlite')
        self.assertEqual(env['RAPHAEL_VERIFY_CLAIMS'],'1')
        self.assertEqual(env['RAPHAEL_STREAM_EVIDENCE_BACKEND'],'qdrant')
        self.assertEqual(env['RAPHAEL_DEV_SYNTHETIC_ONLY'],'1')
