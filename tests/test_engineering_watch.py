"""Read-only engineering report contracts,not persistent-runner acceptance."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.engineering_watch import audit,summarize_runs,write_report

class EngineeringWatchTests(unittest.TestCase):
 def setUp(self):
  self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup);self.root=Path(self.directory.name);(self.root/'docs').mkdir();self.head='a'*40
  self.rows=[{'requirement_id':'TEST-001','state':'VERIFIED','priority':'P0'},{'requirement_id':'TEST-002','state':'PARTIAL','priority':'P0'}];self.save()
  (self.root/'docs/RAPHAEL_FINAL_ACCEPTANCE.json').write_text(json.dumps({'release_decisions':{'synthetic':'PASS','production':'FAIL'}}))
 def save(self): (self.root/'docs/ENGINEERING_LEDGER.json').write_text(json.dumps({'requirement_count':len(self.rows),'requirements':self.rows}))
 def test_scoped_report_proposals_preserve_unverified_gates(self):
  before=(self.root/'docs/ENGINEERING_LEDGER.json').read_bytes();r=audit(self.root,head=self.head,clean=False)
  self.assertEqual(r['proposed_work'],[{'requirement_id':'TEST-002','state':'PARTIAL','priority':'P0'}]);self.assertFalse(r['github_metadata_checked']);self.assertEqual(r['recorded_release_gates']['production'],'FAIL');self.assertEqual(before,(self.root/'docs/ENGINEERING_LEDGER.json').read_bytes());self.assertEqual(r['persistent_execution'],'not_proven_by_one_shot')
 def test_duplicate_or_unknown_state_and_count_refused(self):
  self.rows.append(self.rows[0]);self.save()
  with self.assertRaises(ValueError):audit(self.root,head=self.head,clean=True)
  self.rows=self.rows[:2];self.rows[0]['state']='CLINICALLY_SAFE';self.save()
  with self.assertRaises(ValueError):audit(self.root,head=self.head,clean=True)
 def test_current_source_only_and_no_pending_success_inference(self):
  r=summarize_runs({'workflow_runs':[{'id':7,'head_sha':self.head,'status':'in_progress','conclusion':None},{'id':8,'head_sha':'b'*40,'status':'completed','conclusion':'success'}]},self.head)
  self.assertEqual(len(r),1);self.assertIsNone(r[0]['conclusion']);self.assertEqual(r[0]['run_id'],7)
 def test_forged_status_and_unbounded_inventory_refused(self):
  for payload in [{'workflow_runs':[{}]*21},{'workflow_runs':[{'head_sha':self.head,'id':1,'status':'clinically_validated','conclusion':'success'}]}]:
   with self.assertRaises(ValueError):summarize_runs(payload,self.head)
 def test_report_exclusive_private_and_symlink_refusal(self):
  path=self.root/'report.json';write_report(path,{'ok':True});self.assertEqual(path.stat().st_mode&0o777,0o600)
  with self.assertRaises(FileExistsError):write_report(path,{'overwritten':True})
  link=self.root/'symlink.json';link.symlink_to(path)
  with self.assertRaises(OSError):write_report(link,{'overwritten':True})
  self.assertEqual(json.loads(path.read_text()),{'ok':True})
