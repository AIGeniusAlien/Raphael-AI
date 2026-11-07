"""Count transport/profile gate tests;never evidence that Xcode ran locally."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('native_profile_summary',Path(__file__).resolve().parents[1]/'.github/scripts/native_test_summary.py');summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)
class NativeProfileSummaryTests(unittest.TestCase):
 def bundle(self,p):return Path('mobile/artifacts/NativeRun.'+p+'.abcdef/NativeTests.xcresult')
 def counts(self,n,failed=0,skipped=0):return dict(passedTests=n,failedTests=failed,skippedTests=skipped,totalTestCount=n+failed+skipped)
 def test_distinct_real_profile_counts_required_not_single_success(self):
  data=summary.summarize([self.bundle('development'),self.bundle('consumer_candidate')],lambda p:self.counts(28 if 'development' in str(p) else 1))
  self.assertEqual(data['passedTests'],29);self.assertTrue(summary.accepted(data,{'development':28,'consumer_candidate':1}))
  one=summary.summarize([self.bundle('development')],lambda p:self.counts(29));self.assertFalse(summary.accepted(one,{'development':28,'consumer_candidate':1}))
 def test_duplicate_profile_skips_failures_and_missing_counts_fail(self):
  for reader in [lambda p:self.counts(28,skipped=1),lambda p:self.counts(28,failed=1),lambda p:{},lambda p:dict(passedTests=True,failedTests=0,skippedTests=0,totalTestCount=1)]:
   data=summary.summarize([self.bundle('development')],reader);self.assertFalse(summary.accepted(data,{'development':28}))
  data=summary.summarize([self.bundle('development'),Path('mobile/artifacts/NativeRun.development.other/NativeTests.xcresult')],lambda p:self.counts(28));self.assertFalse(summary.accepted(data,{'development':28}))
