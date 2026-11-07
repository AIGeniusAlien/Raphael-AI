"""Entered-record arithmetic and minimization tests,not medical validation."""
from datetime import datetime,timezone
import json
import unittest
from longitudinal import review,provider_context


AT=datetime(2026,11,1,tzinfo=timezone.utc)


def entry(identifier=1,**extra):
    return {'id':identifier,'kind':'lab','label':'Synthetic analyte','recorded_at':'2020-01-01T00:00:00+00:00','status':'recorded','revision':1,'data':{'value':5,'unit':'test-unit','source':'Synthetic laboratory'},**extra}


class LongitudinalTests(unittest.TestCase):
    def test_chronology_arithmetic_provenance_and_age_are_not_clinical_inference(self):
        rows=[entry(2,recorded_at='2020-01-02T00:00:00+00:00',data={'value':15,'unit':'test-unit','source':'Synthetic laboratory'}),entry()]
        result=review(rows,AT)
        self.assertEqual([r['entry_id'] for r in result['timeline']],[1,2])
        self.assertEqual(result['trends'][0]['delta_first_to_last'],10)
        self.assertEqual(result['timeline'][0]['classification'],'reported_record')
        self.assertGreater(result['timeline'][0]['age_days'],2000)
        self.assertIn('no unit conversion,diagnosis',result['limitations'])

    def test_units_and_sources_are_not_silently_mixed(self):
        rows=[entry(),entry(2,data={'value':10,'unit':'different-unit','source':'Synthetic laboratory'}),entry(3,data={'value':20,'unit':'test-unit','source':'Different laboratory'})]
        result=review(rows,AT)
        self.assertEqual(len(result['trends']),3)
        self.assertTrue(all(t['delta_first_to_last'] is None for t in result['trends']))

    def test_conflicting_same_time_values_block_arithmetic_and_are_in_provider_context(self):
        rows=[entry(),entry(2,data={'value':15,'unit':'test-unit','source':'Synthetic laboratory'})]
        trend=review(rows,AT)['trends'][0]
        self.assertTrue(trend['comparison_blocked']);self.assertIsNone(trend['delta_first_to_last'])
        context=provider_context(rows,'Synthetic analyte',AT)
        self.assertTrue(all('same_time_conflicting_values' in r['review_flags'] for r in context['records']))

    def test_draft_future_and_invalid_dates_do_not_enter_provider_context_or_trends(self):
        rows=[entry(),entry(2,status='draft'),entry(3,recorded_at='2099-01-01T00:00:00+00:00'),entry(4,recorded_at='invalid')]
        result=review(rows,AT)
        self.assertEqual(result['excluded_from_trends'],{'draft':1,'future':1,'invalid_time':1})
        self.assertEqual([r['id'] for r in provider_context(rows,'question',AT)['records']],[1])
        self.assertEqual(len(result['timeline']),4)

    def test_context_prefers_question_matches_but_preserves_whole_bounded_facts_and_hashes(self):
        rows=[entry(1,kind='history',label='Needle',data={'text':'RELEVANT '+'x'*1200,'source':'Manual'}),entry(2,kind='history',recorded_at='2021-01-01T00:00:00+00:00',data={'text':'IRRELEVANT '+'x'*1200,'source':'Manual'})]
        context=provider_context(rows,'Needle',AT,max_chars=2500)
        self.assertLessEqual(len(json.dumps(context)),2500)
        self.assertEqual([r['id'] for r in context['records']],[1])
        self.assertEqual(context['records'][0]['data']['text'],rows[0]['data']['text'])
        self.assertEqual(len(context['records'][0]['snapshot_sha256']),64)
        self.assertEqual(context['records_omitted'],1)

    def test_large_histories_are_explicitly_windowed_and_future_values_do_not_become_current(self):
        rows=[entry(i) for i in range(1,202)]
        result=review(rows,AT);self.assertTrue(result['records_truncated']);self.assertEqual(len(result['timeline']),200)
        self.assertTrue(provider_context(rows,'Synthetic analyte',AT)['older_records_omitted'])
        with self.assertRaises(ValueError):provider_context(rows,'question',datetime(2026,11,1))

    def test_extreme_arithmetic_is_blocked_and_json_remains_finite(self):
        rows=[entry(1,data={'value':-1e308,'unit':'test-unit','source':'Synthetic laboratory'}),entry(2,recorded_at='2020-01-02T00:00:00+00:00',data={'value':1e308,'unit':'test-unit','source':'Synthetic laboratory'})]
        result=review(rows,AT)
        self.assertTrue(result['trends'][0]['comparison_blocked'])
        json.dumps(result,allow_nan=False)
