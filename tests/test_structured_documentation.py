"""Actual authorized chart -> clinician fields -> versioned approval -> patient publication."""
import json
import unittest
from tests import test_intelligence as fixture
from clinical_documentation import STRUCTURED_FORMATS

class StructuredDocumentationTests(unittest.TestCase):
 setUp=fixture.IntelligenceTests.setUp
 login=fixture.IntelligenceTests.login
 session=fixture.IntelligenceTests.session
 def test_all_six_formats_preserve_provenance_edit_approval_and_patient_visibility(self):
  for format,names in STRUCTURED_FORMATS.items():
   with self.subTest(format=format):
    self.login();sid=self.session();path=f'/api/dev/intelligence/sessions/{sid}'
    fields={name:{'text':'Synthetic clinician input for '+name,'chart_entry_ids':[],'transcript_segment_ids':[]} for name in names}
    response=self.client.post(path+'/notes',json={'expected_version':0,'format':format,'fields':fields});self.assertEqual(response.status_code,200,response.text)
    self.assertIn('No diagnosis, order, claim',response.json()['content'])
    first=self.client.get(path).json()['notes'][0];snapshot=json.loads(first['source_snapshot']);self.assertEqual(snapshot['clinician_fields'],fields)
    self.client.post(path+'/notes',json={'expected_version':1,'format':format,'content':response.json()['content']+'\nClinician reviewed.'}).raise_for_status()
    self.assertEqual(self.client.get(path).json()['notes'][1]['source_snapshot'],first['source_snapshot'])
    self.assertEqual(self.client.post(path+'/approve',json={'expected_version':1}).status_code,409)
    self.client.post(path+'/approve',json={'expected_version':2}).raise_for_status()
    self.assertEqual(self.client.post(path+'/notes',json={'expected_version':2,'format':format,'fields':fields}).status_code,409)
    self.login('patient');published=self.client.get(f"/api/dev/patients/{self.patient['id']}/notes");self.assertTrue(any(format in row['content'] and 'Clinician reviewed.' in row['content'] for row in published.json()))
 def test_missing_fields_are_explicit_and_unrelated_provenance_rejected(self):
  sid=self.session();path=f'/api/dev/intelligence/sessions/{sid}/notes'
  bad={'expected_version':0,'format':'care_plan','fields':{'goals':{'text':'Synthetic','chart_entry_ids':[999999]}}}
  self.assertEqual(self.client.post(path,json=bad).status_code,400)
  bad['fields']['goals']={'text':'Synthetic','transcript_segment_ids':[999999]};self.assertEqual(self.client.post(path,json=bad).status_code,400)
  bad['fields']={'unknown':{'text':'Synthetic'}};self.assertEqual(self.client.post(path,json=bad).status_code,400)
  bad['fields']={'goals':{'text':'Synthetic goal'}};r=self.client.post(path,json=bad);r.raise_for_status();self.assertIn('Follow Up: NOT DOCUMENTED',r.json()['content'])
  record=self.client.get(f'/api/dev/intelligence/sessions/{sid}').json()['notes'][0];self.assertIn('follow_up',json.loads(record['source_snapshot'])['missing_fields'])
 def test_metadata_and_forms_are_clinician_only_and_session_owner_scoped(self):
  sid=self.session();path=f'/api/dev/intelligence/sessions/{sid}/notes'
  self.assertEqual(len(self.client.get('/api/dev/intelligence/documentation/formats').json()['formats']),13)
  self.login('second');self.assertEqual(self.client.post(path,json={'expected_version':0,'format':'care_plan'}).status_code,404)
  self.login('patient');self.assertEqual(self.client.get('/api/dev/intelligence/documentation/formats').status_code,403);self.assertEqual(self.client.post(path,json={'expected_version':0,'format':'care_plan'}).status_code,403)
