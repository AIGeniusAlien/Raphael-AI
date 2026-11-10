"""Synthetic scheduling API integration; no live healthcare appointments."""
import unittest
from datetime import datetime, timedelta, timezone
import test_account as fixture


class SchedulingTests(unittest.TestCase):
    setUp=fixture.AccountTests.setUp
    login=fixture.AccountTests.login

    def appointment(self,patient=None,resources=None,start=None):
        start=start or datetime.now(timezone.utc)+timedelta(days=3)
        start=start.replace(microsecond=0)
        return {'patient_id':patient or self.pid,'start':start.isoformat(),
                'end':(start+timedelta(minutes=30)).isoformat(),
                'resources':['clinician:doctor','room:one'] if resources is None else resources,
                'reason':'Synthetic consultation'}

    def test_booking_conflicts_cancel_and_patient_visibility(self):
        self.login('doctor')
        payload=self.appointment()
        first=self.client.post('/api/dev/scheduling/appointments',json=payload)
        self.assertEqual(first.status_code,201,first.text)
        self.assertFalse(first.json()['external_booking_confirmed'])
        self.assertEqual(self.client.post('/api/dev/scheduling/appointments',json=payload).status_code,409)
        self.login()
        visible=self.client.get('/api/dev/scheduling/appointments')
        self.assertEqual(visible.status_code,200,visible.text)
        self.assertEqual(len(visible.json()),1)
        self.assertEqual(self.client.post('/api/dev/scheduling/appointments',json=payload).status_code,403)
        self.assertEqual(self.client.post('/api/dev/scheduling/appointments/1/cancel',json={}).status_code,403)
        self.login('doctor')
        self.client.post('/api/dev/scheduling/appointments/'+str(first.json()['id'])+'/cancel',json={}).raise_for_status()
        self.assertEqual(self.client.post('/api/dev/scheduling/appointments',json=payload).status_code,201)

    def test_conflict_preflight_is_private_and_not_a_booking_guarantee(self):
        self.login('doctor')
        payload=self.appointment()
        params={'start':payload['start'],'end':payload['end'],'resource':'room:one'}
        initial=self.client.get('/api/dev/scheduling/conflicts',params=params)
        self.assertEqual(initial.status_code,200,initial.text)
        self.assertFalse(initial.json()['conflict'])
        self.assertFalse(initial.json()['availability_confirmed'])
        self.client.post('/api/dev/scheduling/appointments',json=payload).raise_for_status()
        self.assertTrue(self.client.get('/api/dev/scheduling/conflicts',params=params).json()['conflict'])
        self.login()
        self.assertEqual(self.client.get('/api/dev/scheduling/conflicts',params=params).status_code,403)
        self.login('foreign',tenant='two')
        self.assertFalse(self.client.get('/api/dev/scheduling/conflicts',params=params).json()['conflict'])

    def test_tenant_and_patient_isolation(self):
        self.login('doctor')
        self.assertEqual(self.client.post('/api/dev/scheduling/appointments',json=self.appointment(patient=999999)).status_code,404)
        self.client.post('/api/dev/scheduling/appointments',json=self.appointment()).raise_for_status()
        self.login('other')
        self.assertEqual(self.client.get('/api/dev/scheduling/appointments').json(),[])
        self.login('foreign',tenant='two')
        self.assertEqual(self.client.get('/api/dev/scheduling/appointments').json(),[])

    def test_invalid_resources_and_time_rejected(self):
        self.login('doctor')
        for resources in ([],['room:one','room:one'],[' '],[' room:one']):
            self.assertEqual(self.client.post('/api/dev/scheduling/appointments',
                             json=self.appointment(resources=resources)).status_code,422)
        payload=self.appointment()
        payload['start']='2026-11-09T12:00:00'
        self.assertEqual(self.client.post('/api/dev/scheduling/appointments',json=payload).status_code,422)
