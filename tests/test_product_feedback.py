"""Real synthetic feedback/task state machine;no training,deployment or clinical proof."""
import json
from pathlib import Path
import unittest
from uuid import uuid4
from sqlalchemy import text
from tests import test_intelligence as fixture
from dev_portal import create_dev_user
from dev_feedback import init_feedback_schema,install_feedback

class ProductFeedbackTests(unittest.TestCase):
    login=fixture.IntelligenceTests.login
    def setUp(self):
        fixture.IntelligenceTests.setUp(self);init_feedback_schema(self.engine)
        for tenant,name in [('one','admin'),('one','backup_admin'),('two','foreign_admin')]:create_dev_user(self.engine,tenant,name,self.password,'org_admin')
        install_feedback(self.app,self.ctx)
    def report(self,**changes):
        body={'request_id':str(uuid4()),'category':'bug','impact':'blocking','title':'Synthetic menu issue','detail':'Synthetic menu does not retain the expected focus. No patient content.','synthetic_only_ack':True,'no_sensitive_data_ack':True};body.update(changes)
        return self.client.post('/api/dev/feedback/reports',json=body)
    def action(self,row,action,**changes):
        body={'request_id':str(uuid4()),'expected_revision':row['revision'],'action':action,'note':'Synthetic operator review,not clinical validation.'};body.update(changes)
        return self.client.post(f"/api/dev/feedback/reports/{row['id']}/actions",json=body)
    def triaged(self,**changes):
        row=self.report(**changes).json();self.login('admin');response=self.action(row,'triage',priority='P1');self.assertEqual(response.status_code,200,response.text);return response.json()
    def task(self,**changes):
        row=self.triaged(**changes);response=self.action(row,'create_task',requirement_id='CORE-007');self.assertEqual(response.status_code,200,response.text);return response.json()

    def test_repeat_migration_preserves_reports_tasks_and_rejects_older_runtime(self):
        from dev_runtime import migrate,COMPONENTS
        row=self.task();migrate(self.engine);migrate(self.engine)
        current=self.client.get(f"/api/dev/feedback/reports/{row['id']}").json()
        self.assertEqual(current['task']['id'],row['task']['id']);self.assertEqual(current['revision'],row['revision'])
        with __import__('unittest').mock.patch('dev_runtime.COMPONENTS',tuple(item for item in COMPONENTS if item[0]!='product_feedback')):
            with self.assertRaisesRegex(RuntimeError,'Unsupported'):migrate(self.engine)
        self.assertEqual(self.client.get(f"/api/dev/feedback/reports/{row['id']}").json()['detail'],row['detail'])

    def test_submission_acknowledgment_idempotency_and_sensitive_input(self):
        request=str(uuid4());first=self.report(request_id=request);self.assertEqual(first.status_code,200,first.text)
        self.assertEqual(self.report(request_id=request).json()['id'],first.json()['id'])
        self.assertEqual(self.report(request_id=request,detail='Different synthetic request body.').status_code,409)
        self.assertEqual(self.report(no_sensitive_data_ack=False).status_code,422)
        self.assertEqual(self.report(detail='password=DO-NOT-STORE-SYNTHETIC-CREDENTIAL').status_code,422)
        self.assertEqual(self.report(detail='sk-SYNTHETIC_LONG_NOT_A_REAL_KEY').status_code,422)
        self.assertFalse(first.json()['external_delivery']);self.assertFalse(first.json()['clinical_validation'])
        with self.engine.connect() as c:self.assertEqual(c.execute(text('SELECT count(*) FROM dev_product_feedback')).scalar(),1)

    def test_owner_role_and_tenant_boundaries_hide_private_admin_notes(self):
        row=self.report().json();self.login('second');self.assertEqual(self.client.get(f"/api/dev/feedback/reports/{row['id']}").status_code,404);self.assertEqual(self.action(row,'triage').status_code,403)
        self.assertEqual(self.client.get('/api/dev/feedback/reports').json()['items'],[])
        self.login('foreign_admin','two');self.assertEqual(self.client.get(f"/api/dev/feedback/reports/{row['id']}").status_code,404);self.assertEqual(self.action(row,'triage').status_code,404)
        self.login('admin');updated=self.action(row,'triage',note='INTERNAL-SYNTHETIC-ADMIN-NOTE').json();self.assertIn('INTERNAL-SYNTHETIC-ADMIN-NOTE',json.dumps(updated))
        self.login();own=self.client.get(f"/api/dev/feedback/reports/{row['id']}").json();self.assertNotIn('INTERNAL-SYNTHETIC-ADMIN-NOTE',json.dumps(own));self.assertEqual(own['state'],'triaged')
        self.assertEqual(self.client.get('/api/dev/feedback/requirements?q=CORE-007').status_code,403)

    def test_complete_admin_task_workflow_records_unverified_pointer_without_changing_guidance(self):
        before=Path('docs/ENGINEERING_LEDGER.json').read_bytes();row=self.task();self.assertEqual(row['task']['state'],'planned')
        row=self.action(row,'start_task').json()
        for pointer in ('http://127.0.0.1/private','docs/../../private-key','https://foreign.invalid/ci'):
            self.assertEqual(self.action(row,'resolve',evidence_reference=pointer).status_code,422)
        response=self.action(row,'resolve',evidence_reference='docs/TEST_RESULTS.md');self.assertEqual(response.status_code,200,response.text);row=response.json()
        self.assertEqual(row['state'],'resolved');self.assertEqual(row['task']['state'],'done');self.assertFalse(row['evidence_pointer_verified']);self.assertFalse(row['clinical_validation'])
        self.assertEqual(self.action(row,'note').status_code,409);self.assertTrue(Path('docs/ENGINEERING_LEDGER.json').read_bytes()==before)
        with self.engine.connect() as c:self.assertEqual(c.execute(text('SELECT count(*) FROM dev_ai_messages')).scalar(),0)

    def test_action_replay_and_stale_versions_cannot_duplicate_or_rewrite_tasks(self):
        row=self.triaged();request=str(uuid4())
        first=self.action(row,'create_task',request_id=request,requirement_id='CORE-007');self.assertEqual(first.status_code,200,first.text)
        replay=self.action(row,'create_task',request_id=request,requirement_id='CORE-007');self.assertTrue(replay.json()['operation_replayed'])
        self.assertEqual(self.action(row,'create_task',request_id=request,requirement_id='UNKNOWN-000').status_code,409)
        self.assertEqual(self.action(row,'reject').status_code,409)
        with self.engine.connect() as c:self.assertEqual(c.execute(text('SELECT count(*) FROM dev_feedback_tasks')).scalar(),1)

    def test_clinical_safety_feedback_requires_qualified_review_not_admin_resolution(self):
        for category,impact in [('clinical_information','routine'),('bug','possible_safety_concern')]:
            self.login();row=self.task(category=category,impact=impact);row=self.action(row,'start_task').json()
            response=self.action(row,'resolve',evidence_reference='docs/TEST_RESULTS.md');self.assertEqual(response.status_code,409);self.assertIn('qualified-review',response.text)
            unchanged=self.client.get(f"/api/dev/feedback/reports/{row['id']}").json();self.assertEqual(unchanged['state'],'in_progress');self.assertFalse(unchanged['clinical_validation'])

    def test_requirement_and_assignee_validation(self):
        row=self.triaged();self.assertEqual(self.action(row,'create_task',requirement_id='NOT-A-REAL-REQUIREMENT').status_code,422)
        with self.engine.connect() as c:foreign=c.execute(text("SELECT id FROM dev_users WHERE username='foreign_admin'")).scalar();backup=c.execute(text("SELECT id FROM dev_users WHERE username='backup_admin'")).scalar()
        self.assertEqual(self.action(row,'create_task',requirement_id='CORE-007',assignee_id=foreign).status_code,404)
        row=self.action(row,'create_task',requirement_id='CORE-007',assignee_id=backup).json();self.assertEqual(row['task']['assigned_to'],backup)
        self.assertEqual(self.action(row,'assign_task',assignee_id=foreign).status_code,404)
        ids=[item['id'] for item in self.client.get('/api/dev/feedback/assignees').json()];self.assertNotIn(foreign,ids)
        result=self.client.get('/api/dev/feedback/requirements?q=CORE-007').json()['items'];self.assertTrue(any(item['id']=='CORE-007' for item in result))

    def test_revocation_after_dependency_authentication_prevents_submission(self):
        def stale_authenticated_user(request):
            user=self.ctx.user(request)
            with self.engine.begin() as c:c.execute(text('DELETE FROM dev_sessions WHERE user_id=:u'),{'u':user['id']})
            return user
        # Preserve the real dependency's Request annotation for FastAPI.
        from fastapi import Request
        stale_authenticated_user.__annotations__={'request':Request}
        self.app.dependency_overrides[self.ctx.user]=stale_authenticated_user
        response=self.report();self.assertEqual(response.status_code,401)
        with self.engine.connect() as c:self.assertEqual(c.execute(text('SELECT count(*) FROM dev_product_feedback')).scalar(),0)

    def test_pagination_is_owner_scoped_and_scripts_remain_literal_data(self):
        row=self.report(title='<script>Synthetic script title</script>').json();self.assertIn('<script>',row['title'])
        self.assertEqual(self.client.get('/api/dev/feedback/reports?after='+str(row['id'])).json()['items'],[])
        self.assertEqual(self.client.get('/api/dev/feedback/reports?after=-1').status_code,422)
        self.assertIsNone(self.client.get('/api/dev/feedback/reports').json()['next_after'])
        self.login('admin')
        for _ in range(21):
            response=self.action(row,'note');self.assertEqual(response.status_code,200,response.text);row=response.json()
        page=self.client.get('/api/dev/feedback/reports').json()['items'][0]
        self.assertEqual(len(page['actions']),20);self.assertTrue(page['actions_truncated'])
        detail=self.client.get(f"/api/dev/feedback/reports/{row['id']}").json()
        self.assertEqual(len(detail['actions']),21);self.assertFalse(detail['actions_truncated'])
