"""Real auth/persistence tests using nonclinical, test-owned checklist text."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4
from fastapi import FastAPI,Request,HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine,text
from dev_knowledge import init_knowledge_schema,install_knowledge
from dev_portal import init_portal_schema,create_dev_user,install_dev_portal
from dev_procedures import init_procedure_schema,install_procedures,SECTIONS,export_patient_procedures

class ProcedureTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        flag=patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'});flag.start();self.addCleanup(flag.stop)
        self.engine=create_engine('sqlite:///'+str(Path(directory.name)/'synthetic.db'));self.addCleanup(self.engine.dispose)
        init_portal_schema(self.engine);init_knowledge_schema(self.engine);init_procedure_schema(self.engine)
        self.password='Synthetic-procedure-test-password'
        for tenant,name,role in [('a','doctor','clinician'),('a','receiver','clinician'),('a','patient','patient'),('a','other','patient'),('a','admin','org_admin'),('b','outsider','clinician')]:create_dev_user(self.engine,tenant,name,self.password,role)
        app=FastAPI();ctx=install_dev_portal(app,self.engine);install_knowledge(app,ctx);install_procedures(app,ctx);self.app=app;self.ctx=ctx
        self.client=TestClient(app);self.addCleanup(self.client.close);self.login()
        self.patient=self.post('/api/dev/patients',{'name':'Synthetic pathway fixture','patient_username':'patient'}).json()
        self.other=self.post('/api/dev/patients',{'name':'Other synthetic pathway fixture','patient_username':'other'}).json()
        self.source=self.post('/api/dev/knowledge/sources',{'title':'Synthetic calibration procedure fixture','url':'https://example.org/procedure-fixture','content':'Violet calibrator checklist fixture. Review the synthetic fixture marker before acknowledging the synthetic checklist. No medical instructions.','license_declaration':'Test-owned nonclinical synthetic fixture, permitted in tests.','permission_attested':True,'org':'Synthetic test'}).json()
        self.post(f"/api/dev/knowledge/sources/{self.source['id']}/approve",{})
    def post(self,path,body):
        r=self.client.post(path,json=body);self.assertEqual(r.status_code,200,r.text);return r
    def login(self,name='doctor',tenant='a'):
        r=self.client.post('/api/dev/login',json={'tenant':tenant,'username':name,'password':self.password});self.assertEqual(r.status_code,200,r.text)
    def template(self,**changes):
        value={'slug':'synthetic_calibration','title':'Synthetic nonclinical checklist','specialty':'Synthetic fixture','source_id':self.source['id'],'source_version_sha256':self.source['version_sha256'],'sections':{key:'Not established: nonclinical synthetic fixture only.' for key in SECTIONS},'steps':[{'key':'patient_marker','phase':'preparation','assignee':'patient','instruction':'Review the nonclinical synthetic marker.'},{'key':'clinician_marker','phase':'safety','assignee':'clinician','instruction':'Acknowledge the nonclinical fixture marker.'}]};value.update(changes)
        return self.post('/api/dev/procedures/templates',value).json()
    def approve(self,value,target='approve',**changes):
        body={'content_sha256':value['content_sha256'],'source_version_sha256':value['source_version_sha256'],'content_reviewed':True,'permission_reviewed':True};body.update(changes)
        return self.client.post(f"/api/dev/procedures/templates/{value['id']}/{target}",json=body)
    def start(self):
        template=self.template()
        if template['status']=='pending':
            r=self.approve(template);self.assertEqual(r.status_code,200,r.text)
        value=self.post(f"/api/dev/procedures/patients/{self.patient['id']}/runs",{'template_id':template['id'],'idempotency_key':str(uuid4())}).json();return template,value
    def publish(self,run):return self.post(f"/api/dev/procedures/runs/{run['id']}/publish",{'expected_revision':run['revision'],'content_reviewed':True}).json()
    def authorize(self,run,authorize=True):return self.post(f"/api/dev/procedures/runs/{run['id']}/authorization",{'expected_revision':run['revision'],'authorize':authorize,'understands_coordination_only':True}).json()
    def get(self,run):return self.client.get(f"/api/dev/procedures/runs/{run['id']}")
    def test_patient_list_changes_only_after_committed_clinician_publication(self):
        _, run = self.start()
        route = f"/api/dev/procedures/patients/{self.patient['id']}/runs"
        self.login('patient')
        self.assertEqual(self.client.get(route).json()['runs'], [])
        self.login('other')
        self.assertEqual(self.client.get(route).status_code, 404)
        self.login('outsider', 'b')
        self.assertEqual(self.client.get(route).status_code, 404)
        self.login()
        response = self.publish(run)
        self.assertEqual(response['state'], 'published')
        with self.engine.connect() as connection:
            actual = connection.execute(
                text("SELECT state, revision, published_at FROM dev_procedure_runs WHERE id=:id"),
                {'id': run['id']}
            ).mappings().one()
            self.assertEqual(actual['state'], 'published')
            self.assertIsNotNone(actual['published_at'])
        self.login('patient')
        published = self.client.get(route)
        self.assertEqual(published.status_code, 200, published.text)
        self.assertEqual([item['id'] for item in published.json()['runs']], [run['id']])
        self.assertEqual(published.json()['runs'][0]['state'], 'published')

    def test_complete_clinician_patient_publication_authorization_and_signoff(self):
        template,run=self.start();self.login('patient');self.assertEqual(self.get(run).status_code,404)
        self.login();run=self.publish(run);self.assertFalse(run['actionable'])
        self.assertEqual(self.client.post(f"/api/dev/procedures/runs/{run['id']}/complete",json={'expected_revision':run['revision'],'content_reviewed':True}).status_code,409)
        self.login('patient');self.assertEqual(self.get(run).status_code,200);self.authorize(run);run=self.get(run).json()
        run=self.post(f"/api/dev/procedures/runs/{run['id']}/steps/patient_marker",{'expected_revision':run['revision'],'status':'acknowledged','note':'Synthetic marker reviewed.'}).json()
        self.assertEqual(self.client.post(f"/api/dev/procedures/runs/{run['id']}/steps/clinician_marker",json={'expected_revision':run['revision'],'status':'acknowledged'}).status_code,403)
        self.login();run=self.post(f"/api/dev/procedures/runs/{run['id']}/steps/clinician_marker",{'expected_revision':run['revision'],'status':'not_applicable','note':'Nonclinical fixture has no intervention.'}).json()
        run=self.post(f"/api/dev/procedures/runs/{run['id']}/complete",{'expected_revision':run['revision'],'content_reviewed':True}).json()
        self.assertEqual(run['state'],'completed');self.assertFalse(run['actionable']);self.assertIn('not medical consent',run['notice'])
        self.assertEqual(self.client.post(f"/api/dev/procedures/runs/{run['id']}/steps/patient_marker",json={'expected_revision':run['revision'],'status':'pending'}).status_code,409)
        self.login('patient');final=self.get(run).json();self.assertEqual(final['snapshot']['source_version_sha256'],self.source['version_sha256'])
        self.assertEqual(final['snapshot']['content_sha256'],template['content_sha256'])
        events=self.client.get(f"/api/dev/procedures/runs/{run['id']}/events").json();self.assertEqual([r['revision'] for r in events],list(range(1,7)))
        self.assertEqual(events[-1]['kind'],'complete_procedure_checklist')
    def test_idempotence_schema_repeat_and_immutable_template_versions(self):
        first=self.template();self.assertTrue(self.template()['deduplicated']);self.approve(first)
        body={'template_id':first['id'],'idempotency_key':str(uuid4())};path=f"/api/dev/procedures/patients/{self.patient['id']}/runs"
        run=self.post(path,body).json();self.assertEqual(self.post(path,body).json()['id'],run['id'])
        self.assertEqual(self.client.post(f"/api/dev/procedures/patients/{self.other['id']}/runs",json=body).status_code,409)
        init_procedure_schema(self.engine);self.assertEqual(self.get(run).json()['snapshot'],run['snapshot'])
        revised=self.template(title='Revised synthetic nonclinical checklist');self.assertEqual(revised['version'],2);self.approve(revised)
        self.assertFalse(self.get(run).json()['active_source_and_template']);self.assertEqual(self.get(run).json()['snapshot']['title'],first['title'])
        self.assertEqual(self.client.post(f"/api/dev/procedures/runs/{run['id']}/publish",json={'expected_revision':run['revision'],'content_reviewed':True}).status_code,409)
    def test_source_revoke_and_supersede_disable_patient_instructions_and_actions(self):
        template,run=self.start();run=self.publish(run);self.login('patient');self.authorize(run);self.login()
        self.post(f"/api/dev/knowledge/sources/{self.source['id']}/revoke",{})
        self.assertFalse(self.get(run).json()['active_source_and_template'])
        self.assertEqual(self.client.get('/api/dev/procedures/coverage').json()['facets'],[])
        self.login('patient');self.assertEqual(self.get(run).status_code,409)
        # Revocation is still possible even when old instructions are withheld.
        self.authorize({'id':run['id'],'revision':3},False)
        self.login();self.assertEqual(self.get(run).status_code,404)
    def test_patient_coordination_and_chart_revocation_block_clinicians(self):
        _,run=self.start();run=self.publish(run);self.login('patient');auth=self.authorize(run)
        self.authorize({'id':run['id'],'revision':auth['revision']},False)
        self.login();self.assertEqual(self.get(run).status_code,404)
        self.assertEqual(self.client.get(f"/api/dev/procedures/patients/{self.patient['id']}/runs").json()['runs'],[])
        self.login('patient');run=self.get(run).json();self.authorize(run)
        with self.engine.begin() as c:doctor_id=c.execute(text("SELECT id FROM dev_users WHERE username='doctor' AND tenant='a'")).scalar()
        self.post(f"/api/dev/patients/{self.patient['id']}/chart-access/{doctor_id}/revoke",{})
        self.login();self.assertEqual(self.get(run).status_code,404)
    def test_cross_patient_tenant_role_and_logout_denial(self):
        _,run=self.start();run=self.publish(run)
        for name,tenant in [('other','a'),('receiver','a'),('outsider','b')]:
            self.login(name,tenant);self.assertEqual(self.get(run).status_code,404)
        self.login('admin');self.assertEqual(self.get(run).status_code,403);self.assertEqual(self.client.get('/api/dev/procedures/templates').status_code,403)
        self.login('patient');self.assertEqual(self.client.get('/api/dev/procedures/templates').status_code,403)
        self.post('/api/dev/logout',{});self.assertEqual(self.get(run).status_code,401)
    def test_stale_revision_and_incomplete_signoff_fail_without_lost_update(self):
        _,run=self.start();published=self.publish(run)
        self.assertEqual(self.client.post(f"/api/dev/procedures/runs/{run['id']}/publish",json={'expected_revision':run['revision'],'content_reviewed':True}).status_code,409)
        self.login('patient');self.authorize(published);current=self.get(run).json()
        self.post(f"/api/dev/procedures/runs/{run['id']}/steps/patient_marker",{'expected_revision':current['revision'],'status':'acknowledged'})
        self.assertEqual(self.client.post(f"/api/dev/procedures/runs/{run['id']}/steps/patient_marker",json={'expected_revision':current['revision'],'status':'pending'}).status_code,409)
        self.login();latest=self.get(run).json();self.assertEqual(latest['steps']['patient_marker']['status'],'acknowledged')
        self.assertEqual(self.client.post(f"/api/dev/procedures/runs/{run['id']}/complete",json={'expected_revision':latest['revision'],'content_reviewed':True}).status_code,409)
    def test_explicit_exact_review_input_bounds_and_not_applicable_rationale(self):
        template=self.template();self.assertEqual(self.approve(template,permission_reviewed=False).status_code,409)
        self.assertEqual(self.approve(template,content_sha256='0'*64).status_code,409)
        self.approve(template);self.assertEqual(self.approve(template).status_code,409)
        _,run=self.start();run=self.publish(run);self.login('patient')
        self.assertEqual(self.client.post(f"/api/dev/procedures/runs/{run['id']}/authorization",json={'expected_revision':run['revision'],'authorize':True,'understands_coordination_only':False}).status_code,422)
        self.authorize(run);self.login();current=self.get(run).json()
        self.assertEqual(self.client.post(f"/api/dev/procedures/runs/{run['id']}/steps/clinician_marker",json={'expected_revision':current['revision'],'status':'not_applicable'}).status_code,422)
        self.assertEqual(self.client.post('/api/dev/procedures/templates',json={'slug':'bad','title':'bad'}).status_code,422)

    def test_source_version_supersession_requires_new_template_review(self):
        template,run=self.start();run=self.publish(run)
        source=self.post('/api/dev/knowledge/sources',{'title':'Revised nonclinical source','url':self.source['url'],'content':'Revised synthetic calibration text. This is nonclinical test content only.','license_declaration':'Test-owned synthetic fixture content.','permission_attested':True,'org':'Synthetic test'}).json()
        self.post(f"/api/dev/knowledge/sources/{source['id']}/approve",{})
        self.assertFalse(self.get(run).json()['active_source_and_template'])
        self.login('patient');self.assertEqual(self.get(run).status_code,409)
        self.login();self.assertEqual(self.client.get('/api/dev/procedures/coverage').json()['facets'],[])
    def test_concurrent_stale_updates_only_one_commits(self):
        from concurrent.futures import ThreadPoolExecutor
        _,run=self.start();run=self.publish(run);self.login('patient');self.authorize(run);self.login();current=self.get(run).json()
        def change(note):return self.client.post(f"/api/dev/procedures/runs/{run['id']}/steps/clinician_marker",json={'expected_revision':current['revision'],'status':'acknowledged','note':note}).status_code
        with ThreadPoolExecutor(max_workers=2) as pool:statuses=list(pool.map(change,['First synthetic edit','Second synthetic edit']))
        self.assertEqual(sorted(statuses),[200,409]);self.assertEqual(self.get(run).json()['revision'],current['revision']+1)

    def test_template_revocation_never_reactivates_deduplicated_content(self):
        template,run=self.start();run=self.publish(run)
        r=self.approve(template,target='revoke');self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(self.template()['status'],'revoked')
        self.assertEqual(self.approve(template).status_code,409)
        self.assertFalse(self.get(run).json()['active_source_and_template'])
        self.login('patient');self.assertEqual(self.get(run).status_code,409)

    def test_session_revoked_after_authentication_cannot_mutate(self):
        template,run=self.start();pending=self.template(title='Pending independent review fixture')
        original=self.ctx.user
        def revoke_after_dependency(request:Request):
            user=original(request)
            with self.engine.begin() as c:c.execute(text('DELETE FROM dev_sessions WHERE user_id=:u'),{'u':user['id']})
            return user
        def denied(name,path,body):
            self.login(name);self.app.dependency_overrides[original]=revoke_after_dependency
            try:self.assertEqual(self.client.post(path,json=body).status_code,401)
            finally:self.app.dependency_overrides.clear()
        review=lambda value:{'content_sha256':value['content_sha256'],'source_version_sha256':value['source_version_sha256'],'content_reviewed':True,'permission_reviewed':True}
        denied('doctor','/api/dev/procedures/templates',pending['content'])
        denied('doctor',f"/api/dev/procedures/templates/{pending['id']}/approve",review(pending))
        denied('doctor',f"/api/dev/procedures/templates/{template['id']}/revoke",review(template))
        denied('doctor',f"/api/dev/procedures/patients/{self.patient['id']}/runs",{'template_id':template['id'],'idempotency_key':str(uuid4())})
        denied('doctor',f"/api/dev/procedures/runs/{run['id']}/publish",{'expected_revision':1,'content_reviewed':True})
        self.login();run=self.publish(run)
        denied('patient',f"/api/dev/procedures/runs/{run['id']}/authorization",{'expected_revision':2,'authorize':True,'understands_coordination_only':True})
        self.login('patient');self.authorize(run);self.login();run=self.get(run).json()
        denied('doctor',f"/api/dev/procedures/runs/{run['id']}/steps/patient_marker",{'expected_revision':run['revision'],'status':'acknowledged'})
        denied('doctor',f"/api/dev/procedures/runs/{run['id']}/complete",{'expected_revision':run['revision'],'content_reviewed':True})
        self.login();unchanged=self.get(run).json();self.assertEqual(unchanged['revision'],run['revision']);self.assertEqual(unchanged['state'],'published');self.assertTrue(all(v['status']=='pending' for v in unchanged['steps'].values()))
    def exported(self,allow_instructions=True):
        self.login('patient');user=self.client.get('/api/dev/session').json()
        with self.engine.begin() as c:return export_patient_procedures(c,user,allow_instructions)
    def test_patient_export_scopes_source_invalidation_and_candidate_boundary(self):
        _,draft=self.start();self.assertEqual(self.exported()['runs'],[])
        self.login();run=self.publish(draft);self.login('patient');self.authorize(run);self.login();run=self.get(run).json()
        run=self.post(f"/api/dev/procedures/runs/{run['id']}/steps/patient_marker",{'expected_revision':run['revision'],'status':'acknowledged','note':'Private synthetic note for this patient only.'}).json()
        exported=self.exported()['runs'][0];self.assertFalse(exported['instructions_withheld']);self.assertIn('content',exported['snapshot']);self.assertIn('Private synthetic note',json.dumps(exported))
        candidate=self.exported(False)['runs'][0];self.assertTrue(candidate['instructions_withheld']);self.assertEqual(candidate['withholding_reason'],'release_scope_restriction');self.assertNotIn('content',candidate['snapshot']);self.assertNotIn('Private synthetic note',json.dumps(candidate))
        self.login('other');user=self.client.get('/api/dev/session').json()
        with self.engine.begin() as c:self.assertEqual(export_patient_procedures(c,user)['runs'],[])
        self.login('patient');self.authorize(run,False);revoked=self.exported()['runs'][0];self.assertEqual(revoked['withholding_reason'],'coordination_revoked');self.assertNotIn('Private synthetic note',json.dumps(revoked))
        self.login();self.post(f"/api/dev/knowledge/sources/{self.source['id']}/revoke",{})
        withheld=self.exported()['runs'][0];self.assertTrue(withheld['instructions_withheld']);self.assertNotIn('content',withheld['snapshot']);self.assertEqual(withheld['id'],run['id'])
    def test_patient_export_refuses_oversize_event_history(self):
        _,run=self.start();self.publish(run)
        with self.engine.begin() as c:
            c.execute(text("INSERT INTO dev_procedure_events(run_id,revision,actor_id,kind,detail,created_at) VALUES (:id,:revision,:actor,'synthetic_history','{}','2026-01-01T00:00:00+00:00')"),[{'id':run['id'],'revision':i,'actor':self.patient['patient_user_id']} for i in range(3,503)])
        with self.assertRaises(HTTPException) as caught:self.exported()
        self.assertEqual(caught.exception.status_code,413)

    def test_chart_granted_team_can_collaborate_only_after_private_draft_publication(self):
        _,run=self.start();self.login('patient')
        self.post(f"/api/dev/patients/{self.patient['id']}/chart-access",{'clinician_username':'receiver','expires_in_minutes':60,'purpose':'Synthetic care-team coordination','authorize_full_chart':True})
        self.login('receiver');self.assertEqual(self.get(run).status_code,404);self.assertEqual(self.client.get(f"/api/dev/procedures/patients/{self.patient['id']}/runs").json()['runs'],[])
        self.assertEqual(self.client.post(f"/api/dev/procedures/runs/{run['id']}/publish",json={'expected_revision':1,'content_reviewed':True}).status_code,404)
        self.login();run=self.publish(run);self.login('patient');self.authorize(run)
        self.login('receiver');run=self.get(run).json()
        for key in run['steps']:
            run=self.post(f"/api/dev/procedures/runs/{run['id']}/steps/{key}",{'expected_revision':run['revision'],'status':'acknowledged','note':'Recorded by authorized synthetic care-team clinician.'}).json()
        run=self.post(f"/api/dev/procedures/runs/{run['id']}/complete",{'expected_revision':run['revision'],'content_reviewed':True}).json();self.assertEqual(run['state'],'completed')
        self.login('patient');self.authorize(run,False);self.login('receiver');self.assertEqual(self.get(run).status_code,404)
    def test_database_guards_preserve_immutable_content_and_append_only_history(self):
        _,run=self.start()
        for statement in ["UPDATE dev_procedure_templates SET content='{}'","UPDATE dev_procedure_runs SET snapshot='{}'","DELETE FROM dev_procedure_events","UPDATE dev_procedure_events SET kind='tampered'"]:
            with self.engine.begin() as c:
                with self.assertRaises(Exception):c.execute(text(statement))
        self.assertEqual(self.get(run).json()['revision'],1)
