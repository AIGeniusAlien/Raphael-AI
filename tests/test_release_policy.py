"""Real authenticated synthetic HTTP tests for server release scope enforcement."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
from sqlalchemy import create_engine, text
from dev_runtime import migrate
from dev_portal import create_dev_user, install_dev_portal
from dev_chart import install_chart
from dev_account import install_account
from dev_feedback import install_feedback
from dev_verify import isolated_environment
from release_policy import ReleasePolicy, install_release_policy


class ReleasePolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='raphael-release-policy-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        env = isolated_environment()
        env.update(RAPHAEL_DEV_PORTAL='1', RAPHAEL_DEV_SYNTHETIC_ONLY='1')
        flag = patch.dict(os.environ, env, clear=True)
        flag.start(); self.addCleanup(flag.stop)
        self.engine = create_engine('sqlite:///' + str(self.directory / 'synthetic.db'))
        self.addCleanup(self.engine.dispose)
        migrate(self.engine)
        self.password = 'Synthetic-release-only-password'
        for tenant, name, role in [('one', 'patient', 'patient'), ('one', 'other', 'patient'),
                                    ('one', 'doctor', 'clinician'), ('one', 'admin', 'org_admin'),
                                    ('two', 'foreign', 'patient')]:
            create_dev_user(self.engine, tenant, name, self.password, role)
        boot, _ = self.build(None)
        with TestClient(boot) as client:
            self.login(client, 'doctor')
            self.pid = client.post('/api/dev/patients', json={'name':'Synthetic own record', 'patient_username':'patient'}).json()['id']
            self.other = client.post('/api/dev/patients', json={'name':'OTHER-MUST-NOT-LEAK', 'patient_username':'other'}).json()['id']
            draft = client.post(f'/api/dev/patients/{self.pid}/notes', json={'content':'UNSIGNED-PRIVATE-DRAFT'}).json()
            approved = client.post(f'/api/dev/patients/{self.pid}/notes', json={'content':'Approved synthetic shared note'}).json()
            client.post(f"/api/dev/notes/{approved['id']}/approve", json={}).raise_for_status()
            self.draft = draft['id']
        self.app, self.invoked = self.build(ReleasePolicy('consumer_candidate'))
        self.client = TestClient(self.app); self.addCleanup(self.client.close)

    def build(self, policy, response_race=None):
        app = FastAPI(); context = install_dev_portal(app, self.engine)
        install_chart(app, context); install_account(app, context); install_feedback(app, context)
        app.mount('/static', StaticFiles(directory='static'), name='static')
        invoked = []
        @app.post('/api/dev/intelligence/threads')
        def experimental(): invoked.append('generation'); return {'unexpected':True}
        @app.post('/api/dev/new-future-tool')
        def future(): invoked.append('future'); return {'unexpected':True}
        @app.get('/healthz')
        def health(): return {'ok':True}
        if response_race is not None:
            @app.middleware('http')
            async def mutate_after_handler(request, call_next):
                response=await call_next(request)
                if request.url.path=='/api/dev/patients':response_race()
                return response
        if policy is not None: install_release_policy(app, context, policy)
        return app, invoked

    def login(self, client=None, name='patient', tenant='one'):
        client = client or self.client
        result = client.post('/api/dev/login', json={'tenant':tenant,'username':name,'password':self.password})
        self.assertEqual(result.status_code,200,result.text); return result

    def test_candidate_manifest_authentication_and_owned_records(self):
        manifest = self.client.get('/api/dev/release-policy').json()
        self.assertEqual(manifest['scope'],'consumer_candidate'); self.assertFalse(manifest['public_distribution_authorized'])
        self.assertFalse(manifest['capabilities']['clinical_generation'])
        self.assertEqual(self.client.get('/api/dev/patients').status_code,401)
        self.login()
        own = self.client.get('/api/dev/patients'); self.assertEqual([x['id'] for x in own.json()],[self.pid])
        notes = self.client.get(f'/api/dev/patients/{self.pid}/notes')
        self.assertIn('Approved synthetic shared note',notes.text); self.assertNotIn('UNSIGNED-PRIVATE-DRAFT',notes.text)
        self.assertEqual(self.client.get(f'/api/dev/patients/{self.other}/notes').status_code,404)
        self.login(name='foreign',tenant='two')
        self.assertEqual(self.client.get(f'/api/dev/patients/{self.pid}').status_code,404)
        self.assertEqual(self.client.get('/api/dev/patients').json(),[])

    def test_direct_urls_mutation_methods_and_future_routes_are_denied(self):
        self.login()
        for path in ['/api/dev/intelligence/threads','/api/dev/patient-ai/threads','/api/dev/procedures/1',
                     '/api/dev/jobs','/api/dev/admin/users','/api/raphael/query','/api/dev/new-future-tool',
                     f'/api/dev/patients/{self.pid}/assessment', f'/api/dev/patients/{self.pid}/entries']:
            self.assertEqual(self.client.post(path,json={}).status_code,403,path)
        self.assertEqual(self.invoked,[])
        for path in ['/api/dev/intelligence/threads','/api/dev/procedures','/api/dev/jobs',
                     '/api/dev/admin/users','/docs','/openapi.json','/static/intelligence.html',
                     '/static/procedures.html','/static/knowledge.js','/static/portal.html']:
            self.assertEqual(self.client.get(path).status_code,403,path)
        self.assertEqual(self.client.get('/api/dev/patients',headers={'X-Raphael-Release-Scope':'development'}).status_code,200)
        self.assertEqual(self.client.post('/api/dev/new-future-tool',json={},headers={'X-Raphael-Release-Scope':'development'}).status_code,403)

    def test_nonpatient_roles_have_no_candidate_record_access_but_can_sign_out(self):
        for name in ['doctor','admin']:
            self.login(name=name)
            self.assertEqual(self.client.get('/api/dev/patients').status_code,403)
            self.assertEqual(self.client.get('/api/dev/account').status_code,403)
            self.assertEqual(self.client.post('/api/dev/logout',json={}).status_code,200)
            self.assertEqual(self.client.get('/api/dev/session').status_code,401)

    def test_profile_feedback_and_review_request_persist_without_clinical_actions(self):
        self.login()
        self.client.post('/api/dev/account/profile',json={'display_name':'Synthetic preference','language':'en','time_zone':'UTC','internal_notifications':False}).raise_for_status()
        body={'request_id':str(uuid4()),'category':'bug','impact':'routine','title':'Synthetic feedback report','detail':'Synthetic reproducible issue for this software test only.','synthetic_only_ack':True,'no_sensitive_data_ack':True}
        first=self.client.post('/api/dev/feedback/reports',json=body);first.raise_for_status()
        self.assertEqual(self.client.post('/api/dev/feedback/reports',json=body).json()['id'],first.json()['id'])
        self.client.post('/api/dev/account/deletion-requests',json={'confirm_review_request':True}).raise_for_status()
        self.client.post('/api/dev/logout',json={}).raise_for_status();self.login()
        self.assertEqual(self.client.get('/api/dev/account').json()['profile']['display_name'],'Synthetic preference')
        self.assertEqual(len(self.client.get('/api/dev/feedback/reports').json()['items']),1)
        self.assertEqual(self.client.get('/api/dev/account/deletion-requests').json()[0]['state'],'REQUESTED')
        self.assertEqual(self.client.get('/api/dev/patients').json()[0]['name'],'Synthetic own record')
        self.assertEqual(self.invoked,[])

    def test_operator_pause_recovers_without_restart_and_preserves_account_controls(self):
        control=self.directory/'operator'; control.mkdir(mode=0o700); pause=control/'paused'
        config=ReleasePolicy.from_environment({'RAPHAEL_RELEASE_SCOPE':'consumer_candidate','RAPHAEL_DEV_PORTAL':'1','RAPHAEL_DEV_SYNTHETIC_ONLY':'1','RAPHAEL_RELEASE_PAUSE_FILE':str(pause)})
        app,_=self.build(config)
        with TestClient(app) as client:
            self.login(client)
            self.assertEqual(client.get('/api/dev/patients').status_code,200)
            fd=os.open(pause,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
            self.assertTrue(client.get('/api/dev/release-policy').json()['operator_paused'])
            self.assertEqual(client.get('/api/dev/patients').status_code,503)
            self.assertEqual(client.get('/api/dev/feedback/reports').status_code,503)
            self.assertEqual(client.get('/api/dev/account/export').status_code,503)
            self.assertEqual(client.get('/api/dev/account').status_code,200)
            pause.unlink()
            self.assertEqual(client.get('/api/dev/patients').status_code,200)
            control.chmod(0o755)
            self.assertEqual(client.get('/api/dev/patients').status_code,503)
            control.chmod(0o700)
            renamed=control.with_name('moved-control');control.rename(renamed)
            self.assertEqual(client.get('/api/dev/patients').status_code,503)
            self.assertEqual(client.get('/api/dev/account/export').status_code,503)
            renamed.rename(control)
            self.assertEqual(client.get('/api/dev/patients').status_code,200)
            client.post('/api/dev/logout',json={}).raise_for_status()

    def test_ambiguous_paths_api_versions_static_and_websocket_boundary(self):
        self.login()
        for path in ['/api/dev/patients/%31','/api/dev//patients','/api/dev/patients/1/notes/', '/static/%69ntelligence.html']:
            self.assertEqual(self.client.get(path).status_code,403,path)
        self.assertEqual(self.client.get('/api/dev/patients',headers={'X-Raphael-API-Version':'2'}).status_code,426)
        for path in ['/static/release-preview.html','/static/release-preview.js','/static/release-preview.css']:
            response=self.client.get(path);self.assertEqual(response.status_code,200)
            self.assertEqual(response.headers['permissions-policy'],'microphone=(), camera=(), geolocation=()')
            self.assertEqual(response.headers['cache-control'],'no-store')
        self.assertEqual(self.client.get('/',follow_redirects=False).headers['location'],'/static/release-preview.html')
        with self.assertRaises(WebSocketDisconnect) as caught:
            with self.client.websocket_connect('/api/dev/intelligence/socket'): pass
        self.assertEqual(caught.exception.code,1008)

    def test_development_passthrough_and_bad_configuration_fail_closed(self):
        app,invoked=self.build(ReleasePolicy())
        with TestClient(app) as client:
            self.assertEqual(client.post('/api/dev/new-future-tool',json={}).status_code,200)
        self.assertEqual(invoked,['future'])
        for env in [{'RAPHAEL_RELEASE_SCOPE':'public'}, {'RAPHAEL_RELEASE_SCOPE':'consumer_candidate'},
                    {'RAPHAEL_RELEASE_SCOPE':'consumer_candidate','RAPHAEL_DEV_PORTAL':'1','RAPHAEL_DEV_SYNTHETIC_ONLY':'0'},
                    {'RAPHAEL_RELEASE_PAUSE_FILE':'/tmp/not-private'}]:
            with self.assertRaises(ValueError):ReleasePolicy.from_environment(env)
        with self.assertRaises(ValueError):install_release_policy(FastAPI(),None,ReleasePolicy('consumer_candidate'))

    def test_response_role_or_tenant_race_withholds_real_handler_records(self):
        for field,value in [('role','clinician'),('tenant','two'),('session',None)]:
            with self.engine.begin() as c:
                c.execute(text("UPDATE dev_users SET role='patient',tenant='one' WHERE username='patient'"))
            def race():
                with self.engine.begin() as c:
                    if field=='role':c.execute(text("UPDATE dev_users SET role=:value WHERE username='patient'"),{'value':value})
                    elif field=='tenant':c.execute(text("UPDATE dev_users SET tenant=:value WHERE username='patient'"),{'value':value})
                    else:c.execute(text("DELETE FROM dev_sessions WHERE user_id IN (SELECT id FROM dev_users WHERE username='patient')"))
            app,_=self.build(ReleasePolicy('consumer_candidate'),response_race=race)
            with TestClient(app) as client:
                self.login(client)
                result=client.get('/api/dev/patients')
                self.assertEqual(result.status_code,401 if field=='session' else 403,result.text)
                self.assertNotIn('Synthetic own record',result.text)
                self.assertIn('protected response withheld',result.text)

    def test_policy_fingerprint_tracks_route_and_static_configuration(self):
        import release_policy
        policy=ReleasePolicy('consumer_candidate')
        original=policy.manifest()['policy_sha256']
        with patch.object(release_policy,'CANDIDATE_ROUTES',release_policy.CANDIDATE_ROUTES+(('GET',r'/future-reviewed-route'),)):
            self.assertNotEqual(policy.manifest()['policy_sha256'],original)
        with patch.object(release_policy,'STATIC_ASSETS',release_policy.STATIC_ASSETS|{'/static/future-reviewed.html'}):
            self.assertNotEqual(policy.manifest()['policy_sha256'],original)
        self.assertEqual(policy.manifest()['policy_sha256'],original)
        manifest=policy.manifest();manifest['capabilities']['clinical_generation']=True
        self.assertFalse(policy.manifest()['capabilities']['clinical_generation'])
        self.assertNotIn(str(self.directory),str(policy.manifest()))

    def test_actual_main_candidate_startup_skips_qdrant_initialization(self):
        env=isolated_environment();env.update(DATABASE_URL='sqlite:///'+str(self.directory/'main-candidate.db'),RAPHAEL_DEV_PORTAL='1',RAPHAEL_DEV_SYNTHETIC_ONLY='1',RAPHAEL_RELEASE_SCOPE='consumer_candidate',QDRANT_URL='https://unreachable.synthetic.example',STATIC_PATH='static')
        with patch.dict(os.environ,env,clear=True):
            spec=importlib.util.spec_from_file_location('raphael_release_startup_test',Path('main.py'))
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            module.ensure_qdrant_collection=Mock(side_effect=AssertionError('Unexpected provider initialization'))
            try:
                with TestClient(module.app) as client:
                    self.assertEqual(client.get('/healthz').status_code,200)
                    self.assertEqual(client.get('/readyz').status_code,200)
                    self.assertEqual(client.get('/api/dev/release-policy').json()['scope'],'consumer_candidate')
                    self.assertEqual(client.get('/api/corpus/sources').status_code,403)
                module.ensure_qdrant_collection.assert_not_called()
            finally:module.engine.dispose()
