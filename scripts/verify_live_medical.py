"""Opt-in actual Chromium + remote Qdrant/OpenAI verification; synthetic DB only.

Run from repository root:python -m scripts.verify_live_medical --live
Creates no default accounts,uses no fixtures,and never approves clinical output.
"""
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time

import httpx
from playwright.sync_api import sync_playwright,expect
from sqlalchemy import create_engine
from dev_runtime import migrate
from dev_portal import create_dev_user
from curated_evidence import fetch_sources,search,collection,MODEL,DIMENSIONS


def run(output):
    stage='setup';result={'status':'failed','uses_mock_services':False}
    try:
        with tempfile.TemporaryDirectory(prefix='raphael-live-medical-') as directory:
            url='sqlite:///'+directory+'/synthetic.db';password=secrets.token_urlsafe(30)
            previous=os.environ.get('RAPHAEL_DEV_SYNTHETIC_ONLY');os.environ['RAPHAEL_DEV_SYNTHETIC_ONLY']='1'
            engine=create_engine(url)
            try:
                migrate(engine)
                for username,role in (('live-physician','clinician'),('live-patient','patient')):create_dev_user(engine,'live-synthetic',username,password,role)
            finally:
                engine.dispose()
                if previous is None:os.environ.pop('RAPHAEL_DEV_SYNTHETIC_ONLY',None)
                else:os.environ['RAPHAEL_DEV_SYNTHETIC_ONLY']=previous
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            base=f'http://127.0.0.1:{port}'
            env=dict(os.environ,DATABASE_URL=url,RAPHAEL_DEV_PORTAL='1',RAPHAEL_DEV_SYNTHETIC_ONLY='1',RAPHAEL_EMBEDDING_BACKEND='openai',RAPHAEL_STREAM_EVIDENCE_BACKEND='qdrant',EMBED_DIM='384',PYTHONDONTWRITEBYTECODE='1')
            with tempfile.TemporaryFile(mode='w+') as log:
                process=subprocess.Popen([sys.executable,'-m','uvicorn','main:app','--host','127.0.0.1','--port',str(port)],env=env,stdout=log,stderr=log,start_new_session=True)
                try:
                    stage='startup'
                    with httpx.Client(base_url=base,timeout=15,trust_env=False) as doctor,httpx.Client(base_url=base,timeout=15,trust_env=False) as patient:
                        deadline=time.monotonic()+60
                        while True:
                            if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Application readiness failed.')
                            try:
                                if doctor.get('/healthz').status_code==200:break
                            except httpx.TransportError:pass
                            time.sleep(.1)
                        for web,username in ((doctor,'live-physician'),(patient,'live-patient')):web.post('/api/dev/login',json={'tenant':'live-synthetic','username':username,'password':password}).raise_for_status()
                        record=doctor.post('/api/dev/patients',json={'name':'Synthetic medical evidence demonstration','patient_username':'live-patient','dob':'1994-01-01'});record.raise_for_status();patient_id=record.json()['id']
                        intake=patient.post(f'/api/dev/care/patients/{patient_id}/intakes',json={'kind':'pre_visit','symptoms':'Synthetic adult with cough,sore throat and rhinorrhea. No examination,testing,allergy verification or red-flag assessment has been performed.'});intake.raise_for_status();intake_id=intake.json()['id']
                        receiver=next(x['id'] for x in patient.get('/api/dev/care/clinicians').json() if x['username']=='live-physician')
                        patient.post(f'/api/dev/care/intakes/{intake_id}/authorize',json={'recipient_id':receiver,'purpose':'pre_visit_review','expires_in_minutes':60,'allow_ai':True}).raise_for_status()
                    stage='browser_live_conversation'
                    with sync_playwright() as playwright:
                        browser=playwright.chromium.launch(executable_path=os.getenv('CHROMIUM_EXECUTABLE','/usr/bin/chromium'),headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
                        try:
                            context=browser.new_context(base_url=base,viewport={'width':1280,'height':900});page=context.new_page();errors=[];page.on('pageerror',lambda error:errors.append(type(error).__name__))
                            page.goto('/');page.locator('#tenant').fill('live-synthetic');page.locator('#username').fill('live-physician');page.locator('#password').fill(password);page.locator('#login-form button').click();expect(page.locator('#workspace')).to_be_visible()
                            page.get_by_role('link',name='Connected intake and emergency information').click();expect(page.locator('#intakes button')).to_have_count(1);page.locator('#intakes button').click();page.locator('#assistant').click();expect(page.locator('#thread-scope')).to_contain_text('No live chart is included')
                            question='For this synthetic adult,answer in one short paragraph only: according to the supplied CDC evidence,do antibiotics work against viruses? Cite the exact evidence label and source date. State that this is general education requiring clinician review,not an individual treatment decision.'
                            page.locator('#prompt').fill(question);page.locator('#prompt-form button').click();page.wait_for_function("() => /Draft complete|blocked|failed|unavailable|No reviewed/.test(document.querySelector('#status').textContent)",timeout=90000)
                            stage='evidence_display'
                            expect(page.locator('#evidence')).to_contain_text('Source: CDC');expect(page.locator('#evidence')).to_contain_text('Official CDC')
                            rendered_resolvers=page.locator('#evidence a').filter(has_text='Resolve stored citation')
                            assert rendered_resolvers.count()>0
                            for resolver in rendered_resolvers.all():
                                href=resolver.get_attribute('href')
                                assert href.startswith('/api/dev/intelligence/evidence/')
                                assert context.request.get(href).status==200
                            result['rendered_citation_links_resolve']=True
                            threads=context.request.get('/api/dev/intelligence/threads').json();thread=context.request.get('/api/dev/intelligence/threads/'+str(threads[0]['id'])).json();message=thread['messages'][-1]
                            assert message['status'] in ('complete','blocked') and message['role']=='assistant'
                            if message['status']=='blocked':assert message.get('claim_screen',{}).get('status')=='blocked'
                            ids=re.findall(r'\[([^\[\]]+)\]',message['content']);assert ids and set(ids)<={x['chunk_id'] for x in message['evidence']}
                            for evidence in message['evidence']:
                                response=context.request.get('/api/dev/intelligence/evidence/'+evidence['chunk_id']);assert response.status==200 and response.json()['chunk_id']==evidence['chunk_id']
                            page.reload();expect(page.locator('#evidence')).to_contain_text('Source: CDC',timeout=30000)
                            assert not errors
                            page.screenshot(path=str(output.with_suffix('.png')),full_page=False)
                            result.update(physician_status=message['status'],claim_screen=message.get('claim_screen'),question=question,response=message['content'],generation_id=message['generation_id'],model=message['model'],citation_ids=ids,retrieved_evidence=message['evidence'],browser_page_errors=errors,clinician_approved=False)
                            stage='patient_live_conversation'
                            patient_context=browser.new_context(base_url=base,viewport={'width':390,'height':844})
                            patient_context.request.post('/api/dev/login',data={'tenant':'live-synthetic','username':'live-patient','password':password})
                            patient_page=patient_context.new_page();patient_page.goto('/static/patient-ai.html')
                            expect(patient_page.locator('#patient-ai')).to_be_visible()
                            patient_page.locator('#title').fill('Synthetic live CDC education')
                            patient_page.locator('#consent').check();patient_page.locator('#new-thread button').click()
                            expect(patient_page.locator('#thread-title')).to_have_text('Synthetic live CDC education')
                            patient_page.locator('#prompt').fill('In one short paragraph, explain from the CDC evidence whether antibiotics work on viruses. Cite the evidence and its date. This is general education,not a request for treatment.')
                            patient_page.locator('#prompt-form button').click()
                            patient_page.wait_for_function("() => /complete|failed|unavailable|blocked|did not pass/.test(document.querySelector('#status').textContent)",timeout=90000)
                            patient_threads=patient_context.request.get('/api/dev/patient-ai/threads').json()
                            patient_thread=patient_context.request.get('/api/dev/patient-ai/threads/'+str(patient_threads[0]['id'])).json()
                            patient_message=patient_thread['messages'][-1]
                            result['patient_workflow']={'status':patient_message['status'],'response':patient_message['content'],'evidence':patient_message['evidence'],'claim_screen':patient_message.get('claim_screen'),'failure_code':patient_message.get('failure_code'),'clinician_approved':False}
                            assert patient_message['status'] in ('complete','blocked')
                            if patient_message['status']=='blocked':assert patient_message.get('claim_screen',{}).get('status')=='blocked'
                            for item in patient_message['evidence']:
                                citation=patient_context.request.get('/api/dev/patient-ai/threads/'+str(patient_threads[0]['id'])+'/evidence/'+item['chunk_id'])
                                assert citation.status==200
                            patient_page.reload();patient_page.locator('#threads button').first.click()
                            expect(patient_page.locator('#messages')).to_contain_text(patient_message['content'][:40])
                            patient_page.screenshot(path=str(output.with_name(output.stem+'-patient.png')),full_page=False)

                        except Exception:
                            result['ui_status']=page.locator('#status').inner_text()
                            try:
                                listed=context.request.get('/api/dev/intelligence/threads').json()
                                if listed:result['diagnostic_thread']=context.request.get('/api/dev/intelligence/threads/'+str(listed[0]['id'])).json()
                            except Exception:pass
                            raise
                        finally:browser.close()
                finally:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        stage='original_source_verification'
        originals={s['url']:s for s in fetch_sources()}
        for evidence in result['retrieved_evidence']+result.get('patient_workflow',{}).get('evidence',[]):
            source=originals[evidence['url']]
            assert source['content_sha256']==evidence['content_sha256'] and evidence['text'] in source['text']
            assert source['source_date']==evidence['source_date'] and source['title']==evidence['title']
        stage='live_missing_evidence_check'
        missing=search('violet gyroscopic calibrator xylophone')
        assert not missing
        result.update(status='passed',original_source_hash_and_excerpt_checks=True,live_missing_evidence_results=0,collection=collection(),embedding_model=MODEL,actual_dimensions=DIMENSIONS)
    except Exception as error:result.update(failed_stage=stage,error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('response','retrieved_evidence','question','generation_id','citation_ids','diagnostic_thread','patient_workflow','claim_screen')}))
    return 0 if result['status']=='passed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--live',required=True,action='store_true');parser.add_argument('--output',type=Path,default=Path('/workspace/.raphael/live-medical-workflow.json'));args=parser.parse_args();return run(args.output)


if __name__=='__main__':raise SystemExit(main())
