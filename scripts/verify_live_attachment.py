"""Opt-in real physician-portal attachment/evidence/privacy verification;synthetic only."""
import argparse
import sys
import json,os,secrets,signal,socket,subprocess,tempfile,time,re
from pathlib import Path
from unittest.mock import patch
import httpx
from cryptography.fernet import Fernet
from sqlalchemy import create_engine,text
from playwright.sync_api import sync_playwright,expect
from dev_runtime import migrate
from dev_portal import create_dev_user
from curated_evidence import fetch_sources


ROOT=Path(__file__).resolve().parents[1]

def run(output):
    result={'status':'failed','uses_mock_services':False,'production_data_used':False,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()};stage='setup'
    try:
     with tempfile.TemporaryDirectory(prefix='raphael-live-attachment-') as directory,tempfile.TemporaryFile(mode='w+') as log:
      url='sqlite:///'+directory+'/synthetic.db';password=secrets.token_urlsafe(30);engine=create_engine(url)
      try:
       with patch.dict(os.environ,{'RAPHAEL_DEV_SYNTHETIC_ONLY':'1'}):
        migrate(engine)
        for name,role in [('doctor','clinician'),('patient','patient')]:create_dev_user(engine,'live-synthetic',name,password,role)
      finally:engine.dispose()
      with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
      base=f'http://127.0.0.1:{port}'
      env=dict(os.environ,DATABASE_URL=url,RAPHAEL_DEV_PORTAL='1',RAPHAEL_DEV_SYNTHETIC_ONLY='1',RAPHAEL_DOCUMENT_ENCRYPTION_KEY=Fernet.generate_key().decode(),RAPHAEL_EMBEDDING_BACKEND='openai',RAPHAEL_STREAM_EVIDENCE_BACKEND='qdrant',RAPHAEL_VERIFY_CLAIMS='1',EMBED_DIM='384')
      process=subprocess.Popen([sys.executable,'-m','uvicorn','main:app','--host','127.0.0.1','--port',str(port)],cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
      try:
       stage='startup'
       with httpx.Client(base_url=base,timeout=3,trust_env=False) as check:
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
         if process.poll() is not None:raise RuntimeError('Startup exited')
         try:
          if check.get('/readyz').status_code==200:break
         except httpx.TransportError:pass
         time.sleep(.1)
        else:raise RuntimeError('Readiness deadline')
       with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=os.getenv('CHROMIUM_EXECUTABLE','/usr/bin/chromium'),headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        try:
         context=browser.new_context(base_url=base,viewport={'width':1280,'height':900});page=context.new_page();errors=[];page.on('pageerror',lambda error:errors.append(type(error).__name__))
         page.goto('/');page.locator('#tenant').fill('live-synthetic');page.locator('#username').fill('doctor');page.locator('#password').fill(password);page.locator('#login-form button').click();expect(page.locator('#workspace')).to_be_visible()
         patient=context.request.post('/api/dev/patients',data={'name':'Synthetic live attachment demonstration','dob':'1994-01-01','patient_username':'patient'});assert patient.status==200;patient_id=patient.json()['id']
         page.goto('/static/intelligence.html?patient='+str(patient_id));expect(page.locator('#intelligence')).to_be_visible();page.locator('#title').fill('Actual live reviewed synthetic attachment');page.locator('#thread-form button').click();expect(page.locator('#thread-title')).to_have_text('Actual live reviewed synthetic attachment')
         page.locator('#attachments-section summary').click();page.locator('#attachment-title').fill('Synthetic patient-reported context')
         marker='SYNTHETIC-ATTACHMENT-PROVENANCE-CHECK'
         content=f'RAPHAEL_SYNTHETIC_PATIENT_ID: {patient_id}\n{marker}. Patient-reported synthetic cough. Examination,testing and allergies have not been verified. This is unverified reported context,not external medical evidence.'
         page.locator('#attachment-file').set_input_files({'name':'synthetic.txt','mimeType':'text/plain','buffer':content.encode()});page.locator('#attachment-upload-button').click();expect(page.locator('#attachment-list')).to_contain_text('quarantined')
         page.get_by_role('button',name='Preview reported text',exact=True).click();expect(page.locator('#attachment-preview')).to_contain_text(marker);page.once('dialog',lambda dialog:dialog.accept());page.get_by_role('button',name='Authorize reviewed text for provider use',exact=True).click();expect(page.locator('#attachment-list')).to_contain_text('reviewed');page.get_by_label('Use Synthetic patient-reported context as reported context',exact=True).check()
         question='In one short paragraph only: according to the supplied CDC evidence,do antibiotics work against viruses? Cite the exact evidence label and its source date. State that the attachment is unverified patient-reported context and that this general education requires clinician review,not an individual treatment decision.'
         page.locator('#prompt').fill(question);page.locator('#send').click();expect(page.locator('#status')).to_contain_text('Authorize selected text');result['missing_per_question_consent_blocked']=True
         page.locator('#attachment-ai-consent').check();stage='actual_provider_generation';page.locator('#send').click();page.wait_for_function("() => /Draft complete|blocked|failed|unavailable|No reviewed/.test(document.querySelector('#status').textContent)",timeout=90000)
         listed=context.request.get('/api/dev/intelligence/threads').json();thread_id=listed[0]['id'];path='/api/dev/intelligence/threads/'+str(thread_id);stored=context.request.get(path).json();message=stored['messages'][-1]
         result.update(conversation_status=message['status'],claim_screen=message.get('claim_screen'),model=message.get('model'),response=message['content'],citation_ids=re.findall(r'\[([^\[\]]+)\]',message['content']),retrieved_evidence=message.get('evidence'),attachment_metadata=message.get('patient_context',{}).get('conversation_attachments'),browser_page_errors=errors,clinician_approved=False)
         assert result['attachment_metadata'] and all('text' not in item for item in result['attachment_metadata'])
         assert result['citation_ids'] and set(result['citation_ids'])<={item['chunk_id'] for item in message['evidence']}
         for item in message['evidence']:assert context.request.get('/api/dev/intelligence/evidence/'+item['chunk_id']).status==200
         result['rendered_citation_resolvers_passed']=True
         metrics=context.request.get('/api/dev/intelligence/performance');assert metrics.status==200;result['actual_generation_metrics']=metrics.json();assert result['actual_generation_metrics']['generation_usage_reports']==1
         page.screenshot(path=str(output.with_suffix('.png')),full_page=False)
         stage='persisted_encryption_and_revocation'
         with engine.connect() as c:
          encrypted=c.execute(text('SELECT encrypted_content FROM dev_conversation_attachments')).scalar();assert marker.encode() not in bytes(encrypted)
          snapshot=c.execute(text('SELECT snapshot FROM dev_ai_context_snapshots')).scalar();assert marker not in snapshot
         result['encrypted_payload_and_hash_only_snapshot_verified']=True
         page.reload();expect(page.locator('#intelligence')).to_be_visible();page.get_by_role('button',name='Actual live reviewed synthetic attachment',exact=True).click();expect(page.locator('#messages')).to_contain_text(message['content'][:40]);result['conversation_persisted_after_reload']=True
         page.locator('#attachments-section summary').click();page.once('dialog',lambda dialog:dialog.accept());page.get_by_role('button',name='Delete attachment payload',exact=True).click();expect(page.locator('#messages')).not_to_contain_text(message['content'][:40])
         blocked=context.request.post(path+'/stream',data={'content':'Synthetic followup after attachment deletion','allow_attachment_ai':True});assert blocked.status==409
         result['deleted_source_withholds_derived_history_and_blocks_reuse']=True
         result['status']='passed' if message['status']=='complete' and (message.get('claim_screen') or {}).get('status')=='passed' and not errors else 'blocked'
        finally:browser.close()
      finally:
       os.killpg(process.pid,signal.SIGTERM)
       try:process.wait(timeout=10)
       except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
       engine.dispose()
     stage='original_source_verification'
     originals={item['url']:item for item in fetch_sources()}
     for item in result.get('retrieved_evidence',[]):
      original=originals[item['url']];assert original['content_sha256']==item['content_sha256'] and item['text'] in original['text']
     result['original_source_hash_and_excerpt_checks']=True
    except Exception as error:result.update(status='failed',failed_stage=stage,error_type=type(error).__name__)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('response','retrieved_evidence','claim_screen')}))
    return 0 if result['status']=='passed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',required=True,action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new report path;existing evidence is never overwritten.')
    return run(args.output)
if __name__=='__main__':raise SystemExit(main())
