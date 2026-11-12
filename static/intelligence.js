(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const state = {thread: null, session: null, sessionPatientId: null, version: 0, dirty: false, generation: null, controller: null, lastPrompt: '', busy: false, epoch: 0, attachmentItems:[], attachmentPreview:null, attachmentSelected:new Set(), attachmentEncryption:false};
  const node = (tag, value) => { const n = document.createElement(tag); n.textContent = value; return n; };
  const status = message => { $('status').textContent = message; };
  async function api(path, body, method='POST') {
    const response = await fetch('/api/dev' + path, {method: body === undefined ? 'GET' : method, credentials:'same-origin', headers:{'Content-Type':'application/json'}, body: body === undefined ? undefined : JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) { if (response.status === 401) { voice.stop(); $('intelligence').hidden = true; } throw new Error(typeof data.detail === 'string' ? data.detail : 'Request failed.'); }
    return data;
  }
  async function action(fn) { try { await fn(); } catch (error) { status(error.message); } }
  function entry(role, content, result='') { const item=node('article','');item.className='chat-entry';item.classList.add(role==='user'?'from-user':'from-raphael');item.append(node('strong',role==='assistant'?'Raphael · '+result:role+' · '+result),node('p',content));item.lastChild.className='chat-text';$('messages').append(item);return item.lastChild; }
  function renderMarkdown(holder, text) {
    // Deliberately restricted Markdown: no HTML, embedded images or model URLs.
    const pane=$('messages');
    const follow=pane.scrollHeight-pane.scrollTop-pane.clientHeight<110;
    holder.replaceChildren();
    for (const line of text.split('\n')) {
      const heading = line.match(/^#{1,3}\s+(.+)/);
      const bullet = line.match(/^[-*]\s+(.+)/);
      holder.append(node(heading ? 'strong' : 'div', heading ? heading[1] : bullet ? '• ' + bullet[1] : line));
    }
    // Keep up with streaming only when the clinician has not scrolled back
    // to inspect earlier responses or evidence.
    if(follow)pane.scrollTop=pane.scrollHeight;
  }
  function renderEvidence(sources) {
    $('evidence').replaceChildren();
    for(const source of sources){const article=node('article','');article.append(node('strong',source.title),node('p',source.text),node('code',source.chunk_id));const a=node('a','Resolve stored citation');a.href=['qdrant_hosted_embedding','qdrant_semantic_bm25_rrf'].includes(source.retrieval_method)?'/api/dev/intelligence/evidence/'+encodeURIComponent(source.chunk_id):'/api/dev/knowledge/citations/'+encodeURIComponent(source.chunk_id);a.target='_blank';a.rel='noopener';article.append(a);try{const url=new URL(source.url);if(url.protocol==='https:'&&!url.username&&!url.password){const original=node('a','Open supplied source URL');original.href=url.href;original.target='_blank';original.rel='noopener';article.append(original);}}catch(_){}article.append(node('p',(source.evidence_quality||'Supplied reviewed excerpt')+' · '+(source.source_date||'Date unknown')+' · '+(source.freshness_status||'Freshness not checked')),node('p',source.attribution_notice||''));article.append(node('p','Evidence snapshot at generation time. Source membership is not factuality or current-guideline verification.'));$('evidence').append(article);}
  }
  let documentationFormats={};
  function documentFields(){
    const container=$('documentation-fields');if(!container)return;container.replaceChildren();$('generate-note').textContent=(documentationFormats[$('format').value]||[]).length?'Prepare clinician document':'Update transcript template';
    for(const name of documentationFormats[$('format').value]||[]){
      const label=node('label',name.replaceAll('_',' ').replace(/\b[a-z]/g,c=>c.toUpperCase())+' — clinician input'),input=node('textarea','');
      input.id='document-field-'+name;input.maxLength=2000;input.oninput=()=>{state.structuredDirty=true;};label.htmlFor=input.id;container.append(label,input);
    }
  }
  function capturedFields(){const fields={};for(const name of documentationFormats[$('format').value]||[]){const value=$('document-field-'+name)?.value.trim();if(value)fields[name]={text:value,chart_entry_ids:[],transcript_segment_ids:[]};}return fields;}
  async function loadThreads() { const rows=await api('/intelligence/threads?q='+encodeURIComponent($('search').value));$('threads').replaceChildren();for(const t of rows){const b=node('button',t.title);b.type='button';b.dataset.threadId=String(t.id);b.setAttribute('aria-current',String(state.thread?.id===t.id));b.onclick=()=>action(()=>selectThread(t.id));$('threads').append(b);} }
  function renderStructuredAssessment(assessment) {
    const holder=$('structured-assessment');
    holder.replaceChildren();
    const header=node('section','');
    header.className='assessment-review';
    header.append(node('h3','Structured clinical reasoning — provisional draft'),
      node('p','Clinician review required. Candidate diagnoses, evidence associations and urgency have not been independently validated. This does not authorize orders or treatment.'));
    const addGroup=(title,items)=>{
      const section=node('section','');
      section.append(node('h4',title));
      const list=node('ul','');
      if(Array.isArray(items)&&items.length){
        for(const item of items)list.append(node('li',String(item)));
      }else list.append(node('li','Not provided — assess clinically.'));
      section.append(list);header.append(section);
    };
    const addText=(title,value)=>{
      const section=node('section','');
      section.append(node('h4',title),node('p',value||'Not established — clinician review required.'));
      header.append(section);
    };
    addText('Problem representation',assessment.problem_representation);
    const differential=node('section','');
    differential.append(node('h4','Ranked differential diagnosis'));
    const entries=Array.isArray(assessment.ranked_differential)?assessment.ranked_differential:[];
    if(!entries.length)differential.append(node('p','No candidate diagnoses provided; insufficient output for ranking.'));
    entries.forEach((item,index)=>{
      const entryBox=node('article','');
      entryBox.className='assessment-candidate';
      entryBox.append(node('h5',String(index+1)+'. '+item.diagnosis),
        node('p','Model likelihood label: '+item.likelihood+' — not a calibrated probability.'));
      const reasons=node('ul','');
      for(const phrase of item.supports||[])reasons.append(node('li','Supporting observation: '+phrase));
      for(const phrase of item.argues_against||[])reasons.append(node('li','Counterpoint: '+phrase));
      entryBox.append(reasons,node('p','Unverified citation IDs: '+(item.citations||[]).join(', ')));
      differential.append(entryBox);
    });
    header.append(differential);
    addText('Working diagnosis (not confirmed)',assessment.working_diagnosis);
    addGroup('Rule-in / rule-out considerations',assessment.rule_in_out_plan);
    addGroup('Potential must-not-miss conditions',assessment.cant_miss);
    addGroup('Red flags and escalation for clinician assessment',assessment.red_flags_escalation);
    addGroup('Missing clinical information',assessment.missing_data);
    addText('Uncertainty / confidence statement',assessment.confidence);
    addGroup('Safety limitations',assessment.safety_notes);
    const provenance=node('details','');
    provenance.append(node('summary','Inspect unverified evidence metadata'),
      node('pre',JSON.stringify(assessment.evidence_used||[],null,2)));
    header.append(provenance);
    holder.append(header);
  }
  function renderConversation(thread) {
    $('messages').replaceChildren();$('evidence').replaceChildren();
    for(const m of thread.messages){if(m.status==='provisional_assessment'){const inspect=node('button','Review provisional diagnostic assessment '+m.id);inspect.type='button';inspect.onclick=()=>{try{renderStructuredAssessment(JSON.parse(m.content));}catch(_){status('Stored assessment could not be displayed; no clinical content was inferred.');}};$('messages').append(node('p','Saved structured assessment — unreviewed, not used as clinical context.'),inspect);continue;}entry(m.role,m.content,m.status);if(m.patient_context && m.role==='assistant'){const details=node('details'),summary=node('summary','Inspect prepared patient context for response '+m.id),body=node('pre',JSON.stringify(m.patient_context,null,2));details.append(summary,node('p','Authorized preparation snapshot,not independently verified findings or proof of transmission. Omitted facts are unknown,not negative findings.'),body);$('messages').append(details);}if(m.evidence?.length){const inspect=node('button','Inspect evidence for response '+m.id);inspect.onclick=()=>renderEvidence(m.evidence);$('messages').append(inspect);}}
    const latestEvidence=[...thread.messages].reverse().find(m=>m.evidence?.length);
    if(latestEvidence)renderEvidence(latestEvidence.evidence);const latestScreen=[...thread.messages].reverse().find(m=>m.claim_screen);if(latestScreen)status('Stored automated claim-support screen: '+latestScreen.claim_screen.status+'. Clinician review still required.');
  }
  function attachmentPath(){if(!state.thread)throw new Error('Select a patient conversation.');return '/intelligence/threads/'+state.thread.id+'/attachments';}
  function attachmentIdle(){if(state.busy)throw new Error('Stop or finish generation before changing attachments.');}
  function renderAttachmentItems(){
    $('attachment-list').replaceChildren();
    for(const item of state.attachmentItems){
      const box=node('article','');box.dataset.attachmentId=item.id;box.append(node('strong',item.title),node('p','Status: '+item.state));
      if(item.state!=='unavailable'){
        const preview=node('button','Preview reported text');preview.type='button';preview.onclick=()=>action(async()=>{
          attachmentIdle();const id=state.thread.id,epoch=state.epoch;
          const value=await api(attachmentPath()+'/'+item.id+'/preview');if(epoch!==state.epoch||id!==state.thread?.id)return;
          state.attachmentPreview=value;$('attachment-preview').replaceChildren(node('h3','Private reported attachment '+value.id),node('p','Source SHA-256: '+value.sha256),node('p',value.notice),node('pre',value.text));renderAttachmentItems();
        });box.append(preview);
        if(item.state==='quarantined'){
          const review=node('button','Authorize reviewed text for provider use');review.type='button';review.disabled=state.attachmentPreview?.id!==item.id||!state.attachmentPreview?.reviewable;
          review.onclick=()=>action(async()=>{attachmentIdle();if(!confirm('Authorize this exact reviewed text for the configured development provider? It is unverified reported data,not clinical evidence.'))return;await api(attachmentPath()+'/'+item.id+'/review',{sha256:item.sha256,authorize_provider_use:true});await loadAttachments();status('Attachment reviewed. Select it and explicitly authorize each provider request.');});box.append(review);
        }
        if(item.state==='reviewed'){
          const label=node('label','Use '+item.title+' as reported context'),check=document.createElement('input');check.type='checkbox';check.value=item.id;check.checked=state.attachmentSelected.has(item.id);
          check.onchange=()=>action(()=>{if(state.busy){check.checked=state.attachmentSelected.has(item.id);throw new Error('Finish generation before changing context.');}if(check.checked&&state.attachmentSelected.size>=2){check.checked=false;throw new Error('Select at most two attachments.');}if(check.checked)state.attachmentSelected.add(item.id);else state.attachmentSelected.delete(item.id);});label.prepend(check);box.append(label);
        }
      }
      const remove=node('button','Delete attachment payload');remove.type='button';remove.onclick=()=>action(async()=>{
        attachmentIdle();if(!confirm('Delete this attachment payload? Prior derived drafts will be withheld and reuse blocked. Original chart records and audit history remain. Copies already disclosed cannot be recalled.'))return;
        const id=state.thread.id,epoch=state.epoch;await api(attachmentPath()+'/'+item.id+'/delete',{confirm:true});if(epoch!==state.epoch||id!==state.thread?.id)return;
        const thread=await api('/intelligence/threads/'+id);if(epoch!==state.epoch)return;state.thread=thread;renderConversation(thread);await loadAttachments();status('Attachment payload deleted logically. This is not original-chart or account erasure. Start a new conversation if prior context is no longer authorized.');
      });box.append(remove);$('attachment-list').append(box);
    }
  }
  async function loadAttachments(){
    state.attachmentSelected.clear();state.attachmentPreview=null;$('attachment-ai-consent').checked=false;$('attachment-preview').replaceChildren();
    const thread=state.thread,epoch=state.epoch;const eligible=thread?.patient_id&&!thread?.intake_id;$('attachments-section').hidden=!eligible;
    if(!eligible){state.attachmentItems=[];$('attachment-list').replaceChildren();return;}
    const result=await api(attachmentPath());if(epoch!==state.epoch||thread.id!==state.thread?.id)return;
    state.attachmentItems=result.items;state.attachmentEncryption=result.encryption_available;
    $('attachment-header').textContent='RAPHAEL_SYNTHETIC_PATIENT_ID: '+thread.patient_id;
    $('attachment-limits').textContent=result.encryption_available?result.scope:'Protected attachment storage unavailable. Ask the operator to configure its document encryption key. Existing records have not been overwritten.';
    $('attachment-upload-button').disabled=!result.encryption_available;$('attachment-import').disabled=!result.encryption_available;renderAttachmentItems();
    const documents=await api('/patients/'+thread.patient_id+'/documents');if(epoch!==state.epoch)return;
    $('attachment-document').replaceChildren(node('option','Select authorized text document'));$('attachment-document').firstChild.value='';
    for(const source of documents.filter(v=>v.file_type==='txt')){const option=node('option',source.title);option.value=source.id;$('attachment-document').append(option);}
  }
  async function releaseEncounterSessionBeforeNavigation() {
    voice.stop();
    if (!state.session) return;
    // A successful server response is required before the browser discards
    // the active patient's documentation-session binding.
    await api('/intelligence/sessions/'+state.session+'/consent',{consent:false});
  }
  async function selectThread(id) {
    if(state.busy)throw new Error('Stop generation before changing conversations.');
    await releaseEncounterSessionBeforeNavigation();
    state.epoch++;state.attachmentSelected.clear();state.attachmentPreview=null;$('attachments-section').hidden=true;$('attachment-preview').replaceChildren();$('attachment-ai-consent').checked=false;state.session=null;state.sessionPatientId=null;state.version=0;state.dirty=false;$('note').value='';$('transcript').textContent='';$('audit').replaceChildren();$('versions').replaceChildren();$('note-status').textContent='';$('session-id').value='';$('consent').checked=false;state.structuredDirty=false;documentFields();
    const epoch=state.epoch;
    const thread=await api('/intelligence/threads/'+id);
    if(epoch!==state.epoch)return;
    state.thread=thread;for(const choice of $('threads').querySelectorAll('button[data-thread-id]'))choice.setAttribute('aria-current',String(choice.dataset.threadId===String(thread.id)));$('thread-scope').textContent=thread.intake_id?'Uses only the authorized intake snapshot. No live chart is included; expired or revoked consent denies future access.':thread.patient_id?'A selected authorized chart snapshot is prepared. Omitted facts remain unknown; inspect saved context.':'General research conversation; no patient chart is included.';$('thread-title').textContent=state.thread.title;$('patient').value=state.thread.patient_id||'';$('patient').disabled=true;$('messages').replaceChildren();$('evidence').replaceChildren();$('structured-assessment').replaceChildren();
    renderConversation(thread);
    $('intake-return').hidden=!thread.intake_id;
    $('intake-return').href=thread.intake_id?'/static/care.html?intake='+thread.intake_id+'&thread='+thread.id:'/static/care.html';
    const interrupted = state.thread.messages.find(m=>m.status==='streaming');
    state.generation=interrupted?.generation_id||null;state.busy=!!interrupted;
    $('cancel').disabled=!interrupted;$('send').disabled=!!interrupted;
    if(interrupted)status('A previous generation is unfinished. Stop it to recover this conversation.');
    $('export').disabled=false;$('retry').disabled=!state.thread.messages.some(m=>m.role==='user');
    state.lastPrompt=[...state.thread.messages].reverse().find(m=>m.role==='user')?.content||'';
    await encounters();await loadAttachments();if(thread.intake_id)$('thread-title').textContent=thread.title+' · consented snapshot only';
  }
  async function encounters() {
    $('encounter').replaceChildren();const id=$('patient').value;$('chart-link').href=id?'/static/chart.html?patient_id='+encodeURIComponent(id):'/';$('imaging-link').href=id?'/static/imaging.html?patient='+encodeURIComponent(id):'/static/imaging.html';
    if(!id)return;
    const rows=await api('/patients/'+id+'/encounters');for(const e of rows.filter(e=>e.status==='open')){const o=node('option',e.title);o.value=e.id;$('encounter').append(o);}
  }
  async function send(prompt) {
    if(!state.thread)throw new Error('Create or select a conversation.');if(state.busy)throw new Error('A generation is already running.');if(!prompt.trim())return;
    const selected=[...state.attachmentSelected],authorized=$('attachment-ai-consent').checked;
    if((selected.length||state.thread.requires_attachment_authorization)&&!authorized)throw new Error('Authorize selected text and attachment-derived history for this question before sending.');
    state.lastPrompt=prompt;state.busy=true;state.generation=null;state.controller=new AbortController();const controller=state.controller,threadId=state.thread.id,epoch=state.epoch;const active=()=>state.controller===controller&&state.epoch===epoch&&state.thread?.id===threadId&&!controller.signal.aborted;$('cancel').disabled=false;$('send').disabled=true;$('retry').disabled=true;entry('user',prompt);const output=entry('Raphael','Provisional stream — awaiting clinician review.');let full='';
    try {
      const response=await fetch('/api/dev/intelligence/threads/'+threadId+'/stream',{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify({content:prompt,attachment_ids:selected,allow_attachment_ai:authorized}),signal:controller.signal});
      if(!active())return;
      if(!response.ok){
        if(response.status===401||response.status===403){
          // A streamed request bypasses api(); fail closed on lost credentials
          // or clinician role, rather than leaving protected chart text visible.
          voice.stop();$('intelligence').hidden=true;
        }
        const error=await response.json();
        throw new Error(typeof error.detail==='string'?error.detail:'Provider request failed.');
      }
      if(!active())return;
      // Clear only the exact question the server accepted. A failed request
      // leaves the clinician's draft intact for correction or retry.
      if ($('prompt').value === prompt) $('prompt').value = '';
      if(selected.length)state.thread.requires_attachment_authorization=true;
      const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
      for(;;){const {done,value}=await reader.read();if(!active()){await reader.cancel();return;}buffer+=decoder.decode(value||new Uint8Array(),{stream:!done});buffer=buffer.replaceAll('\r\n','\n');let boundary;
        while((boundary=buffer.indexOf('\n\n'))>=0){const frame=buffer.slice(0,boundary);buffer=buffer.slice(boundary+2);const type=frame.split('\n').find(l=>l.startsWith('event: '))?.slice(7);const payload=frame.split('\n').filter(l=>l.startsWith('data: ')).map(l=>l.slice(6)).join('\n');if(!payload)continue;const data=JSON.parse(payload);
          if(type==='start')state.generation=data.generation_id;
          if(type==='delta'){full+=data.text;renderMarkdown(output,full);}
          if(type==='evidence')renderEvidence(data);
          if(type==='canonical'){full=data.text;renderMarkdown(output,full);}
          if(type==='support_check')status('Automated claim-support screening: '+data.status+'. This is not clinical validation.');
          if(type==='blocked'){status('Draft blocked: '+data.detail);output.textContent=full;}
          if(type==='error'||type==='abstained'){status(data.detail);output.textContent=full||data.detail;}
          if(type==='complete'){status('Draft complete. Citation membership checked; clinical factuality remains unvalidated.');if($('speak').checked)voice.speak(full);}
          if(type==='cancelled')status('Generation cancelled.');
        }if(done)break;
      }
    }catch(error){if(!controller.signal.aborted&&active())throw error;if(active())status('Generation stopped.');}finally{if(state.controller===controller){$('attachment-ai-consent').checked=false;state.busy=false;state.generation=null;state.controller=null;$('cancel').disabled=true;$('send').disabled=false;$('retry').disabled=false;}}
  }
  async function cancel(){voice.stop();try{if(state.generation)await api('/intelligence/generations/'+encodeURIComponent(state.generation)+'/cancel',{});}finally{state.controller?.abort();state.busy=false;state.generation=null;$('cancel').disabled=true;$('send').disabled=false;status('Generation stopped.');} }
  async function loadSession(){if(!state.session)throw new Error('Start or resume a documentation session.');const session=state.session,epoch=state.epoch;const s=await api('/intelligence/sessions/'+session);if(session!==state.session||epoch!==state.epoch)return null;if(!s.consent)voice.stop();state.sessionPatientId=String(s.patient_id);$('transcript').textContent=s.segments.map(v=>v.origin+': '+v.content).join('\n');$('consent').checked=!!s.consent;const latest=s.notes.at(-1);state.version=latest?.version||0;if(!state.dirty){$('note').value=latest?.content||'';if(latest)$('format').value=latest.format;if(!state.structuredDirty){documentFields();const fields=latest?.source_snapshot?JSON.parse(latest.source_snapshot).clinician_fields||{}:{};for(const [key,field] of Object.entries(fields)){const input=$('document-field-'+key);if(input)input.value=field.text;}}}$('note-status').textContent=latest?'Version '+latest.version+' · '+latest.status:'';$('versions').replaceChildren();for(const v of s.notes){const b=node('button','View version '+v.version+' · '+v.status);b.onclick=()=>{status('Historical note: '+v.content);};$('versions').append(b);}const audit=await api('/intelligence/sessions/'+session+'/audit');if(session!==state.session||epoch!==state.epoch)return null;$('audit').replaceChildren(...audit.map(e=>node('p',e.created_at+' · '+e.event)));return s;}
  async function updateNote(edited=false){if(!state.session)throw new Error('Start a documentation session.');if(state.dirty&&!edited)throw new Error('Preserve your edits: save or merge them explicitly.');const session=state.session,epoch=state.epoch;const body={expected_version:state.version,format:$('format').value,fields:edited?{}:capturedFields()};if(edited)body.content=$('note').value;const note=await api('/intelligence/sessions/'+session+'/notes',body);if(session!==state.session||epoch!==state.epoch)return;state.version=note.version;state.dirty=false;state.structuredDirty=false;await loadSession();}
  let segmentQueue=Promise.resolve();
  function appendSegment(content,origin){const session=state.session,epoch=state.epoch;segmentQueue=segmentQueue.catch(()=>{}).then(async()=>{if(!session||!$('consent').checked||session!==state.session||epoch!==state.epoch)throw new Error('Recording consent/session is not active.');await api('/intelligence/sessions/'+session+'/segments',{content,origin});await loadSession();if(session!==state.session||epoch!==state.epoch)return;if(!state.dirty)await updateNote();});return segmentQueue;}
  const voice=new window.RaphaelVoice({onStatus:message=>{$('voice-status').textContent=message;},onText:text=>{if(state.session){appendSegment(text,'browser_speech').catch(error=>{voice.stop();$('voice-status').textContent='Speech documentation halted; verify session authorization and consent before re-enabling microphone.';status(error.message||'Unable to save authorized transcript.');});}if($('mode').value==='standard')$('prompt').value=text;},onCommand:command=>action(()=>send(command))});
  async function boot(){
    const user=await api('/session');if(user.role!=='clinician')throw new Error('The intelligence workspace requires a clinician account.');$('identity').textContent=user.username+' · '+user.tenant;documentationFormats=(await api('/intelligence/documentation/formats')).formats;documentFields();const config=await api('/intelligence/config');$('public-evidence-notice').textContent=config.evidence_notice||'';$('public-evidence-notice').hidden=!config.evidence_notice;$('model').textContent='Model: '+config.model+' · '+(config.provider_configured?'Configured, live validation not established':'Provider credentials missing');
    const patients=await api('/patients');for(const p of patients){const o=node('option',p.name);o.value=p.id;$('patient').append(o);}const selected=new URLSearchParams(location.search).get('patient');if(patients.some(p=>String(p.id)===selected))$('patient').value=selected;
    $('intelligence').hidden=false;await loadThreads();await encounters();const linked=new URLSearchParams(location.search).get('thread');if(linked&&/^\d+$/.test(linked))await selectThread(Number(linked));$('voice-status').textContent=voice.supported()?'Microphone off. Recognition available; explicit permission required.':'Speech recognition unavailable. Manual synthetic transcript entry supported.';
  }
  $('format').onchange=()=>{state.structuredDirty=false;documentFields();};
  $('assessment').onclick=()=>action(async()=>{if(state.attachmentSelected.size||state.thread?.requires_attachment_authorization)throw new Error('Attachment-derived context requires the reviewed streaming workflow. Start a clean thread for structured assessment.');if(!state.thread||state.busy)throw new Error('Select a conversation and finish generation first.');const epoch=state.epoch;const output=await api('/intelligence/threads/'+state.thread.id+'/assessment',{content:$('prompt').value});if(epoch!==state.epoch)return;renderStructuredAssessment(output);status('Structured assessment saved. No note was approved and no message was sent.');});
  $('new-draft').onclick=()=>action(async()=>{if(state.busy)throw new Error('Stop generation before starting a new conversation.');await releaseEncounterSessionBeforeNavigation();state.epoch++;state.thread=null;state.attachmentSelected.clear();state.attachmentItems=[];state.attachmentPreview=null;$('attachment-preview').replaceChildren();$('attachment-list').replaceChildren();$('attachments-section').hidden=true;$('attachment-ai-consent').checked=false;$('intake-return').hidden=true;state.session=null;state.version=0;state.dirty=false;$('patient').disabled=false;$('messages').replaceChildren();$('evidence').replaceChildren();$('structured-assessment').replaceChildren();$('note').value='';$('transcript').textContent='';$('audit').replaceChildren();$('versions').replaceChildren();$('consent').checked=false;$('session-id').value='';$('note-status').textContent='';$('thread-title').textContent='New conversation';state.structuredDirty=false;documentFields();$('export').disabled=true;$('retry').disabled=true;await encounters();});
  $('coverage-refresh').onclick=()=>action(async()=>{const r=await api('/intelligence/coverage');const holder=$('coverage-map');holder.replaceChildren(node('p',r.supported_use));if(!r.index_checked)return;holder.append(node('p',`${r.active_documents} active documents · ${r.active_vectors} vectors. ${r.freshness}`));for(const source of r.sources){const item=node('p',`${source.domain.replaceAll('_',' ')} · ${source.population} · ${source.indexed_documents} documents. ${source.scope_note}`),a=node('a','Open original source');a.href=source.url;a.target='_blank';a.rel='noopener noreferrer';holder.append(item,a);}holder.append(node('p','Not validated: '+r.not_validated.join('; ')));});
  $('workflow-refresh').onclick=()=>action(async()=>{const r=await api('/intelligence/workflow-checks');$('workflow-checks').textContent=`${r.accessible_generations} accessible generations: ${r.outcomes.complete} complete, ${r.outcomes.blocked} blocked, ${r.outcomes.failed} failed. Support screens: ${r.claim_screens.passed} passed, ${r.claim_screens.blocked} blocked, ${r.claim_screens.missing} missing. Checker-only usage: ${r.checker_tokens} tokens from ${r.checker_usage_reports} reports. ${r.scope}`;const metrics=await api('/intelligence/performance');const timing=metrics.timings.total_ms;$('workflow-checks').textContent+=` Observed total latency: ${timing.samples} samples, p50 ${timing.p50??'unknown'} ms, p95 ${timing.p95??'unknown'} ms, p99 ${timing.p99??'unknown'} ms. Generation tokens: ${metrics.generation_tokens??'unknown'} from ${metrics.generation_usage_reports} actual usage reports. Cost unknown. ${metrics.scope}`;});
  $('attachment-upload').onsubmit=e=>{e.preventDefault();action(async()=>{
    attachmentIdle();const id=state.thread?.id,epoch=state.epoch;const file=$('attachment-file').files[0];if(!id||!file)throw new Error('Select a patient conversation and synthetic text file.');
    const form=new FormData();form.append('title',$('attachment-title').value);form.append('file',file);
    const response=await fetch('/api/dev'+attachmentPath(),{method:'POST',credentials:'same-origin',body:form});const value=await response.json();if(!response.ok){if(response.status===401){voice.stop();$('intelligence').hidden=true;}throw new Error(typeof value.detail==='string'?value.detail:'Attachment upload failed.');}
    if(epoch!==state.epoch||id!==state.thread?.id)return;$('attachment-file').value='';$('attachment-title').value='';await loadAttachments();status('Encrypted attachment quarantined. Preview and explicitly review it before model use.');
  });};
  $('attachment-import').onclick=()=>action(async()=>{attachmentIdle();const value=Number($('attachment-document').value);if(!value)throw new Error('Select an authorized matching text document.');await api(attachmentPath()+'/from-document',{document_id:value});await loadAttachments();status('Authorized chart source copied into encrypted quarantine. Original-source consent remains required.');});
  $('attachment-refresh').onclick=()=>action(async()=>{attachmentIdle();const id=state.thread?.id,epoch=state.epoch;await loadAttachments();if(id&&epoch===state.epoch){const thread=await api('/intelligence/threads/'+id);if(epoch===state.epoch){state.thread=thread;renderConversation(thread);}}status('Attachment permissions refreshed;selection and per-question consent cleared.');});
  $('search-button').onclick=()=>action(loadThreads);$('patient').onchange=()=>action(async()=>{if(state.session){voice.stop();$('patient').value=state.sessionPatientId||'';throw new Error('Finish the active documentation session and create a new conversation before switching patients. Microphone stopped.');}if(state.thread){$('patient').value=state.thread.patient_id||'';throw new Error('Create a new thread to change its patient context.');}await encounters();});
  $('thread-form').onsubmit=e=>{e.preventDefault();action(async()=>{const t=await api('/intelligence/threads',{title:$('title').value,patient_id:$('patient').value?Number($('patient').value):null});await loadThreads();await selectThread(t.id);});};
  $('prompt-form').onsubmit=e=>{e.preventDefault();action(()=>send($('prompt').value));};window.RaphaelComposer?.install($('prompt-form'),$('prompt'));$('cancel').onclick=()=>action(cancel);$('retry').onclick=()=>action(()=>send(state.lastPrompt));
  $('export').onclick=()=>action(async()=>{const thread=await api('/intelligence/threads/'+state.thread.id);const url=URL.createObjectURL(new Blob([JSON.stringify(thread,null,2)],{type:'application/json'}));const a=node('a','');a.href=url;a.download='raphael-synthetic-conversation.json';a.click();URL.revokeObjectURL(url);});
  $('new-encounter').onclick=()=>action(async()=>{if(!$('patient').value)throw new Error('Select a synthetic patient.');const e=await api('/patients/'+$('patient').value+'/encounters',{title:'Synthetic clinical encounter',summary:''});await encounters();$('encounter').value=e.id;});
  $('start-session').onclick=()=>action(async()=>{if(!$('consent').checked)throw new Error('Confirm recording consent first.');if(!$('patient').value||!$('encounter').value)throw new Error('Select a patient and open encounter.');await releaseEncounterSessionBeforeNavigation();const epoch=state.epoch;const patient=$('patient').value;const s=await api('/intelligence/patients/'+patient+'/sessions',{encounter_id:Number($('encounter').value),consent:true});if(epoch!==state.epoch||patient!==$('patient').value)return;state.session=s.id;state.sessionPatientId=String(patient);state.dirty=false;$('session-id').value=s.id;await loadSession();});
  $('resume').onclick=()=>action(async()=>{voice.stop();const epoch=state.epoch;const s=await api('/intelligence/sessions/'+Number($('session-id').value));if(epoch!==state.epoch)return;if(String(s.patient_id)!==$('patient').value)throw new Error('Select the matching patient before resuming.');if(state.session&&state.session!==s.id)await releaseEncounterSessionBeforeNavigation();if(epoch!==state.epoch)return;state.session=s.id;state.sessionPatientId=String(s.patient_id);state.dirty=false;await loadSession();});
  $('listen').onclick=()=>action(async()=>{
    if(!state.session||!$('consent').checked)throw new Error('Start an authorized documentation session first.');
    const sessionId=state.session,epoch=state.epoch;
    // A different device can revoke consent after the current page loaded.
    // Recheck the server immediately before requesting microphone access.
    const active=await api('/intelligence/sessions/'+sessionId);
    if(epoch!==state.epoch||state.session!==sessionId)return;
    if(!active.consent||active.state!=='active'||String(active.patient_id)!==state.sessionPatientId){
      voice.stop();$('consent').checked=false;
      throw new Error('Recording consent was revoked or changed. Reauthorize an encounter before listening.');
    }
    await voice.start({consent:true,mode:$('mode').value,wakePhrase:$('wake').value});
  });
  $('pause').onclick=()=>{voice.stop();$('voice-status').textContent='Microphone paused.';};
  $('stop-session').onclick=()=>action(async()=>{voice.stop();$('consent').checked=false;if(state.session)await api('/intelligence/sessions/'+state.session+'/consent',{consent:false});$('voice-status').textContent='Microphone off; recording consent revoked.';});
  $('consent').onchange=()=>{if(!$('consent').checked)$('stop-session').click();};
  $('append').onclick=()=>action(async()=>{await appendSegment($('manual').value,'manual_synthetic');$('manual').value='';});$('note').oninput=()=>{state.dirty=true;};$('generate-note').onclick=()=>action(()=>updateNote());$('save-note').onclick=()=>action(()=>updateNote(true));
  $('approve').onclick=()=>action(async()=>{if(state.dirty)throw new Error('Save your edits before approval.');if(!confirm('Approve this saved synthetic draft and publish it to the patient record? Review every missing field and factual statement first. This is a test workflow, not a legally verified signature.'))return;await api('/intelligence/sessions/'+state.session+'/approve',{expected_version:state.version});await loadSession();});
  window.addEventListener('pagehide',()=>{voice.stop();state.controller?.abort();});
  action(boot);
})();
