(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const state = {thread: null, session: null, version: 0, dirty: false, generation: null, controller: null, lastPrompt: '', busy: false, epoch: 0};
  const node = (tag, value) => { const n = document.createElement(tag); n.textContent = value; return n; };
  const status = message => { $('status').textContent = message; };
  async function api(path, body, method='POST') {
    const response = await fetch('/api/dev' + path, {method: body === undefined ? 'GET' : method, credentials:'same-origin', headers:{'Content-Type':'application/json'}, body: body === undefined ? undefined : JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) { if (response.status === 401) { voice.stop(); $('intelligence').hidden = true; } throw new Error(typeof data.detail === 'string' ? data.detail : 'Request failed.'); }
    return data;
  }
  async function action(fn) { try { await fn(); } catch (error) { status(error.message); } }
  function entry(role, content, result='') { const item=node('article','');item.className='chat-entry';item.append(node('strong',role+' · '+result),node('p',content));item.lastChild.className='chat-text';$('messages').append(item);return item.lastChild; }
  function renderMarkdown(holder, text) {
    // Deliberately restricted Markdown: no HTML, embedded images or model URLs.
    holder.replaceChildren();
    for (const line of text.split('\n')) {
      const heading = line.match(/^#{1,3}\s+(.+)/);
      const bullet = line.match(/^[-*]\s+(.+)/);
      holder.append(node(heading ? 'strong' : 'div', heading ? heading[1] : bullet ? '• ' + bullet[1] : line));
    }
  }
  function renderEvidence(sources) {
    $('evidence').replaceChildren();
    for(const source of sources){const article=node('article','');article.append(node('strong',source.title),node('p',source.text),node('code',source.chunk_id));const a=node('a','Resolve stored citation');a.href=source.retrieval_method==='qdrant_hosted_embedding'?'/api/dev/intelligence/evidence/'+encodeURIComponent(source.chunk_id):'/api/dev/knowledge/citations/'+encodeURIComponent(source.chunk_id);a.target='_blank';a.rel='noopener';article.append(a);try{const url=new URL(source.url);if(url.protocol==='https:'&&!url.username&&!url.password){const original=node('a','Open supplied source URL');original.href=url.href;original.target='_blank';original.rel='noopener';article.append(original);}}catch(_){}article.append(node('p',(source.evidence_quality||'Supplied reviewed excerpt')+' · '+(source.source_date||'Date unknown')+' · '+(source.freshness_status||'Freshness not checked')),node('p',source.attribution_notice||''));article.append(node('p','Evidence snapshot at generation time. Source membership is not factuality or current-guideline verification.'));$('evidence').append(article);}
  }
  async function loadThreads() { const rows=await api('/intelligence/threads?q='+encodeURIComponent($('search').value));$('threads').replaceChildren();for(const t of rows){const b=node('button',t.title);b.onclick=()=>action(()=>selectThread(t.id));$('threads').append(b);} }
  async function selectThread(id) {
    if(state.busy)throw new Error('Stop generation before changing conversations.');
    voice.stop();state.epoch++;state.session=null;state.version=0;state.dirty=false;$('note').value='';$('transcript').textContent='';$('audit').replaceChildren();$('versions').replaceChildren();$('note-status').textContent='';$('session-id').value='';$('consent').checked=false;
    const epoch=state.epoch;
    const thread=await api('/intelligence/threads/'+id);
    if(epoch!==state.epoch)return;
    state.thread=thread;$('thread-scope').textContent=thread.intake_id?'Uses only the authorized intake snapshot. No live chart is included; expired or revoked consent denies future access.':thread.patient_id?'Authorized current chart context is included.':'General research conversation; no patient chart is included.';$('thread-title').textContent=state.thread.title;$('patient').value=state.thread.patient_id||'';$('patient').disabled=true;$('messages').replaceChildren();$('evidence').replaceChildren();$('structured-assessment').replaceChildren();
    for(const m of state.thread.messages){entry(m.role,m.content,m.status);if(m.evidence?.length){const inspect=node('button','Inspect evidence for response '+m.id);inspect.onclick=()=>renderEvidence(m.evidence);$('messages').append(inspect);}}
    const latestEvidence=[...state.thread.messages].reverse().find(m=>m.evidence?.length);
    if(latestEvidence)renderEvidence(latestEvidence.evidence);const latestScreen=[...state.thread.messages].reverse().find(m=>m.claim_screen);if(latestScreen)status('Stored automated claim-support screen: '+latestScreen.claim_screen.status+'. Clinician review still required.');
    $('intake-return').hidden=!thread.intake_id;
    $('intake-return').href=thread.intake_id?'/static/care.html?intake='+thread.intake_id+'&thread='+thread.id:'/static/care.html';
    const interrupted = state.thread.messages.find(m=>m.status==='streaming');
    state.generation=interrupted?.generation_id||null;state.busy=!!interrupted;
    $('cancel').disabled=!interrupted;$('send').disabled=!!interrupted;
    if(interrupted)status('A previous generation is unfinished. Stop it to recover this conversation.');
    $('export').disabled=false;$('retry').disabled=!state.thread.messages.some(m=>m.role==='user');
    state.lastPrompt=[...state.thread.messages].reverse().find(m=>m.role==='user')?.content||'';
    await encounters();if(thread.intake_id)$('thread-title').textContent=thread.title+' · consented snapshot only';
  }
  async function encounters() {
    $('encounter').replaceChildren();const id=$('patient').value;$('chart-link').href=id?'/static/chart.html?patient_id='+encodeURIComponent(id):'/';
    if(!id)return;
    const rows=await api('/patients/'+id+'/encounters');for(const e of rows.filter(e=>e.status==='open')){const o=node('option',e.title);o.value=e.id;$('encounter').append(o);}
  }
  async function send(prompt) {
    if(!state.thread)throw new Error('Create or select a conversation.');if(state.busy)throw new Error('A generation is already running.');if(!prompt.trim())return;
    state.lastPrompt=prompt;state.busy=true;state.generation=null;state.controller=new AbortController();$('cancel').disabled=false;$('send').disabled=true;$('retry').disabled=true;entry('user',prompt);const output=entry('Raphael','Provisional stream — awaiting clinician review.');let full='';
    try {
      const response=await fetch('/api/dev/intelligence/threads/'+state.thread.id+'/stream',{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify({content:prompt}),signal:state.controller.signal});
      if(!response.ok){const error=await response.json();throw new Error(error.detail||'Provider request failed.');}
      const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
      for(;;){const {done,value}=await reader.read();buffer+=decoder.decode(value||new Uint8Array(),{stream:!done});buffer=buffer.replaceAll('\r\n','\n');let boundary;
        while((boundary=buffer.indexOf('\n\n'))>=0){const frame=buffer.slice(0,boundary);buffer=buffer.slice(boundary+2);const type=frame.split('\n').find(l=>l.startsWith('event: '))?.slice(7);const payload=frame.split('\n').filter(l=>l.startsWith('data: ')).map(l=>l.slice(6)).join('\n');if(!payload)continue;const data=JSON.parse(payload);
          if(type==='start')state.generation=data.generation_id;
          if(type==='delta'){full+=data.text;renderMarkdown(output,full);}
          if(type==='evidence')renderEvidence(data);
          if(type==='canonical'){full=data.text;renderMarkdown(output,full);}
          if(type==='support_check')status('Automated claim-support screening: '+data.status+'. This is not clinical validation.');
          if(type==='blocked'){status(data.detail);output.textContent=full;}
          if(type==='error'||type==='abstained'){status(data.detail);output.textContent=full||data.detail;}
          if(type==='complete'){status('Draft complete. Citation membership checked; clinical factuality remains unvalidated.');if($('speak').checked)voice.speak(full);}
          if(type==='cancelled')status('Generation cancelled.');
        }if(done)break;
      }
    }catch(error){if(!state.controller?.signal.aborted)throw error;status('Generation stopped.');}finally{state.busy=false;state.generation=null;state.controller=null;$('cancel').disabled=true;$('send').disabled=false;$('retry').disabled=false;}
  }
  async function cancel(){voice.stop();try{if(state.generation)await api('/intelligence/generations/'+encodeURIComponent(state.generation)+'/cancel',{});}finally{state.controller?.abort();state.busy=false;state.generation=null;$('cancel').disabled=true;$('send').disabled=false;status('Generation stopped.');} }
  async function loadSession(){if(!state.session)throw new Error('Start or resume a documentation session.');const session=state.session,epoch=state.epoch;const s=await api('/intelligence/sessions/'+session);if(session!==state.session||epoch!==state.epoch)return null;$('transcript').textContent=s.segments.map(v=>v.origin+': '+v.content).join('\n');$('consent').checked=!!s.consent;const latest=s.notes.at(-1);state.version=latest?.version||0;if(!state.dirty)$('note').value=latest?.content||'';$('note-status').textContent=latest?'Version '+latest.version+' · '+latest.status:'';$('versions').replaceChildren();for(const v of s.notes){const b=node('button','View version '+v.version+' · '+v.status);b.onclick=()=>{status('Historical note: '+v.content);};$('versions').append(b);}const audit=await api('/intelligence/sessions/'+session+'/audit');if(session!==state.session||epoch!==state.epoch)return null;$('audit').replaceChildren(...audit.map(e=>node('p',e.created_at+' · '+e.event)));return s;}
  async function updateNote(edited=false){if(!state.session)throw new Error('Start a documentation session.');if(state.dirty&&!edited)throw new Error('Preserve your edits: save or merge them explicitly.');const session=state.session,epoch=state.epoch;const body={expected_version:state.version,format:$('format').value};if(edited)body.content=$('note').value;const note=await api('/intelligence/sessions/'+session+'/notes',body);if(session!==state.session||epoch!==state.epoch)return;state.version=note.version;state.dirty=false;await loadSession();}
  let segmentQueue=Promise.resolve();
  function appendSegment(content,origin){const session=state.session,epoch=state.epoch;segmentQueue=segmentQueue.catch(()=>{}).then(async()=>{if(!session||!$('consent').checked||session!==state.session||epoch!==state.epoch)throw new Error('Recording consent/session is not active.');await api('/intelligence/sessions/'+session+'/segments',{content,origin});await loadSession();if(session!==state.session||epoch!==state.epoch)return;if(!state.dirty)await updateNote();});return segmentQueue;}
  const voice=new window.RaphaelVoice({onStatus:message=>{$('voice-status').textContent=message;},onText:text=>{if(state.session)action(()=>appendSegment(text,'browser_speech'));if($('mode').value==='standard')$('prompt').value=text;},onCommand:command=>action(()=>send(command))});
  async function boot(){
    const user=await api('/session');if(user.role!=='clinician')throw new Error('The intelligence workspace requires a clinician account.');$('identity').textContent=user.username+' · '+user.tenant;const config=await api('/intelligence/config');$('public-evidence-notice').textContent=config.evidence_notice||'';$('public-evidence-notice').hidden=!config.evidence_notice;$('model').textContent='Model: '+config.model+' · '+(config.provider_configured?'Configured, live validation not established':'Provider credentials missing');
    const patients=await api('/patients');for(const p of patients){const o=node('option',p.name);o.value=p.id;$('patient').append(o);}const selected=new URLSearchParams(location.search).get('patient');if(patients.some(p=>String(p.id)===selected))$('patient').value=selected;
    $('intelligence').hidden=false;await loadThreads();await encounters();const linked=new URLSearchParams(location.search).get('thread');if(linked&&/^\d+$/.test(linked))await selectThread(Number(linked));$('voice-status').textContent=voice.supported()?'Microphone off. Recognition available; explicit permission required.':'Speech recognition unavailable. Manual synthetic transcript entry supported.';
  }
  $('assessment').onclick=()=>action(async()=>{if(!state.thread||state.busy)throw new Error('Select a conversation and finish generation first.');const epoch=state.epoch;const output=await api('/intelligence/threads/'+state.thread.id+'/assessment',{content:$('prompt').value});if(epoch!==state.epoch)return;const holder=$('structured-assessment');holder.replaceChildren(node('p','Structured clinician draft — requires review.'));for(const [key,value] of Object.entries(output)){holder.append(node('h3',key.replaceAll('_',' ')),node('pre',typeof value==='string'?value:JSON.stringify(value,null,2)));}status('Structured assessment saved. No note was approved and no message was sent.');});
  $('new-draft').onclick=()=>action(async()=>{if(state.busy)throw new Error('Stop generation before starting a new conversation.');voice.stop();state.epoch++;state.thread=null;$('intake-return').hidden=true;state.session=null;state.version=0;state.dirty=false;$('patient').disabled=false;$('messages').replaceChildren();$('evidence').replaceChildren();$('structured-assessment').replaceChildren();$('note').value='';$('transcript').textContent='';$('audit').replaceChildren();$('versions').replaceChildren();$('consent').checked=false;$('session-id').value='';$('note-status').textContent='';$('thread-title').textContent='New conversation';$('export').disabled=true;$('retry').disabled=true;await encounters();});
  $('search-button').onclick=()=>action(loadThreads);$('patient').onchange=()=>action(async()=>{if(state.thread){$('patient').value=state.thread.patient_id||'';throw new Error('Create a new thread to change its patient context.');}await encounters();});
  $('thread-form').onsubmit=e=>{e.preventDefault();action(async()=>{const t=await api('/intelligence/threads',{title:$('title').value,patient_id:$('patient').value?Number($('patient').value):null});await loadThreads();await selectThread(t.id);});};
  $('prompt-form').onsubmit=e=>{e.preventDefault();action(()=>send($('prompt').value));};$('cancel').onclick=()=>action(cancel);$('retry').onclick=()=>action(()=>send(state.lastPrompt));
  $('export').onclick=()=>action(async()=>{const thread=await api('/intelligence/threads/'+state.thread.id);const url=URL.createObjectURL(new Blob([JSON.stringify(thread,null,2)],{type:'application/json'}));const a=node('a','');a.href=url;a.download='raphael-synthetic-conversation.json';a.click();URL.revokeObjectURL(url);});
  $('new-encounter').onclick=()=>action(async()=>{if(!$('patient').value)throw new Error('Select a synthetic patient.');const e=await api('/patients/'+$('patient').value+'/encounters',{title:'Synthetic clinical encounter',summary:''});await encounters();$('encounter').value=e.id;});
  $('start-session').onclick=()=>action(async()=>{if(!$('consent').checked)throw new Error('Confirm recording consent first.');if(!$('patient').value||!$('encounter').value)throw new Error('Select a patient and open encounter.');voice.stop();const epoch=state.epoch;const patient=$('patient').value;const s=await api('/intelligence/patients/'+patient+'/sessions',{encounter_id:Number($('encounter').value),consent:true});if(epoch!==state.epoch||patient!==$('patient').value)return;state.session=s.id;state.dirty=false;$('session-id').value=s.id;await loadSession();});
  $('resume').onclick=()=>action(async()=>{voice.stop();const epoch=state.epoch;const s=await api('/intelligence/sessions/'+Number($('session-id').value));if(epoch!==state.epoch)return;if(String(s.patient_id)!==$('patient').value)throw new Error('Select the matching patient before resuming.');state.session=s.id;state.dirty=false;await loadSession();});
  $('listen').onclick=()=>action(async()=>{if(!state.session||!$('consent').checked)throw new Error('Start an authorized documentation session first.');await voice.start({consent:true,mode:$('mode').value,wakePhrase:$('wake').value});});
  $('pause').onclick=()=>{voice.stop();$('voice-status').textContent='Microphone paused.';};
  $('stop-session').onclick=()=>action(async()=>{voice.stop();$('consent').checked=false;if(state.session)await api('/intelligence/sessions/'+state.session+'/consent',{consent:false});$('voice-status').textContent='Microphone off; recording consent revoked.';});
  $('consent').onchange=()=>{if(!$('consent').checked)$('stop-session').click();};
  $('append').onclick=()=>action(async()=>{await appendSegment($('manual').value,'manual_synthetic');$('manual').value='';});$('note').oninput=()=>{state.dirty=true;};$('generate-note').onclick=()=>action(()=>updateNote());$('save-note').onclick=()=>action(()=>updateNote(true));
  $('approve').onclick=()=>action(async()=>{if(state.dirty)throw new Error('Save your edits before approval.');if(!confirm('Approve and sign this saved synthetic draft? This is an explicit test workflow, not a legally verified signature.'))return;await api('/intelligence/sessions/'+state.session+'/approve',{expected_version:state.version});await loadSession();});
  window.addEventListener('pagehide',()=>{voice.stop();state.controller?.abort();});
  action(boot);
})();
