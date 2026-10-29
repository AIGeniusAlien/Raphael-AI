(() => {
  'use strict';
  const state = { user: null, threads: [], id: null, version: 0, editing: null };
  const el = id => document.getElementById(id);
  const node = (tag, text, className) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = String(text); if (className) n.className = className; return n; };
  const status = (text = '', error = false) => { el('status').textContent = text; el('status').dataset.error = String(error); };
  function clear() { state.version++; state.user = null; state.id = null; state.editing = null; el('thread').hidden = true; el('inbox').replaceChildren(); el('messages').replaceChildren(); el('drafts').replaceChildren(); ['message','draft-content','instruction','due-at'].forEach(id => {el(id).value='';}); }
  async function api(path, options = {}) {
    const response = await fetch('/api/dev' + path, {credentials:'same-origin',...options,headers:{...(!(options.body instanceof FormData) ? {'Content-Type':'application/json'} : {}),...(options.headers || {})}});
    const data = await response.json();
    if (!response.ok) { if (response.status === 401) clear(); throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${response.status}).`); }
    return data;
  }
  const post = (path,body={}) => api(path,{method:'POST',body:JSON.stringify(body)});
  async function action(fn) {status();try{return await fn();}catch(error){status(error.message,true);}}
  function button(text, fn) { const n=node('button',text);n.type='button';n.addEventListener('click',()=>action(fn));return n; }
  async function loadInbox() {
    const version=state.version; const box=await api('/inbox'); if(version!==state.version || !state.user)return;
    state.threads=box.threads;el('counts').textContent=`${box.unread_count} unread internal messages · ${box.due_followups} routine follow-ups due`;
    el('inbox').replaceChildren();
    for(const thread of box.threads)el('inbox').appendChild(button(`${thread.title} · ${thread.patient_name} · ${thread.unread_count} unread`,()=>select(thread.id)));
    if(!box.threads.length)el('inbox').appendChild(node('p','No authorized internal conversations. Create a conversation in the physician portal first.'));
  }
  async function select(id) {
    state.version++;state.id=id;state.editing=null;el('messages').replaceChildren();el('drafts').replaceChildren();['message','draft-content','instruction','due-at'].forEach(k=>{el(k).value='';});el('cancel-edit').hidden=true;
    await loadThread();
  }
  async function loadThread() {
    const id=state.id,version=state.version;if(!id)return;const data=await api(`/inbox/${id}`);if(version!==state.version || id!==state.id)return;
    el('thread').hidden=false;el('thread-title').textContent=data.conversation.title;el('thread-meta').textContent='Internal synthetic portal only. Read status reflects a portal acknowledgement, not external delivery.';
    el('message-form').hidden=state.user.role!=='patient';el('draft-section').hidden=state.user.role!=='clinician';el('messages').replaceChildren();
    for(const message of data.messages){const item=node('li');item.appendChild(node('p',`${message.sender_username} (${message.sender_role}) · ${message.created_at}`,'meta'));item.appendChild(node('p',message.content));item.appendChild(node('p',message.read_by_recipient?'Recipient acknowledged reading in this portal.':'Stored in portal; no recipient read acknowledgement.','meta'));
      for(const attachment of message.attachments){const link=node('a',`Download ${attachment.filename} (${attachment.byte_count} bytes)`);link.href=`/api/dev/message-attachments/${attachment.id}/content`;link.download='';item.appendChild(link);}
      if(message.sender_id===state.user.id){const form=node('form');const label=node('label','Attach a routine synthetic document (UTF-8 text or validated PDF; max 2 MiB)');const input=node('input');input.type='file';input.accept='.txt,.md,.pdf';label.appendChild(input);form.appendChild(label);const submit=node('button','Upload attachment');submit.type='submit';form.appendChild(submit);form.addEventListener('submit',event=>{event.preventDefault();return action(async()=>{if(version!==state.version || id!==state.id)throw new Error('Select the current conversation again.');if(!input.files[0])throw new Error('Select a file.');submit.disabled=true;try{const fd=new FormData();fd.append('file',input.files[0]);await api(`/messages/${message.id}/attachments`,{method:'POST',body:fd});if(version===state.version)await loadThread();}finally{submit.disabled=false;}});});item.appendChild(form);}el('messages').appendChild(item);}
    renderDrafts(data.drafts,version,id);
    if(data.messages.length){await post(`/conversations/${id}/read`,{last_read_id:data.messages.at(-1).id});if(version!==state.version)return;await loadInbox();}
  }
  function renderDrafts(drafts,version,id){el('drafts').replaceChildren();for(const draft of drafts){const item=node('article');item.appendChild(node('h4',`${draft.status} · ${draft.source==='manual'?'Manual draft':'Configured AI provider draft — clinician review required'}`));item.appendChild(node('p',draft.content,'content'));if(draft.due_at)item.appendChild(node('p',`Routine follow-up: ${draft.due_at}`,'meta'));
    const current=()=>{if(version!==state.version || id!==state.id)throw new Error('This control belongs to a previous conversation.');};
    if(draft.status==='draft'){item.appendChild(button('Edit draft',async()=>{current();state.editing=draft.id;el('draft-content').value=draft.content;el('due-at').value=draft.due_at?new Date(new Date(draft.due_at).getTime()-new Date(draft.due_at).getTimezoneOffset()*60000).toISOString().slice(0,16):'';el('cancel-edit').hidden=false;el('draft-content').focus();}));item.appendChild(button('Approve draft (does not send)',async()=>{current();await post(`/message-drafts/${draft.id}/approve`);if(version===state.version){await loadThread();status('Draft approved. It has not been sent.');}}));}
    if(draft.status==='approved')item.appendChild(button('Send approved draft internally',async()=>{current();await post(`/message-drafts/${draft.id}/send`);if(version===state.version){await loadThread();await loadInbox();status('Approved draft stored as an internal message. No external message was delivered.');}}));
    if(draft.status==='sent')item.appendChild(node('p','Already stored internally. This draft cannot be sent again.','meta'));el('drafts').appendChild(item);}}
  function bindForm(id,fn){el(id).addEventListener('submit',event=>{event.preventDefault();const buttons=[...el(id).querySelectorAll('button')];buttons.forEach(b=>{b.disabled=true;});return action(fn).finally(()=>buttons.forEach(b=>{b.disabled=false;}));});}
  bindForm('message-form',async()=>{if(!state.id)throw new Error('Select a conversation.');const id=state.id,version=state.version;await post(`/conversations/${id}/messages`,{content:el('message').value.trim()});if(version===state.version){el('message').value='';await loadThread();status('Patient message stored internally; no external delivery.');}});
  bindForm('draft-form',async()=>{if(!state.id)throw new Error('Select a conversation.');const id=state.id,version=state.version;const body={content:el('draft-content').value.trim(),due_at:el('due-at').value?new Date(el('due-at').value).toISOString():null,urgency:'routine'};if(state.editing)await api(`/message-drafts/${state.editing}`,{method:'PATCH',body:JSON.stringify(body)});else await post(`/conversations/${id}/drafts`,body);if(version===state.version){state.editing=null;el('draft-content').value='';el('due-at').value='';el('cancel-edit').hidden=true;await loadThread();status('Unapproved draft saved. Nothing has been sent.');}});
  bindForm('assist-form',async()=>{if(!state.id)throw new Error('Select a conversation.');const id=state.id,version=state.version;await post(`/conversations/${id}/draft-assist`,{instruction:el('instruction').value.trim()});if(version===state.version){await loadThread();status('AI-assisted draft saved for review. Nothing was approved or sent automatically.');}});
  el('cancel-edit').addEventListener('click',()=>{state.editing=null;el('draft-content').value='';el('due-at').value='';el('cancel-edit').hidden=true;});
  el('refresh').addEventListener('click',()=>action(async()=>{await loadInbox();if(state.id)await loadThread();}));
  async function boot(){await action(async()=>{state.user=await api('/session');if(!['patient','clinician'].includes(state.user.role)){clear();throw new Error('Use the administration workspace for this role.');}el('session').textContent=`${state.user.username} · ${state.user.role} · ${state.user.tenant}`;await loadInbox();});}
  window.RaphaelInbox={state,boot,select,loadInbox,loadThread};window.RaphaelInbox.ready=boot();
})();
