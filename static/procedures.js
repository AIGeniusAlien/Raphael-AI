(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  const sections=['indications','contraindications','patient_selection','preparation','tests_and_imaging','consent_requirements','safety_checks','equipment_and_resources','procedure_stages','complications','monitoring','follow_up','escalation'];
  const element=(tag,value)=>{const e=document.createElement(tag);if(value!==undefined)e.textContent=String(value);return e;};
  let user,epoch=0,selection=0,run,nonce=crypto.randomUUID();
  const status=(value='',error=false)=>{$('status').textContent=value;$('status').dataset.error=String(error);};
  async function api(path,body){
    const response=await fetch('/api/dev'+path,{credentials:'same-origin',headers:{'Content-Type':'application/json'},...(body===undefined?{}:{method:'POST',body:JSON.stringify(body)})});
    const data=await response.json();
    if(!response.ok){if([404,409].includes(response.status)&&path.startsWith('/procedures/runs/')){selection++;run=null;$('run-detail').replaceChildren();}if(response.status===401||response.status===403){epoch++;$('workspace').hidden=true;$('run-detail').replaceChildren();$('templates').replaceChildren();$('runs').replaceChildren();}throw new Error(typeof data.detail==='string'?data.detail:'Request failed;check the supplied fields.');}
    return data;
  }
  async function action(fn){status();try{await fn();}catch(error){status(error.message||'Request failed.',true);}}
  function button(label,fn,disabled=false){const e=element('button',label);e.type='button';e.disabled=disabled;e.addEventListener('click',()=>{e.disabled=true;action(fn).finally(()=>{e.disabled=disabled;});});return e;}
  function checkbox(holder,label){const wrapper=element('label');const input=element('input');input.type='checkbox';wrapper.append(input,document.createTextNode(label));holder.appendChild(wrapper);return input;}
  function textField(holder,id,label,tag='textarea'){const lab=element('label',label);lab.htmlFor=id;const input=element(tag);input.id=id;input.required=true;holder.append(lab,input);return input;}
  function sourceLink(url){const a=element('a','Read bound source');const parsed=new URL(url);if(parsed.protocol!=='https:')throw new Error('Unsupported source URL.');a.href=parsed.href;a.target='_blank';a.rel='noopener noreferrer';return a;}
  async function refresh(){const expected=epoch;await loadRuns();if(user.role==='clinician')await loadTemplates();if(epoch!==expected)return;}
  async function loadRuns(){
    const expected=epoch,patient=$('patient').value;$('runs').replaceChildren();if(!patient)return;
    const data=await api(`/procedures/patients/${patient}/runs`);if(epoch!==expected||$('patient').value!==patient)return;
    if(!data.runs.length)$('runs').appendChild(element('p','No visible checklists for this patient.'));
    for(const row of data.runs)$('runs').appendChild(button(`Checklist ${row.id} · ${row.state} · coordination ${row.consent_state}${row.active_source_and_template?'':' · reference no longer current'}`,()=>showRun(row.id)));
  }
  async function loadTemplates(){
    const expected=epoch;const [data,sources,coverage]=await Promise.all([api('/procedures/templates'),api('/knowledge/sources'),api('/procedures/coverage')]);if(epoch!==expected)return;
    $('templates').replaceChildren();$('source').replaceChildren();$('run-template').replaceChildren();
    for(const [holder,label] of [[$('source'),'Select a reviewed source'],[$('run-template'),'Select a current approved template']]){const option=element('option',label);option.value='';holder.appendChild(option);}
    for(const source of sources.filter(v=>v.status==='approved')){const option=element('option',`${source.title} · version ${source.version}`);option.value=String(source.id);option.dataset.hash=source.version_sha256;$('source').appendChild(option);}
    $('coverage').textContent=coverage.facets.length?coverage.facets.map(v=>`${v.specialty}: ${v.templates} current templates; independent clinical validation absent`).join('; '):'No current approved procedure coverage. No comprehensive coverage claim.';
    for(const value of data.templates){
      if(value.active_for_use){const option=element('option',`${value.title} · version ${value.version}`);option.value=String(value.id);$('run-template').appendChild(option);}
      const card=element('article');card.dataset.template=String(value.id);card.append(element('h3',`${value.title} · version ${value.version}`),element('p',`${value.status} · ${value.active_for_use?'current for synthetic use':'inactive for new actions'} · ${value.specialty}`),sourceLink(value.source_url),element('p',`Source version ${value.source_version_sha256}; template ${value.content_sha256}`));
      const details=element('details'),summary=element('summary','Read template and checklist content');details.append(summary);
      for(const key of sections)details.append(element('h4',key.replaceAll('_',' ')),element('p',value.content.sections[key]));
      for(const step of value.content.steps)details.append(element('p',`${step.assignee} · ${step.phase}: ${step.instruction}`));card.appendChild(details);
      if(value.status!=='revoked'){
        const content=checkbox(card,' I reviewed this exact content and bound source version.');const permission=checkbox(card,' I reviewed permission to use this material. These are attestations, not independent clinical validation.');
        for(const target of value.status==='pending'?['approve','revoke']:['revoke'])card.appendChild(button(target==='approve'?'Approve template version':'Revoke template version',async()=>{
          if(!content.checked||!permission.checked)throw new Error('Explicit content and permission review are required.');
          await api(`/procedures/templates/${value.id}/${target}`,{content_sha256:value.content_sha256,source_version_sha256:value.source_version_sha256,content_reviewed:true,permission_reviewed:true});if(epoch!==expected)return;await refresh();$('run-detail').replaceChildren();status(target==='approve'?'Template approved for synthetic coordination.':'Template revoked;previous instructions are not actionable.');
        }));
      }
      $('templates').appendChild(card);
    }
  }
  async function showRun(id){
    const request=++selection,expected=epoch;$('run-detail').replaceChildren();run=null;
    const value=await api(`/procedures/runs/${id}`);if(request!==selection||epoch!==expected)return;run=value;renderRun();
  }
  async function mutate(path,extra){const selected=run.id,expected=epoch;await api(`/procedures/runs/${selected}/${path}`,{expected_revision:run.revision,...extra});if(epoch!==expected)return;await loadRuns();await showRun(selected);}
  function renderRun(){
    const holder=$('run-detail');holder.replaceChildren();const value=run;
    holder.append(element('h3',`${value.snapshot.title} · checklist ${value.id}`),element('p',`${value.state} · coordination ${value.consent_state} · revision ${value.revision}`),element('p',value.notice),sourceLink(value.snapshot.source_url),element('p',`Bound source version ${value.snapshot.source_version_sha256}`));
    if(!value.active_source_and_template)holder.appendChild(element('p','Source/template is no longer current. Further checklist actions are blocked;historical clinician review only.'));
    const references=element('details');references.appendChild(element('summary','Read published reference sections'));
    for(const key of sections)references.append(element('h4',key.replaceAll('_',' ')),element('p',value.snapshot.content.sections[key]));holder.appendChild(references);
    if(user.role==='clinician'&&value.state==='draft'){
      const reviewed=checkbox(holder,' I reviewed this exact plan before publishing it to the linked patient.');holder.appendChild(button('Publish reviewed checklist',async()=>{if(!reviewed.checked)throw new Error('Read and explicitly review the plan first.');await mutate('publish',{content_reviewed:true});},!value.active_source_and_template));
    }
    if(user.role==='patient'){
      const understood=checkbox(holder,' I authorize checklist coordination. This is not medical procedure consent and does not authorize an intervention.');
      holder.appendChild(button('Authorize coordination',async()=>{if(!understood.checked)throw new Error('Acknowledge the limited coordination scope first.');await mutate('authorization',{authorize:true,understands_coordination_only:true});},value.consent_state==='granted'||!value.active_source_and_template));
      holder.appendChild(button('Revoke coordination',()=>mutate('authorization',{authorize:false,understands_coordination_only:false}),value.consent_state==='revoked'));
    }
    for(const definition of value.snapshot.content.steps){
      const state=value.steps[definition.key],card=element('article');card.dataset.step=definition.key;card.append(element('h4',definition.instruction),element('p',`${definition.assignee} · ${definition.phase} · ${state.status}`));const note=textField(card,`note-${definition.key}`,'Acknowledgement note (synthetic information only)');note.required=false;note.maxLength=2000;note.value=state.note;
      const canAct=value.actionable&&(user.role==='clinician'||definition.assignee==='patient');note.disabled=!canAct;
      for(const target of user.role==='clinician'?['acknowledged','pending','not_applicable']:['acknowledged','pending'])card.appendChild(button(target==='acknowledged'?'Acknowledge step':target==='pending'?'Reset step':'Mark not applicable',()=>mutate(`steps/${definition.key}`,{status:target,note:note.value.trim()}),!canAct));holder.appendChild(card);
    }
    if(user.role==='clinician'&&value.state==='published'){
      const reviewed=checkbox(holder,' I reviewed all recorded acknowledgements. Completion records a checklist review, not procedure performance.');holder.appendChild(button('Sign off completed checklist',async()=>{if(!reviewed.checked)throw new Error('Explicit final clinician review is required.');await mutate('complete',{content_reviewed:true});},!value.actionable||Object.values(value.steps).some(v=>v.status==='pending')));
    }
    holder.appendChild(button('Read event history',async()=>{const selected=value.id,expected=epoch;const events=await api(`/procedures/runs/${selected}/events`);if(epoch!==expected||run?.id!==selected)return;holder.appendChild(element('p','Latest500event records are shown;older stored audit events are retained.'));const list=element('ol');for(const event of events)list.appendChild(element('li',`Revision ${event.revision}: ${event.kind} · ${event.created_at}`));holder.appendChild(list);}));
  }
  function addStep(){
    const holder=$('step-fields');if(holder.children.length>=30)throw new Error('Maximum30steps.');const row=element('article'),index=crypto.randomUUID();row.className='step-editor';
    textField(row,`step-${index}`,'Supplied checklist instruction').maxLength=1500;
    const assignee=textField(row,`assignee-${index}`,'Responsible role','select');for(const role of ['clinician','patient']){const o=element('option',role);o.value=role;assignee.appendChild(o);}
    const phase=textField(row,`phase-${index}`,'Checklist phase','select');for(const value of ['preparation','procedure','monitoring','follow_up','safety']){const o=element('option',value);o.value=value;phase.appendChild(o);}row.appendChild(button('Remove step',()=>{row.remove();}));holder.appendChild(row);
  }
  function bindForm(id,fn){$(id).addEventListener('submit',event=>{event.preventDefault();const submit=$(id).querySelector('button:not([type=button])');submit.disabled=true;action(fn).finally(()=>{submit.disabled=false;});});}
  async function boot(){
    user=await api('/session');$('identity').textContent=`${user.username} · ${user.tenant} · ${user.role}`;if(!['patient','clinician'].includes(user.role))throw new Error('Patient or clinician account required.');
    const patients=await api('/patients');for(const patient of patients){const option=element('option',patient.name);option.value=String(patient.id);$('patient').appendChild(option);}
    $('workspace').hidden=false;$('clinician-workspace').hidden=user.role!=='clinician';
    $('patient').addEventListener('change',()=>{selection++;$('run-detail').replaceChildren();action(loadRuns);});$('refresh').addEventListener('click',()=>{selection++;$('run-detail').replaceChildren();action(refresh);});
    if(user.role==='clinician'){
      for(const key of sections)textField($('reference-fields'),`reference-${key}`,key.replaceAll('_',' ')).maxLength=2000;
      addStep();$('add-step').addEventListener('click',()=>action(async()=>addStep()));
      bindForm('template-form',async()=>{const expected=epoch,source=$('source').selectedOptions[0];if(!source?.value)throw new Error('Select a current approved source.');const rows=[...$('step-fields').children];if(!rows.length)throw new Error('At least one checklist step is required.');
        await api('/procedures/templates',{slug:$('slug').value,title:$('title').value,specialty:$('specialty').value,source_id:Number(source.value),source_version_sha256:source.dataset.hash,sections:Object.fromEntries(sections.map(key=>[key,$(`reference-${key}`).value])),steps:rows.map((row,i)=>({key:`step_${i+1}`,instruction:row.querySelector('textarea').value,assignee:row.querySelectorAll('select')[0].value,phase:row.querySelectorAll('select')[1].value}))});if(epoch!==expected)return;await loadTemplates();status('Pending immutable template saved. Read and explicitly review it before patient use.');});
      bindForm('run-form',async()=>{const expected=epoch;if(!$('patient').value||!$('run-template').value)throw new Error('Select a patient and current approved template.');const result=await api(`/procedures/patients/${$('patient').value}/runs`,{template_id:Number($('run-template').value),idempotency_key:nonce});if(epoch!==expected)return;nonce=crypto.randomUUID();await loadRuns();await showRun(result.id);status('Private draft created. Patient visibility requires explicit clinician publication.');});
    }
    await refresh();
  }
  action(boot);
})();
