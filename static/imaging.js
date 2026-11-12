(() => {
  'use strict';
  const el = id => document.getElementById(id);
  const state = { user: null, patient: null, studies: [], study: null, epoch: 0 };
  const clinician = () => state.user?.role === 'clinician';
  const make = (tag, text) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; return n; };
  function status(text = '', error = false) { el('status').textContent = text; el('status').dataset.error = String(error); }
  async function api(path, options = {}) {
    const r = await fetch('/api/dev' + path, { credentials: 'same-origin', ...options });
    const data = await r.json();
    if (!r.ok) {
      if (r.status === 401 || r.status === 403) {
        // Any revoked clinical session must not keep showing cached chart
        // pixels, report drafts, or sandbox record previews.
        reset(); state.patient = null;
        el('workspace').hidden = true;
        el('patient-heading').textContent = '';
        el('studies').replaceChildren();
        el('patient').value = '';
      }
      throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${r.status}).`);
    }
    return data;
  }
  const post = (path, body = {}) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  async function action(fn) { status(); try { await fn(); } catch (e) { status(e.message || 'Request failed.', true); } }
  function reset() {
    state.epoch++; state.study = null; state.studies = [];
    el('viewer').hidden = true; el('prior-view').hidden = true; el('reports').replaceChildren();
    el('current-image').removeAttribute('src'); el('prior-image').removeAttribute('src');
    el('metadata').textContent = ''; el('fhir-output').textContent = ''; el('dicomweb-output').textContent = '';
    for (const id of ['findings', 'impression', 'recommendation', 'dicom-file', 'fhir-id']) el(id).value = '';
  }
  async function loadStudies(epoch = state.epoch) {
    const items = await api(`/patients/${state.patient.id}/imaging/studies`);
    if (epoch !== state.epoch) return;
    state.studies = items; el('studies').replaceChildren();
    if (!items.length) el('studies').appendChild(make('p', 'No synthetic images stored.'));
    for (const item of items) {
      const queue = clinician() && item.report_summary ? ` · ${item.report_summary.drafts} drafts · ${item.report_summary.approved} approved` : '';
      const button = make('button', `${item.metadata.Modality} · ${item.created_at} · ${item.metadata.rows}×${item.metadata.columns}${queue}`);
      button.type = 'button'; button.addEventListener('click', () => action(() => selectStudy(item)));
      el('studies').appendChild(button);
    }
  }
  async function selectPatient(id) {
    reset(); const epoch = state.epoch;
    const patient = await api(`/patients/${id}`);
    if (epoch !== state.epoch) return;
    state.patient = patient; el('patient-heading').textContent = patient.name;
    el('chart-link').href = '/static/chart.html?patient_id=' + encodeURIComponent(String(patient.id));
    el('workspace').hidden = false;
    await loadStudies(epoch);
  }
  async function loadReports(epoch = state.epoch, studyId = state.study?.id) {
    const items = await api(`/imaging/studies/${studyId}/reports`);
    if (epoch !== state.epoch || state.study?.id !== studyId) return;
    el('reports').replaceChildren();
    if (!items.length) el('reports').appendChild(make('p', 'No visible reports for this image.'));
    for (const report of items) {
      const article = make('article'); article.appendChild(make('h4', report.status === 'approved' ? 'Clinician-approved report (synthetic test workflow)' : report.status === 'withdrawn' ? 'Withdrawn report — historical record only' : 'Draft — clinician review required'));
      if (report.status === 'withdrawn') article.appendChild(make('p', 'Withdrawal reason: ' + (report.withdrawal_reason || 'Not supplied')));
      for (const key of ['findings', 'impression', 'recommendation']) { article.appendChild(make('strong', key)); article.appendChild(make('p', report[key])); }
      if (clinician() && report.status === 'draft') {
        const button = make('button', 'Approve report for patient visibility'); button.type = 'button';
        button.addEventListener('click', () => action(async () => {
          if (epoch !== state.epoch || state.study?.id !== studyId) return;
          await post(`/imaging/reports/${report.id}/approve`);
          await loadReports(epoch, studyId); await loadStudies(epoch);
        })); article.appendChild(button);
      }
      if (clinician() && report.status === 'approved') {
        // This is an attestation ledger, not a pager, alerting service, or
        // automatic recognition of urgent findings. Never imply delivery.
        const section = make('section');
        section.append(make('h5', 'Manually document an already completed result communication'),
          make('p', 'Development tracking only. Raphael does not detect critical findings, call anyone, or deliver notifications. Contact the receiving team through an approved external clinical workflow.'));
        const recipient = make('input');
        recipient.maxLength = 120; recipient.placeholder = 'Recipient role (synthetic only)'; recipient.setAttribute('aria-label', 'Communication recipient role');
        const method = make('select');
        method.setAttribute('aria-label', 'Communication method');
        for (const [key,label] of [['phone','Phone'],['in_person','In person'],['secure_internal_message','External secure message'],['other','Other external method']]) {
          const option = make('option',label); option.value=key;method.append(option);
        }
        const summary = make('textarea');
        summary.maxLength=1000;summary.placeholder='Document what was communicated, using synthetic data only';summary.setAttribute('aria-label','Communication summary');
        const agreed = make('input'); agreed.type='checkbox';
        const agreement = make('label', 'I confirm this communication already occurred outside Raphael');
        agreement.prepend(agreed);
        const save = make('button', 'Record completed communication');save.type='button';
        const history = make('button', 'View communication history');history.type='button';
        const historyPane = make('div');
        const load = async()=> {
          const records=await api(`/imaging/reports/${report.id}/communications`);
          if(epoch!==state.epoch || state.study?.id!==studyId)return;
          historyPane.replaceChildren();
          if(!records.length)historyPane.append(make('p','No communications attested in Raphael. This does not establish whether anyone was contacted.'));
          for(const row of records)historyPane.append(make('p',row.recorded_at+' · '+row.method+' · '+row.recipient_role+' · '+row.documented_message));
        };
        history.addEventListener('click',()=>action(load));
        save.addEventListener('click',()=>action(async()=>{
          if(epoch!==state.epoch || state.study?.id!==studyId)return;
          await post(`/imaging/reports/${report.id}/communications`,{
            recipient_role:recipient.value,method:method.value,documented_message:summary.value,
            clinician_confirms_communication_occurred:agreed.checked
          });
          recipient.value='';summary.value='';agreed.checked=false;
          await load();
          status('External communication attestation recorded; Raphael did not deliver a notification.');
        }));
        section.append(recipient,method,summary,agreement,save,history,historyPane);
        article.append(section);
      }
      if (clinician() && report.status === 'approved') {
        const label = make('label', 'Reason to withdraw approved synthetic report');
        const reason = make('textarea'); reason.maxLength = 500; reason.minLength = 5;
        label.appendChild(reason);
        const button = make('button', 'Withdraw report from patient visibility'); button.type = 'button';
        button.addEventListener('click', () => action(async () => {
          if (epoch !== state.epoch || state.study?.id !== studyId) return;
          const explanation = reason.value.trim();
          if (explanation.length < 5) throw new Error('Enter a withdrawal reason of at least five characters.');
          await post(`/imaging/reports/${report.id}/withdraw`, {reason: explanation});
          await loadReports(epoch, studyId); await loadStudies(epoch);
          status('Report withdrawn from patient visibility. Historical clinician record retained.');
        }));
        article.append(label, button);
      }
      el('reports').appendChild(article);
    }
  }
  async function selectStudy(study) {
    state.epoch++; state.study = study; const epoch = state.epoch;
    el('viewer').hidden = false; el('prior-view').hidden = true; el('prior-image').removeAttribute('src'); el('reports').replaceChildren();
    el('metadata').textContent = JSON.stringify(study.metadata, null, 2); el('current-caption').textContent = `Stored synthetic image ${study.id}`;
    el('frame').value = '0'; el('frame').max = String(study.metadata.frames - 1);
    el('current-image').src = `/api/dev/imaging/studies/${study.id}/preview.png?frame=0`;
    el('prior').replaceChildren(make('option', 'No comparison')); el('prior').firstChild.value = '';
    for (const other of state.studies.filter(item => item.id !== study.id)) { const option = make('option', `${other.metadata.Modality} · ${other.created_at}`); option.value = String(other.id); el('prior').appendChild(option); }
    for (const id of ['findings', 'impression', 'recommendation']) el(id).value = '';
    await loadReports(epoch, study.id);
  }
  async function boot() {
    state.user = await api('/session');
    if (!['patient', 'clinician'].includes(state.user.role)) throw new Error('Clinical chart access is unavailable for this role.');
    el('session-label').textContent = `${state.user.username} · ${state.user.role} · synthetic development only`;
    el('upload-form').hidden = !clinician(); el('report-form').hidden = !clinician(); el('fhir-section').hidden = !clinician(); el('dicomweb-section').hidden = !clinician();
    const patients = await api('/patients');
    for (const patient of patients) { const option = make('option', patient.name); option.value = String(patient.id); el('patient').appendChild(option); }
    el('patient').addEventListener('change', () => action(async () => { if (el('patient').value) await selectPatient(Number(el('patient').value)); else { reset(); state.patient = null; el('workspace').hidden = true; } }));
    el('upload-form').addEventListener('submit', event => { event.preventDefault(); action(async () => {
      if (!state.patient) throw new Error('Select a synthetic record.');
      const epoch = state.epoch; const file = el('dicom-file').files[0];
      if (!file || file.size > 10 * 1024 * 1024) throw new Error('Choose a synthetic DICOM file no larger than 10 MiB.');
      await api(`/patients/${state.patient.id}/imaging/studies`, { method: 'POST', headers: { 'Content-Type': 'application/dicom', 'X-Raphael-Upload': 'synthetic' }, body: file });
      if (epoch !== state.epoch) return; el('dicom-file').value = ''; await loadStudies(epoch); status('Synthetic image stored; no diagnostic findings were generated.');
    }); });
    el('view-frame').addEventListener('click', () => action(async () => {
      if (!state.study) return; const frame = Number(el('frame').value);
      if (!Number.isInteger(frame) || frame < 0 || frame >= state.study.metadata.frames) throw new Error('Frame is out of range.');
      el('current-image').src = `/api/dev/imaging/studies/${state.study.id}/preview.png?frame=${frame}`;
    }));
    el('prior').addEventListener('change', () => { const id = Number(el('prior').value); el('prior-view').hidden = !id; if (id) el('prior-image').src = `/api/dev/imaging/studies/${id}/preview.png?frame=0`; else el('prior-image').removeAttribute('src'); });
    el('current-image').addEventListener('error', () => status('The image could not be displayed. Check authorization and supported pixel format.', true));
    el('prior-image').addEventListener('error', () => status('Comparison image is unavailable.', true));
    el('report-form').addEventListener('submit', event => { event.preventDefault(); action(async () => {
      if (!state.study) throw new Error('Select an image.'); const epoch = state.epoch; const studyId = state.study.id;
      await post(`/imaging/studies/${studyId}/reports`, { findings: el('findings').value, impression: el('impression').value, recommendation: el('recommendation').value });
      if (epoch !== state.epoch) return; for (const id of ['findings', 'impression', 'recommendation']) el(id).value = ''; await loadReports(epoch, studyId); status('Manual report draft saved. It is not visible to the patient until approved.');
    }); });
    el('dicomweb-preview').addEventListener('click', () => action(async () => {
      if (!clinician() || !state.patient) throw new Error('Select an authorized synthetic record.');
      const epoch=state.epoch,patientId=state.patient.id;
      const result=await api(`/patients/${patientId}/imaging/dicomweb-studies`);
      if(epoch!==state.epoch||state.patient?.id!==patientId)return;
      el('dicomweb-output').textContent=JSON.stringify(result,null,2);
      status('Read-only synthetic DICOMweb metadata preview completed; no images imported.');
    }));
    el('fhir-form').addEventListener('submit', event => { event.preventDefault(); action(async () => {
      if (!state.patient) throw new Error('Select a synthetic record.'); const epoch = state.epoch;
      const result = await api(`/patients/${state.patient.id}/fhir-preview?resource_id=${encodeURIComponent(el('fhir-id').value)}`);
      if (epoch !== state.epoch) return; el('fhir-output').textContent = JSON.stringify(result, null, 2); status('Read-only synthetic sandbox preview; nothing was imported.');
    }); });
    const requested = Number(new URLSearchParams(location.search).get('patient'));
    if (patients.some(patient => patient.id === requested)) { el('patient').value = String(requested); await selectPatient(requested); }
  }
  action(boot);
})();
