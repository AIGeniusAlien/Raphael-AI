(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const patientId = Number(new URLSearchParams(location.search).get('patient_id'));
  let user = null, chart = null, generation = 0;
  const clinician = () => user?.role === 'clinician';
  function node(tag, text, className) {
    const n = document.createElement(tag);
    if (text !== undefined) n.textContent = String(text);
    if (className) n.className = className;
    return n;
  }
  function status(text = '', error = false) {
    byId('status').textContent = text;
    byId('status').setAttribute('data-error', String(error));
  }
  async function api(path, options = {}) {
    const response = await fetch('/api/dev' + path, {credentials: 'same-origin', ...options,
      headers: options.body instanceof FormData ? {} : {'Content-Type': 'application/json'}});
    const data = await response.json();
    if (!response.ok) {
      if (response.status === 401) {
        user = chart = null;
        generation++;
        byId('chart-content').hidden = true;
        byId('patient-name').textContent = 'Sign in through the portal to view this chart.';
      }
      throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${response.status}); check the required fields.`);
    }
    return data;
  }
  const post = (path, body) => api(path, {method: 'POST', body: JSON.stringify(body)});
  const patch = (path, body) => api(path, {method: 'PATCH', body: JSON.stringify(body)});
  const base = () => `/patients/${patientId}`;
  function button(text, fn) {
    const b = node('button', text); b.type = 'button';
    b.addEventListener('click', () => action(fn, b));
    return b;
  }
  async function action(fn, control) {
    if (control?.disabled) return;
    if (control) control.disabled = true;
    status();
    try { await fn(); }
    catch (error) { status(error.message || 'Request failed.', true); }
    finally { if (control) control.disabled = false; }
  }
  function summaryRecord(title, text, meta = '') {
    const item = node('article', undefined, 'record');
    item.append(node('h3', title), node('p', text), node('p', meta, 'meta'));
    return item;
  }
  function entryName(kind) { return ({history: 'Medical history', medication: 'Medication', allergy: 'Allergy',
    lab: 'Laboratory', vital: 'Vital observation', imaging_report: 'Imaging report', care_plan: 'Care plan'})[kind] || kind; }
  function renderEntries() {
    const holder = byId('entries'); holder.replaceChildren();
    if (!chart.entries.length) holder.append(node('p', 'No visible structured records yet.'));
    const observations = chart.entries.filter(e => ['lab', 'vital'].includes(e.kind));
    if (observations.length) {
      holder.append(node('h3', 'Observation trends — chronological entered values'));
      const wrapper = node('div', undefined, 'table-wrap'), table = node('table');
      const head = node('tr');
      for (const label of ['Time', 'Observation', 'Value / unit', 'Entered range', 'Comparison flag']) head.append(node('th', label));
      const thead = node('thead'); thead.append(head); table.append(thead);
      const tbody = node('tbody');
      for (const e of observations) {
        const d = e.data, row = node('tr');
        for (const cell of [e.recorded_at, e.label, `${d.value} ${d.unit}`, `${d.reference_low ?? '—'} to ${d.reference_high ?? '—'}`,
          e.reference_flag?.replaceAll('_', ' ') || 'No entered range']) row.append(node('td', cell));
        tbody.append(row);
      }
      table.append(tbody); wrapper.append(table); holder.append(wrapper);
    }
    for (const e of chart.entries) {
      const d = e.data;
      const details = [d.text, d.dose ? `Entered dose: ${d.dose}` : '', d.reaction ? `Reported reaction: ${d.reaction} (${d.severity || 'unknown'})` : '',
        d.value !== null ? `Entered value: ${d.value} ${d.unit || ''}` : ''].filter(Boolean).join('\n');
      const item = summaryRecord(`${entryName(e.kind)} · ${e.label}`, details,
        `${e.recorded_at} · ${e.status} · revision ${e.revision} · source: ${d.source}`);
      if (e.status === 'draft') item.classList.add('draft');
      if (clinician()) {
        item.append(button('Edit record', () => editEntry(e)));
        if (e.status === 'draft') item.append(button('Approve this draft for patient visibility', async () => {
          await post(`/chart-entries/${e.id}/approve`, {revision: e.revision});
          await load(); status('Draft explicitly approved for this synthetic patient chart.');
        }));
      }
      holder.append(item);
    }
  }
  function renderEncounters() {
    const holder = byId('encounters'); holder.replaceChildren();
    if (!chart.encounters.length) holder.append(node('p', 'No encounters recorded.'));
    for (const e of chart.encounters) {
      const item = summaryRecord(e.title, e.summary || 'No entered summary.', `${e.occurred_at} · ${e.status}`);
      if (clinician()) {
        const label = node('label', 'Edit entered summary'), field = node('textarea'); field.value = e.summary; field.maxLength = 10000; label.append(field);
        item.append(label, button(e.status === 'closed' ? 'Save and reopen encounter' : 'Save and close encounter', async () => {
          await patch(`/encounters/${e.id}`, {status: e.status === 'closed' ? 'open' : 'closed', summary: field.value});
          await load(); status('Encounter updated.');
        }));
      }
      holder.append(item);
    }
    const selected = byId('entry-encounter').value;
    byId('entry-encounter').replaceChildren(node('option', 'No encounter link'));
    byId('entry-encounter').firstChild.value = '';
    for (const e of chart.encounters) { const option = node('option', `${e.title} (${e.status})`); option.value = String(e.id); byId('entry-encounter').append(option); }
    byId('entry-encounter').value = selected;
  }
  function renderAppointments() {
    const holder = byId('appointments'); holder.replaceChildren();
    if (!chart.appointments.length) holder.append(node('p', 'No internal appointment records.'));
    for (const a of chart.appointments) {
      const item = summaryRecord(a.purpose, `${a.location}\n${a.follow_up}`, `${a.starts_at} · ${a.status} · No external notification delivered`);
      if (clinician()) for (const next of ['completed', 'cancelled']) {
        if (a.status !== next) item.append(button(`Mark ${next}`, async () => { await patch(`/appointments/${a.id}`, {status: next}); await load(); status('Internal appointment status updated.'); }));
      }
      holder.append(item);
    }
  }
  function renderDocuments() {
    const holder = byId('documents'); holder.replaceChildren();
    if (!chart.documents.length) holder.append(node('p', 'No documents authorized for this account.'));
    for (const d of chart.documents) {
      const item = summaryRecord(d.title, 'Uploaded document — not clinician-approved advice.', `${d.file_type} · ${d.size_bytes} bytes · ${d.visibility} · ${d.created_at}`);
      const a = node('a', 'Download authorized attachment'); a.href = `/api/dev/documents/${Number(d.id)}/content`;
      item.append(a, node('p', `SHA-256: ${d.sha256}`, 'meta')); holder.append(item);
    }
  }
  async function load() {
    const version = ++generation;
    const data = await api(base() + '/chart');
    if (version !== generation || !user) return;
    chart = data;
    byId('patient-name').textContent = `${chart.patient.name} · Synthetic chart`;
    byId('chart-content').hidden = false;
    renderEntries(); renderEncounters(); renderAppointments(); renderDocuments();
    if (user.role === 'patient') {
      const consents = await api(base() + '/document-consents');
      if (version !== generation || !user) return;
      byId('consents').replaceChildren();
      for (const c of consents) byId('consents').append(node('p', `${c.clinician_username}: ${c.granted ? 'granted' : 'revoked'} private-document access`));
      if (!consents.length) byId('consents').append(node('p', 'No private-document grants.'));
    }
  }
  function localTime(value) {
    if (!value) return '';
    const date = new Date(value);
    return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0,16);
  }
  function iso(id) { return byId(id).value ? new Date(byId(id).value).toISOString() : null; }
  function nullableNumber(id) { return byId(id).value === '' ? null : Number(byId(id).value); }
  function fields() {
    const kind = byId('entry-kind').value, numeric = ['lab','vital'].includes(kind);
    byId('numeric-fields').hidden = !numeric;
    byId('entry-value').required = byId('entry-unit').required = numeric;
    byId('dose-field').hidden = kind !== 'medication'; byId('entry-dose').required = kind === 'medication';
    byId('allergy-fields').hidden = kind !== 'allergy'; byId('entry-reaction').required = kind === 'allergy';
    byId('entry-text').required = ['history','imaging_report','care_plan'].includes(kind);
  }
  function clearEntry() { byId('entry-form').reset(); byId('entry-id').value = byId('entry-revision').value = ''; byId('entry-kind').disabled = false; byId('entry-form-heading').textContent = 'Add a structured record'; fields(); }
  function editEntry(e) {
    byId('entry-id').value = String(e.id); byId('entry-revision').value = String(e.revision);
    byId('entry-kind').value = e.kind; byId('entry-kind').disabled = true;
    byId('entry-label').value = e.label; byId('entry-time').value = localTime(e.recorded_at);
    byId('entry-encounter').value = e.encounter_id ?? '';
    for (const [id, key] of [['text','text'],['value','value'],['unit','unit'],['low','reference_low'],['high','reference_high'],['dose','dose'],['reaction','reaction'],['severity','severity'],['source','source']]) byId('entry-' + id).value = e.data[key] ?? '';
    byId('entry-form-heading').textContent = `Edit record ${e.id} — revision ${e.revision}`;
    fields(); byId('entry-form').scrollIntoView({block:'start', behavior:'smooth'});
  }
  function bind(id, fn) {
    byId(id).addEventListener('submit', event => {
      event.preventDefault(); const submit = byId(id).querySelector('button');
      return action(fn, submit);
    });
  }
  async function boot() {
    if (!Number.isSafeInteger(patientId) || patientId <= 0) throw new Error('Open a chart from an authorized patient record in the portal.');
    user = await api('/session');
    if (!['clinician','patient'].includes(user.role)) throw new Error('This account does not have chart access.');
    byId('session').textContent = `${user.username} · ${user.role} · ${user.tenant}`;
    for (const id of ['entry-form','encounter-form','appointment-form']) byId(id).hidden = !clinician();
    byId('consent-panel').hidden = user.role !== 'patient';
    byId('entry-kind').addEventListener('change', fields);
    byId('entry-cancel').addEventListener('click', clearEntry);
    byId('refresh').addEventListener('click', () => action(load, byId('refresh')));
    bind('entry-form', async () => {
      const data = {kind:byId('entry-kind').value,label:byId('entry-label').value,recorded_at:iso('entry-time'),
        encounter_id:nullableNumber('entry-encounter'),text:byId('entry-text').value,value:nullableNumber('entry-value'),unit:byId('entry-unit').value || null,
        reference_low:nullableNumber('entry-low'),reference_high:nullableNumber('entry-high'),dose:byId('entry-dose').value || null,reaction:byId('entry-reaction').value || null,
        severity:byId('entry-severity').value || null,source:byId('entry-source').value};
      if (byId('entry-id').value) await patch(`/chart-entries/${Number(byId('entry-id').value)}`, {...data,revision:Number(byId('entry-revision').value)});
      else await post(base() + '/entries', data);
      clearEntry(); await load(); status('Entered chart record saved. Draft care plans and imaging reports still require approval.');
    });
    bind('encounter-form', async () => { await post(base() + '/encounters', {title:byId('encounter-title').value,occurred_at:iso('encounter-time'),summary:byId('encounter-summary').value}); byId('encounter-form').reset(); await load(); status('Synthetic encounter recorded.'); });
    bind('appointment-form', async () => { await post(base() + '/appointments', {starts_at:iso('appointment-time'),purpose:byId('appointment-purpose').value,location:byId('appointment-location').value,follow_up:byId('appointment-followup').value}); byId('appointment-form').reset(); await load(); status('Internal appointment recorded. No reminder was delivered.'); });
    bind('document-form', async () => {
      const file = byId('document-file').files[0];
      if (!file || file.size > 5 * 1024 * 1024) throw new Error('Choose a synthetic .txt or .pdf file up to 5 MiB.');
      const data = new FormData(); data.append('title',byId('document-title').value);data.append('visibility',byId('document-visibility').value);data.append('file',file);
      await api(base() + '/documents',{method:'POST',body:data});byId('document-form').reset();await load();status('Synthetic document stored with the selected sharing scope.');
    });
    bind('consent-form', async () => { await post(base() + '/document-consents', {clinician_username:byId('consent-clinician').value,granted:byId('consent-granted').value === 'true'}); await load();status('Private-document permission updated. Other chart policies were not changed.'); });
    fields(); await load();
  }
  window.RaphaelChart = {boot, load, editEntry, clearEntry};
  window.RaphaelChart.ready = boot().catch(error => status(error.message || 'Chart unavailable.', true));
})();
