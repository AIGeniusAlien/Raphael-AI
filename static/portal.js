(() => {
  'use strict';
  const state = { user: null, patients: [], patient: null, conversations: [], conversation: null, version: 0, threadVersion: 0, sessionEpoch: 0, patientsRequest: 0, authBusy: false };
  const byId = id => document.getElementById(id);
  const clinician = () => state.user?.role === 'clinician';
  const snapshot = () => ({ epoch: state.sessionEpoch, version: state.version,
    patientId: state.patient?.id, conversationId: state.conversation?.id, threadVersion: state.threadVersion });
  const sameSession = mark => state.sessionEpoch === mark.epoch && Boolean(state.user);
  const sameChart = mark => sameSession(mark) && state.version === mark.version
    && state.patient?.id === mark.patientId;
  const sameThread = mark => sameChart(mark) && state.conversation?.id === mark.conversationId
    && state.threadVersion === mark.threadVersion;


  function status(message = '', error = false) {
    byId('status').textContent = message;
    byId('status').setAttribute('data-error', String(error));
  }

  function node(tag, text, className) {
    const item = document.createElement(tag);
    if (text !== undefined) item.textContent = String(text);
    if (className) item.className = className;
    return item;
  }

  function clearChart() {
    state.version += 1;
    state.threadVersion += 1;
    state.patient = null;
    state.conversation = null;
    state.conversations = [];
    byId('chart').hidden = true;
    byId('thread').hidden = true;
    byId('draft-note').hidden = true;
    for (const id of ['conversations', 'messages', 'notes', 'assessment-output']) byId(id).replaceChildren();
    for (const id of ['message-content', 'note-content', 'assessment-question']) byId(id).value = '';
  }

  function clearSession() {
    state.sessionEpoch += 1;
    state.user = null;
    state.patients = [];
    clearChart();
    byId('patients').replaceChildren();
    byId('workspace').hidden = true;
    byId('login-panel').hidden = false;
    byId('logout').hidden = true;
    byId('session-label').textContent = '';
    byId('password').value = '';
  }

  async function api(path, options = {}) {
    const epoch = state.sessionEpoch;
    const version = state.version;
    try {
      const response = await fetch('/api/dev' + path, {
        credentials: 'same-origin',
        ...options,
        headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      });
      const data = await response.json();
      if (!response.ok) {
        const error = new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${response.status}).`);
        if (response.status === 401 && path !== '/login' && epoch === state.sessionEpoch) {
          clearSession();
          error.sessionCleared = true;
          error.epoch = state.sessionEpoch;
          error.version = state.version;
        } else error.epoch = epoch;
        throw error;
      }
      return data;
    } catch (error) {
      if (error.epoch === undefined) error.epoch = epoch;
      if (error.version === undefined) error.version = version;
      throw error;
    }
  }

  const post = (path, body = {}) => api(path, { method: 'POST', body: JSON.stringify(body) });

  function renderPatients() {
    byId('patients').replaceChildren();
    if (!state.patients.length) byId('patients').appendChild(node('p', 'No authorized synthetic records are available.'));
    const epoch = state.sessionEpoch;
    for (const patient of state.patients) {
      const button = node('button', patient.name);
      button.type = 'button';
      button.setAttribute('aria-pressed', String(state.patient?.id === patient.id));
      button.addEventListener('click', () => action(() => selectPatient(patient.id, epoch)));
      byId('patients').appendChild(button);
    }
  }

  async function loadPatients(epoch = state.sessionEpoch) {
    if (epoch !== state.sessionEpoch || !state.user) return;
    const request = ++state.patientsRequest;
    const patients = await api('/patients');
    if (epoch !== state.sessionEpoch || !state.user || request !== state.patientsRequest) return;
    state.patients = patients;
    renderPatients();
  }

  async function enter(user, expectedEpoch) {
    if (state.sessionEpoch !== expectedEpoch) return false;
    state.sessionEpoch += 1;
    clearChart();
    state.user = user;
    if (user.role === 'org_admin') {
      window.location.assign('/static/admin.html');
      return true;
    }
    const epoch = state.sessionEpoch;
    byId('login-panel').hidden = true;
    byId('workspace').hidden = false;
    byId('logout').hidden = false;
    byId('session-label').textContent = `${user.username} · ${user.role === 'clinician' ? 'Physician workspace' : 'Patient workspace'} · ${user.tenant}`;
    byId('patient-form').hidden = !clinician();
    byId('conversation-form').hidden = !clinician();
    byId('note-form').hidden = !clinician();
    byId('assessment-panel').hidden = !clinician();
    byId('knowledge-link').hidden = !clinician();
    byId('notes-explanation').textContent = clinician()
      ? 'Draft notes require your explicit review and approval before a patient account can view them. Approval here is a development workflow, not a qualified electronic signature.'
      : 'Only notes explicitly approved by a clinician account are shown. General AI output is not a clinician-approved care instruction.';
    await loadPatients(epoch);
    if (state.sessionEpoch !== epoch || !state.user) return false;
    if (state.patients.length === 1) await selectPatient(state.patients[0].id, epoch);
    return state.sessionEpoch === epoch;
  }

  async function login() {
    if (state.authBusy) throw new Error('Wait for the current sign-in or sign-out request to finish.');
    const credentials = { tenant: byId('tenant').value.trim(), username: byId('username').value.trim(), password: byId('password').value };
    clearSession();
    const epoch = state.sessionEpoch;
    state.authBusy = true;
    try {
      const user = await post('/login', credentials);
      if (await enter(user, epoch)) status('Signed in to the synthetic development portal.');
    } finally {
      state.authBusy = false;
      byId('password').value = '';
    }
  }

  async function logout() {
    if (state.authBusy) throw new Error('Wait for the current sign-in or sign-out request to finish.');
    state.authBusy = true;
    const pending = post('/logout');
    clearSession(); // Hide data immediately, even if revocation is delayed.
    const epoch = state.sessionEpoch;
    try {
      await pending;
      if (state.sessionEpoch === epoch) status('Signed out.');
    } catch (error) {
      if (state.sessionEpoch === epoch) status('Local records cleared; server sign-out could not be confirmed. Try signing out again before changing accounts.', true);
    } finally {
      state.authBusy = false;
    }
  }

  async function selectPatient(id, epoch = state.sessionEpoch) {
    if (epoch !== state.sessionEpoch || !state.user) return;
    clearChart();
    const version = state.version;
    const patient = await api(`/patients/${id}`);
    if (state.version !== version || state.sessionEpoch !== epoch || !state.user) return;
    state.patient = patient;
    byId('chart').hidden = false;
    byId('chart-heading').textContent = patient.name;
    byId('chart-summary').textContent = `Synthetic record ${patient.id}${patient.dob ? ' · DOB ' + patient.dob : ''}${patient.mrn ? ' · MRN ' + patient.mrn : ''}`;
    byId('structured-chart-link').href = `/static/chart.html?patient_id=${encodeURIComponent(patient.id)}`;
    byId('imaging-link').href = `/static/imaging.html?patient=${encodeURIComponent(patient.id)}`;
    renderPatients();
    await Promise.all([loadConversations(version, epoch), loadNotes(version, epoch)]);
  }

  async function loadConversations(version = state.version, epoch = state.sessionEpoch) {
    if (!state.patient || state.sessionEpoch !== epoch || state.version !== version) return;
    const mark = snapshot();
    const items = await api(`/patients/${state.patient.id}/conversations`);
    if (!sameChart(mark)) return;
    state.conversations = items;
    byId('conversations').replaceChildren();
    if (!items.length) byId('conversations').appendChild(node('p', 'No internal conversations yet.'));
    for (const conversation of items) {
      const button = node('button', conversation.title);
      button.type = 'button';
      button.setAttribute('aria-pressed', String(state.conversation?.id === conversation.id));
      button.addEventListener('click', () => action(async () => {
        if (sameChart(mark)) await selectConversation(conversation);
      }));
      byId('conversations').appendChild(button);
    }
  }

  async function selectConversation(conversation) {
    if (!state.user || !state.patient || conversation.patient_id !== state.patient.id) return;
    state.threadVersion += 1;
    state.conversation = conversation;
    byId('messages').replaceChildren();
    byId('assessment-output').replaceChildren();
    byId('assessment-question').value = '';
    byId('message-content').value = '';
    byId('thread-heading').textContent = conversation.title;
    byId('thread').hidden = false;
    byId('draft-note').hidden = !clinician();
    await Promise.all([loadMessages(), loadConversations()]);
  }

  async function loadMessages() {
    if (!state.conversation) return;
    const mark = snapshot();
    const id = state.conversation.id;
    const items = await api(`/conversations/${id}/messages`);
    if (!sameThread(mark)) return;
    byId('messages').replaceChildren();
    for (const message of items) {
      const item = node('li');
      item.appendChild(node('p', `${message.sender_username} (${message.sender_role}) · ${message.created_at}`, 'meta'));
      item.appendChild(node('p', message.content, 'message-content'));
      item.appendChild(node('p', 'Stored in this internal test portal. External delivery is not available.', 'meta'));
      byId('messages').appendChild(item);
    }
  }

  async function sendMessage() {
    if (!state.conversation) throw new Error('Select an internal conversation first.');
    const content = byId('message-content').value.trim();
    if (!content) throw new Error('Enter a message.');
    const mark = snapshot();
    const id = state.conversation.id;
    await post(`/conversations/${id}/messages`, { content });
    if (!sameThread(mark)) return;
    if (state.conversation?.id === id) {
      byId('message-content').value = '';
      await loadMessages();
    }
    if (!sameThread(mark)) return;
    status('Message stored in the internal test conversation. No external message was delivered.');
  }

  async function loadNotes(version = state.version, epoch = state.sessionEpoch) {
    if (!state.patient || state.sessionEpoch !== epoch || state.version !== version) return;
    const mark = snapshot();
    const items = await api(`/patients/${state.patient.id}/notes`);
    if (!sameChart(mark)) return;
    byId('notes').replaceChildren();
    if (!items.length) byId('notes').appendChild(node('p', clinician() ? 'No notes yet.' : 'No clinician-approved notes are available.'));
    for (const note of items) {
      const item = node('article', undefined, 'note');
      item.appendChild(node('h4', note.status === 'approved' ? 'Clinician-approved note (test workflow)' : 'Draft — awaiting clinician review'));
      item.appendChild(node('p', `${note.author_username || 'Clinician'} · ${note.created_at}`, 'meta'));
      item.appendChild(node('p', note.content, 'note-content'));
      if (clinician() && note.status !== 'approved') {
        const approve = node('button', 'Approve this note for the patient account');
        approve.type = 'button';
        approve.addEventListener('click', () => action(async () => {
          if (!sameChart(mark)) return;
          await post(`/notes/${note.id}/approve`);
          if (!sameChart(mark)) return;
          await loadNotes(version, epoch);
          if (!sameChart(mark)) return;
          status('Note explicitly approved in the synthetic test workflow.');
        }));
        item.appendChild(approve);
      }
      byId('notes').appendChild(item);
    }
  }

  async function assessment() {
    if (!clinician() || !state.patient || !state.conversation) throw new Error('Select a patient and conversation in the physician workspace first.');
    const mark = snapshot();
    const output = await post(`/patients/${state.patient.id}/assessment`, {
      conversation_id: state.conversation.id, question: byId('assessment-question').value.trim(),
    });
    if (!sameThread(mark)) return;
    const holder = byId('assessment-output');
    holder.replaceChildren(node('p', 'Draft AI assessment — clinician review required. Not shared with the patient.'));
    for (const [key, value] of Object.entries(output)) {
      const section = node('section', undefined, 'assessment-section');
      section.appendChild(node('h4', key.replaceAll('_', ' ')));
      section.appendChild(node('pre', typeof value === 'string' ? value : JSON.stringify(value, null, 2), 'assessment-content'));
      holder.appendChild(section);
    }
    status('Draft assessment available for clinician review. No message or order was sent.');
  }

  async function action(fn) {
    status();
    const mark = snapshot();
    try { return await fn(); } catch (error) {
      if (sameChart(mark) || (error.epoch === state.sessionEpoch
          && (error.version === state.version || !state.user))) {
        status(error.message || 'Request failed.', true);
      }
    }
  }

  function bindForm(id, fn) {
    byId(id).addEventListener('submit', event => {
      event.preventDefault();
      const buttons = Array.from(byId(id).querySelectorAll('button'));
      buttons.forEach(button => { button.disabled = true; });
      return action(fn).finally(() => buttons.forEach(button => { button.disabled = false; }));
    });
  }

  async function boot() {
    bindForm('login-form', login);
    bindForm('patient-form', async () => {
      if (!clinician()) return;
      const mark = snapshot();
      const patient = await post('/patients', { name: byId('patient-name').value.trim(), patient_username: byId('patient-account').value.trim() || null });
      if (!sameSession(mark)) return;
      await loadPatients(mark.epoch);
      if (!sameChart(mark)) return;
      byId('patient-name').value = '';
      byId('patient-account').value = '';
      await selectPatient(patient.id, mark.epoch);
      if (sameSession(mark)) status('Synthetic record created.');
    });
    bindForm('conversation-form', async () => {
      if (!clinician() || !state.patient) throw new Error('Select a synthetic record in the physician workspace first.');
      const mark = snapshot();
      const conversation = await post(`/patients/${mark.patientId}/conversations`, { title: byId('conversation-title').value.trim() });
      if (!sameChart(mark)) return;
      byId('conversation-title').value = '';
      await selectConversation(conversation);
    });
    bindForm('message-form', sendMessage);
    bindForm('note-form', async () => {
      if (!state.patient) throw new Error('Select a synthetic record first.');
      const mark = snapshot();
      await post(`/patients/${mark.patientId}/notes`, { content: byId('note-content').value.trim() });
      if (!sameChart(mark)) return;
      byId('note-content').value = '';
      await loadNotes(mark.version, mark.epoch);
      if (!sameChart(mark)) return;
      status('Draft saved for clinician review; it is not patient-visible.');
    });
    bindForm('assessment-form', assessment);
    byId('draft-note').addEventListener('click', () => action(async () => {
      if (!state.patient || !state.conversation) throw new Error('Select a conversation first.');
      const mark = snapshot();
      await post(`/patients/${mark.patientId}/note-draft`, { conversation_id: mark.conversationId });
      if (!sameThread(mark)) return;
      await loadNotes(mark.version, mark.epoch);
      if (!sameThread(mark)) return;
      status('Transcript draft saved. This deterministic draft is not AI-generated and requires review.');
    }));
    byId('logout').addEventListener('click', () => action(logout));
    const epoch = state.sessionEpoch;
    try { await enter(await api('/session'), epoch); }
    catch (error) {
      if (state.sessionEpoch !== epoch && error.epoch !== state.sessionEpoch) return;
      if (state.sessionEpoch === epoch) clearSession();
      if (error.message !== 'Authentication required.') status(error.message, true);
    }
  }

  window.RaphaelPortal = { state, boot, login, selectPatient, selectConversation, sendMessage, assessment, clearSession, logout, action };
  window.RaphaelPortal.ready = boot();
})();
