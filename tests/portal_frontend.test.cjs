// Controller tests execute the shipped JS using a minimal DOM double, not a browser.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function element(tag = 'div') {
  const item = {
    tagName: tag, value: '', hidden: false, children: [], attributes: {}, listeners: {},
    _text: '',
    set textContent(value) { this._text = String(value); this.children = []; },
    get textContent() { return this._text; },
    set innerHTML(_) { throw new Error('Untrusted HTML assignment is forbidden'); },
    setAttribute(name, value) { this.attributes[name] = value; },
    replaceChildren(...children) { this.children = children; this._text = ''; },
    appendChild(child) { this.children.push(child); return child; },
    addEventListener(name, handler) { this.listeners[name] = handler; },
    querySelectorAll() { return []; },
  };
  return item;
}

async function controller(routes, waitForBoot = true) {
  const elements = new Map();
  for (const match of fs.readFileSync('static/portal.html', 'utf8').matchAll(/id="([^"]+)"/g)) {
    elements.set(match[1], element());
  }
  const calls = [];
  const context = vm.createContext({
    window: {},
    document: {
      getElementById(id) {
        if (!elements.has(id)) elements.set(id, element());
        return elements.get(id);
      },
      createElement: element,
    },
    fetch: async (path, options) => {
      calls.push({ path, options });
      const key = (options.method || 'GET') + ' ' + path;
      assert.ok(Object.hasOwn(routes, key), `Unexpected request: ${key}`);
      const response = typeof routes[key] === 'function' ? await routes[key](options) : routes[key];
      return { ok: (response.status || 200) < 400, status: response.status || 200,
        json: async () => response.data };
    },
  });
  vm.runInContext(fs.readFileSync('static/portal.js', 'utf8'), context);
  if (waitForBoot) await context.window.RaphaelPortal.ready;
  return { api: context.window.RaphaelPortal, elements, calls, context };
}

const signedOut = { 'GET /api/dev/session': { status: 401, data: { detail: 'Authentication required.' } } };
const physician = { id: 1, tenant: 'synthetic', username: 'synthetic-physician', role: 'clinician' };
const patientUser = { id: 2, tenant: 'synthetic', username: 'synthetic-patient', role: 'patient' };
const record = { id: 3, name: 'Synthetic patient' };
const thread = { id: 9, patient_id: 3, title: 'Synthetic internal conversation' };

function recordRoutes(user = physician) {
  return {
    'GET /api/dev/session': { data: user },
    'GET /api/dev/patients': { data: [record] },
    'GET /api/dev/patients/3': { data: record },
    'GET /api/dev/patients/3/conversations': { data: [thread] },
    'GET /api/dev/patients/3/notes': { data: [] },
    'GET /api/dev/conversations/9/messages': { data: [] },
  };
}

test('sign in sends credentials only to same-origin login, clears password, and enables physician workspace', async () => {
  const routes = { ...signedOut, 'POST /api/dev/login': { data: physician }, 'GET /api/dev/patients': { data: [] } };
  const { api, elements, calls } = await controller(routes);
  elements.get('tenant').value = 'synthetic';
  elements.get('username').value = 'synthetic-physician';
  elements.get('password').value = 'synthetic-password-only';
  await api.login();
  assert.equal(api.state.user.role, 'clinician');
  assert.equal(elements.get('password').value, '');
  assert.equal(elements.get('workspace').hidden, false);
  assert.equal(elements.get('patient-form').hidden, false);
  const request = calls.find(call => call.path === '/api/dev/login');
  assert.equal(request.options.credentials, 'same-origin');
  assert.equal(request.options.headers['Content-Type'], 'application/json');
  assert.equal(JSON.parse(request.options.body).password, 'synthetic-password-only');
});

test('failed login clears password and reports the server error', async () => {
  const { api, elements } = await controller({ ...signedOut,
    'POST /api/dev/login': { status: 401, data: { detail: 'Invalid development credentials.' } } });
  elements.get('password').value = 'synthetic-wrong-password';
  await api.action(api.login);
  assert.equal(elements.get('password').value, '');
  assert.equal(elements.get('status').textContent, 'Invalid development credentials.');
  assert.equal(api.state.user, null);
});

test('patient workspace hides clinician assessment and note-approval controls', async () => {
  const { api, elements } = await controller(recordRoutes(patientUser));
  assert.equal(api.state.patient.id, 3);
  assert.equal(elements.get('patient-form').hidden, true);
  assert.equal(elements.get('note-form').hidden, true);
  assert.equal(elements.get('assessment-panel').hidden, true);
  assert.equal(elements.get('conversation-form').hidden, true);
  assert.match(elements.get('notes-explanation').textContent, /Only notes explicitly approved/);
});

test('internal messages require explicit submission and never claim external delivery', async () => {
  const routes = recordRoutes();
  let sent = false;
  routes['POST /api/dev/conversations/9/messages'] = options => {
    assert.equal(JSON.parse(options.body).content, '<img src=x onerror=alert(1)>');
    sent = true;
    return { data: { id: 1, delivery_status: 'stored_in_portal' } };
  };
  routes['GET /api/dev/conversations/9/messages'] = () => ({ data: sent ? [{
    sender_username: 'synthetic-physician', sender_role: 'clinician', created_at: 'synthetic time',
    content: '<img src=x onerror=alert(1)>', delivery_status: 'stored_in_portal',
  }] : [] });
  const { api, elements, calls } = await controller(routes);
  await api.selectConversation(thread);
  assert.equal(calls.filter(call => call.options.method === 'POST').length, 0);
  elements.get('message-content').value = '<img src=x onerror=alert(1)>';
  await api.sendMessage();
  const message = elements.get('messages').children[0];
  assert.equal(message.children[1].textContent, '<img src=x onerror=alert(1)>');
  assert.match(message.children[2].textContent, /External delivery is not available/);
  assert.equal(elements.get('message-content').value, '');
  assert.match(elements.get('status').textContent, /No external message was delivered/);
});

test('chart selection clears the previous internal conversation and draft inputs', async () => {
  const routes = { ...recordRoutes(),
    'GET /api/dev/patients/4': { data: { id: 4, name: '<script>synthetic</script>' } },
    'GET /api/dev/patients/4/conversations': { data: [] },
    'GET /api/dev/patients/4/notes': { data: [] },
  };
  const { api, elements } = await controller(routes);
  await api.selectConversation(thread);
  elements.get('message-content').value = 'synthetic unsent message';
  await api.selectPatient(4);
  assert.equal(api.state.conversation, null);
  assert.equal(elements.get('thread').hidden, true);
  assert.equal(elements.get('message-content').value, '');
  assert.equal(elements.get('chart-heading').textContent, '<script>synthetic</script>');
});

test('expired session clears chart and hides authorized workspace', async () => {
  const routes = recordRoutes();
  routes['GET /api/dev/conversations/9/messages'] = { status: 401, data: { detail: 'Authentication required.' } };
  const { api, elements } = await controller(routes);
  await api.action(() => api.selectConversation(thread));
  assert.equal(api.state.user, null);
  assert.equal(api.state.patient, null);
  assert.equal(elements.get('workspace').hidden, true);
  assert.equal(elements.get('messages').children.length, 0);
});

test('logout revokes via JSON POST and clears all displayed records', async () => {
  const { api, elements, calls } = await controller({ ...recordRoutes(), 'POST /api/dev/logout': { data: { status: 'ok' } } });
  await elements.get('logout').listeners.click();
  assert.equal(api.state.user, null);
  assert.equal(elements.get('patients').children.length, 0);
  assert.equal(elements.get('login-panel').hidden, false);
  const request = calls.find(call => call.path === '/api/dev/logout');
  assert.equal(request.options.body, '{}');
});

test('assessment renders draft data as text and performs no message send', async () => {
  const routes = recordRoutes();
  routes['POST /api/dev/patients/3/assessment'] = { data: {
    problem_representation: '<svg onload=alert(1)>', confidence: 'Low (synthetic test)',
    evidence_used: [{ chunk_id: 'synthetic-test-only' }],
  } };
  const { api, elements, calls } = await controller(routes);
  await api.selectConversation(thread);
  elements.get('assessment-question').value = 'Synthetic question';
  await api.assessment();
  assert.equal(elements.get('assessment-output').children[1].children[1].textContent, '<svg onload=alert(1)>');
  assert.equal(calls.some(call => call.options.method === 'POST' && call.path.endsWith('/messages')), false);
  assert.match(elements.get('status').textContent, /No message or order was sent/);
});

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

function secondChart() {
  return {
    'GET /api/dev/patients/4': { data: { id: 4, name: 'Another synthetic patient' } },
    'GET /api/dev/patients/4/conversations': { data: [] },
    'GET /api/dev/patients/4/notes': { data: [] },
  };
}

test('late conversation creation cannot select a prior patient thread under a new chart', async () => {
  const response = deferred();
  const { api, elements, calls } = await controller({ ...recordRoutes(), ...secondChart(),
    'POST /api/dev/patients/3/conversations': () => response.promise });
  elements.get('conversation-title').value = 'New internal conversation';
  const creation = elements.get('conversation-form').listeners.submit({ preventDefault() {} });
  await api.selectPatient(4);
  response.resolve({ data: { id: 10, patient_id: 3, title: 'Prior patient thread' } });
  await creation;
  assert.equal(api.state.patient.id, 4);
  assert.equal(api.state.conversation, null);
  assert.equal(elements.get('thread').hidden, true);
  assert.equal(calls.some(call => call.path === '/api/dev/conversations/10/messages'), false);
});

test('a stale 401 response cannot clear a newer authenticated session', async () => {
  const response = deferred();
  let newLogin = false;
  const routes = recordRoutes();
  routes['GET /api/dev/conversations/9/messages'] = () => response.promise;
  routes['POST /api/dev/login'] = () => { newLogin = true; return { data: patientUser }; };
  routes['GET /api/dev/patients'] = () => ({ data: newLogin ? [] : [record] });
  const { api, elements } = await controller(routes);
  const oldRequest = api.action(() => api.selectConversation(thread));
  elements.get('tenant').value = 'synthetic';
  elements.get('username').value = patientUser.username;
  elements.get('password').value = 'synthetic-password';
  await api.login();
  response.resolve({ status: 401, data: { detail: 'Old session expired.' } });
  await oldRequest;
  assert.equal(api.state.user.username, patientUser.username);
  assert.equal(elements.get('workspace').hidden, false);
  assert.match(elements.get('status').textContent, /Signed in/);
});

test('a delayed prior-session patient list cannot populate after logout and new login', async () => {
  const oldPatients = deferred();
  const requested = deferred();
  let patientRequests = 0;
  const routes = { ...recordRoutes(),
    'GET /api/dev/patients': () => {
      patientRequests += 1;
      if (patientRequests === 1) { requested.resolve(); return oldPatients.promise; }
      return { data: [] };
    },
    'POST /api/dev/logout': { data: { status: 'ok' } },
    'POST /api/dev/login': { data: patientUser },
  };
  const { api, elements } = await controller(routes, false);
  await requested.promise;
  await api.logout();
  elements.get('tenant').value = 'synthetic';
  elements.get('username').value = patientUser.username;
  elements.get('password').value = 'synthetic-password';
  await api.login();
  oldPatients.resolve({ data: [record] });
  await api.ready;
  assert.equal(api.state.user.username, patientUser.username);
  assert.equal(api.state.patients.length, 0);
  assert.equal(api.state.patient, null);
  assert.equal(elements.get('chart').hidden, true);
});

test('retired note approval controls cannot approve after changing charts', async () => {
  const routes = { ...recordRoutes(), ...secondChart() };
  routes['GET /api/dev/patients/3/notes'] = { data: [{ id: 8, content: 'Prior draft', status: 'draft' }] };
  const { api, elements, calls } = await controller(routes);
  const approve = elements.get('notes').children[0].children.at(-1);
  await api.selectPatient(4);
  await approve.listeners.click();
  assert.equal(calls.some(call => call.path === '/api/dev/notes/8/approve'), false);
  assert.equal(api.state.patient.id, 4);
});

test('late note creation cannot clear another chart’s draft or overwrite its notes', async () => {
  const response = deferred();
  const { api, elements } = await controller({ ...recordRoutes(), ...secondChart(),
    'POST /api/dev/patients/3/notes': () => response.promise });
  elements.get('note-content').value = 'Synthetic first patient draft';
  const creation = elements.get('note-form').listeners.submit({ preventDefault() {} });
  await api.selectPatient(4);
  elements.get('note-content').value = 'Synthetic second patient draft';
  response.resolve({ data: { id: 5, patient_id: 3, status: 'draft' } });
  await creation;
  assert.equal(elements.get('note-content').value, 'Synthetic second patient draft');
  assert.equal(api.state.patient.id, 4);
});

test('current chart-loading errors remain visible while stale chart errors are suppressed', async () => {
  const oldResponse = deferred();
  const { api, elements } = await controller({ ...recordRoutes(), ...secondChart(),
    'GET /api/dev/patients/5': { status: 404, data: { detail: 'Synthetic record not found.' } },
    'GET /api/dev/patients/6': () => oldResponse.promise });
  await api.action(() => api.selectPatient(5));
  assert.equal(elements.get('status').textContent, 'Synthetic record not found.');
  const oldRequest = api.action(() => api.selectPatient(6));
  await api.selectPatient(4);
  oldResponse.resolve({ status: 404, data: { detail: 'Prior chart error.' } });
  await oldRequest;
  assert.equal(elements.get('status').textContent, '');
  assert.equal(api.state.patient.id, 4);
});

test('switching conversations clears assessment context and ignores an older assessment after reselecting its thread', async () => {
  const pending = deferred();
  const routes = { ...recordRoutes(),
    'GET /api/dev/conversations/10/messages': { data: [] },
    'POST /api/dev/patients/3/assessment': () => pending.promise,
  };
  const { api, elements } = await controller(routes);
  await api.selectConversation(thread);
  elements.get('assessment-question').value = 'Question about first synthetic thread';
  elements.get('assessment-output').textContent = 'Earlier assessment';
  const assessment = api.assessment();
  await api.selectConversation({ id: 10, patient_id: 3, title: 'Second synthetic thread' });
  assert.equal(elements.get('assessment-output').children.length, 0);
  assert.equal(elements.get('assessment-output').textContent, '');
  assert.equal(elements.get('assessment-question').value, '');
  await api.selectConversation(thread);
  pending.resolve({ data: { problem_representation: 'Stale first-thread assessment' } });
  await assessment;
  assert.equal(elements.get('assessment-output').children.length, 0);
});
