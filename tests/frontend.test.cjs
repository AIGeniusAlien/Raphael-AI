// Execute the real frontend boot/selection logic with a minimal DOM test double.
// This verifies controller behavior, not browser layout or rendering compatibility.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

async function frontend(responses) {
  const elements = new Map();
  function element() {
    return { style: {}, children: [], value: '', scrollHeight: 0,
      appendChild(child) { this.children.push(child); },
      addEventListener() {}, querySelector() { return element(); },
      get firstChild() { return element(); },
      set innerHTML(value) { this.children = []; },
    };
  }
  const calls = [];
  const context = vm.createContext({
    document: {
      getElementById(id) {
        if (!elements.has(id)) elements.set(id, element());
        return elements.get(id);
      },
      createElement: element,
    },
    fetch: async path => {
      calls.push(path);
      assert.ok(Object.hasOwn(responses, path), `Unexpected API request: ${path}`);
      return { ok: true, headers: new Map([['content-type', 'application/json']]),
        json: async () => responses[path] };
    },
    setTimeout, clearTimeout,
  });
  const source = fs.readFileSync('static/app.js', 'utf8');
  vm.runInContext(source.replace(/boot\(\);\s*$/, 'globalThis.bootPromise = boot();'), context);
  await context.bootPromise;
  return { context, calls, elements };
}

test('browser boot loads patients and binds controls without a nonexistent /api/env', async () => {
  const { calls, elements } = await frontend({ '/api/patients': [] });
  assert.deepEqual(calls, ['/api/patients']);
  assert.equal(typeof elements.get('add-patient').onclick, 'function');
  assert.equal(typeof elements.get('add-conv').onclick, 'function');
  assert.equal(typeof elements.get('send').onclick, 'function');
});

test('selecting a patient with no conversations clears the prior conversation and messages', async () => {
  const { context } = await frontend({ '/api/patients': [], '/api/conversations?patient_id=2': [] });
  await vm.runInContext(`
    state.currentPatient = {id: 2};
    state.currentConversation = {id: 1, patient_id: 1};
    state.messages = [{role: 'user', content: 'synthetic previous patient'}];
    loadConversations();
  `, context);
  assert.equal(vm.runInContext('state.currentConversation', context), null);
  assert.equal(vm.runInContext('state.messages.length', context), 0);
});

test('selecting another patient loads that patient’s conversation and messages', async () => {
  const { context, calls } = await frontend({
    '/api/patients': [],
    '/api/conversations?patient_id=2': [{id: 22, patient_id: 2, title: 'Synthetic'}],
    '/api/messages?conversation_id=22': [{role: 'assistant', content: 'Synthetic reply'}],
  });
  await vm.runInContext(`
    state.currentPatient = {id: 2};
    state.currentConversation = {id: 11, patient_id: 1};
    loadConversations();
  `, context);
  assert.equal(vm.runInContext('state.currentConversation.id', context), 22);
  assert.equal(vm.runInContext('state.messages[0].content', context), 'Synthetic reply');
  assert.ok(calls.includes('/api/messages?conversation_id=22'));
});
