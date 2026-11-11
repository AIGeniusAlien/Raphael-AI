// Evaluate the real navigation-release helper, not a duplicate implementation.
// Browser speech is mocked; these tests do not activate a microphone.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync('static/intelligence.js', 'utf8');
const start = source.indexOf('  async function releaseEncounterSessionBeforeNavigation() {');
const end = source.indexOf('  async function selectThread(id) {', start);
assert.ok(start >= 0 && end > start, 'The clinician navigation release function must exist');
const extracted = source.slice(start, end);

function harness(session, error = null) {
  const events = [], state = {session, sessionPatientId: 'synthetic-patient'};
  const voice = {stop() { events.push('microphone-stopped'); }};
  async function api(path, body) {
    events.push('server-revocation');
    assert.equal(path, '/intelligence/sessions/' + session + '/consent');
    assert.deepEqual(JSON.parse(JSON.stringify(body)), {consent:false});
    if (error) throw error;
    return {consent:false};
  }
  const context = vm.createContext({voice, api, state});
  const release = vm.runInContext(extracted + '\nreleaseEncounterSessionBeforeNavigation', context);
  return {release, events, state};
}

test('switching away from authorized encounter stops microphone then confirms revocation', async () => {
  const h = harness(12);
  await h.release();
  assert.deepEqual(h.events, ['microphone-stopped', 'server-revocation']);
  assert.equal(h.state.session, 12, 'Session identity remains available until caller finishes navigation');
});

test('no active encounter does not send a spurious revocation request', async () => {
  const h = harness(null);
  await h.release();
  assert.deepEqual(h.events, ['microphone-stopped']);
});

test('failed consent revocation prevents navigation without clearing patient binding', async () => {
  const h = harness(13, new Error('server unavailable'));
  await assert.rejects(h.release(), /server unavailable/);
  assert.deepEqual(h.events, ['microphone-stopped', 'server-revocation']);
  assert.equal(h.state.session, 13);
  assert.equal(h.state.sessionPatientId, 'synthetic-patient');
});

test('both conversation changes call the same encounter release safeguard', () => {
  assert.match(source, /async function selectThread\(id\)[\s\S]*?await releaseEncounterSessionBeforeNavigation\(\);/);
  assert.match(source, /\$\('new-draft'\)\.onclick=[\s\S]*?await releaseEncounterSessionBeforeNavigation\(\);/);
});
