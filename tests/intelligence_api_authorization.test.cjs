// Direct tests of the shipping clinician JSON API's access-denial boundary.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync('static/intelligence.js','utf8');
const start=source.indexOf('  async function api(');
const end=source.indexOf('  async function action(',start);
assert.ok(start>=0&&end>start,'Clinician API authorization boundary must exist');
function fixture(status){
 const workspace={hidden:false},events=[];
 const api=vm.runInNewContext(source.slice(start,end)+'\napi',{
  fetch:async()=>({ok:status===200,status,json:async()=>({detail:'Synthetic scoped denial'})}),
  voice:{stop(){events.push('microphone-stopped');}},
  $:()=>workspace,JSON,Error
 });
 return {api,workspace,events};
}
test('clinician scoped 404 grant revocation conceals cached chart and stops voice',async()=>{
 for(const path of ['/intelligence/sessions/42','/intelligence/threads/8','/patients/3']){
  const h=fixture(404);
  await assert.rejects(h.api(path),/Synthetic scoped denial/);
  assert.equal(h.workspace.hidden,true);
  assert.deepEqual(h.events,['microphone-stopped']);
 }
});
test('clinician denied session or role also conceals workspace',async()=>{
 for(const status of [401,403]){
  const h=fixture(status);
  await assert.rejects(h.api('/intelligence/sessions/42'),/Synthetic scoped denial/);
  assert.equal(h.workspace.hidden,true);
  assert.deepEqual(h.events,['microphone-stopped']);
 }
});
test('unrelated missing citation and ordinary 409 do not erase authorized chart',async()=>{
 for(const [status,path] of [[404,'/knowledge/citations/absent'],[409,'/intelligence/sessions/42']]){
  const h=fixture(status);
  await assert.rejects(h.api(path),/Synthetic scoped denial/);
  assert.equal(h.workspace.hidden,false);
  assert.deepEqual(h.events,[]);
 }
});
test('authorized clinician read does not stop voice or conceal conversation',async()=>{
 const h=fixture(200);
 const result=await h.api('/intelligence/threads/8');
 assert.equal(result.detail,'Synthetic scoped denial');
 assert.equal(h.workspace.hidden,false);
 assert.deepEqual(h.events,[]);
});
