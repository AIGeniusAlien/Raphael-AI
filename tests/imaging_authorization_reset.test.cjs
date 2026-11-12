// Real imaging API/reset functions extracted from the shipping browser script.
// All fixtures are synthetic and never require a medical or imaging model.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

const code=fs.readFileSync('static/imaging.js','utf8');
const begin=code.indexOf('  async function api(path, options = {}) {');
const end=code.indexOf('  const post =',begin);
const resetStart=code.indexOf('  function reset() {');
const resetEnd=code.indexOf('  async function loadStudies',resetStart);
assert.ok(begin>=0&&end>begin&&resetStart>=0&&resetEnd>resetStart,
  'Real imaging controller authorization and reset functions should be preserved');

function fixture(status){
 const ids=['workspace','viewer','prior-view','reports','studies','current-image',
   'prior-image','metadata','fhir-output','dicomweb-output','findings','impression','recommendation',
   'dicom-file','fhir-id','patient-heading','patient'];
 const elements=Object.fromEntries(ids.map(id=>[id,{
   hidden:false,value:id,textContent:'Synthetic protected content',
   children:['synthetic fixture'],
   replaceChildren(){this.children=[];},
   removeAttribute(name){if(name==='src')this.src=undefined;},
   src:'synthetic:image'
 }]));
 const state={patient:{id:1},epoch:2,study:{id:5},studies:[{id:5}]};
 const sandbox=vm.createContext({
  el:id=>elements[id],state,
  fetch:async()=>({ok:status===200,status,async json(){return {detail:'Synthetic access rejected'};}}),
  Error,JSON
 });
 const codeUnderTest=code.slice(resetStart,resetEnd)+'\n'+code.slice(begin,end)+'\napi';
 return {api:vm.runInContext(codeUnderTest,sandbox),state,elements};
}
test('unauthorized synthetic imaging API request clears all protected cached presentation',async()=>{
 for(const status of [401,403]){
  const {api,state,elements}=fixture(status);
  await assert.rejects(api('/imaging/studies/5/reports'),/Synthetic access rejected/);
  assert.equal(state.patient,null);
  assert.equal(state.study,null);
  assert.equal(state.studies.length,0);
  assert.equal(elements.workspace.hidden,true);
  assert.equal(elements.viewer.hidden,true);
  assert.equal(elements['prior-view'].hidden,true);
  assert.equal(elements['current-image'].src,undefined);
  assert.equal(elements['prior-image'].src,undefined);
  assert.equal(elements.metadata.textContent,'');
  assert.equal(elements['fhir-output'].textContent,'');
  assert.equal(elements['dicomweb-output'].textContent,'');
  assert.equal(elements['patient-heading'].textContent,'');
  assert.equal(elements.studies.children.length,0);
  assert.equal(elements.patient.value,'');
 }
});
test('ordinary synthetic request error does not erase an otherwise authorized workspace',async()=>{
 const {api,state,elements}=fixture(409);
 await assert.rejects(api('/imaging/studies/5/reports'),/Synthetic access rejected/);
 assert.equal(state.patient.id,1);
 assert.equal(elements.workspace.hidden,false);
});
