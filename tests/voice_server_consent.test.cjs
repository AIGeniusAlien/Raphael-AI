// Exercise the actual clinician microphone activation handler under synthetic
// backend fixtures. No microphone is ever activated in this test process.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

const source=fs.readFileSync('static/intelligence.js','utf8');
const begin=source.indexOf("  $('listen').onclick=");
const end=source.indexOf("  $('pause').onclick=",begin);
assert.ok(begin>=0&&end>begin,'Real microphone permission handler must exist');
const handler=source.slice(begin,end);

function harness(result){
 const fields={
  listen:{onclick:null},consent:{checked:true},
  mode:{value:'standard'},wake:{value:'Raphael'}
 };
 const state={session:42,sessionPatientId:'7',epoch:8};
 const events=[];
 let resolve;
 const request=new Promise(r=>resolve=r);
 const voice={stop:()=>events.push('stopped'),start:async()=>events.push('microphone_requested')};
 const api=async()=>result??request;
 const context=vm.createContext({state,voice,$:id=>fields[id],api,action:fn=>fn(),Error,String});
 vm.runInContext(handler,context);
 return {state,fields,events,resolve,click:()=>fields.listen.onclick()};
}
test('server-revoked recording consent denies microphone and clears local consent',async()=>{
 const h=harness({patient_id:7,consent:false,state:'stopped'});
 await assert.rejects(h.click(),/Recording consent was revoked/);
 assert.deepEqual(h.events,['stopped']);
 assert.equal(h.fields.consent.checked,false);
});
test('same-patient active server consent allows clinician to request microphone',async()=>{
 const h=harness({patient_id:7,consent:true,state:'active'});
 await h.click();
 assert.deepEqual(h.events,['microphone_requested']);
});
test('server session for a different patient fails closed',async()=>{
 const h=harness({patient_id:19,consent:true,state:'active'});
 await assert.rejects(h.click(),/Recording consent was revoked/);
 assert.deepEqual(h.events,['stopped']);
});
test('late response after encounter switch does not activate stale microphone',async()=>{
 const h=harness(null);
 const pending=h.click();
 h.state.epoch++;
 h.state.session=44;
 h.resolve({patient_id:7,consent:true,state:'active'});
 await pending;
 assert.deepEqual(h.events,[]);
});
test('server unavailability does not attempt microphone permission',async()=>{
 const h=harness(null);
 const pending=h.click();
 h.resolve(Promise.reject(new Error('Synthetic server unavailable')));
 await assert.rejects(pending,/Synthetic server unavailable/);
 assert.deepEqual(h.events,[]);
});
