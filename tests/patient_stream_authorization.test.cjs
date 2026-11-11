// Test the actual patient education stream authorization boundary, using
// a synthetic HTTP fixture and no model, microphone, or medical information.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

const source=fs.readFileSync('static/patient-ai.js','utf8');
const start=source.indexOf(' async function send(question){');
const end=source.indexOf(" $('new-thread').onsubmit",start);
assert.ok(start>=0&&end>start,'Patient education stream handler must remain present');
const extracted=source.slice(start,end);

function setup(){
 const elements={
   'patient-ai':{hidden:false},
   send:{disabled:false},regenerate:{disabled:false},stop:{disabled:true},
   prompt:{value:'Synthetic educational question'}
 };
 const events=[],state={thread:{id:10,consent_ai:true},busy:false,abort:null,last:'',epoch:2};
 let resolveFetch;const fetchPromise=new Promise(resolve=>resolveFetch=resolve);
 const context=vm.createContext({
  state,$:id=>elements[id],fetch:()=>fetchPromise,
  status:value=>events.push('status:'+value),
  message:(role,content)=>({textContent:content}),
  sources:()=>events.push('evidence'),
  AbortController,TextDecoder,Uint8Array,JSON
 });
 const send=vm.runInContext(extracted+'\nsend',context);
 return {send,state,elements,events,resolve:resolveFetch};
}
test('patient streaming authorization denial conceals saved education history and retains unsent draft',async()=>{
 for(const status of [401,403]){
  const h=setup(),pending=h.send('Synthetic educational question');
  h.resolve({ok:false,status,async json(){return {detail:'Access no longer authorized.'};}});
  await assert.rejects(pending,/Access no longer authorized/);
  assert.equal(h.elements['patient-ai'].hidden,true);
  assert.equal(h.elements.prompt.value,'Synthetic educational question');
  assert.equal(h.state.busy,false);
  assert.equal(h.state.abort,null);
  assert.equal(h.elements.stop.disabled,true);
 }
});
test('patient stream rejection without revoked session does not conceal accessible workspace',async()=>{
 const h=setup(),pending=h.send('Synthetic educational question');
 h.resolve({ok:false,status:409,async json(){return {detail:'Consent not active.'};}});
 await assert.rejects(pending,/Consent not active/);
 assert.equal(h.elements['patient-ai'].hidden,false);
 assert.equal(h.elements.prompt.value,'Synthetic educational question');
});

test('late accepted patient response after stop cannot consume old stream or clear a new draft',async()=>{
 const h=setup(),pending=h.send('Synthetic educational question');
 h.state.abort.abort();
 h.state.busy=false;
 h.elements.prompt.value='Another synthetic draft';
 let touched=false;
 h.resolve({ok:true,body:{getReader(){touched=true;throw Error('Canceled response accessed a stream');}}});
 await pending;
 assert.equal(touched,false);
 assert.equal(h.elements.prompt.value,'Another synthetic draft');
 assert.equal(h.events.length,0);
});
test('old response after switching patient context cannot modify new conversation',async()=>{
 const h=setup(),pending=h.send('Synthetic educational question');
 h.state.epoch++;
 h.state.thread={id:88,consent_ai:true};
 h.elements.prompt.value='New synthetic patient question';
 let touched=false;
 h.resolve({ok:true,body:{getReader(){touched=true;throw Error('Wrong patient stream consumed');}}});
 await pending;
 assert.equal(touched,false);
 assert.equal(h.elements.prompt.value,'New synthetic patient question');
});
