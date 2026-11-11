// Real extracted clinician stream implementation under deterministic synthetic
// browser mocks. No medical data, inference provider or microphone is used.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

const source=fs.readFileSync('static/intelligence.js','utf8');
const start=source.indexOf('  async function send(prompt) {');
const end=source.indexOf('  async function cancel()',start);
assert.ok(start>=0&&end>start,'Clinician streaming implementation should be present');
const extracted=source.slice(start,end);
function harness(){
 const elements={
   'attachment-ai-consent':{checked:false},
   cancel:{disabled:true},send:{disabled:false},retry:{disabled:false},
   prompt:{value:'Original synthetic request'},speak:{checked:false}
 };
 const statuses=[],outputs=[],events=[];
 const state={thread:{id:41,requires_attachment_authorization:false},busy:false,attachmentSelected:new Set(),
  lastPrompt:'',generation:null,controller:null,epoch:10};
 let finishFetch;
 const request=new Promise(resolve=>{finishFetch=resolve;});
 const context=vm.createContext({
   state,
   $:id=>elements[id],
   entry:(role,content)=>{const holder={role,textContent:content};outputs.push(holder);return holder;},
   renderMarkdown:(holder,value)=>{holder.textContent=value;},
   renderEvidence:()=>{events.push('evidence');},
   status:value=>statuses.push(value),
   voice:{speak:()=>events.push('speech')},
   fetch:()=>request,
   AbortController,TextDecoder,Uint8Array,JSON
 });
 const send=vm.runInContext(extracted+'\nsend',context);
 return {send,state,elements,statuses,outputs,events,resolve:finishFetch};
}
test('late accepted response after canceled patient switch cannot clear a new draft or read old stream',async()=>{
 const h=harness(),pending=h.send('Original synthetic request');
 h.state.controller.abort();h.state.epoch++;h.state.thread={id:99};
 h.elements.prompt.value='New patient synthetic draft';
 let readerCreated=false;
 h.resolve({ok:true,body:{getReader(){readerCreated=true;throw Error('Old stream must not be consumed');}}});
 await pending;
 assert.equal(readerCreated,false);
 assert.equal(h.elements.prompt.value,'New patient synthetic draft');
 assert.deepEqual(h.events,[]);
 assert.equal(h.statuses.length,0);
});
test('accepted response in same conversation clears only the submitted draft',async()=>{
 const h=harness(),pending=h.send('Original synthetic request');
 h.resolve({ok:true,body:{getReader(){return {async read(){return {done:true,value:new Uint8Array()};}};}}});
 await pending;
 assert.equal(h.elements.prompt.value,'');
 assert.equal(h.state.busy,false);
 assert.equal(h.state.controller,null);
});
test('a clinician editing while a request is pending retains updated text',async()=>{
 const h=harness(),pending=h.send('Original synthetic request');
 h.elements.prompt.value='Updated synthetic question';
 h.resolve({ok:true,body:{getReader(){return {async read(){return {done:true,value:new Uint8Array()};}};}}});
 await pending;
 assert.equal(h.elements.prompt.value,'Updated synthetic question');
});
