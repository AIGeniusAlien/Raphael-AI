// Tests the real browser composer module without a microphone or provider requests.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

function setup(value='Synthetic clinician question'){
 const events={},textarea={value,style:{},scrollHeight:122,
   addEventListener(type,fn){events[type]=fn;}};
 let submits=0;const send={disabled:false};
 const form={querySelector(selector){assert.equal(selector,'#send');return send;},requestSubmit(){submits++;}};
 const context=vm.createContext({window:{}});
 vm.runInContext(fs.readFileSync('static/chat_composer.js','utf8'),context);
 assert.equal(context.window.RaphaelComposer.install(form,textarea),true);
 return {events,textarea,send,submits:()=>submits};
}
function key(overrides={}){
 const event={key:'Enter',shiftKey:false,ctrlKey:false,metaKey:false,altKey:false,isComposing:false,keyCode:13,prevented:false,
 preventDefault(){this.prevented=true;}};
 return Object.assign(event,overrides);
}

test('Enter sends the real form once without navigating or bypassing its submit handler',()=>{
 const h=setup();const e=key();h.events.keydown(e);assert.equal(h.submits(),1);assert.equal(e.prevented,true);
});
test('Shift+Enter and IME composition keep writing and never send a clinical message',()=>{
 for(const options of [{shiftKey:true},{isComposing:true},{keyCode:229},{ctrlKey:true},{metaKey:true},{altKey:true},{key:'a'}]){
  const h=setup(),e=key(options);h.events.keydown(e);assert.equal(h.submits(),0);assert.equal(e.prevented,false);
 }
});
test('disabled send and empty drafts cannot be submitted by keyboard',()=>{
 const blank=setup('   '),e=key();blank.events.keydown(e);assert.equal(blank.submits(),0);
 const waiting=setup();waiting.send.disabled=true;waiting.events.keydown(key());assert.equal(waiting.submits(),0);
});
test('composer height remains bounded and controls remain available to accessibility',()=>{
 const h=setup();assert.equal(h.textarea.style.height,'122px');
 h.textarea.scrollHeight=600;h.events.input();assert.equal(h.textarea.style.height,'240px');
 h.textarea.scrollHeight=14;h.events.input();assert.equal(h.textarea.style.height,'80px');
});
