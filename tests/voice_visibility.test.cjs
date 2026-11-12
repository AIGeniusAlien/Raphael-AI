// Synthetic-only test of the real clinician tab-privacy handler.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync('static/intelligence.js','utf8');
const begin=source.indexOf("  document.addEventListener('visibilitychange'");
const end=source.indexOf("  window.addEventListener('pagehide'",begin);
assert.ok(begin>=0&&end>begin,'Clinician tab visibility microphone safeguard required');
const extracted=source.slice(begin,end);
function setup(hidden,active){
 const events=[],field={textContent:''};
 const document={hidden,addEventListener(type,handler){events.push(type);this.callback=handler;}};
 const voice={active,stop(){events.push('microphone-stopped');this.active=false;}};
 vm.runInNewContext(extracted,{document,voice,$:()=>field});
 return {document,voice,field,events};
}
test('hidden clinician tab stops active microphone without automatic restart',()=>{
 const h=setup(true,true);
 h.document.callback();
 assert.equal(h.voice.active,false);
 assert.deepEqual(h.events,['visibilitychange','microphone-stopped']);
 assert.match(h.field.textContent,/Start listening explicitly/);
 h.document.hidden=false;h.document.callback();
 assert.deepEqual(h.events,['visibilitychange','microphone-stopped']);
});
test('visible clinician tab does not interrupt active recognition',()=>{
 const h=setup(false,true);h.document.callback();
 assert.equal(h.voice.active,true);
 assert.deepEqual(h.events,['visibilitychange']);
});
test('hidden tab without active recording does not claim an interruption',()=>{
 const h=setup(true,false);h.document.callback();
 assert.deepEqual(h.events,['visibilitychange']);
 assert.equal(h.field.textContent,'');
});
