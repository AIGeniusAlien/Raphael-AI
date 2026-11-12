// Browser-service fixtures only: these tests do not exercise a real microphone.
const {test}=require('node:test');const assert=require('node:assert/strict');const vm=require('node:vm');const fs=require('node:fs');
function setup({available=true,media}={}){
 const instances=[],texts=[],commands=[],statuses=[];let stopped=0,speaks=0,cancels=0,timer,watchdog;
 class Recognition{constructor(){instances.push(this);}start(){this.started=true;}stop(){this.stopped=true;}}
 const window={speechSynthesis:{cancel(){cancels++;},speak(){speaks++;}},SpeechSynthesisUtterance:class{constructor(text){this.text=text;}}};if(available)window.SpeechRecognition=Recognition;
 const context=vm.createContext({window,navigator:{mediaDevices:{getUserMedia:media||async function(){return{getTracks:()=>[{stop(){stopped++;}}]};}}},setTimeout(fn){timer=fn;return 1;},clearTimeout(){},setInterval(fn){watchdog=fn;return 2;},clearInterval(){watchdog=null;}});
 vm.runInContext(fs.readFileSync('static/voice.js','utf8'),context);const voice=new window.RaphaelVoice({onText:t=>texts.push(t),onCommand:t=>commands.push(t),onStatus:t=>statuses.push(t)});
 return{voice,instances,texts,commands,statuses,window,tracksStopped:()=>stopped,speaks:()=>speaks,cancels:()=>cancels,timeout:()=>timer(),watchdog:()=>watchdog&&watchdog()};
}
test('consent and device support gate microphone access',async()=>{let calls=0;const a=setup({media:async()=>{calls++;}});await assert.rejects(a.voice.start({consent:false}),/consent/);assert.equal(calls,0);const b=setup({available:false});await assert.rejects(b.voice.start({consent:true}),/unavailable/);});
test('permission denial never enables recognition',async()=>{const a=setup({media:async()=>{throw new Error('permission denied');}});await assert.rejects(a.voice.start({consent:true}),/denied/);assert.equal(a.instances.length,0);assert.equal(a.voice.active,false);});
test('standard mode emits only final text and never wake commands',async()=>{const a=setup();await a.voice.start({consent:true,mode:'standard',wakePhrase:'Hey Raphael'});const r=a.instances[0];r.onresult({resultIndex:0,results:[Object.assign([{transcript:'partial'}],{isFinal:false})]});assert.equal(a.texts.length,0);r.onresult({resultIndex:0,results:[Object.assign([{transcript:'Hey Raphael synthetic'}],{isFinal:true})]});assert.deepEqual(a.texts,['Hey Raphael synthetic']);assert.equal(a.commands.length,0);assert.equal(a.tracksStopped(),1);a.voice.stop();assert.equal(r.stopped,true);});
test('handsfree activation requires configured prefix and supports interruption',async()=>{const a=setup();await a.voice.start({consent:true,mode:'handsfree',wakePhrase:'Hey Raphael'});const r=a.instances[0];for(const [index,text] of ['ordinary transcript','HEY RAPHAEL, synthetic question'].entries())r.onresult({resultIndex:index,results:Array.from({length:index+1},(_,i)=>Object.assign([{transcript:i===index?text:'ordinary transcript'}],{isFinal:true}))});assert.deepEqual(a.commands,['synthetic question']);assert.deepEqual(a.texts,['ordinary transcript']);a.voice.speak('Synthetic');assert.equal(a.speaks(),1);a.voice.stop();assert.ok(a.cancels()>0);});
test('late permission grant after stop cannot restart recording',async()=>{let resolve;const a=setup({media:()=>new Promise(r=>resolve=r)});const pending=a.voice.start({consent:true,mode:'handsfree',wakePhrase:'Hey Raphael'});a.voice.stop();let stopped=false;resolve({getTracks:()=>[{stop(){stopped=true;}}]});await pending;assert.equal(stopped,true);assert.equal(a.voice.active,false);assert.equal(a.instances.length,0);});
test('timeout and browser service errors stop listening',async()=>{const a=setup();await a.voice.start({consent:true,mode:'handsfree',wakePhrase:'Hey Raphael'});a.timeout();assert.equal(a.voice.active,false);await a.voice.start({consent:true,mode:'handsfree',wakePhrase:'Hey Raphael'});a.instances.at(-1).onerror({error:'not-allowed'});assert.equal(a.voice.active,false);assert.match(a.statuses.at(-1),/Microphone stopped/);});

test('retired recognition callbacks cannot enter a new voice session',async()=>{const a=setup();await a.voice.start({consent:true,mode:'handsfree',wakePhrase:'Hey Raphael'});const old=a.instances[0];a.voice.stop();await a.voice.start({consent:true,mode:'handsfree',wakePhrase:'Hey Raphael'});old.onresult({resultIndex:0,results:[Object.assign([{transcript:'Hey Raphael retired patient'}],{isFinal:true})]});old.onerror({error:'not-allowed'});assert.equal(a.texts.length,0);assert.equal(a.commands.length,0);assert.equal(a.voice.active,true);a.voice.stop();});

test('repeated final results cannot duplicate transcript segments or handsfree commands',async()=>{
 const a=setup();await a.voice.start({consent:true,mode:'handsfree',wakePhrase:'Hey Raphael'});
 const r=a.instances[0];
 const partial=Object.assign([{transcript:'Hey Raphael synthetic'}],{isFinal:false});
 const final=Object.assign([{transcript:'Hey Raphael synthetic question'}],{isFinal:true});
 r.onresult({resultIndex:0,results:[partial]});
 r.onresult({resultIndex:0,results:[final]});
 r.onresult({resultIndex:0,results:[final]});
 r.onresult({resultIndex:0,results:[final,Object.assign([{transcript:'Second final'}],{isFinal:true})]});
 assert.deepEqual(a.texts,['Second final']);
 assert.deepEqual(a.commands,['synthetic question']);
 a.voice.stop();
});
test('blank final result never causes a transcript or wake command',async()=>{
 const a=setup();await a.voice.start({consent:true,mode:'handsfree',wakePhrase:'Hey Raphael'});
 const r=a.instances[0];
 r.onresult({resultIndex:0,results:[Object.assign([{transcript:'   '}],{isFinal:true})]});
 assert.deepEqual(a.texts,[]);assert.deepEqual(a.commands,[]);
 a.voice.stop();
});

test('wake phrase must be delimited; command-like utterances are never dictated',async()=>{
 const a=setup();await a.voice.start({consent:true,mode:'handsfree',wakePhrase:'Hey Raphael'});
 const r=a.instances[0];
 for(const [index,utterance] of ['Hey Raphaelite is not a command','Hey Raphael','Hey Raphael, explain the synthetic example'].entries()){
  const results=Array.from({length:index+1},(_,i)=>Object.assign([{transcript:
    ['Hey Raphaelite is not a command','Hey Raphael','Hey Raphael, explain the synthetic example'][i]}],{isFinal:true}));
  r.onresult({resultIndex:index,results});
 }
 assert.deepEqual(a.texts,['Hey Raphaelite is not a command']);
 assert.deepEqual(a.commands,['explain the synthetic example']);
 a.voice.stop();
});

test('handsfree restart accepts new final result at reset browser index zero',async()=>{
 const a=setup();await a.voice.start({consent:true,mode:'handsfree',wakePhrase:'Hey Raphael'});
 const r=a.instances[0],result=transcript=>Object.assign([{transcript}],{isFinal:true});
 r.onresult({resultIndex:0,results:[result('First synthetic dictation')]});
 r.onend();
 assert.equal(r.started,true);
 r.onresult({resultIndex:0,results:[result('Second synthetic dictation')]});
 assert.deepEqual(a.texts,['First synthetic dictation','Second synthetic dictation']);
 a.voice.stop();
});

test('active voice watchdog stops vendor capture when consent is revoked remotely',async()=>{
 const a=setup();
 await a.voice.start({consent:true,mode:'standard',wakePhrase:'Raphael',verifyConsent:async()=>false});
 assert.equal(a.voice.active,true);
 await a.watchdog();
 assert.equal(a.voice.active,false);
 assert.equal(a.instances[0].stopped,true);
 assert.match(a.statuses.at(-1),/consent or encounter authorization could not be verified/);
});
test('active voice watchdog retains listening only after affirmative authorization',async()=>{
 const a=setup();
 await a.voice.start({consent:true,mode:'standard',wakePhrase:'Raphael',verifyConsent:async()=>true});
 await a.watchdog();
 assert.equal(a.voice.active,true);
 a.voice.stop();
 assert.equal(a.watchdog(),null);
});
test('active voice watchdog fails closed on network exception',async()=>{
 const a=setup();
 await a.voice.start({consent:true,mode:'standard',wakePhrase:'Raphael',verifyConsent:async()=>{throw Error('offline');}});
 await a.watchdog();
 assert.equal(a.voice.active,false);
});
test('stale watchdog response cannot interfere with newer listening session',async()=>{
 const a=setup();
 let settle;
 await a.voice.start({consent:true,mode:'standard',wakePhrase:'Raphael',
   verifyConsent:()=>new Promise(resolve=>{settle=resolve;})});
 const earlier=a.watchdog();
 a.voice.stop();
 await a.voice.start({consent:true,mode:'standard',wakePhrase:'Raphael',verifyConsent:async()=>true});
 settle(false); await earlier;
 assert.equal(a.voice.active,true);
 assert.equal(a.instances[1].stopped,undefined);
 a.voice.stop();
});

test('stalled remote consent verification has bounded failure time and stops microphone',async()=>{
 const a=setup();
 await a.voice.start({consent:true,mode:'standard',wakePhrase:'Raphael',
   verifyConsent:()=>new Promise(()=>{})});
 const checking=a.watchdog();
 // Exercise only the 5-second authorization deadline callback in the
 // deterministic fake timer; no network, audio, or wall-clock sleeps.
 a.timeout();
 await checking;
 assert.equal(a.voice.active,false);
 assert.equal(a.instances[0].stopped,true);
 assert.match(a.statuses.at(-1),/could not be verified/);
});
