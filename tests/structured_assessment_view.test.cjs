// Clinical-review presentation contract: synthetic input, DOM text nodes only.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

const source=fs.readFileSync('static/intelligence.js','utf8');
const start=source.indexOf('  function renderStructuredAssessment(assessment) {');
const end=source.indexOf('  function renderConversation(thread) {',start);
assert.ok(start>=0&&end>start,'Clinician differential renderer should remain present');
const excerpt=source.slice(start,end);
function element(tag,text=''){
 return {tag,textContent:text,className:'',children:[],replaceChildren(...children){this.children=children;},append(...children){this.children.push(...children);}};
}
function render(assessment){
 const holder=element('div'),context=vm.createContext({
  $:id=>{assert.equal(id,'structured-assessment');return holder;},
  node:(tag,text)=>element(tag,text),
  JSON,String,Array
 });
 vm.runInContext(excerpt+'\nrenderStructuredAssessment',context)(assessment);
 return holder;
}
function walk(element,parts=[]){
 parts.push(String(element.textContent||''));
 for(const child of element.children||[])walk(child,parts);
 return parts.join(' ');
}
test('structured assessment visibly separates rank, contraindications, red flags, and missing data',()=>{
 const holder=render({
  problem_representation:'Synthetic abdominal concern',
  ranked_differential:[{
   diagnosis:'SYNTHETIC-A',likelihood:'Moderate',
   supports:['Synthetic observation'],argues_against:['Unconfirmed historical feature'],
   citations:['fixture-id']
  }],
  working_diagnosis:null,
  rule_in_out_plan:['Physician review first'],cant_miss:['Synthetic critical differential'],
  red_flags_escalation:['Escalation must be reviewed'],missing_data:['Allergies unknown'],
  confidence:'Not calibrated',evidence_used:[{chunk_id:'fixture-id'}],
  safety_notes:['No real clinical evidence']
 });
 const words=walk(holder);
 for(const item of ['1. SYNTHETIC-A','not a calibrated probability','Unverified citation IDs: fixture-id',
  'Missing clinical information','Allergies unknown','Potential must-not-miss conditions',
  'Synthetic critical differential','Not established — clinician review required.']){
  assert.ok(words.includes(item),item);
 }
});
test('untrusted assessment text is never inserted as executable markup',()=>{
 const injected='<img src=x onerror=alert(1)>';
 const holder=render({
  problem_representation:injected,
  ranked_differential:[{diagnosis:injected,likelihood:'Low',supports:[],argues_against:[],citations:[]}],
  missing_data:[],safety_notes:[],evidence_used:[]
 });
 assert.ok(walk(holder).includes(injected));
 assert.ok(!excerpt.includes('.innerHTML'));
});
