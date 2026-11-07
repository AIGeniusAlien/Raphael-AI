(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let epoch = 0, timer = null, pending = null;
  const element = (tag, text) => { const item = document.createElement(tag); if (text !== undefined) item.textContent = String(text); return item; };
  const status = (message = '') => { $('status').textContent = message; };
  function stop() { epoch++; clearTimeout(timer); timer = null; $('workspace').hidden = true; $('jobs').replaceChildren(); $('worker-status').textContent = ''; $('counts').textContent = ''; pending = null; }
  async function api(path, body) {
    const response = await fetch('/api/dev' + path, {credentials:'same-origin',headers:{'Content-Type':'application/json'},...(body === undefined ? {} : {method:'POST',body:JSON.stringify(body)})});
    const data = await response.json();
    if (!response.ok) { if (response.status === 401 || response.status === 403) stop(); throw new Error(typeof data.detail === 'string' ? data.detail : 'Request refused. Check the fields and permissions.'); }
    return data;
  }
  async function action(fn) { try { return await fn(); } catch (error) { status(error.message || 'Request failed.'); } }
  function date(seconds) { return new Date(seconds * 1000).toLocaleString(); }
  async function load() {
    const expected = epoch;
    const [data, health] = await Promise.all([api('/jobs'),api('/jobs/status')]);
    if (epoch !== expected) return;
    $('worker-status').textContent = health.fresh_worker_heartbeat ? 'A recent bounded-worker heartbeat was observed. This does not establish continuous availability.' : 'No recent active worker heartbeat. Jobs may remain queued until an operator starts a worker.';
    $('counts').textContent = `My jobs: ${JSON.stringify(health.own_jobs_by_state)}. Permanent scheduler configured: no.`;
    $('jobs').replaceChildren();
    if (!data.items.length) $('jobs').appendChild(element('p','No retained jobs.'));
    for (const job of data.items) {
      const card = element('article'); card.dataset.jobId = String(job.id);
      card.appendChild(element('h3',`Job ${job.id}: ${job.kind}`));
      card.appendChild(element('p',`${job.state} · attempts ${job.attempts}/${job.max_attempts} · next eligible ${date(job.available_at)}`));
      if (job.last_error_code) card.appendChild(element('p',`Last safe error category: ${job.last_error_code}`));
      if (job.result) card.appendChild(element('pre',JSON.stringify(job.result,null,2)));
      const history = element('details'); history.appendChild(element('summary','Retained job history'));
      const entries = element('ol'); for (const event of job.events) entries.appendChild(element('li',`${date(event.created_at)} · ${event.event} · attempt ${event.attempt}`)); history.appendChild(entries); card.appendChild(history);
      if (['queued','retry_wait','running'].includes(job.state)) {
        const button = element('button','Cancel this job'); button.type='button';
        button.addEventListener('click',() => action(async() => { button.disabled=true; try { await api(`/jobs/${job.id}/cancel`,{}); if(epoch!==expected)return; await load(); status('Job cancelled. A stale worker cannot publish a result.'); } finally {button.disabled=false;} })); card.appendChild(button);
      }
      $('jobs').appendChild(card);
    }
  }
  function schedule() { clearTimeout(timer); if ($('auto-refresh').checked && !$('workspace').hidden) timer=setTimeout(() => action(load).finally(schedule),5000); }
  $('refresh').addEventListener('click',() => action(load));
  $('auto-refresh').addEventListener('change',schedule);
  $('job-form').addEventListener('submit', event => {
    event.preventDefault(); const expected=epoch;
    if (!pending) pending={request_id:crypto.randomUUID(),kind:$('kind').value,delay_seconds:Number($('delay').value),priority:Number($('priority').value)};
    $('enqueue').disabled=true;
    action(async() => { const job=await api('/jobs',pending); if(epoch!==expected)return; pending=null; await load(); status(`Job ${job.id} ${job.deduplicated ? 'already retained; no duplicate added' : 'queued'}. Execution requires a separate worker.`); }).finally(() => {$('enqueue').disabled=false;});
  });
  action(async() => {
    const user=await api('/session'); $('identity').textContent=`${user.username} · ${user.role} · ${user.tenant}`;
    const health=await api('/jobs/status'); $('kind').replaceChildren();
    for(const kind of health.supported_kinds){const option=element('option',kind.replaceAll('_',' ')); option.value=kind; $('kind').appendChild(option);}
    $('workspace').hidden=false; await load();
  });
  window.addEventListener('pagehide',() => clearTimeout(timer));
})();
