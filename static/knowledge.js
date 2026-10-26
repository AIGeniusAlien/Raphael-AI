(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let generation = 0;
  let sourceGeneration = 0;
  const element = (tag, value, className) => {
    const item = document.createElement(tag);
    if (value !== undefined) item.textContent = String(value);
    if (className) item.className = className;
    return item;
  };
  const status = (message = '', error = false) => {
    $('status').textContent = message;
    $('status').setAttribute('data-error', String(error));
  };
  async function api(path, body) {
    const response = await fetch('/api/dev' + path, {
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      ...(body === undefined ? {} : { method: 'POST', body: JSON.stringify(body) }),
    });
    const data = await response.json();
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) {
        generation++;
        $('workspace').hidden = true;
        $('sources').replaceChildren(); $('results').replaceChildren();
        $('citation').replaceChildren(); $('source-detail').replaceChildren();
      }
      throw new Error(typeof data.detail === 'string' ? data.detail : 'Request failed. Check the fields and source permissions.');
    }
    return data;
  }
  async function action(fn) {
    status();
    try { return await fn(); } catch (error) { status(error.message || 'Request failed.', true); }
  }
  function sourceLink(url) {
    const link = element('a', url);
    // Server validates public HTTPS URLs. Validate protocol again before assigning.
    const parsed = new URL(url);
    if (parsed.protocol !== 'https:') throw new Error('Unsupported source URL.');
    link.href = parsed.href;
    link.target = '_blank'; link.rel = 'noopener noreferrer';
    return link;
  }
  async function coverage() {
    const expected = generation;
    const data = await api('/knowledge/status');
    if (generation !== expected) return;
    $('coverage').textContent = `${data.approved_chunks} approved chunks. Source versions: ${JSON.stringify(data.source_versions_by_status)}. ${data.coverage}`;
  }
  async function loadSources() {
    const expected = generation;
    const items = await api('/knowledge/sources');
    if (generation !== expected) return;
    $('sources').replaceChildren();
    if (!items.length) $('sources').appendChild(element('p', 'No supplied sources. Nothing is indexed until intake and review.'));
    for (const source of items) {
      const row = element('article');
      row.appendChild(element('h3', source.title));
      row.appendChild(element('p', `Version ${source.version} · ${source.status} · ${source.source_date || 'publication date unknown'}`, 'meta'));
      const review = element('button', 'Read excerpt and review');
      review.type = 'button';
      review.addEventListener('click', () => action(() => showSource(source.id)));
      row.appendChild(review);
      $('sources').appendChild(row);
    }
    await coverage();
  }
  async function showSource(id) {
    const expected = ++sourceGeneration;
    const session = generation;
    const source = await api(`/knowledge/sources/${id}`);
    if (sourceGeneration !== expected || generation !== session) return;
    const holder = $('source-detail'); holder.replaceChildren();
    holder.appendChild(element('h3', `${source.title} — version ${source.version}`));
    holder.appendChild(sourceLink(source.url));
    holder.appendChild(element('p', `${source.status} · ${source.org} · ${source.source_type} · supplied tier ${source.tier}`));
    holder.appendChild(element('p', `Stored excerpt SHA-256: ${source.content_sha256}`, 'meta'));
    holder.appendChild(element('p', `Version SHA-256: ${source.version_sha256}`, 'meta'));
    holder.appendChild(element('p', `Permission declaration: ${source.license_declaration}`));
    holder.appendChild(element('p', source.verification_notice, 'warning'));
    holder.appendChild(element('pre', source.content));
    const label = element('label', undefined, 'checkbox');
    const checked = document.createElement('input'); checked.type = 'checkbox';
    label.appendChild(checked);
    label.appendChild(element('span', 'I reviewed this supplied excerpt and its permission declaration. Approval is a clinician attestation, not independent verification.'));
    holder.appendChild(label);
    const approve = element('button', 'Approve this version for local retrieval'); approve.type = 'button';
    approve.disabled = true;
    checked.addEventListener('change', () => { approve.disabled = !checked.checked; });
    approve.addEventListener('click', () => action(async () => {
      if (sourceGeneration !== expected || generation !== session || !checked.checked) return;
      await api(`/knowledge/sources/${id}/approve`, {});
      if (generation !== session || sourceGeneration !== expected) return;
      await loadSources(); await showSource(id);
      status('Version explicitly reviewed for local retrieval. No legal or clinical certification is implied.');
    }));
    holder.appendChild(approve);
    if (source.status === 'approved') {
      const revoke = element('button', 'Revoke this version from retrieval'); revoke.type = 'button';
      revoke.addEventListener('click', () => action(async () => {
        if (sourceGeneration !== expected || generation !== session) return;
        await api(`/knowledge/sources/${id}/revoke`, {});
        if (generation !== session || sourceGeneration !== expected) return;
        $('results').replaceChildren(); $('citation').replaceChildren();
        await loadSources(); await showSource(id);
        status('Source revoked from search and citation resolution. Its stored history is preserved.');
      }));
      holder.appendChild(revoke);
    }
  }
  async function search() {
    const expected = generation;
    const data = await api('/knowledge/search', { query: $('query').value.trim() });
    if (generation !== expected) return;
    $('results').replaceChildren(); $('citation').replaceChildren();
    if (!data.results.length) $('results').appendChild(element('p', 'No approved excerpt matched. No answer or citation is invented.'));
    for (const hit of data.results) {
      const section = element('section');
      section.appendChild(element('h3', hit.title));
      section.appendChild(sourceLink(hit.url));
      section.appendChild(element('p', `Version ${hit.source_version} · ${hit.reviewed_by} · lexical query coverage ${Math.round(hit.score * 100)}% · ${hit.freshness_status}`, 'meta'));
      section.appendChild(element('pre', hit.text));
      const resolve = element('button', 'Resolve stored citation and checksum'); resolve.type = 'button';
      resolve.addEventListener('click', () => action(async () => {
        const citation = await api(`/knowledge/citations/${hit.chunk_id}`);
        if (generation !== expected) return;
        $('citation').replaceChildren(element('h3', 'Resolved stored excerpt'), sourceLink(citation.url),
          element('p', `Version ${citation.version}; reviewed by ${citation.reviewed_by}; SHA-256 ${citation.content_sha256}`, 'meta'),
          element('pre', citation.body));
      }));
      section.appendChild(resolve); $('results').appendChild(section);
    }
    status(data.notice);
  }
  async function intake() {
    const expected = generation;
    const source = await api('/knowledge/sources', {
      title: $('title').value.trim(), url: $('url').value.trim(), org: $('org').value.trim(),
      jurisdiction: $('jurisdiction').value.trim(), source_type: $('source-type').value.trim(),
      tier: $('tier').value, evidence_strength: $('strength').value || null,
      source_date: $('source-date').value || null, license_declaration: $('license').value.trim(),
      content: $('content').value, permission_attested: $('permission').checked,
    });
    if (generation !== expected) return;
    $('source-form').reset();
    await loadSources(); await showSource(source.id);
    status(source.deduplicated ? 'Identical immutable version already exists; no duplicate was created.' : 'Pending version saved. Read the excerpt and explicitly review it before retrieval.');
  }
  function bindForm(id, fn) {
    $(id).addEventListener('submit', event => {
      event.preventDefault();
      const buttons = Array.from($(id).querySelectorAll('button'));
      buttons.forEach(button => { button.disabled = true; });
      action(fn).finally(() => buttons.forEach(button => { button.disabled = false; }));
    });
  }
  async function boot() {
    bindForm('source-form', intake); bindForm('search-form', search);
    $('refresh').addEventListener('click', () => action(loadSources));
    const user = await api('/session');
    $('account').textContent = `${user.username} · ${user.tenant} · ${user.role}`;
    if (user.role !== 'clinician') throw new Error('This reviewed evidence workspace requires a clinician role.');
    $('workspace').hidden = false;
    await loadSources();
  }
  action(boot);
})();
