let state = {
  env: null,
  patients: [],
  currentPatient: null,
  conversations: [],
  currentConversation: null,
  messages: [],
};

async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || r.statusText);
  }
  return r.headers.get("content-type")?.includes("application/json") ? r.json() : r.text();
}

function el(html) {
  const d = document.createElement("div");
  d.innerHTML = html.trim();
  return d.firstChild;
}

function renderPatients() {
  const holder = document.getElementById("patient-list");
  holder.innerHTML = "";
  state.patients.forEach(p => {
    const n = el(`<div class="row ${state.currentPatient?.id===p.id?'active':''}">${p.name}</div>`);
    n.onclick = async () => {
      state.currentPatient = p;
      await loadConversations();
      renderPatients();
    };
    holder.appendChild(n);
  });
}

function renderConversations() {
  const holder = document.getElementById("conv-list");
  holder.innerHTML = "";
  state.conversations.forEach(c => {
    const n = el(`<div class="row ${state.currentConversation?.id===c.id?'active':''}">${c.title}</div>`);
    n.onclick = async () => {
      state.currentConversation = c;
      await loadMessages();
      renderConversations();
    };
    holder.appendChild(n);
  });
}

function renderMessages() {
  const list = document.getElementById("messages");
  list.innerHTML = "";
  document.getElementById("center-title").style.display = state.messages.length ? "none" : "flex";
  state.messages.forEach(m => {
    const bubble = el(`
      <div class="msg ${m.role}">
        <div class="role">${m.role === 'assistant' ? 'Raphael' : 'You'}</div>
        <div class="content"></div>
      </div>`);
    bubble.querySelector(".content").textContent = m.content;
    list.appendChild(bubble);
  });
  list.scrollTop = list.scrollHeight;
}

async function boot() {
  state.env = await api("/api/env");
  await loadPatients();
  bindUI();
}

async function loadPatients() {
  state.patients = await api("/api/patients");
  if (!state.currentPatient && state.patients.length) state.currentPatient = state.patients[0];
  renderPatients();
  if (state.currentPatient) await loadConversations();
}

async function loadConversations() {
  state.conversations = await api(`/api/conversations?patient_id=${state.currentPatient.id}`);
  if (!state.currentConversation && state.conversations.length) state.currentConversation = state.conversations[0];
  renderConversations();
  if (state.currentConversation) await loadMessages();
  else {
    state.messages = [];
    renderMessages();
  }
}

async function loadMessages() {
  state.messages = await api(`/api/messages?conversation_id=${state.currentConversation.id}`);
  renderMessages();
}

async function send() {
  const ta = document.getElementById("prompt");
  const text = ta.value.trim();
  if (!text || !state.currentConversation) return;
  ta.value = "";
  // optimistic render
  state.messages.push({ role: "user", content: text });
  renderMessages();
  try {
    const res = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        conversation_id: state.currentConversation.id,
        text,
        patient: state.currentPatient || {},
      }),
    });
    state.messages.push({ role: "assistant", content: res.reply });
    renderMessages();
  } catch (e) {
    state.messages.push({ role: "assistant", content: `⚠️ ${e.message}` });
    renderMessages();
  }
}

function bindUI() {
  document.getElementById("add-patient").onclick = async () => {
    const name = prompt("Patient name");
    if (!name) return;
    await api("/api/patients", { method: "POST", body: JSON.stringify({ name }) });
    await loadPatients();
  };

  document.getElementById("add-conv").onclick = async () => {
    if (!state.currentPatient) return alert("Select a patient first.");
    const title = prompt("Conversation title") || "New Chat";
    await api("/api/conversations", {
      method: "POST",
      body: JSON.stringify({ patient_id: state.currentPatient.id, title }),
    });
    await loadConversations();
  };

  document.getElementById("send").onclick = send;
  document.getElementById("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  // doc upload
  const form = document.getElementById("doc-upload");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    try {
      const r = await fetch("/api/docs/upload", { method: "POST", body: fd });
      if (!r.ok) throw new Error(await r.text());
      alert("Uploaded");
      form.reset();
    } catch (err) {
      alert(err.message);
    }
  });

  // doc search
  const search = document.getElementById("doc-search");
  const results = document.getElementById("doc-results");
  let timer = null;
  search.addEventListener("input", () => {
    clearTimeout(timer);
    const q = search.value.trim();
    if (!q) { results.innerHTML = ""; return; }
    timer = setTimeout(async () => {
      const hits = await api(`/api/docs/search?q=${encodeURIComponent(q)}`);
      results.innerHTML = hits.map(h =>
        `<div class="hit"><div class="h-title">${h.title}</div><div class="h-snippet">${h.snippet}</div></div>`
      ).join("");
    }, 250);
  });
}

boot();
