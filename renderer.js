// renderer.js — UI logic for Raphael Electron app
// Requires preload.js exposing window.raphael

function $(sel) {
  return document.querySelector(sel);
}

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderRaphaelOutput(out) {
  const diff = out.ranked_differential || [];
  const evidence = out.evidence_used || [];
  const notes = out.safety_notes || [];

  const diffHtml = diff.length
    ? diff
        .map(
          (d) => `
      <div class="r-card">
        <div class="r-card-title">
          ${escapeHtml(d.diagnosis)}
          <span class="r-pill">${escapeHtml(d.likelihood)}</span>
        </div>
        <div class="r-line"><b>Supports:</b> ${escapeHtml((d.supports || []).join(" | "))}</div>
        <div class="r-line"><b>Argues against:</b> ${escapeHtml((d.argues_against || []).join(" | "))}</div>
        <div class="r-line"><b>Citations:</b> <span class="mono">${escapeHtml((d.citations || []).join(", "))}</span></div>
      </div>
    `
        )
        .join("")
    : `<div class="muted">No differential returned (insufficient evidence or refused output).</div>`;

  const evidenceHtml = evidence.length
    ? evidence
        .map(
          (e) => `
      <div class="r-evidence">
        <div class="r-evidence-top">
          <b>${escapeHtml(e.title)}</b>
          <span class="r-pill">${escapeHtml(e.tier)}</span>
        </div>
        <div class="muted mono">chunk_id: ${escapeHtml(e.chunk_id)} | score: ${Number(e.score).toFixed(
            3
          )} | section: ${escapeHtml(e.section || "")}</div>
        ${e.url ? `<div class="muted mono">url: ${escapeHtml(e.url)}</div>` : ""}
        <details>
          <summary>Show chunk text</summary>
          <div class="mono r-chunk">${escapeHtml(e.text || "")}</div>
        </details>
      </div>
    `
        )
        .join("")
    : `<div class="muted">No evidence retrieved.</div>`;

  const list = (arr) =>
    (arr || []).length
      ? `<ul>${arr.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>`
      : `<div class="muted">—</div>`;

  return `
    <div class="r-wrap">
      <div class="r-section">
        <h3>Problem representation</h3>
        <div>${escapeHtml(out.problem_representation || "")}</div>
      </div>

      <div class="r-section">
        <h3>Working diagnosis</h3>
        <div>${escapeHtml(out.working_diagnosis || "—")}</div>
      </div>

      <div class="r-section">
        <h3>Ranked differential</h3>
        ${diffHtml}
      </div>

      <div class="r-section">
        <h3>Rule-in / Rule-out plan</h3>
        ${list(out.rule_in_out_plan)}
      </div>

      <div class="r-section">
        <h3>Can’t-miss diagnoses</h3>
        ${list(out.cant_miss)}
      </div>

      <div class="r-section">
        <h3>Red flags & escalation</h3>
        ${list(out.red_flags_escalation)}
      </div>

      <div class="r-section">
        <h3>Missing data</h3>
        ${list(out.missing_data)}
      </div>

      <div class="r-section">
        <h3>Confidence</h3>
        <div>${escapeHtml(out.confidence || "")}</div>
      </div>

      <div class="r-section">
        <h3>Safety notes</h3>
        ${list(notes)}
      </div>

      <div class="r-section">
        <h3>Evidence used</h3>
        ${evidenceHtml}
      </div>
    </div>
  `;
}

// ---------- API calls ----------
async function raphaelQuery({ conversation_id, question, mode = "radiology" }) {
  return await window.raphael.postJson("/api/raphael/query", {
    conversation_id,
    question,
    mode,
  });
}

async function ingestEvidence({ title, text, tier = "TIER_1", source_date = null, url = null, section = null }) {
  return await window.raphael.postJson("/api/evidence/ingest", {
    title,
    text,
    tier,
    source_date,
    url,
    section,
  });
}

// ---------- Wire to your UI ----------
// This assumes your HTML has these elements. If your IDs differ, change them here.
document.addEventListener("DOMContentLoaded", () => {
  // Required UI elements (adjust IDs if needed)
  const convIdEl = $("#conversationId");
  const questionEl = $("#question");
  const outputEl = $("#assistantPane");
  const runBtn = $("#runRaphael");

  // Evidence ingest elements (optional, but recommended)
  const evTitle = $("#evTitle");
  const evText = $("#evText");
  const evBtn = $("#evIngestBtn");
  const evStatus = $("#evStatus");

  if (runBtn) {
    runBtn.addEventListener("click", async () => {
      try {
        outputEl.innerHTML = `<div class="muted">Running Raphael…</div>`;
        const out = await raphaelQuery({
          conversation_id: convIdEl.value.trim(),
          question: questionEl.value.trim(),
          mode: "radiology",
        });
        outputEl.innerHTML = renderRaphaelOutput(out);
      } catch (e) {
        outputEl.innerHTML = `<div class="r-error">Error: ${escapeHtml(e.message || String(e))}</div>`;
      }
    });
  }

  if (evBtn) {
    evBtn.addEventListener("click", async () => {
      try {
        evStatus.textContent = "Indexing evidence…";
        const res = await ingestEvidence({
          title: (evTitle?.value || "").trim(),
          text: (evText?.value || "").trim(),
          tier: "TIER_1",
        });
        evStatus.textContent = `Indexed chunks: ${res.chunks_indexed}`;
      } catch (e) {
        evStatus.textContent = `Error: ${e.message || String(e)}`;
      }
    });
  }
});
