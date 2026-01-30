import json
from typing import List
from rag.schemas import OutputV1, EvidenceChunk
from rag.retriever import retrieve
from rag.safety_gate import safety_check
from rag.prompts import SYSTEM_PROMPT, build_user_prompt, CRITIC_SYSTEM, build_critic_prompt
from rag.citation_guard import enforce_citations
from rag.llm import OpenAICompatibleLLM

def _format_evidence(chunks: List[EvidenceChunk], max_chars: int) -> str:
    out, total = [], 0
    for c in chunks:
        block = (
            f"[chunk_id={c.chunk_id} | tier={c.tier} | title={c.title} | section={c.section} | date={c.source_date}]\n"
            f"{c.text}\n"
        )
        if total + len(block) > max_chars:
            break
        out.append(block)
        total += len(block)
    return "\n---\n".join(out)

def _fallback(chunks: List[EvidenceChunk], notes: List[str]) -> OutputV1:
    return OutputV1(
        problem_representation="Insufficient evidence retrieved to safely generate a diagnostic output.",
        ranked_differential=[],
        working_diagnosis=None,
        rule_in_out_plan=[
            "Clarify chief complaint, onset, duration, severity, and relevant risk factors.",
            "Collect vitals, focused exam findings, and key labs/imaging based on setting.",
            "Use local protocols; escalate to in-person evaluation if red flags are present."
        ],
        cant_miss=[],
        red_flags_escalation=["If red-flag symptoms/signs are present, seek urgent/emergent evaluation."],
        missing_data=["Higher-quality evidence sources and more case context are required."],
        confidence="Low (retrieval insufficient).",
        evidence_used=chunks,
        safety_notes=notes
    )

def generate(*, case_context: str, question: str, mode: str,
             qdrant_url: str, collection: str, embed_model: str, embed_dim: int,
             top_k: int, min_sources: int, min_avg_sim: float, max_context_chars: int,
             llm_base_url: str, llm_api_key: str, llm_model: str) -> OutputV1:

    chunks = retrieve(
        question,
        qdrant_url=qdrant_url,
        collection=collection,
        embed_model=embed_model,
        embed_dim=embed_dim,
        top_k=top_k
    )

    ok, safety_notes = safety_check(chunks, min_sources=min_sources, min_avg_sim=min_avg_sim)
    if not ok:
        return _fallback(chunks, safety_notes)

    evidence_block = _format_evidence(chunks, max_context_chars)
    llm = OpenAICompatibleLLM(llm_base_url, llm_api_key, llm_model)

    # Pass 1: draft
    user_prompt = build_user_prompt(mode, case_context, question, evidence_block)
    raw = llm.complete(SYSTEM_PROMPT, user_prompt)
    try:
        draft = json.loads(raw)
    except json.JSONDecodeError:
        return _fallback(chunks, ["Model did not return valid JSON (draft)."] + safety_notes)

    # Pass 2: critic
    critic_prompt = build_critic_prompt(json.dumps(draft), evidence_block)
    raw_c = llm.complete(CRITIC_SYSTEM, critic_prompt)
    try:
        critique = json.loads(raw_c)
    except json.JSONDecodeError:
        critique = {"ok": False, "issues": ["Critic JSON parse failed"], "required_fixes": ["Regenerate with stricter safety + citations."]}

    if not critique.get("ok", False):
        fixes = "\n".join(f"- {x}" for x in critique.get("required_fixes", []))
        regen_q = question + "\n\nSAFETY_FIXES_REQUIRED:\n" + fixes
        raw2 = llm.complete(SYSTEM_PROMPT, build_user_prompt(mode, case_context, regen_q, evidence_block))
        try:
            draft = json.loads(raw2)
        except json.JSONDecodeError:
            return _fallback(chunks, ["Model did not return valid JSON (regen)."] + safety_notes + critique.get("issues", []))

    enforce_citations(draft)

    out = OutputV1(**draft)
    out.evidence_used = chunks
    out.safety_notes = list(dict.fromkeys(out.safety_notes + safety_notes + critique.get("issues", [])))
    return out
