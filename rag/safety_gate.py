from typing import List, Tuple
from rag.schemas import EvidenceChunk

def safety_check(chunks: List[EvidenceChunk], *, min_sources: int, min_avg_sim: float) -> Tuple[bool, list]:
    notes: list[str] = []

    if len(chunks) < min_sources:
        notes.append(f"Insufficient sources retrieved: {len(chunks)} < {min_sources}.")
        return False, notes

    avg = sum(c.score for c in chunks) / max(1, len(chunks))
    if avg < min_avg_sim:
        notes.append(f"Low retrieval similarity: avg={avg:.3f} < {min_avg_sim}.")
        return False, notes

    tiers = {c.tier for c in chunks}
    if "TIER_1" not in tiers:
        notes.append("No Tier 1 evidence retrieved; output must be conservative and low-confidence.")

    return True, notes
