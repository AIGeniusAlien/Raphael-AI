import hashlib
from datetime import datetime

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def normalize_date(date_str: str | None) -> str | None:
    if not date_str:
        return None
    return date_str.strip()

def doc_id_for(url: str, title: str) -> str:
    return sha256(f"{url}|{title}")

def build_payload(*, chunk_id: str, doc_id: str, title: str, tier: str, source_type: str,
                  org: str, jurisdiction: str, source_date: str | None, url: str | None,
                  section: str | None, evidence_strength: str | None, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "title": title,
        "tier": tier,  # "A/B/C/D"
        "source_type": source_type,
        "org": org,
        "jurisdiction": jurisdiction,
        "source_date": normalize_date(source_date),
        "url": url,
        "section": section,
        "evidence_strength": evidence_strength or None,  # "high/moderate/low/very_low"
        "indexed_at": datetime.utcnow().isoformat(),
        "text": text,
    }
