from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

Tier = Literal["TIER_1", "TIER_2", "TIER_3"]

class EvidenceDocIn(BaseModel):
    title: str
    tier: Tier = "TIER_1"
    source_date: Optional[str] = None
    url: Optional[str] = None
    section: Optional[str] = None
    text: str

class EvidenceChunk(BaseModel):
    chunk_id: str
    title: str
    tier: Tier
    source_date: Optional[str] = None
    url: Optional[str] = None
    section: Optional[str] = None
    text: str
    score: float

class CaseArtifactIn(BaseModel):
    kind: Literal["note","radiology_report","labs","meds","voice_transcript","fhir_bundle"]
    text: str

class QueryIn(BaseModel):
    case_id: str
    question: str
    mode: Literal["radiology","general"] = "radiology"

class DifferentialItem(BaseModel):
    diagnosis: str
    likelihood: Literal["High","Moderate","Low"]
    supports: List[str]
    argues_against: List[str]
    citations: List[str]  # chunk_ids

class OutputV1(BaseModel):
    problem_representation: str
    ranked_differential: List[DifferentialItem]
    working_diagnosis: Optional[str]
    rule_in_out_plan: List[str]
    cant_miss: List[str]
    red_flags_escalation: List[str]
    missing_data: List[str]
    confidence: str
    evidence_used: List[EvidenceChunk]
    safety_notes: List[str]
