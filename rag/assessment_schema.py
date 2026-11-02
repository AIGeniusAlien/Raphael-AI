"""Runtime contract for the active main.py assessment pipeline.

This validates structure; it is not clinical validation or source verification.
Legacy rag.schemas retains its separate historical evidence-tier contract.
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Differential(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    diagnosis: str = Field(min_length=1)
    likelihood: Literal["High", "Moderate", "Low"]
    supports: list[str]
    argues_against: list[str]
    citations: list[str] = Field(min_length=1)


class Assessment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    problem_representation: str = Field(min_length=1)
    ranked_differential: list[Differential]
    working_diagnosis: str | None
    rule_in_out_plan: list[str]
    cant_miss: list[str]
    red_flags_escalation: list[str]
    missing_data: list[str]
    confidence: str = Field(min_length=1)
    evidence_used: list[dict[str, Any]]
    safety_notes: list[str]


class Critique(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    ok: bool
    issues: list[str] = Field(default_factory=list)
    required_fixes: list[str] = Field(default_factory=list)
