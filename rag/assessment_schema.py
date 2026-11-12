"""Bounded runtime structure for provisional synthetic clinician assessments.

Fields are constrained for availability and safe rendering, not clinical
accuracy, citation truth, patient applicability or regulatory approval.
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Differential(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    diagnosis: str = Field(min_length=1, max_length=240)
    likelihood: Literal["High", "Moderate", "Low"]
    supports: list[str] = Field(max_length=12)
    argues_against: list[str] = Field(max_length=12)
    citations: list[str] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def bounded_prose_and_refs(self):
        if any(not 1 <= len(item.strip()) <= 1000 for item in self.supports + self.argues_against):
            raise ValueError("Support and contra-indication notes must be bounded and nonempty")
        if any(not isinstance(ref, str) or not 1 <= len(ref) <= 128 for ref in self.citations):
            raise ValueError("Citations must be bounded source identifiers")
        if len(set(self.citations)) != len(self.citations):
            raise ValueError("Duplicate citation identifiers are not allowed")
        return self


class Assessment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    problem_representation: str = Field(min_length=1, max_length=3000)
    ranked_differential: list[Differential] = Field(max_length=10)
    working_diagnosis: str | None = Field(max_length=240)
    rule_in_out_plan: list[str] = Field(max_length=20)
    cant_miss: list[str] = Field(max_length=20)
    red_flags_escalation: list[str] = Field(max_length=20)
    missing_data: list[str] = Field(max_length=20)
    confidence: str = Field(min_length=1, max_length=1500)
    evidence_used: list[dict[str, Any]] = Field(max_length=12)
    safety_notes: list[str] = Field(max_length=20)

    @model_validator(mode="after")
    def coherent_bounded_review(self):
        if any(not 1 <= len(item.strip()) <= 1200 for group in (
            self.rule_in_out_plan, self.cant_miss, self.red_flags_escalation,
            self.missing_data, self.safety_notes,
        ) for item in group):
            raise ValueError("Assessment review text must be bounded and nonempty")
        diagnoses = [item.diagnosis.strip().casefold() for item in self.ranked_differential]
        if len(diagnoses) != len(set(diagnoses)):
            raise ValueError("Ranked differential repeats a diagnosis")
        # evidence_used is not a trusted source collection. Prevent arbitrary
        # nested objects or large unbounded provider output from being persisted
        # as a purportedly reviewed clinical source.
        import json
        if len(json.dumps(self.evidence_used, ensure_ascii=True)) > 24000:
            raise ValueError("Assessment evidence field exceeds bounded metadata budget")
        return self


class Critique(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    ok: bool
    issues: list[str] = Field(default_factory=list, max_length=20)
    required_fixes: list[str] = Field(default_factory=list, max_length=20)
