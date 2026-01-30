"""
Public + legally ingestible sources only.

Tiers:
A = Canonical truth (guidelines, regulator labels, systematic reviews)
B = Point-of-care public equivalents (pathways, society summaries, high-quality reviews)
C = Primary literature (PubMed indexed)
D = Education layer (teaching frameworks, public teaching cases)

We do NOT include proprietary sources (e.g., UpToDate) unless licensed.
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Source:
    name: str
    tier: str                # "A" | "B" | "C" | "D"
    source_type: str         # guideline | regulator_label | safety_comm | systematic_review | pathway | review | teaching
    org: str
    jurisdiction: str
    url: str
    cadence: str             # monthly | weekly | nightly | manual
    notes: Optional[str] = None

# Minimal starter set (expand over time)
PUBLIC_SOURCES: List[Source] = [
    # Tier A: Canonical
    Source(
        name="USPSTF Recommendations (landing)",
        tier="A",
        source_type="guideline",
        org="USPSTF",
        jurisdiction="US",
        url="https://www.uspreventiveservicestaskforce.org/",
        cadence="monthly",
        notes="In v1, ingest manually or via URL-to-text. Full crawling is optional."
    ),
    Source(
        name="CDC Guidelines (landing)",
        tier="A",
        source_type="guideline",
        org="CDC",
        jurisdiction="US",
        url="https://www.cdc.gov/",
        cadence="monthly",
        notes="Use targeted URL ingestion for specific topics."
    ),
    Source(
        name="WHO (landing)",
        tier="A",
        source_type="guideline",
        org="WHO",
        jurisdiction="Global",
        url="https://www.who.int/",
        cadence="monthly",
        notes="Use targeted URL ingestion for specific guidelines."
    ),
    Source(
        name="NICE Guidance (landing)",
        tier="A",
        source_type="guideline",
        org="NICE",
        jurisdiction="UK",
        url="https://www.nice.org.uk/guidance",
        cadence="monthly",
        notes="Ingest specific NICE guideline pages or PDFs."
    ),

    # Tier B: Public point-of-care equivalents
    Source(
        name="ACR Appropriateness Criteria (landing)",
        tier="A",
        source_type="guideline",
        org="ACR",
        jurisdiction="US",
        url="https://www.acr.org/Clinical-Resources/ACR-Appropriateness-Criteria",
        cadence="monthly",
        notes="Some content may be gated; only ingest public pages/docs."
    ),
]

def list_sources() -> List[Source]:
    return PUBLIC_SOURCES
