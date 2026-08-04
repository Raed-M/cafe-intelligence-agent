"""Merge and rank critic-approved findings, capping the final set at
`max_final_findings` per plan section 11.7 / ADR-010."""
from __future__ import annotations

from src.schemas.findings import AnalystFinding


def _score(finding: AnalystFinding, recent_titles: set[str]) -> float:
    evidence_quality = min(len(finding.get("evidence", [])) / 3.0, 1.0)
    confidence = finding.get("confidence", 0.5)
    impact = finding.get("business_impact_score", 0.5)
    multi_source = 1.0 if len(finding.get("source_names", [])) >= 2 else 0.5
    novelty = 0.5 if finding.get("title", "").strip().lower() in recent_titles else 1.0
    actionability = finding.get("actionability_score", 0.5)
    return (
        0.25 * evidence_quality + 0.2 * confidence + 0.25 * impact
        + 0.1 * multi_source + 0.1 * novelty + 0.1 * actionability
    )


def rank_findings(
    approved_findings: list[AnalystFinding],
    max_final_findings: int,
    recent_finding_titles: list[str] | None = None,
) -> list[AnalystFinding]:
    recent = {t.strip().lower() for t in (recent_finding_titles or [])}
    scored = sorted(approved_findings, key=lambda f: _score(f, recent), reverse=True)
    return scored[:max_final_findings]
