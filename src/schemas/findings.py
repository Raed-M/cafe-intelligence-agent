from __future__ import annotations

from typing import Any, Literal, TypedDict

from src.schemas.artifacts import ArtifactRef


class MetricEvidence(TypedDict):
    metric_name: str
    value: int | float | str | None
    unit: str | None
    numerator: int | float | None
    denominator: int | float | None
    period_start: str
    period_end: str
    comparison_period_start: str | None
    comparison_period_end: str | None
    source_names: list[str]
    result_path: str
    result_key: str


class AnalystFinding(TypedDict):
    finding_id: str
    analyst_name: str
    title: str
    claim: str
    finding_type: str
    evidence: list[MetricEvidence]
    source_names: list[str]
    code_artifact: ArtifactRef
    result_artifact: ArtifactRef
    sample_size: int | None
    coverage_notes: list[str]
    assumptions: list[str]
    confidence: float
    business_impact_score: float
    actionability_score: float
    revision_count: int
    execution_metadata: dict[str, Any]


class RevisionRequest(TypedDict):
    finding_id: str
    analyst_name: str
    reason_code: str
    explanation: str
    required_fix: str


class SemanticReviewResult(TypedDict):
    decision: Literal["approve", "revise", "reject", "insufficient_evidence"]
    explanation: str
    required_fix: str


class CriticOutput(TypedDict):
    approved_findings: list[str]
    rejected_findings: list[str]
    revision_requests: list[RevisionRequest]
    removed_after_cap: list[str]
    total_rejections: int
    notes: list[str]


class CrossDomainFindingDraft(TypedDict):
    """One synthesis proposal. `metric_refs` are pool keys (analyst__result_key)
    from the supplied evidence pool, and every number in the free-text fields
    must be written as a `<<pool_key>>` placeholder rather than a literal --
    src/graph/cross_domain_nodes.py substitutes the real values afterwards, so
    a synthesized claim cannot state a figure no analyst actually computed."""

    title: str
    claim: str
    finding_type: str
    metric_refs: list[str]
    assumptions: list[str]
    coverage_notes: list[str]
    confidence: float


class CrossDomainSynthesisOutput(TypedDict):
    items: list[CrossDomainFindingDraft]
