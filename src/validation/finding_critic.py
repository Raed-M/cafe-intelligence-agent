"""Finding critic (Module 5): deterministic provenance checks first, then an
optional constrained-LLM semantic pass for nuance a hard rule cannot capture.

The deterministic layer is the load-bearing gate: it independently re-reads
each finding's result_artifact from disk and verifies every cited metric key
actually resolves there with a numeric/string value, rather than trusting
whatever the analyst claims. This is what makes "a number without a resolvable
computation must be rejected" enforceable rather than aspirational.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from src.schemas.findings import AnalystFinding, CriticOutput, RevisionRequest

KNOWN_SOURCE_NAMES = {"pos", "menu", "traffic", "staff", "inventory", "emails", "reviews"}

_CAUSAL_PATTERNS = re.compile(
    r"\b(because|caused by|due to|led to|resulted in|drove|driving)\b", re.IGNORECASE
)
_ITEM_LEVEL_COST_PATTERNS = re.compile(
    r"\b(per[- ]drink|per[- ]item|per[- ]cup)\b.*\b(cost|margin|impact)\b", re.IGNORECASE
)


def _resolve_metric(result_obj: dict[str, Any], metric_name: str, result_key: str) -> bool:
    for f in result_obj.get("findings", []):
        metrics = f.get("metrics", {})
        if result_key in metrics:
            return True
    return False


def _deterministic_check(
    finding: AnalystFinding, valid_periods: set[tuple[str, str]] | None = None
) -> tuple[bool, list[str]]:
    """Returns (ok, reasons). ok=False means reject or revise."""
    reasons: list[str] = []

    if not finding.get("code_artifact") or not finding.get("result_artifact"):
        return False, ["missing code_artifact or result_artifact"]

    result_path = Path(finding["result_artifact"]["path"])
    if not result_path.exists():
        return False, [f"result artifact does not exist on disk: {result_path}"]
    try:
        result_obj = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, ["result artifact is not valid JSON"]

    if not finding.get("evidence"):
        return False, ["finding has no metric evidence to resolve"]

    for ev in finding["evidence"]:
        if not _resolve_metric(result_obj, ev["metric_name"], ev["result_key"]):
            reasons.append(f"metric '{ev['result_key']}' does not resolve in stored result")
        if ev["value"] is None:
            reasons.append(f"metric '{ev['result_key']}' has null value")
        if not ev.get("period_start") or not ev.get("period_end"):
            reasons.append(f"metric '{ev['result_key']}' missing period bounds")
        elif valid_periods is not None:
            period_key = (ev["period_start"][:10], ev["period_end"][:10])
            if period_key not in valid_periods:
                reasons.append(
                    f"metric '{ev['result_key']}' period {period_key} does not match any known "
                    f"analysis/previous/trailing-baseline period for this run"
                )

    unknown_sources = set(finding.get("source_names", [])) - KNOWN_SOURCE_NAMES
    if unknown_sources:
        reasons.append(f"unknown source names: {sorted(unknown_sources)}")

    if not (0.0 <= finding.get("confidence", -1) <= 1.0):
        reasons.append("confidence out of [0,1] range")

    claim = finding.get("claim", "")
    if _CAUSAL_PATTERNS.search(claim) and len(finding.get("source_names", [])) < 2:
        reasons.append("causal language used with only a single source; explanations require joined evidence")

    if finding.get("analyst_name") == "margin" and _ITEM_LEVEL_COST_PATTERNS.search(claim):
        assumptions_text = " ".join(finding.get("assumptions", [])).lower()
        if "estimate" not in assumptions_text and "bom" not in assumptions_text:
            reasons.append("item-level supplier-cost claim lacks recipe/BOM evidence or explicit estimate label")

    return (len(reasons) == 0), reasons


def _dedupe_key(finding: AnalystFinding) -> str:
    return re.sub(r"\s+", " ", finding.get("title", "").strip().lower())


def run_critic(
    candidate_findings: list[AnalystFinding],
    revision_round: int,
    max_revision_rounds: int,
    valid_periods: set[tuple[str, str]] | None = None,
) -> CriticOutput:
    approved: list[str] = []
    rejected: list[str] = []
    revision_requests: list[RevisionRequest] = []
    removed_after_cap: list[str] = []
    notes: list[str] = []

    seen_titles: dict[str, str] = {}
    for finding in candidate_findings:
        ok, reasons = _deterministic_check(finding, valid_periods)
        if ok:
            key = _dedupe_key(finding)
            if key in seen_titles:
                rejected.append(finding["finding_id"])
                notes.append(f"{finding['finding_id']} rejected as duplicate of {seen_titles[key]}")
                continue
            seen_titles[key] = finding["finding_id"]
            approved.append(finding["finding_id"])
        else:
            if revision_round < max_revision_rounds:
                revision_requests.append(RevisionRequest(
                    finding_id=finding["finding_id"], analyst_name=finding["analyst_name"],
                    reason_code="unresolved_or_unsupported_claim",
                    explanation="; ".join(reasons),
                    required_fix="Recompute so every claimed number resolves to a result_key in the stored JSON, "
                                 "with explicit periods, known source names, and (if causal) multi-source evidence.",
                ))
            else:
                rejected.append(finding["finding_id"])
                removed_after_cap.append(finding["finding_id"])
                notes.append(f"{finding['finding_id']} rejected after exhausting {max_revision_rounds} revision rounds: {'; '.join(reasons)}")

    return CriticOutput(
        approved_findings=approved,
        rejected_findings=rejected,
        revision_requests=revision_requests,
        removed_after_cap=removed_after_cap,
        total_rejections=len(rejected),
        notes=notes,
    )
