"""Finding critic (Module 5): deterministic provenance checks first, then an
optional constrained-LLM semantic pass for nuance a hard rule cannot capture
(prompts/critic.md, plan section 16.9).

The deterministic layer is the load-bearing gate: it independently re-reads
each finding's result_artifact from disk and verifies every cited metric key
actually resolves there with a numeric/string value, rather than trusting
whatever the analyst claims. This is what makes "a number without a resolvable
computation must be rejected" enforceable rather than aspirational. The LLM
layer only runs on findings that already passed that gate, and only adds
scrutiny (correlation-vs-causation overstatement, assumption/data
compatibility) that a regex cannot judge -- it never overrides a deterministic
rejection, and if it is unavailable (no key, network failure) the finding
keeps its deterministic decision rather than the whole run failing closed.

Also deterministic: a causal claim whose period overlaps Ramadan/Eid must
disclose that calendar swing or be sent back. Proven necessary against this
dataset -- a real staffing change (senior barista's last shift ~2026-03-15)
showed no attributable effect because Eid al-Fitr traffic (~3x door counts)
landed the same week; without this check a model could easily misattribute
one for the other.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from src.schemas.findings import AnalystFinding, CriticOutput, RevisionRequest, SemanticReviewResult

SemanticReviewer = Callable[[str, dict[str, Any]], dict[str, Any]]

CRITIC_PROMPT_PATH = Path("prompts/critic.md")

KNOWN_SOURCE_NAMES = {"pos", "menu", "traffic", "staff", "inventory", "emails", "reviews"}

_CAUSAL_PATTERNS = re.compile(
    r"\b(because|caused by|due to|led to|resulted in|drove|driving)\b", re.IGNORECASE
)
_ITEM_LEVEL_COST_PATTERNS = re.compile(
    r"\b(per[- ]drink|per[- ]item|per[- ]cup)\b.*\b(cost|margin|impact)\b", re.IGNORECASE
)


def _calendar_confound(ev_period_start: str, ev_period_end: str) -> list[str]:
    """Returns the name(s) of every Ramadan/Eid window overlapping this
    finding's period -- these are exactly the calendar swings large enough to
    dwarf a same-week domain-specific signal (e.g. a staffing change),
    proven concretely against this dataset: a senior-to-junior barista
    handover around 2026-03-15 showed no attributable conversion-rate effect
    because Eid al-Fitr traffic (door counts up ~3x) landed the same week --
    which itself falls inside the tail of Ramadan, so a period can overlap
    both; a causal claim must disclose at least one of the windows it spans.
    Shares its Hijri-window logic with src.analysis.correlation_hints, which
    surfaces the same overlap to analysts pre-emptively."""
    from src.analysis.correlation_hints import calendar_overlaps

    return calendar_overlaps({"start": ev_period_start, "end": ev_period_end})


_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers_in_claim(claim: str) -> list[float]:
    """Extracts numeric tokens from claim text, deterministically -- this is
    the same class of check content_validator.py already does for content
    ideas, applied here to analyst findings. Proven necessary empirically: a
    cheap model will sometimes write a claim sentence that doesn't match the
    JSON its own executed code just produced (e.g. claiming "34.5%" when the
    stored result says 26.93%) -- catching that here, deterministically, at
    round 0 is far cheaper than waiting for the semantic LLM pass to notice
    it after 1-2 wasted revision rounds."""
    text = _ISO_DATE_RE.sub("", claim)
    numbers = []
    for m in _NUMBER_RE.finditer(text):
        raw = m.group(0).replace(",", "")
        try:
            n = float(raw)
        except ValueError:
            continue
        if 2020 <= n <= 2099 and "." not in raw and "," not in m.group(0):
            continue  # looks like a bare year
        numbers.append(n)
    return numbers


def _grounded_values(finding: AnalystFinding) -> set[float]:
    values: set[float] = set()
    for ev in finding.get("evidence", []):
        for raw in (ev.get("value"), ev.get("numerator"), ev.get("denominator")):
            if isinstance(raw, (int, float)):
                for nd in (0, 1, 2, 3, 4):
                    values.add(round(float(raw), nd))
                if -1.0 <= raw <= 1.0:
                    for nd in (0, 1, 2):
                        values.add(round(float(raw) * 100, nd))
    for key in ("sample_size", "confidence"):
        raw = finding.get(key)
        if isinstance(raw, (int, float)):
            values.add(round(float(raw), 2))
    return values


def _claim_numbers_are_grounded(finding: AnalystFinding) -> list[str]:
    grounded = _grounded_values(finding)
    if not grounded:
        return []
    ungrounded = []
    for n in _numbers_in_claim(finding.get("claim", "")):
        tolerance = max(0.5, abs(n) * 0.02)
        if not any(abs(n - g) <= tolerance for g in grounded):
            ungrounded.append(n)
    return [f"{n:g}" for n in ungrounded]


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

    ungrounded_numbers = _claim_numbers_are_grounded(finding)
    if ungrounded_numbers:
        reasons.append(
            f"claim states number(s) {ungrounded_numbers} that do not match any evidence value on "
            f"this finding -- the prose claim must restate what the executed code actually computed"
        )

    if not (0.0 <= finding.get("confidence", -1) <= 1.0):
        reasons.append("confidence out of [0,1] range")

    claim = finding.get("claim", "")
    if _CAUSAL_PATTERNS.search(claim) and len(finding.get("source_names", [])) < 2:
        reasons.append("causal language used with only a single source; explanations require joined evidence")
    elif _CAUSAL_PATTERNS.search(claim) and finding.get("evidence"):
        ev0 = finding["evidence"][0]
        confounds = _calendar_confound(ev0.get("period_start", ""), ev0.get("period_end", ""))
        if confounds:
            disclosed_text = f"{claim} {' '.join(finding.get('assumptions', []))} {' '.join(finding.get('coverage_notes', []))}".lower()
            disclosed = any(
                name.lower() in disclosed_text or name.lower().replace(" al-", "-") in disclosed_text
                for name in confounds
            )
            if not disclosed:
                reasons.append(
                    f"causal claim's period overlaps {' / '.join(confounds)}, a calendar swing large enough to "
                    f"dominate most operational metrics on its own; the claim/assumptions must disclose or "
                    f"control for at least one of these before this explanation is trustworthy"
                )

    if finding.get("analyst_name") == "margin" and _ITEM_LEVEL_COST_PATTERNS.search(claim):
        assumptions_text = " ".join(finding.get("assumptions", [])).lower()
        if "estimate" not in assumptions_text and "bom" not in assumptions_text:
            reasons.append("item-level supplier-cost claim lacks recipe/BOM evidence or explicit estimate label")

    return (len(reasons) == 0), reasons


def _dedupe_key(finding: AnalystFinding) -> str:
    return re.sub(r"\s+", " ", finding.get("title", "").strip().lower())


def _default_llm_reviewer(model_name: str) -> SemanticReviewer:
    def _review(system_prompt: str, context: dict[str, Any]) -> dict[str, Any]:
        from src.tools.llm_factory import get_chat_model

        llm = get_chat_model(model_name, temperature=0)
        structured_llm = llm.with_structured_output(SemanticReviewResult)
        user_prompt = (
            f"Candidate finding (already passed deterministic provenance checks):\n"
            f"{json.dumps(context, indent=2, default=str)}\n\n"
            "Decide approve/revise/reject/insufficient_evidence for this single finding."
        )
        result = structured_llm.invoke([("system", system_prompt), ("user", user_prompt)])
        return dict(result) if isinstance(result, dict) else result.__dict__

    return _review


def _semantic_review(
    finding: AnalystFinding, model_name: str | None, reviewer: SemanticReviewer | None,
    cross_domain_hints: list[str] | None = None,
) -> tuple[bool, str, str] | None:
    """Returns (ok, explanation, required_fix), or None if the layer is not
    enabled (no model/reviewer supplied) or unavailable (call failed)."""
    if reviewer is None and not model_name:
        return None
    system_prompt = CRITIC_PROMPT_PATH.read_text(encoding="utf-8")
    context = {
        "finding_id": finding["finding_id"], "analyst_name": finding["analyst_name"],
        "title": finding["title"], "claim": finding["claim"],
        "evidence": finding["evidence"], "source_names": finding.get("source_names", []),
        "assumptions": finding.get("assumptions", []), "coverage_notes": finding.get("coverage_notes", []),
        "confidence": finding.get("confidence"),
        "cross_domain_hints_for_this_period": cross_domain_hints or [],
    }
    review = reviewer or _default_llm_reviewer(model_name)  # type: ignore[arg-type]
    try:
        out = review(system_prompt, context)
    except Exception as e:  # noqa: BLE001
        return None  # layer unavailable -- keep the deterministic decision
    decision = out.get("decision", "approve")
    if decision == "approve":
        return True, "", ""
    return False, out.get("explanation", "semantic review flagged this claim"), out.get("required_fix", "")


def run_critic(
    candidate_findings: list[AnalystFinding],
    revision_round: int,
    max_revision_rounds: int,
    valid_periods: set[tuple[str, str]] | None = None,
    model_name: str | None = None,
    reviewer: SemanticReviewer | None = None,
    cross_domain_hints: list[str] | None = None,
) -> CriticOutput:
    approved: list[str] = []
    rejected: list[str] = []
    revision_requests: list[RevisionRequest] = []
    removed_after_cap: list[str] = []
    notes: list[str] = []

    seen_titles: dict[str, str] = {}
    for finding in candidate_findings:
        ok, reasons = _deterministic_check(finding, valid_periods)
        semantic_fix = ""
        if ok:
            semantic = _semantic_review(finding, model_name, reviewer, cross_domain_hints)
            if semantic is not None:
                sem_ok, sem_explanation, sem_fix = semantic
                if not sem_ok:
                    ok = False
                    reasons = [f"semantic review: {sem_explanation}" if sem_explanation else "semantic review flagged this claim"]
                    semantic_fix = sem_fix
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
                    required_fix=semantic_fix or (
                        "Recompute so every claimed number resolves to a result_key in the stored JSON, "
                        "with explicit periods, known source names, and (if causal) multi-source evidence."
                    ),
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
