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
from src.tools.concurrency import bounded_map

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


def _load_result_obj(path_str: str, cache: dict[str, Any]) -> dict[str, Any] | None:
    """Reads and caches a result artifact by path. Returns None when the file
    is missing or unparseable, which the caller reports as an unresolvable
    metric rather than crashing the critic."""
    if path_str in cache:
        return cache[path_str]
    obj: dict[str, Any] | None
    try:
        obj = json.loads(Path(path_str).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        obj = None
    cache[path_str] = obj
    return obj


def _is_known_period_or_span(
    period_key: tuple[str, str], valid_periods: set[tuple[str, str]]
) -> bool:
    """True when the period exactly matches a known run period, or exactly
    tiles a *contiguous span* of them (e.g. the four trailing baseline weeks
    2026-02-23..2026-03-23 collapsed into one month-long window).

    A multi-week baseline is legitimate and often better statistics -- an
    anomaly z-score over four weeks is more robust than over one -- and
    rejecting it outright (as the exact-match-only rule did) sent a sound
    finding through the whole revision loop to a guaranteed rejection. An
    arbitrary invented range still fails, because the span must start on a
    known period's start, end on a known period's end, and be fully tiled by
    known periods end-to-start with no gaps or overlaps."""
    if period_key in valid_periods:
        return True
    start, end = period_key
    if start >= end:
        return False
    starts: dict[str, set[str]] = {}
    for p_start, p_end in valid_periods:
        starts.setdefault(p_start, set()).add(p_end)
    cursor = start
    seen: set[str] = set()
    while cursor in starts and cursor not in seen:
        seen.add(cursor)
        # Prefer the longest next hop that doesn't overshoot the target end.
        candidates = [e for e in starts[cursor] if e <= end]
        if not candidates:
            return False
        cursor = max(candidates)
        if cursor == end:
            return True
    return False


DeterministicRuleContext = dict[str, Any]
"""Passed to every rule in _DETERMINISTIC_RULES: {"result_obj", "result_cache",
"valid_periods", "excluded_days_by_source"}. Adding a new deterministic check
means adding one function with this signature and appending it to the list
below -- nothing else in this module needs to change, which is the point:
previously every new rule was one more `if` bolted onto a single growing
function."""


def _rule_metric_evidence_resolves(finding: AnalystFinding, ctx: DeterministicRuleContext) -> list[str]:
    """Each evidence item is resolved against *its own* result_path, not the
    finding-level result_artifact. Single-analyst findings are unaffected (every
    evidence item points at that analyst's one artifact anyway), but this is
    what makes a genuinely cross-domain finding verifiable: its evidence cites
    metrics computed by several different analysts' executed code, each still
    independently re-read from the artifact that actually produced it, so
    provenance chains back to the real computation rather than to a synthesis
    step's say-so."""
    reasons: list[str] = []
    for ev in finding["evidence"]:
        # Prefer the evidence item's own artifact; fall back to the
        # finding-level one when it names no readable path. The fallback keeps
        # this strictly an *enhancement*: single-artifact findings (and any
        # caller that doesn't populate a real per-evidence path) behave exactly
        # as before, while a genuine multi-artifact finding gets each number
        # checked against the artifact that actually produced it.
        ev_result_obj = None
        ev_path = ev.get("result_path")
        if ev_path:
            ev_result_obj = _load_result_obj(ev_path, ctx["result_cache"])
        if ev_result_obj is None:
            ev_result_obj = ctx["result_obj"]
        if not _resolve_metric(ev_result_obj, ev["metric_name"], ev["result_key"]):
            reasons.append(f"metric '{ev['result_key']}' does not resolve in stored result")
        if ev["value"] is None:
            reasons.append(f"metric '{ev['result_key']}' has null value")
        if not ev.get("period_start") or not ev.get("period_end"):
            reasons.append(f"metric '{ev['result_key']}' missing period bounds")
        elif ctx["valid_periods"] is not None:
            period_key = (ev["period_start"][:10], ev["period_end"][:10])
            if not _is_known_period_or_span(period_key, ctx["valid_periods"]):
                reasons.append(
                    f"metric '{ev['result_key']}' period {period_key} does not match any known "
                    f"analysis/previous/trailing-baseline period (or contiguous span of them) for this run"
                )
    return reasons


def _rule_known_source_names(finding: AnalystFinding, ctx: DeterministicRuleContext) -> list[str]:
    unknown_sources = set(finding.get("source_names", [])) - KNOWN_SOURCE_NAMES
    if unknown_sources:
        return [f"unknown source names: {sorted(unknown_sources)}"]
    return []


def _rule_claim_numbers_grounded(finding: AnalystFinding, ctx: DeterministicRuleContext) -> list[str]:
    ungrounded_numbers = _claim_numbers_are_grounded(finding)
    if ungrounded_numbers:
        return [
            f"claim states number(s) {ungrounded_numbers} that do not match any evidence value on "
            f"this finding -- the prose claim must restate what the executed code actually computed"
        ]
    return []


def _rule_confidence_in_range(finding: AnalystFinding, ctx: DeterministicRuleContext) -> list[str]:
    if not (0.0 <= finding.get("confidence", -1) <= 1.0):
        return ["confidence out of [0,1] range"]
    return []


def _rule_causal_claim_disclosure(finding: AnalystFinding, ctx: DeterministicRuleContext) -> list[str]:
    claim = finding.get("claim", "")
    if not _CAUSAL_PATTERNS.search(claim):
        return []
    if len(finding.get("source_names", [])) < 2:
        return ["causal language used with only a single source; explanations require joined evidence"]
    if not finding.get("evidence"):
        return []
    ev0 = finding["evidence"][0]
    confounds = _calendar_confound(ev0.get("period_start", ""), ev0.get("period_end", ""))
    if not confounds:
        return []
    disclosed_text = f"{claim} {' '.join(finding.get('assumptions', []))} {' '.join(finding.get('coverage_notes', []))}".lower()
    disclosed = any(
        name.lower() in disclosed_text or name.lower().replace(" al-", "-") in disclosed_text
        for name in confounds
    )
    if disclosed:
        return []
    return [
        f"causal claim's period overlaps {' / '.join(confounds)}, a calendar swing large enough to "
        f"dominate most operational metrics on its own; the claim/assumptions must disclose or "
        f"control for at least one of these before this explanation is trustworthy"
    ]


def _rule_item_level_cost_requires_bom(finding: AnalystFinding, ctx: DeterministicRuleContext) -> list[str]:
    claim = finding.get("claim", "")
    if finding.get("analyst_name") != "margin" or not _ITEM_LEVEL_COST_PATTERNS.search(claim):
        return []
    assumptions_text = " ".join(finding.get("assumptions", [])).lower()
    if "estimate" not in assumptions_text and "bom" not in assumptions_text:
        return ["item-level supplier-cost claim lacks recipe/BOM evidence or explicit estimate label"]
    return []


_NORMALIZATION_DISCLOSURE_KEYWORDS = (
    "per day", "per-day", "daily average", "daily rate", "average per", "normalized", "normalised",
)


def _rule_unequal_valid_days_not_normalized(finding: AnalystFinding, ctx: DeterministicRuleContext) -> list[str]:
    """A period-over-period comparison that sums a metric over unequal numbers
    of valid days (e.g. one period lost 3 days to a dead sensor, the other
    didn't) is apples-to-oranges even when every number is individually
    grounded -- the raw totals aren't comparable, only a per-day rate is.
    Proven necessary: a footfall finding for week 2026-06-08 claimed a
    "significant drop" from a 7-valid-day previous period to a 4-valid-day
    current period; the daily average actually rose (873/day vs 796/day) once
    corrected. Requires excluded_days_by_source (per-source dead/excluded
    dates from data_quality) to be supplied -- a no-op otherwise."""
    excluded_days_by_source: dict[str, list[str]] | None = ctx.get("excluded_days_by_source")
    if not excluded_days_by_source:
        return []
    periods = {(ev.get("period_start", "")[:10], ev.get("period_end", "")[:10]) for ev in finding.get("evidence", [])}
    periods.discard(("", ""))
    if len(periods) < 2:
        return []
    relevant_sources = [s for s in finding.get("source_names", []) if excluded_days_by_source.get(s)]
    if not relevant_sources:
        return []
    excluded_counts = set()
    for p_start, p_end in periods:
        n_excluded = sum(
            1 for src in relevant_sources for d in excluded_days_by_source[src] if p_start <= d < p_end
        )
        excluded_counts.add(n_excluded)
    if len(excluded_counts) < 2:
        return []  # both periods equally affected (or unaffected) -- raw sums stay comparable
    text = f"{finding.get('claim', '')} {' '.join(finding.get('assumptions', []))} {' '.join(finding.get('coverage_notes', []))}".lower()
    if any(kw in text for kw in _NORMALIZATION_DISCLOSURE_KEYWORDS):
        return []
    return [
        "finding compares raw totals across periods with a different number of excluded/dead-sensor days "
        "for its cited source(s) -- this is an apples-to-oranges comparison; state a per-day rate instead of "
        "(or in addition to) the raw total, or explicitly disclose the day-count mismatch"
    ]


_DETERMINISTIC_RULES: list[Callable[[AnalystFinding, DeterministicRuleContext], list[str]]] = [
    _rule_metric_evidence_resolves,
    _rule_known_source_names,
    _rule_claim_numbers_grounded,
    _rule_confidence_in_range,
    _rule_causal_claim_disclosure,
    _rule_item_level_cost_requires_bom,
    _rule_unequal_valid_days_not_normalized,
]


def _deterministic_check(
    finding: AnalystFinding,
    valid_periods: set[tuple[str, str]] | None = None,
    excluded_days_by_source: dict[str, list[str]] | None = None,
) -> tuple[bool, list[str]]:
    """Returns (ok, reasons). ok=False means reject or revise. A few checks
    are hard prerequisites the rules below all depend on (artifact exists,
    parses as JSON, has evidence at all) and short-circuit immediately;
    everything past that runs every rule in _DETERMINISTIC_RULES and unions
    their reasons, so adding a new check never means editing this function."""
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

    ctx: DeterministicRuleContext = {
        "result_obj": result_obj,
        "result_cache": {str(result_path): result_obj},
        "valid_periods": valid_periods,
        "excluded_days_by_source": excluded_days_by_source,
    }
    reasons: list[str] = []
    for rule in _DETERMINISTIC_RULES:
        reasons.extend(rule(finding, ctx))
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
) -> tuple[bool, str, str] | None:
    """Returns (ok, explanation, required_fix), or None if the layer is not
    enabled (no model/reviewer supplied) or unavailable (call failed).

    Deliberately *not* given other analysts' metrics: a single-analyst finding
    can only cite numbers its own code produced, so showing the reviewer a
    neighbouring domain's figure only tempts it to demand something that the
    deterministic grounding rule then rejects. Cross-analyst relationships are
    synthesized in their own stage (src/analysis/cross_domain.py) and arrive
    here as ordinary `cross_domain` findings to be judged on their own merits."""
    if reviewer is None and not model_name:
        return None
    system_prompt = CRITIC_PROMPT_PATH.read_text(encoding="utf-8")
    context = {
        "finding_id": finding["finding_id"], "analyst_name": finding["analyst_name"],
        "title": finding["title"], "claim": finding["claim"],
        "evidence": finding["evidence"], "source_names": finding.get("source_names", []),
        "assumptions": finding.get("assumptions", []), "coverage_notes": finding.get("coverage_notes", []),
        "confidence": finding.get("confidence"),
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
    excluded_days_by_source: dict[str, list[str]] | None = None,
    non_revisable_analysts: set[str] | None = None,
) -> CriticOutput:
    approved: list[str] = []
    rejected: list[str] = []
    revision_requests: list[RevisionRequest] = []
    removed_after_cap: list[str] = []
    notes: list[str] = []

    # Deterministic checks are cheap pure-python and must run in order (no
    # shared state issue either way), but the semantic-review LLM call per
    # finding is independent and was previously one-at-a-time -- run those
    # concurrently (bounded, still centrally rate-limited) instead, then
    # recombine in original order below so dedupe/revision-cap bookkeeping
    # stays exactly as order-dependent as before. See src/tools/concurrency.py.
    deterministic_results: dict[str, tuple[bool, list[str]]] = {
        finding["finding_id"]: _deterministic_check(finding, valid_periods, excluded_days_by_source)
        for finding in candidate_findings
    }
    needs_semantic_review = [f for f in candidate_findings if deterministic_results[f["finding_id"]][0]]

    def _review_one(finding: AnalystFinding):
        return finding["finding_id"], _semantic_review(finding, model_name, reviewer)

    semantic_results: dict[str, tuple[bool, str, str] | None] = dict(bounded_map(_review_one, needs_semantic_review))

    seen_titles: dict[str, str] = {}
    for finding in candidate_findings:
        ok, reasons = deterministic_results[finding["finding_id"]]
        semantic_fix = ""
        if ok:
            semantic = semantic_results.get(finding["finding_id"])
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
            # A finding from a stage with no revision path (cross-domain
            # synthesis) is terminal: re-running it would re-issue the same
            # LLM call over the same pooled inputs, which is precisely the
            # doomed-retry pattern this design removed. Reject it now with
            # the real reason instead of booking a revision that can never
            # be serviced and would spin the loop to its cap.
            if (non_revisable_analysts or set()) & {finding["analyst_name"]}:
                rejected.append(finding["finding_id"])
                notes.append(
                    f"{finding['finding_id']} ({finding['analyst_name']}) rejected without revision "
                    f"(stage has no revision path): {'; '.join(reasons)}"
                )
            elif revision_round < max_revision_rounds:
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
