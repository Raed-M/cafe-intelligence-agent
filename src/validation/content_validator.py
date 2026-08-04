"""Content validator (Module 6, second validation layer). Purely deterministic:
verifies references resolve and structural constraints hold. It does not
create or revise content -- only approves or fails closed with exact repair
instructions, per prompts/content_validator.md.
"""
from __future__ import annotations

import re
from typing import Any

from src.business_time import parse_opening_hours
from src.schemas.content import ContentIdea
from src.schemas.context import ContextBundle
from src.schemas.findings import AnalystFinding

_SUPERLATIVE_RE = re.compile(r"\b(best|guaranteed|#1|number one|always|never fails)\b", re.IGNORECASE)
_CAUSAL_RE = re.compile(r"\b(because|proves|causes|guarantees)\b", re.IGNORECASE)
_MONEY_OR_PCT_RE = re.compile(r"(?:SAR\s*)?(\d+(?:\.\d+)?)\s*%|SAR\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def _acceptable_number_strings(evidence: list[dict[str, Any]], metric_keys: list[str]) -> set[float]:
    accepted: set[float] = set()
    for ev in evidence:
        if ev["result_key"] not in metric_keys:
            continue
        value = ev.get("value")
        if not isinstance(value, (int, float)):
            continue
        for nd in (0, 1, 2):
            accepted.add(round(float(value), nd))
        if -1.0 <= value <= 1.0:
            for nd in (0, 1, 2):
                accepted.add(round(float(value) * 100, nd))
    return accepted


def _numbers_in_text_are_grounded(text: str, accepted: set[float], tolerance: float = 0.5) -> list[str]:
    problems = []
    for m in _MONEY_OR_PCT_RE.finditer(text):
        raw = m.group(1) or m.group(2)
        try:
            n = float(raw)
        except ValueError:
            continue
        if not any(abs(n - a) <= tolerance for a in accepted):
            problems.append(raw)
    return problems


def _within_opening_hours(post_time: str, opening_hours: dict[str, str]) -> bool:
    try:
        interval = parse_opening_hours(opening_hours.get("default", "07:00-23:00"))
        h, m = map(int, post_time.split(":"))
    except (ValueError, AttributeError):
        return False
    t = h * 60 + m
    open_t = interval.open_time.hour * 60 + interval.open_time.minute
    close_t = interval.close_time.hour * 60 + interval.close_time.minute
    if interval.crosses_midnight:
        return t >= open_t or t <= close_t
    return open_t <= t <= close_t


def validate_content_ideas(
    ideas: list[ContentIdea],
    final_findings: list[AnalystFinding],
    context_bundle: ContextBundle,
    active_skus: set[str],
    opening_hours: dict[str, str],
    skus_with_stock_risk: set[str] | None = None,
) -> dict[str, Any]:
    approved_finding_ids = {f["finding_id"] for f in final_findings}
    findings_by_id = {f["finding_id"]: f for f in final_findings}
    context_by_id = {e["context_id"]: e for e in context_bundle["evidence"]}
    windows_by_id = {w["window_id"]: w for w in context_bundle["posting_windows"]}
    skus_with_stock_risk = skus_with_stock_risk or set()

    issues: dict[str, list[str]] = {}

    if len(ideas) != 3:
        issues["_count"] = [f"expected exactly 3 ideas, got {len(ideas)}"]

    hooks_seen: list[tuple[str, str]] = []
    for idea in ideas:
        idea_issues: list[str] = []

        finding = findings_by_id.get(idea["finding_id"])
        if idea["finding_id"] not in approved_finding_ids:
            idea_issues.append(f"finding_id {idea['finding_id']!r} is not a critic-approved finding")
        elif idea["cited_metric_keys"]:
            available_keys = {e["result_key"] for e in finding["evidence"]}
            bad_keys = set(idea["cited_metric_keys"]) - available_keys
            if bad_keys:
                idea_issues.append(f"cited metric keys not on the finding: {sorted(bad_keys)}")
        else:
            idea_issues.append("no cited_metric_keys provided")

        local_ids = idea.get("local_context_ids", [])
        if not local_ids or any(cid not in context_by_id or context_by_id[cid]["kind"] not in ("event", "profile", "email_event") for cid in local_ids):
            idea_issues.append("local_context_ids missing or do not resolve to a local/profile/email context item")

        cal_ids = idea.get("calendar_context_ids", [])
        if not cal_ids or any(cid not in context_by_id or context_by_id[cid]["kind"] not in ("calendar", "prayer") for cid in cal_ids):
            idea_issues.append("calendar_context_ids missing or do not resolve to a calendar/prayer context item")

        window = windows_by_id.get(idea["posting_window_id"])
        if not window:
            idea_issues.append(f"posting_window_id {idea['posting_window_id']!r} does not resolve")
        else:
            if idea.get("timing_metric_keys") and set(idea["timing_metric_keys"]) - set(window["busy_metric_keys"]):
                idea_issues.append("timing_metric_keys do not match the posting window's busy_metric_keys")
            if not _within_opening_hours(idea.get("post_time_local", ""), opening_hours):
                idea_issues.append(f"post_time_local {idea.get('post_time_local')!r} is outside opening hours")

        if idea["product_sku"] not in active_skus:
            idea_issues.append(f"product_sku {idea['product_sku']!r} is not an active menu SKU")
        elif idea["product_sku"] in skus_with_stock_risk and idea.get("inventory_suitability") == "supported":
            idea_issues.append(f"product_sku {idea['product_sku']!r} has known stock/waste risk not addressed by the idea")

        if not idea.get("hook_ar") or not idea.get("hook_en"):
            idea_issues.append("both hook_ar and hook_en are required")

        for field_name in ("hook_en", "rationale_en", "hook_ar", "rationale_ar"):
            text = idea.get(field_name, "")
            if _SUPERLATIVE_RE.search(text) or _CAUSAL_RE.search(text):
                idea_issues.append(f"{field_name} contains an unsupported superlative/causal claim")

        if finding is not None:
            accepted_numbers = _acceptable_number_strings(finding["evidence"], idea.get("cited_metric_keys", []))
            for field_name in ("hook_en", "rationale_en"):
                bad_numbers = _numbers_in_text_are_grounded(idea.get(field_name, ""), accepted_numbers)
                if bad_numbers:
                    idea_issues.append(
                        f"{field_name} states number(s) {bad_numbers} (SAR amount/percentage) that do not match "
                        f"any cited metric value on the finding"
                    )

        hooks_seen.append((idea.get("hook_en", "").strip().lower(), idea.get("hook_ar", "").strip()))

        if idea_issues:
            issues[idea["idea_id"]] = idea_issues

    if len(ideas) == 3:
        en_hooks = [h[0] for h in hooks_seen]
        if len(set(en_hooks)) < len(en_hooks):
            issues.setdefault("_distinctness", []).append("two or more ideas share the same (or near-identical) hook")

    return {"valid": len(issues) == 0, "issues": issues}
