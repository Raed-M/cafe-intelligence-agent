"""Cross-domain synthesis: relating metrics that *different* analysts computed.

Why this is a separate stage rather than something each analyst does
-------------------------------------------------------------------
An individual analyst structurally cannot produce a cross-domain finding, and
the previous design's attempt to make it do so was an unwinnable catch-22
(observed live, trace 2026-03-23):

* the critic's semantic layer told analysts to acknowledge a cross-domain
  hint ("gross margin moved +69.9% in the same week"), and
* the critic's deterministic layer then rejected the very findings that
  complied -- because 69.9 is not in *that analyst's* evidence, and every
  number in a claim must resolve to a metric the claiming analyst's own
  executed code produced (F-operations-6d5ebe1a and F-margin-d7d6b392 were
  both rejected for exactly this).

Comply and you are rejected; ignore it and you are rejected. Every affected
finding burned its full revision budget on a guaranteed failure -- 6 of 12
rejections in that run, and the single largest source of wasted LLM calls.

The resolution is structural: each analyst stays in its own lane and produces
well-grounded single-domain findings, and *this* stage -- which alone sees
every analyst's evidence at once -- relates them. Its findings legitimately
cite several analysts' metrics because each MetricEvidence item carries its
own `result_path`, so the critic re-reads every cited number from the
artifact of the executed code that actually computed it.

Cost discipline
---------------
Everything here is deterministic pandas/python except one optional call:
the pool and the co-movement detection are pure code, and when no material
co-movement exists (the common case) the stage returns without contacting a
model at all. When it does call, it calls exactly once, is capped at
MAX_CROSS_DOMAIN_FINDINGS outputs, and never gets raw data -- only the
already-computed pool. The model chooses *which* metrics relate and writes
the narrative; the numbers themselves are substituted in by us from the pool,
so a cross-domain claim cannot state a value that no analyst computed.
"""
from __future__ import annotations

import re
from typing import Any

from src.schemas.findings import AnalystFinding, MetricEvidence

MATERIALITY_PCT = 10.0
"""Minimum |%| for a change metric to count as having 'moved' -- same
threshold correlation_hints uses, kept consistent deliberately."""

MAX_CO_MOVEMENTS = 6
MAX_CROSS_DOMAIN_FINDINGS = 2

_CHANGE_KEY_RE = re.compile(r"pct|delta|change|growth|_rate\b|rate_", re.IGNORECASE)
_PERCENT_UNITS = {"%", "percent", "pct", "percentage"}


def pool_key(analyst_name: str, result_key: str) -> str:
    """Pool keys use `__` (not `.`) so they are valid `<<placeholder>>` tokens
    for the shared claim-substitution helper in src/analysts/base.py."""
    return f"{analyst_name}__{result_key}"


def build_evidence_pool(candidate_findings: list[AnalystFinding]) -> dict[str, dict[str, Any]]:
    """Flattens every analyst's evidence into one keyed pool. Only complete,
    numeric, period-bearing evidence is admitted -- anything the critic would
    reject on its own is not eligible to be related to something else."""
    pool: dict[str, dict[str, Any]] = {}
    for finding in candidate_findings:
        analyst = finding.get("analyst_name", "")
        if analyst == "cross_domain":
            continue  # never synthesize from a previous synthesis
        for ev in finding.get("evidence", []):
            value = ev.get("value")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if not ev.get("period_start") or not ev.get("period_end") or not ev.get("result_path"):
                continue
            key = pool_key(analyst, ev["result_key"])
            if key in pool:
                continue  # first writer wins; revisions append later, keep the earliest grounded one
            pool[key] = {
                "pool_key": key,
                "analyst_name": analyst,
                "metric_name": ev.get("metric_name", ev["result_key"]),
                "result_key": ev["result_key"],
                "value": float(value),
                "unit": ev.get("unit"),
                "period_start": ev["period_start"],
                "period_end": ev["period_end"],
                "result_path": ev["result_path"],
                "source_names": ev.get("source_names", []),
                "origin_finding_id": finding.get("finding_id", ""),
                "origin_title": finding.get("title", ""),
            }
    return pool


def _is_change_metric(entry: dict[str, Any]) -> bool:
    unit = (entry.get("unit") or "").strip().lower()
    if unit in _PERCENT_UNITS:
        return True
    return bool(_CHANGE_KEY_RE.search(entry["result_key"]))


def detect_co_movements(pool: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically pairs materially-moved change metrics reported by
    *different* analysts for the *same* period. Divergent pairs (one up, one
    down) score highest: 'revenue fell 23% while gross margin rose 70%' is
    precisely the shape of finding that no single analyst can ever state, and
    the one most likely to matter to an owner."""
    movers = [
        e for e in pool.values()
        if _is_change_metric(e) and abs(e["value"]) >= MATERIALITY_PCT
    ]
    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(movers):
        for b in movers[i + 1:]:
            if a["analyst_name"] == b["analyst_name"]:
                continue
            if (a["period_start"][:10], a["period_end"][:10]) != (b["period_start"][:10], b["period_end"][:10]):
                continue
            divergent = (a["value"] > 0) != (b["value"] > 0)
            pairs.append({
                "left": a["pool_key"], "right": b["pool_key"],
                "left_analyst": a["analyst_name"], "right_analyst": b["analyst_name"],
                "left_value": a["value"], "right_value": b["value"],
                "left_unit": a.get("unit"), "right_unit": b.get("unit"),
                "period_start": a["period_start"], "period_end": a["period_end"],
                "divergent": divergent,
                "score": abs(a["value"]) + abs(b["value"]) + (100.0 if divergent else 0.0),
            })
    pairs.sort(key=lambda p: p["score"], reverse=True)
    return pairs[:MAX_CO_MOVEMENTS]


def pool_for_prompt(pool: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact projection of the pool for the model -- no file paths, no
    origin ids, just what is needed to decide which metrics relate."""
    return [
        {
            "pool_key": e["pool_key"], "analyst": e["analyst_name"],
            "metric": e["result_key"], "value": e["value"], "unit": e.get("unit"),
            "period_start": e["period_start"][:10], "period_end": e["period_end"][:10],
        }
        for e in pool.values()
    ]


def evidence_from_pool(pool: dict[str, dict[str, Any]], keys: list[str]) -> list[MetricEvidence]:
    """Rebuilds MetricEvidence straight from pooled entries, preserving each
    one's original result_path so the critic verifies it against the artifact
    of the analyst that actually computed it."""
    evidence: list[MetricEvidence] = []
    for key in keys:
        e = pool[key]
        evidence.append(MetricEvidence(
            metric_name=e["metric_name"], value=e["value"], unit=e.get("unit"),
            numerator=None, denominator=None,
            period_start=e["period_start"], period_end=e["period_end"],
            comparison_period_start=None, comparison_period_end=None,
            source_names=e.get("source_names", []),
            result_path=e["result_path"], result_key=e["result_key"],
        ))
    return evidence


def validate_metric_refs(
    pool: dict[str, dict[str, Any]], refs: list[str]
) -> tuple[list[str], str | None]:
    """A cross-domain finding must cite >=2 pool keys spanning >=2 distinct
    analysts, all of which exist. Returns (clean_refs, rejection_reason)."""
    clean = [r for r in dict.fromkeys(refs) if r in pool]
    unknown = [r for r in refs if r not in pool]
    if unknown:
        return [], f"cites unknown pool key(s) {unknown}"
    if len(clean) < 2:
        return [], "cites fewer than 2 pooled metrics"
    analysts = {pool[r]["analyst_name"] for r in clean}
    if len(analysts) < 2:
        return [], f"all cited metrics come from a single analyst ({analysts})"
    return clean, None
