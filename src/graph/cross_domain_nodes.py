"""Cross-domain synthesis node (runs between analysis_fanin and the critic).

See src/analysis/cross_domain.py for why this is a dedicated stage rather than
something each analyst does. This module is the graph wiring plus the single,
gated LLM call: it degrades to a no-op (returning no findings, spending
nothing) whenever the deterministic pre-pass finds nothing worth relating, and
it never crashes the graph -- a model failure here leaves the run with exactly
the single-domain findings it already had.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.analysis.cross_domain import (
    MAX_CROSS_DOMAIN_FINDINGS,
    build_evidence_pool,
    detect_co_movements,
    evidence_from_pool,
    has_substantive_co_movement,
    placeholder_metrics_from_pool,
    pool_for_prompt,
    validate_metric_refs,
)
from src.analysts.base import _substitute_claim_placeholders
from src.schemas.findings import AnalystFinding, CrossDomainSynthesisOutput
from src.state import CafeIntelligenceState
from src.tools.artifact_io import artifact_dir, write_json

CROSS_DOMAIN_PROMPT_PATH = Path("prompts/cross_domain.md")

CROSS_DOMAIN_ANALYST_NAME = "cross_domain"

Synthesizer = Callable[[str, dict[str, Any]], dict[str, Any]]
"""(system_prompt, context) -> CrossDomainSynthesisOutput-shaped dict. Tests
inject a fake to exercise the whole stage with no network access."""


def _default_llm_synthesizer(model_name: str) -> Synthesizer:
    def _synthesize(system_prompt: str, context: dict[str, Any]) -> dict[str, Any]:
        from src.tools.llm_factory import get_chat_model

        llm = get_chat_model(model_name, temperature=0)
        structured_llm = llm.with_structured_output(CrossDomainSynthesisOutput)
        user_prompt = (
            f"{json.dumps(context, indent=2, default=str)}\n\n"
            "Produce at most "
            f"{MAX_CROSS_DOMAIN_FINDINGS} cross-domain findings in `items`, each citing "
            "metric_refs from at least two different analysts, with every number written "
            "as a <<pool_key>> placeholder. Return an empty items list if nothing "
            "genuinely relates across analysts."
        )
        result = structured_llm.invoke([("system", system_prompt), ("user", user_prompt)])
        return dict(result) if isinstance(result, dict) else result.__dict__

    return _synthesize


def _substitute_all(
    draft: dict[str, Any], placeholder_metrics: dict[str, Any]
) -> tuple[str, list[str], list[str], list[str]]:
    """Substitutes <<pool_key>> in claim/assumptions/coverage_notes using the
    shared helper from src/analysts/base.py (same NaN/inf handling and same
    unresolved-key semantics as single-analyst findings)."""
    unresolved: list[str] = []
    claim, bad = _substitute_claim_placeholders(draft.get("claim", ""), placeholder_metrics)
    unresolved.extend(bad)
    assumptions: list[str] = []
    for a in draft.get("assumptions", []):
        text, bad = _substitute_claim_placeholders(a, placeholder_metrics)
        assumptions.append(text)
        unresolved.extend(bad)
    coverage: list[str] = []
    for c in draft.get("coverage_notes", []):
        text, bad = _substitute_claim_placeholders(c, placeholder_metrics)
        coverage.append(text)
        unresolved.extend(bad)
    return claim, assumptions, coverage, unresolved


def cross_domain_synthesis_node(
    state: CafeIntelligenceState, synthesizer: Synthesizer | None = None
) -> dict[str, Any]:
    candidate_findings = state.get("candidate_findings", [])
    pool = build_evidence_pool(candidate_findings)
    co_movements = detect_co_movements(pool)

    # Gate: no material cross-analyst co-movement -> no call, no cost. This is
    # the common case on a quiet week and is deliberately silent rather than
    # forcing a weak connection. A pool whose only co-movements are the same
    # quantity computed by two analysts is treated as nothing to say.
    if not co_movements or not has_substantive_co_movement(co_movements):
        return {"step_count": 1}

    config = state["config"]
    try:
        from src.analysis.correlation_hints import calendar_overlaps

        overlaps = calendar_overlaps(state["analysis_period"])
    except Exception:  # noqa: BLE001
        overlaps = []

    context = {
        "analysis_period": state["analysis_period"],
        "evidence_pool": pool_for_prompt(pool),
        "co_movements": co_movements,
        "calendar_overlaps": overlaps,
    }

    try:
        system_prompt = CROSS_DOMAIN_PROMPT_PATH.read_text(encoding="utf-8")
        synth = synthesizer or _default_llm_synthesizer(config.app_settings.models.analyst)
        out = synth(system_prompt, context)
    except Exception as e:  # noqa: BLE001
        # Synthesis is additive: if it fails, the run keeps every single-domain
        # finding it already has rather than failing the whole analysis stage.
        return {
            "step_count": 1,
            "errors": [{"node": "cross_domain_synthesis", "error": f"{type(e).__name__}: {e}"}],
        }

    drafts = (out or {}).get("items", []) or []
    findings: list[AnalystFinding] = []
    notes: list[str] = []

    results_dir = artifact_dir(config.artifact_root, state["run_id"], "results", CROSS_DOMAIN_ANALYST_NAME, "synthesis")
    code_dir = artifact_dir(config.artifact_root, state["run_id"], "code", CROSS_DOMAIN_ANALYST_NAME, "synthesis")

    for i, draft in enumerate(drafts[:MAX_CROSS_DOMAIN_FINDINGS], start=1):
        refs, reject_reason = validate_metric_refs(pool, draft.get("metric_refs", []))
        if reject_reason:
            notes.append(f"cross-domain draft {draft.get('title', '')!r} dropped: {reject_reason}")
            continue

        placeholder_metrics = placeholder_metrics_from_pool(pool)
        claim, assumptions, coverage, unresolved = _substitute_all(draft, placeholder_metrics)
        if unresolved:
            notes.append(
                f"cross-domain draft {draft.get('title', '')!r} dropped: unresolved placeholder(s) {unresolved}"
            )
            continue

        cited = [pool[r] for r in refs]
        # Provenance: the result manifest records exactly which pooled metrics
        # were related and where each was originally computed; the evidence
        # items themselves keep their origin result_path, so the critic
        # re-verifies every number against the analyst artifact that produced
        # it rather than against this stage's own output.
        manifest = {
            "findings": [{
                "title": draft.get("title", ""),
                "metrics": {
                    e["result_key"]: {
                        "value": e["value"], "unit": e.get("unit"),
                        "period_start": e["period_start"], "period_end": e["period_end"],
                        "origin_analyst": e["analyst_name"], "origin_result_path": e["result_path"],
                        "origin_finding_id": e["origin_finding_id"],
                    }
                    for e in cited
                },
            }],
            "synthesis_method": "deterministic evidence pooling + co-movement detection; narrative only from LLM",
        }
        result_ref = write_json(manifest, results_dir / f"{i}.json")
        method_ref = write_json(
            {
                "stage": "cross_domain_synthesis",
                "deterministic_inputs": {"co_movements": co_movements, "cited_pool_keys": refs},
                "note": (
                    "No analysis code is generated at this stage. Every cited value was computed by "
                    "another analyst's executed code and is re-verified against that analyst's own "
                    "result artifact; this stage only selects which of those metrics relate and "
                    "writes the narrative around them."
                ),
            },
            code_dir / f"{i}.json",
        )

        source_names = sorted({s for e in cited for s in e.get("source_names", [])})
        findings.append(AnalystFinding(
            finding_id=f"F-{CROSS_DOMAIN_ANALYST_NAME}-{uuid.uuid4().hex[:8]}",
            analyst_name=CROSS_DOMAIN_ANALYST_NAME,
            title=draft.get("title", ""), claim=claim,
            finding_type=draft.get("finding_type", "cross_domain"),
            evidence=evidence_from_pool(pool, refs),
            source_names=source_names,
            code_artifact=method_ref, result_artifact=result_ref,
            sample_size=None, coverage_notes=coverage, assumptions=assumptions,
            confidence=float(draft.get("confidence", 0.5)),
            business_impact_score=float(draft.get("business_impact_score", 0.7)),
            actionability_score=float(draft.get("actionability_score", 0.6)),
            revision_count=0,
            execution_metadata={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "cited_pool_keys": refs,
                "origin_analysts": sorted({e["analyst_name"] for e in cited}),
            },
        ))

    out_state: dict[str, Any] = {"candidate_findings": findings, "step_count": 1}
    if notes:
        out_state["errors"] = [{"node": "cross_domain_synthesis", "dropped_drafts": notes}]
    return out_state
