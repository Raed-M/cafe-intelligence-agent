"""Critic execution + targeted revision routing (Module 5 graph wiring)."""
from __future__ import annotations

from typing import Any, Literal

from langgraph.types import Send

from src.analysts import anomaly, customer, margin, operations, sales
from src.analysts.base import run_analyst
from src.state import CafeIntelligenceState
from src.validation.finding_critic import run_critic
from src.validation.finding_ranker import rank_findings

ANALYST_SPECS = {
    "sales": sales.SPEC, "margin": margin.SPEC, "operations": operations.SPEC,
    "customer": customer.SPEC, "anomaly": anomaly.SPEC,
}


def _valid_periods(state: CafeIntelligenceState) -> set[tuple[str, str]]:
    periods = [state["analysis_period"], state["previous_period"], *state.get("trailing_baseline_periods", [])]
    return {(p["start"][:10], p["end"][:10]) for p in periods}


def _excluded_days_by_source(state: CafeIntelligenceState) -> dict[str, list[str]]:
    """Per-source dead-sensor/excluded dates from the cleaning step's data
    quality report, keyed by source name -- lets the critic's deterministic
    day-count-normalization check (finding_critic._rule_unequal_valid_days_not_normalized)
    tell whether two periods a finding compares were affected unequally."""
    summaries = state.get("data_quality", {}).get("source_summaries", [])
    return {
        s["source_name"]: [p["date"] for p in s.get("excluded_periods", [])]
        for s in summaries if s.get("excluded_periods")
    }


def _non_revisable_analysts(candidate_findings: list[dict[str, Any]]) -> set[str]:
    """Any producing stage that isn't a rerunnable specialist analyst (i.e. has
    no entry in ANALYST_SPECS) cannot be sent through the revision loop --
    run_one_analyst_revision would KeyError on it. Derived from the findings
    actually present rather than hardcoded, so adding another synthesis-style
    stage later needs no change here."""
    return {
        f["analyst_name"] for f in candidate_findings
        if f.get("analyst_name") and f["analyst_name"] not in ANALYST_SPECS
    }


def critic_node(state: CafeIntelligenceState) -> dict[str, Any]:
    limits = state["config"].app_settings.limits
    round_ = state.get("critic_round", 0)
    candidate_findings = state.get("candidate_findings", [])
    result = run_critic(
        candidate_findings, round_, limits.critic_revision_rounds,
        valid_periods=_valid_periods(state),
        model_name=state["config"].app_settings.models.critic,
        excluded_days_by_source=_excluded_days_by_source(state),
        non_revisable_analysts=_non_revisable_analysts(candidate_findings),
    )
    return {"critic_results": result, "critic_round": round_ + 1, "step_count": 1}


class _RevisionJobState(CafeIntelligenceState, total=False):
    _analyst_name: str


def route_after_critic(state: CafeIntelligenceState) -> str | list[Send]:
    critic: dict = state["critic_results"]
    if critic["revision_requests"]:
        # Defense in depth: the critic already refuses to book revisions for
        # non-rerunnable stages, but never dispatch a Send for a name that has
        # no AnalystSpec -- that would KeyError inside the revision node.
        analysts_to_rerun = sorted(
            {r["analyst_name"] for r in critic["revision_requests"] if r["analyst_name"] in ANALYST_SPECS}
        )
        if analysts_to_rerun:
            return [Send("run_one_analyst_revision", {**state, "_analyst_name": name}) for name in analysts_to_rerun]
    if not critic["approved_findings"]:
        return "no_evidence"
    return "rank"


def run_one_analyst_revision(state: _RevisionJobState) -> dict[str, Any]:
    name = state["_analyst_name"]
    spec = ANALYST_SPECS[name]
    config = state["config"]
    critic: dict = state["critic_results"]
    feedback = [
        f"{r['reason_code']}: {r['explanation']} -> {r['required_fix']}"
        for r in critic["revision_requests"] if r["analyst_name"] == name
    ]
    prior = [f for f in state.get("candidate_findings", []) if f["analyst_name"] == name]
    prior_revision = max((f["revision_count"] for f in prior), default=-1)

    result = run_analyst(
        spec=spec, run_id=state["run_id"], config=config,
        cleaned_artifacts=state.get("cleaned_artifacts", {}),
        analysis_period=state["analysis_period"], previous_period=state["previous_period"],
        trailing_baseline_periods=state["trailing_baseline_periods"],
        revision_feedback=feedback,
        invocation_tag=f"revision-{prior_revision + 1}",
    )
    revised_findings = []
    for f in result.findings:
        f = dict(f)
        f["revision_count"] = prior_revision + 1
        revised_findings.append(f)
    return {"candidate_findings": revised_findings, "step_count": 1}


def revision_fanin(state: CafeIntelligenceState) -> dict[str, Any]:
    return {}


def rank_node(state: CafeIntelligenceState) -> dict[str, Any]:
    limits = state["config"].app_settings.limits
    critic: dict = state["critic_results"]
    approved_ids = set(critic["approved_findings"])
    approved = [f for f in state.get("candidate_findings", []) if f["finding_id"] in approved_ids]
    final = rank_findings(approved, limits.max_final_findings)
    return {"final_findings": final, "step_count": 1}


def no_evidence_node(state: CafeIntelligenceState) -> dict[str, Any]:
    return {"final_findings": [], "run_status": "partial", "step_count": 1}
