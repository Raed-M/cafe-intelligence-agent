"""Cross-domain synthesis stage.

Covers the deterministic pooling/detection layer, the cost gate, placeholder
grounding, and the end-to-end property that actually motivated this stage:
a finding relating two analysts' metrics must be ACCEPTED by the critic. The
previous design made that impossible (see src/analysis/cross_domain.py), so
these are regression tests against re-introducing that catch-22.
"""
import json

import pytest

from src.analysis.cross_domain import (
    build_evidence_pool,
    detect_co_movements,
    has_substantive_co_movement,
    placeholder_metrics_from_pool,
    pool_key,
    validate_metric_refs,
)
from src.analysts.base import _substitute_claim_placeholders
from src.graph.cross_domain_nodes import cross_domain_synthesis_node
from src.validation.finding_critic import run_critic

PERIOD = ("2026-03-23T00:00:00+03:00", "2026-03-30T00:00:00+03:00")
VALID_PERIODS = {(PERIOD[0][:10], PERIOD[1][:10])}


def _artifact(path):
    return {"path": str(path), "media_type": "application/json", "sha256": "x",
            "schema_version": "1.0", "row_count": None, "byte_size": 10, "created_at": "now"}


def _finding(tmp_path, fid, analyst, title, claim, metrics, sources):
    """metrics: list of (result_key, value, unit)."""
    p = tmp_path / f"{analyst}_{fid}.json"
    p.write_text(json.dumps(
        {"findings": [{"metrics": {k: {"value": v} for k, v, _ in metrics}}]}), encoding="utf-8")
    art = _artifact(p)
    code = tmp_path / "code.json"
    code.write_text("{}", encoding="utf-8")
    return {
        "finding_id": fid, "analyst_name": analyst, "title": title, "claim": claim,
        "finding_type": "trend",
        "evidence": [{
            "metric_name": k, "value": v, "unit": u, "numerator": None, "denominator": None,
            "period_start": PERIOD[0], "period_end": PERIOD[1],
            "comparison_period_start": None, "comparison_period_end": None,
            "source_names": sources, "result_path": art["path"], "result_key": k,
        } for k, v, u in metrics],
        "source_names": sources,
        "code_artifact": _artifact(code), "result_artifact": art,
        "sample_size": 100, "coverage_notes": [], "assumptions": [], "confidence": 0.9,
        "business_impact_score": 0.5, "actionability_score": 0.5, "revision_count": 0,
        "execution_metadata": {},
    }


@pytest.fixture
def two_analyst_findings(tmp_path):
    """The real shape of the 2026-03-23 week: revenue down (sales), margin rate
    up (margin) -- two analysts, same period, opposite directions."""
    return [
        _finding(tmp_path, "F-sales-1", "sales", "Revenue Decline",
                 "Net revenue decreased.", [("rev_pct", -23.54, "%"), ("tx_pct", -23.64, "%")], ["pos"]),
        _finding(tmp_path, "F-margin-1", "margin", "Gross Profit",
                 "Gross margin rate held.",
                 [("margin_rate", 69.89, "%"), ("gross_profit_val", 31516.75, "SAR")], ["pos", "menu"]),
    ]


class _Cfg:
    class app_settings:
        class models:
            analyst = "fake-model"

    def __init__(self, root):
        self.artifact_root = root


def _state(findings, tmp_path):
    return {"candidate_findings": findings, "config": _Cfg(tmp_path / "artifacts"),
            "run_id": "t", "analysis_period": {"start": PERIOD[0], "end": PERIOD[1]}}


def _synth_returning(items):
    calls = []

    def _synth(system_prompt, context):
        calls.append(context)
        return {"items": items}

    return _synth, calls


# --- deterministic layer ----------------------------------------------------

def test_pool_spans_analysts_and_detects_divergent_co_movement(two_analyst_findings):
    pool = build_evidence_pool(two_analyst_findings)
    assert pool_key("sales", "rev_pct") in pool
    assert pool_key("margin", "margin_rate") in pool

    movements = detect_co_movements(pool)
    assert movements, "revenue down + margin up must be detected"
    assert movements[0]["divergent"], "divergent pairs must rank first"
    assert movements[0]["left_analyst"] != movements[0]["right_analyst"]


def test_same_analyst_metrics_never_pair(tmp_path):
    """Two metrics from one analyst are that analyst's own job, not this stage's."""
    one = [_finding(tmp_path, "F1", "sales", "t", "c",
                    [("rev_pct", -23.5, "%"), ("tx_pct", 40.0, "%")], ["pos"])]
    assert detect_co_movements(build_evidence_pool(one)) == []


def test_same_quantity_computed_twice_is_a_tautology(tmp_path):
    """Sales' revenue delta and margin's revenue delta are one fact, not a
    relationship -- flagged, demoted, and not worth an LLM call on its own."""
    findings = [
        _finding(tmp_path, "F1", "sales", "t", "c", [("revenue_pct", -23.54, "%")], ["pos"]),
        _finding(tmp_path, "F2", "margin", "t", "c", [("net_revenue_pct", -23.51, "%")], ["pos"]),
    ]
    pairs = detect_co_movements(build_evidence_pool(findings))
    assert pairs and pairs[0]["likely_same_quantity"]
    assert not has_substantive_co_movement(pairs)


def test_tautology_demoted_below_genuine_pair(tmp_path):
    findings = [
        _finding(tmp_path, "F1", "sales", "t", "c", [("revenue_pct", -23.54, "%")], ["pos"]),
        _finding(tmp_path, "F2", "margin", "t", "c", [("net_revenue_pct", -23.51, "%")], ["pos"]),
        _finding(tmp_path, "F3", "operations", "t", "c", [("waste_pct", 62.4, "%")], ["inventory"]),
    ]
    pairs = detect_co_movements(build_evidence_pool(findings))
    assert has_substantive_co_movement(pairs)
    assert not pairs[0]["likely_same_quantity"]
    assert pairs[-1]["likely_same_quantity"]


@pytest.mark.parametrize("refs,expected_error", [
    (["sales__rev_pct"], "fewer than 2"),
    (["sales__rev_pct", "sales__tx_pct"], "single analyst"),
    (["sales__rev_pct", "margin__nope"], "unknown pool key"),
])
def test_metric_refs_must_span_two_analysts(two_analyst_findings, refs, expected_error):
    pool = build_evidence_pool(two_analyst_findings)
    clean, err = validate_metric_refs(pool, refs)
    assert clean == [] and expected_error in err


def test_valid_metric_refs_accepted(two_analyst_findings):
    pool = build_evidence_pool(two_analyst_findings)
    clean, err = validate_metric_refs(pool, ["sales__rev_pct", "margin__margin_rate"])
    assert err is None and len(clean) == 2


# --- placeholders -----------------------------------------------------------

def test_abs_placeholder_avoids_double_negative(two_analyst_findings):
    """`fell <<k>>%` on a negative value reads "fell -23.54%"; the __abs variant
    lets the model write natural directional prose without typing a digit."""
    metrics = placeholder_metrics_from_pool(build_evidence_pool(two_analyst_findings))
    signed, bad = _substitute_claim_placeholders("revenue fell <<sales__rev_pct>>%", metrics)
    assert bad == [] and signed == "revenue fell -23.54%"
    absd, bad = _substitute_claim_placeholders("revenue fell <<sales__rev_pct__abs>>%", metrics)
    assert bad == [] and absd == "revenue fell 23.54%"


def test_unknown_placeholder_key_still_rejected(two_analyst_findings):
    metrics = placeholder_metrics_from_pool(build_evidence_pool(two_analyst_findings))
    _, bad = _substitute_claim_placeholders("<<sales__nope__abs>>", metrics)
    assert bad == ["sales__nope__abs"]


# --- the node ---------------------------------------------------------------

def test_no_cross_analyst_co_movement_spends_nothing(tmp_path):
    """The common case on a quiet week: silence, and zero LLM calls."""
    single = [_finding(tmp_path, "F1", "sales", "t", "c", [("rev_pct", -23.54, "%")], ["pos"])]
    synth, calls = _synth_returning([])
    out = cross_domain_synthesis_node(_state(single, tmp_path), synthesizer=synth)
    assert calls == []
    assert not out.get("candidate_findings")


def test_synthesis_grounds_numbers_and_keeps_origin_provenance(two_analyst_findings, tmp_path):
    synth, calls = _synth_returning([{
        "title": "Volume fell while margin rate rose",
        "claim": ("Net revenue fell <<sales__rev_pct__abs>>% in the same week that gross margin "
                  "rate reached <<margin__margin_rate>>%."),
        "finding_type": "cross_domain",
        "metric_refs": ["sales__rev_pct", "margin__margin_rate"],
        "assumptions": ["Single-week co-movement is weak evidence of mechanism."],
        "coverage_notes": [], "confidence": 0.7,
    }])
    out = cross_domain_synthesis_node(_state(two_analyst_findings, tmp_path), synthesizer=synth)

    assert len(calls) == 1, "exactly one LLM call per run"
    (xf,) = out["candidate_findings"]
    assert "<<" not in xf["claim"]
    assert "23.54" in xf["claim"] and "69.89" in xf["claim"]
    assert xf["analyst_name"] == "cross_domain"
    assert sorted(xf["source_names"]) == ["menu", "pos"]
    # each cited number must point at the artifact of the analyst that computed it
    assert {e["result_path"] for e in xf["evidence"]} == {
        two_analyst_findings[0]["result_artifact"]["path"],
        two_analyst_findings[1]["result_artifact"]["path"],
    }


def test_single_analyst_draft_is_dropped(two_analyst_findings, tmp_path):
    synth, _ = _synth_returning([{
        "title": "Not cross domain", "claim": "Revenue fell <<sales__rev_pct__abs>>%.",
        "finding_type": "cross_domain", "metric_refs": ["sales__rev_pct", "sales__tx_pct"],
        "assumptions": [], "coverage_notes": [], "confidence": 0.7,
    }])
    out = cross_domain_synthesis_node(_state(two_analyst_findings, tmp_path), synthesizer=synth)
    assert not out.get("candidate_findings")


def test_model_failure_is_additive_only(two_analyst_findings, tmp_path):
    """A synthesis failure must never cost the run its single-domain findings."""
    def _boom(system_prompt, context):
        raise RuntimeError("model exploded")

    out = cross_domain_synthesis_node(_state(two_analyst_findings, tmp_path), synthesizer=_boom)
    assert not out.get("candidate_findings")
    assert out["errors"]


# --- the property this stage exists for -------------------------------------

def test_critic_accepts_multi_analyst_finding(two_analyst_findings, tmp_path):
    """The regression that matters: under the old design this was rejected for
    citing a number the claiming analyst's own code did not compute."""
    synth, _ = _synth_returning([{
        "title": "Volume fell while margin rate rose",
        "claim": ("Net revenue fell <<sales__rev_pct__abs>>% while gross margin rate reached "
                  "<<margin__margin_rate>>%, consistent with a mix shift."),
        "finding_type": "cross_domain",
        "metric_refs": ["sales__rev_pct", "margin__margin_rate"],
        "assumptions": ["Single-week co-movement is weak evidence of mechanism."],
        "coverage_notes": [], "confidence": 0.7,
    }])
    (xf,) = cross_domain_synthesis_node(
        _state(two_analyst_findings, tmp_path), synthesizer=synth)["candidate_findings"]

    out = run_critic([xf], revision_round=0, max_revision_rounds=2, valid_periods=VALID_PERIODS)
    assert out["approved_findings"] == [xf["finding_id"]], out["notes"]


def test_fabricated_number_in_cross_domain_claim_still_rejected(two_analyst_findings, tmp_path):
    synth, _ = _synth_returning([{
        "title": "t", "claim": "Revenue fell <<sales__rev_pct__abs>>% and margin reached <<margin__margin_rate>>%.",
        "finding_type": "cross_domain", "metric_refs": ["sales__rev_pct", "margin__margin_rate"],
        "assumptions": [], "coverage_notes": [], "confidence": 0.7,
    }])
    (xf,) = cross_domain_synthesis_node(
        _state(two_analyst_findings, tmp_path), synthesizer=synth)["candidate_findings"]
    xf["claim"] = "Net revenue fell 99.99% while margin reached 12.34%."

    out = run_critic([xf], revision_round=2, max_revision_rounds=2, valid_periods=VALID_PERIODS)
    assert out["rejected_findings"] == [xf["finding_id"]]


def test_cross_domain_never_enters_the_revision_loop(two_analyst_findings, tmp_path):
    """Re-running synthesis on identical inputs cannot fix anything, so a bad
    cross-domain finding must be rejected terminally rather than queued."""
    synth, _ = _synth_returning([{
        "title": "t", "claim": "Revenue fell <<sales__rev_pct__abs>>% and margin reached <<margin__margin_rate>>%.",
        "finding_type": "cross_domain", "metric_refs": ["sales__rev_pct", "margin__margin_rate"],
        "assumptions": [], "coverage_notes": [], "confidence": 0.7,
    }])
    (xf,) = cross_domain_synthesis_node(
        _state(two_analyst_findings, tmp_path), synthesizer=synth)["candidate_findings"]
    xf["claim"] = "Net revenue fell 99.99%."

    out = run_critic([xf], revision_round=0, max_revision_rounds=2, valid_periods=VALID_PERIODS,
                     non_revisable_analysts={"cross_domain"})
    assert out["revision_requests"] == []
    assert out["rejected_findings"] == [xf["finding_id"]]
    assert any("no revision path" in n for n in out["notes"])
