"""Grounding / hallucination injection tests (plan section 28.5). Each test
injects one specific defect into a finding or content idea and asserts the
appropriate validation layer rejects it, and that the critic's rejection
counter is non-zero somewhere in the suite (M27).
"""
import json

from src.validation.content_validator import validate_content_ideas
from src.validation.finding_critic import run_critic

VALID_PERIODS = {("2026-01-05", "2026-01-12"), ("2025-12-29", "2026-01-05")}


def _result_ref(tmp_path, obj, name="result.json"):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return {"path": str(p), "media_type": "application/json", "sha256": "x", "schema_version": "1.0",
            "row_count": None, "byte_size": 10, "created_at": "now"}


def _finding(**overrides):
    base = {
        "finding_id": "F1", "analyst_name": "sales", "title": "Revenue changed",
        "claim": "Net revenue increased.", "finding_type": "trend",
        "evidence": [{
            "metric_name": "net_revenue", "value": 1000, "unit": "SAR", "numerator": None, "denominator": None,
            "period_start": "2026-01-05", "period_end": "2026-01-12", "comparison_period_start": None,
            "comparison_period_end": None, "source_names": ["pos"], "result_path": "x", "result_key": "net_revenue",
        }],
        "source_names": ["pos"],
        "code_artifact": {"path": "c.py", "media_type": "text/x-python", "sha256": "x", "schema_version": "1.0",
                           "row_count": None, "byte_size": 1, "created_at": "now"},
        "result_artifact": None, "sample_size": 100, "coverage_notes": [], "assumptions": [], "confidence": 0.9,
        "business_impact_score": 0.7, "actionability_score": 0.6, "revision_count": 0, "execution_metadata": {},
    }
    base.update(overrides)
    return base


def test_false_numerical_finding_rejected_by_critic(tmp_path):
    """A finding claiming a number that was never actually computed (its
    result_key doesn't exist in the stored result JSON) must be rejected."""
    result_ref = _result_ref(tmp_path, {"findings": [{"metrics": {}}]})  # net_revenue absent
    finding = _finding(result_artifact=result_ref)
    out = run_critic([finding], revision_round=2, max_revision_rounds=2)
    assert out["approved_findings"] == []
    assert out["total_rejections"] > 0


def test_correct_number_wrong_period_rejected_by_critic(tmp_path):
    """The number is real and resolvable, but the claimed period doesn't match
    any period this run actually analysed -- must be rejected, not approved
    on the technicality that the number itself resolves."""
    result_ref = _result_ref(tmp_path, {"findings": [{"metrics": {"net_revenue": {"value": 1000}}}]})
    finding = _finding(result_artifact=result_ref)
    finding["evidence"][0]["period_start"] = "2025-06-01"
    finding["evidence"][0]["period_end"] = "2025-06-08"
    out = run_critic([finding], revision_round=2, max_revision_rounds=2, valid_periods=VALID_PERIODS)
    assert out["approved_findings"] == []
    assert out["total_rejections"] > 0


def test_item_level_milk_cost_without_bom_rejected_by_critic(tmp_path):
    result_ref = _result_ref(tmp_path, {"findings": [{"metrics": {"net_revenue": {"value": 1000}}}]})
    finding = _finding(
        analyst_name="margin", result_artifact=result_ref,
        claim="Per-drink cost rose 0.40 SAR due to the milk price increase.",
        assumptions=[],
    )
    out = run_critic([finding], revision_round=2, max_revision_rounds=2)
    assert out["approved_findings"] == []


def test_content_citing_rejected_finding_fails_validation(tmp_path):
    """final_findings only contains critic-approved findings; an idea citing a
    finding_id that isn't in that list (e.g. one the critic actually
    rejected) must fail closed."""
    context_bundle = {
        "recommendation_period_start": "x", "recommendation_period_end": "y", "search_queries": [],
        "evidence": [{"context_id": "ev1", "kind": "event", "title": "t", "date_start": None, "date_end": None,
                       "location": None, "source": "tavily", "source_url_or_artifact": None, "retrieved_at": "n", "summary": ""},
                     {"context_id": "cal1", "kind": "calendar", "title": "t", "date_start": None, "date_end": None,
                       "location": None, "source": "hijridate", "source_url_or_artifact": None, "retrieved_at": "n", "summary": ""}],
        "posting_windows": [{"window_id": "pw1", "post_date": "2026-01-13", "start_time_local": "18:00",
                              "end_time_local": "19:00", "busy_metric_keys": [], "demand_score": 1.0,
                              "prayer_relation": None, "event_context_ids": [], "rationale": ""}],
        "search_status": "success", "warnings": [],
    }
    approved_finding = _finding(finding_id="F-approved")
    idea = {
        "idea_id": "I1", "hook_ar": "a", "hook_en": "b", "format": "reel", "product_sku": "ICE-001",
        "product_name_ar": "x", "product_name_en": "y", "finding_id": "F-rejected-not-in-final-list",
        "cited_metric_keys": ["net_revenue"], "local_context_ids": ["ev1"], "calendar_context_ids": ["cal1"],
        "posting_window_id": "pw1", "timing_metric_keys": [], "rationale_ar": "r", "rationale_en": "r",
        "post_date": "2026-01-13", "post_time_local": "18:30", "timing_reason": "t", "inventory_suitability": "unknown",
    }
    out = validate_content_ideas(
        [idea, dict(idea, idea_id="I2"), dict(idea, idea_id="I3")],
        [approved_finding], context_bundle, {"ICE-001"}, {"default": "07:00-23:00"},
    )
    assert not out["valid"]
    all_reasons = [reason for reasons in out["issues"].values() for reason in reasons]
    assert any("not a critic-approved finding" in reason for reason in all_reasons)


def test_nonexistent_event_context_id_rejected():
    context_bundle = {
        "recommendation_period_start": "x", "recommendation_period_end": "y", "search_queries": [],
        "evidence": [{"context_id": "cal1", "kind": "calendar", "title": "t", "date_start": None, "date_end": None,
                       "location": None, "source": "hijridate", "source_url_or_artifact": None, "retrieved_at": "n", "summary": ""}],
        "posting_windows": [{"window_id": "pw1", "post_date": "2026-01-13", "start_time_local": "18:00",
                              "end_time_local": "19:00", "busy_metric_keys": [], "demand_score": 1.0,
                              "prayer_relation": None, "event_context_ids": [], "rationale": ""}],
        "search_status": "success", "warnings": [],
    }
    finding = _finding(finding_id="F1")
    idea = {
        "idea_id": "I1", "hook_ar": "a", "hook_en": "b", "format": "reel", "product_sku": "ICE-001",
        "product_name_ar": "x", "product_name_en": "y", "finding_id": "F1", "cited_metric_keys": ["net_revenue"],
        "local_context_ids": ["nonexistent-context-id"], "calendar_context_ids": ["cal1"],
        "posting_window_id": "pw1", "timing_metric_keys": [], "rationale_ar": "r", "rationale_en": "r",
        "post_date": "2026-01-13", "post_time_local": "18:30", "timing_reason": "t", "inventory_suitability": "unknown",
    }
    out = validate_content_ideas([idea, dict(idea, idea_id="I2"), dict(idea, idea_id="I3")],
                                  [finding], context_bundle, {"ICE-001"}, {"default": "07:00-23:00"})
    assert not out["valid"]


def test_closed_time_recommendation_rejected():
    context_bundle = {
        "recommendation_period_start": "x", "recommendation_period_end": "y", "search_queries": [],
        "evidence": [{"context_id": "ev1", "kind": "event", "title": "t", "date_start": None, "date_end": None,
                       "location": None, "source": "tavily", "source_url_or_artifact": None, "retrieved_at": "n", "summary": ""},
                     {"context_id": "cal1", "kind": "calendar", "title": "t", "date_start": None, "date_end": None,
                       "location": None, "source": "hijridate", "source_url_or_artifact": None, "retrieved_at": "n", "summary": ""}],
        "posting_windows": [{"window_id": "pw1", "post_date": "2026-01-13", "start_time_local": "03:00",
                              "end_time_local": "04:00", "busy_metric_keys": [], "demand_score": 1.0,
                              "prayer_relation": None, "event_context_ids": [], "rationale": ""}],
        "search_status": "success", "warnings": [],
    }
    finding = _finding(finding_id="F1")
    idea = {
        "idea_id": "I1", "hook_ar": "a", "hook_en": "b", "format": "reel", "product_sku": "ICE-001",
        "product_name_ar": "x", "product_name_en": "y", "finding_id": "F1", "cited_metric_keys": ["net_revenue"],
        "local_context_ids": ["ev1"], "calendar_context_ids": ["cal1"], "posting_window_id": "pw1",
        "timing_metric_keys": [], "rationale_ar": "r", "rationale_en": "r", "post_date": "2026-01-13",
        "post_time_local": "03:00", "timing_reason": "t", "inventory_suitability": "unknown",
    }
    out = validate_content_ideas([idea, dict(idea, idea_id="I2"), dict(idea, idea_id="I3")],
                                  [finding], context_bundle, {"ICE-001"}, {"default": "07:00-23:00"})
    assert not out["valid"]
    assert any("outside opening hours" in r for reasons in out["issues"].values() for r in reasons)


def test_critic_rejection_count_nonzero_across_suite(tmp_path):
    """M27: at least one scenario in this suite produces a non-zero critic
    rejection count."""
    result_ref = _result_ref(tmp_path, {"findings": [{"metrics": {}}]})
    finding = _finding(result_artifact=result_ref)
    out = run_critic([finding], revision_round=2, max_revision_rounds=2)
    assert out["total_rejections"] > 0
