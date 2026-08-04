import json
from pathlib import Path

from src.validation.finding_critic import run_critic
from src.validation.finding_ranker import rank_findings


def _write_result(tmp_path, obj, name="result.json"):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return {
        "path": str(p), "media_type": "application/json", "sha256": "x",
        "schema_version": "1.0", "row_count": None, "byte_size": 10, "created_at": "now",
    }


def _finding(**overrides):
    base = {
        "finding_id": "F1", "analyst_name": "sales", "title": "Revenue up",
        "claim": "Net revenue increased.", "finding_type": "trend",
        "evidence": [{
            "metric_name": "net_revenue", "value": 1000, "unit": "SAR",
            "numerator": None, "denominator": None,
            "period_start": "2026-01-05", "period_end": "2026-01-12",
            "comparison_period_start": None, "comparison_period_end": None,
            "source_names": ["pos"], "result_path": "x", "result_key": "net_revenue",
        }],
        "source_names": ["pos"],
        "code_artifact": {"path": "c.py", "media_type": "text/x-python", "sha256": "x",
                           "schema_version": "1.0", "row_count": None, "byte_size": 1, "created_at": "now"},
        "result_artifact": None,
        "sample_size": 100, "coverage_notes": [], "assumptions": [], "confidence": 0.9,
        "business_impact_score": 0.7, "actionability_score": 0.6, "revision_count": 0,
        "execution_metadata": {},
    }
    base.update(overrides)
    return base


def test_valid_finding_approved(tmp_path):
    result_ref = _write_result(tmp_path, {"findings": [{"metrics": {"net_revenue": {"value": 1000}}}]})
    finding = _finding(result_artifact=result_ref)
    out = run_critic([finding], revision_round=0, max_revision_rounds=2)
    assert out["approved_findings"] == ["F1"]
    assert out["total_rejections"] == 0


def test_unresolved_metric_key_gets_revision_request_before_cap(tmp_path):
    result_ref = _write_result(tmp_path, {"findings": [{"metrics": {}}]})  # net_revenue missing
    finding = _finding(result_artifact=result_ref)
    out = run_critic([finding], revision_round=0, max_revision_rounds=2)
    assert out["approved_findings"] == []
    assert len(out["revision_requests"]) == 1
    assert out["revision_requests"][0]["analyst_name"] == "sales"
    assert out["total_rejections"] == 0


def test_unresolved_metric_key_rejected_after_cap(tmp_path):
    result_ref = _write_result(tmp_path, {"findings": [{"metrics": {}}]})
    finding = _finding(result_artifact=result_ref)
    out = run_critic([finding], revision_round=2, max_revision_rounds=2)
    assert out["approved_findings"] == []
    assert out["rejected_findings"] == ["F1"]
    assert out["total_rejections"] == 1
    assert "F1" in out["removed_after_cap"]


def test_causal_claim_with_single_source_flagged(tmp_path):
    result_ref = _write_result(tmp_path, {"findings": [{"metrics": {"net_revenue": {"value": 1000}}}]})
    finding = _finding(
        result_artifact=result_ref, claim="Revenue fell because of fewer customers.", source_names=["pos"],
    )
    out = run_critic([finding], revision_round=0, max_revision_rounds=2)
    assert out["approved_findings"] == []
    assert out["revision_requests"]


def test_item_level_margin_claim_without_bom_flagged(tmp_path):
    result_ref = _write_result(tmp_path, {"findings": [{"metrics": {"net_revenue": {"value": 1000}}}]})
    finding = _finding(
        analyst_name="margin",
        claim="Per-drink cost rose due to milk price increase.",
        result_artifact=result_ref,
        assumptions=[],
    )
    out = run_critic([finding], revision_round=0, max_revision_rounds=2)
    assert out["approved_findings"] == []


def test_item_level_margin_claim_labelled_estimate_passes(tmp_path):
    result_ref = _write_result(tmp_path, {"findings": [{"metrics": {"net_revenue": {"value": 1000}}}]})
    finding = _finding(
        analyst_name="margin",
        claim="Per-drink cost estimate rose given milk price increase.",
        result_artifact=result_ref,
        assumptions=["This is an estimate assuming standard milk usage per drink."],
    )
    out = run_critic([finding], revision_round=0, max_revision_rounds=2)
    assert out["approved_findings"] == ["F1"]


def test_duplicate_titles_deduped(tmp_path):
    result_ref = _write_result(tmp_path, {"findings": [{"metrics": {"net_revenue": {"value": 1000}}}]})
    f1 = _finding(finding_id="F1", title="Revenue up", result_artifact=result_ref)
    f2 = _finding(finding_id="F2", title="Revenue up", result_artifact=result_ref)
    out = run_critic([f1, f2], revision_round=0, max_revision_rounds=2)
    assert out["approved_findings"] == ["F1"]
    assert out["rejected_findings"] == ["F2"]


def test_ranker_caps_at_max_final_findings(tmp_path):
    result_ref = _write_result(tmp_path, {"findings": [{"metrics": {"net_revenue": {"value": 1000}}}]})
    findings = [
        _finding(finding_id=f"F{i}", title=f"finding {i}", result_artifact=result_ref, confidence=0.5 + i * 0.05)
        for i in range(8)
    ]
    ranked = rank_findings(findings, max_final_findings=5)
    assert len(ranked) == 5
    # Higher-confidence findings (later in the loop) should be preferred.
    assert ranked[0]["finding_id"] == "F7"
