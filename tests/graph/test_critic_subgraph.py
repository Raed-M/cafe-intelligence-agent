"""Proves the critic revision loop: a bad finding from one analyst triggers a
*targeted* rerun of only that analyst, the loop terminates within the
configured cap, and the critic rejection count is non-zero when a finding
never becomes resolvable.
"""
import json
from datetime import date
from pathlib import Path

import pytest

from src.config.runtime_config import resolve_runtime_config
from src.graph.critic_subgraph import build_critic_subgraph

ROOT = Path(__file__).resolve().parents[2]


def _result_ref(tmp_path, obj, name):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return {
        "path": str(p), "media_type": "application/json", "sha256": "x",
        "schema_version": "1.0", "row_count": None, "byte_size": 10, "created_at": "now",
    }


def _finding(finding_id, analyst_name, result_artifact, metric_key="net_revenue",
             claim="Net revenue for the week of 2026-01-05 was SAR 1,000, resolved directly from pos.result_key=net_revenue."):
    return {
        "finding_id": finding_id, "analyst_name": analyst_name, "title": f"title-{finding_id}",
        "claim": claim, "finding_type": "trend",
        "evidence": [{
            "metric_name": metric_key, "value": 1000, "unit": "SAR", "numerator": None, "denominator": None,
            "period_start": "2026-01-05", "period_end": "2026-01-12",
            "comparison_period_start": None, "comparison_period_end": None,
            "source_names": ["pos"], "result_path": result_artifact["path"], "result_key": metric_key,
        }],
        "source_names": ["pos"],
        "code_artifact": {"path": "c.py", "media_type": "text/x-python", "sha256": "x", "schema_version": "1.0",
                           "row_count": None, "byte_size": 1, "created_at": "now"},
        "result_artifact": result_artifact,
        "sample_size": 10, "coverage_notes": [], "assumptions": [], "confidence": 0.8,
        "business_impact_score": 0.6, "actionability_score": 0.6, "revision_count": 0,
        "execution_metadata": {},
    }


def _cfg():
    return resolve_runtime_config(
        profile_path=ROOT / "data" / "qahwa_saihat" / "cafe_profile.json",
        data_dir=ROOT / "data" / "qahwa_saihat",
        app_settings_path=ROOT / "config" / "app_settings.yaml",
        source_registry_path=ROOT / "config" / "source_registry.yaml",
        target_week=date(2026, 1, 5),
        artifact_root=ROOT / "outputs" / "artifacts",
    )


def test_good_finding_reaches_rank_without_revision(monkeypatch, tmp_path):
    config = _cfg()
    good_result = _result_ref(tmp_path, {"findings": [{"metrics": {"net_revenue": {"value": 1000}}}]}, "good.json")
    graph = build_critic_subgraph()
    out = graph.invoke({
        "config": config, "run_id": "critic_test_good",
        "candidate_findings": [_finding("F1", "sales", good_result)],
        "critic_round": 0,
        "cleaned_artifacts": {}, "analysis_period": config.analysis_period,
        "previous_period": config.previous_period, "trailing_baseline_periods": config.trailing_baseline_periods,
    })
    assert [f["finding_id"] for f in out["final_findings"]] == ["F1"]
    assert out["critic_results"]["total_rejections"] == 0


def test_hallucinated_finding_triggers_targeted_revision_then_rejected_after_cap(monkeypatch, tmp_path):
    """One analyst (margin) produces an unresolvable claim every time; the loop
    must only ever rerun `margin` (never `sales`), and after exhausting the
    revision cap the finding is rejected (non-zero rejection count) rather than
    looping forever."""
    config = _cfg()
    bad_result = _result_ref(tmp_path, {"findings": [{"metrics": {}}]}, "bad.json")  # net_revenue unresolvable
    good_result = _result_ref(tmp_path, {"findings": [{"metrics": {"net_revenue": {"value": 500}}}]}, "good2.json")

    call_log = []

    def fake_run_analyst(spec, run_id, config, cleaned_artifacts, analysis_period, previous_period,
                          trailing_baseline_periods, code_generator=None, repair_generator=None,
                          revision_feedback=None, invocation_tag="initial"):
        call_log.append(spec.name)
        from src.analysts.base import AnalystRunResult
        # Margin analyst always regenerates the same unresolvable finding.
        return AnalystRunResult(
            analyst_name=spec.name,
            findings=[_finding("F-margin-new", "margin", bad_result)],
            status="success", attempts=1, notes=[],
        )

    monkeypatch.setattr("src.graph.routers.run_analyst", fake_run_analyst)

    graph = build_critic_subgraph()
    out = graph.invoke({
        "config": config, "run_id": "critic_test_bad",
        "candidate_findings": [
            _finding("F1", "sales", good_result),
            _finding("F2", "margin", bad_result),
        ],
        "critic_round": 0,
        "cleaned_artifacts": {}, "analysis_period": config.analysis_period,
        "previous_period": config.previous_period, "trailing_baseline_periods": config.trailing_baseline_periods,
    }, config={"recursion_limit": 50})

    # Only margin was ever rerun -- sales (which was fine) is never touched.
    assert set(call_log) == {"margin"}
    assert out["critic_results"]["total_rejections"] > 0
    approved_titles = {f["finding_id"] for f in out.get("final_findings", [])}
    assert "F2" not in approved_titles and "F-margin-new" not in approved_titles
