"""Exercises the restricted executor + repair loop through the real analyst
framework, using an injected (non-network) code generator so the test proves
the self-correction mechanics deterministically: attempt 1 is a deliberately
broken program, attempt 2 is the repaired program that succeeds.
"""
import uuid
from pathlib import Path

from src.analysts.base import AnalystSpec, run_analyst
from src.config.runtime_config import resolve_runtime_config
from datetime import date

ROOT = Path(__file__).resolve().parents[2]

GOOD_CODE = '''
import json, os
import pandas as pd
meta = json.load(open(os.environ["ANALYST_INPUTS_JSON"]))
df = pd.read_parquet(meta["inputs"]["pos"])
net_revenue = float(df["line_total_sar"].sum())
result = {
    "status": "success",
    "findings": [{
        "title": "Net revenue",
        "claim": f"Net revenue was {net_revenue:.2f} SAR",
        "finding_type": "metric",
        "metrics": {"net_revenue": {"value": net_revenue, "unit": "SAR",
                                     "numerator": None, "denominator": None,
                                     "period_start": "2026-01-05", "period_end": "2026-01-12"}},
        "source_names": ["pos"],
        "sample_size": len(df),
        "coverage_notes": [],
        "assumptions": [],
        "confidence": 0.9,
    }],
}
json.dump(result, open(meta["output_path"], "w"))
'''

BROKEN_CODE = "this is not valid python !!!"


def _fake_generator_sequence(codes):
    it = iter(codes)

    def gen(system_prompt, context):
        return next(it)

    return gen


def _cfg():
    return resolve_runtime_config(
        profile_path=ROOT / "data" / "qahwa_saihat" / "cafe_profile.json",
        data_dir=ROOT / "data" / "qahwa_saihat",
        app_settings_path=ROOT / "config" / "app_settings.yaml",
        source_registry_path=ROOT / "config" / "source_registry.yaml",
        target_week=date(2026, 1, 5),
        artifact_root=ROOT / "outputs" / "artifacts",
    )


def _pos_artifact(config, run_id):
    from src.graph.ingestion_subgraph import build_ingestion_subgraph
    from src.cleaning.cleaner import clean_and_materialise

    graph = build_ingestion_subgraph()
    ingest_out = graph.invoke({"run_id": run_id, "config": config})
    clean_out = clean_and_materialise({"run_id": run_id, "config": config, "source_results": ingest_out["source_results"]})
    return clean_out["cleaned_artifacts"]


def test_analyst_succeeds_on_first_attempt(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = _cfg()
    run_id = "analyst_ok_" + uuid.uuid4().hex[:8]
    cleaned = _pos_artifact(config, run_id)

    spec = AnalystSpec(name="sales", prompt_path=ROOT / "prompts" / "analysts" / "sales.md", required_artifacts=["pos", "menu"])
    result = run_analyst(
        spec, run_id, config, cleaned, config.analysis_period, config.previous_period,
        config.trailing_baseline_periods, code_generator=_fake_generator_sequence([GOOD_CODE]),
    )
    assert result.status == "success"
    assert result.attempts == 1
    assert len(result.findings) == 1
    assert result.findings[0].get("evidence")[0]["metric_name"] == "net_revenue"


def test_analyst_self_corrects_after_broken_first_attempt(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = _cfg()
    run_id = "analyst_repair_" + uuid.uuid4().hex[:8]
    cleaned = _pos_artifact(config, run_id)

    spec = AnalystSpec(name="sales", prompt_path=ROOT / "prompts" / "analysts" / "sales.md", required_artifacts=["pos", "menu"])
    result = run_analyst(
        spec, run_id, config, cleaned, config.analysis_period, config.previous_period,
        config.trailing_baseline_periods,
        code_generator=_fake_generator_sequence([BROKEN_CODE]),
        repair_generator=lambda prev, stderr, ctx: GOOD_CODE,
    )
    assert result.status == "success"
    assert result.attempts == 2
    assert len(result.notes) == 1
    assert "attempt 1 failed" in result.notes[0]


def test_analyst_fails_closed_after_exhausting_attempts(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = _cfg()
    run_id = "analyst_fail_" + uuid.uuid4().hex[:8]
    cleaned = _pos_artifact(config, run_id)

    spec = AnalystSpec(name="sales", prompt_path=ROOT / "prompts" / "analysts" / "sales.md", required_artifacts=["pos", "menu"])
    result = run_analyst(
        spec, run_id, config, cleaned, config.analysis_period, config.previous_period,
        config.trailing_baseline_periods,
        code_generator=_fake_generator_sequence([BROKEN_CODE]),
        repair_generator=lambda prev, stderr, ctx: BROKEN_CODE,
    )
    assert result.status == "failed"
    assert result.findings == []
    assert result.attempts == config.app_settings.limits.analyst_code_attempts


def test_missing_required_artifact_is_unavailable_not_fabricated(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = _cfg()
    spec = AnalystSpec(name="customer", prompt_path=ROOT / "prompts" / "analysts" / "customer.md", required_artifacts=["reviews"])
    result = run_analyst(
        spec, "run_missing", config, {}, config.analysis_period, config.previous_period,
        config.trailing_baseline_periods,
    )
    assert result.status == "unavailable"
    assert result.findings == []
