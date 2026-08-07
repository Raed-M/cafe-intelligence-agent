"""Fault injection suite (plan section 28.6). Each test forces one specific
failure mode and asserts the system degrades honestly (partial/failed status,
disclosed warnings) rather than crashing or silently fabricating output.
"""
import json
import shutil
import uuid
from datetime import date
from pathlib import Path

import pytest

from src.cleaning.cleaner import clean_and_materialise
from src.config.preflight import run_preflight
from src.config.runtime_config import resolve_runtime_config
from src.graph.ingestion_subgraph import build_ingestion_subgraph
from src.tools.code_executor import CodeExecutionRequest, execute_python_code
from src.tools.tavily_search import run_local_search

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "qahwa_saihat"


def _cfg(data_dir: Path):
    return resolve_runtime_config(
        profile_path=data_dir / "cafe_profile.json", data_dir=data_dir,
        app_settings_path=ROOT / "config" / "app_settings.yaml",
        source_registry_path=ROOT / "config" / "source_registry.yaml",
        target_week=date(2026, 1, 5), artifact_root=ROOT / "outputs" / "artifacts",
    )


# 1. Corrupt inventory_weekly.xlsx -- inventory branch fails; six others continue.
def test_corrupt_excel_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    shutil.copytree(DATA_DIR, tmp_path / "data")
    (tmp_path / "data" / "inventory_weekly.xlsx").write_bytes(b"corrupt, not a real workbook")
    config = _cfg(tmp_path / "data")
    run_id = "fault_excel_" + uuid.uuid4().hex[:6]
    graph = build_ingestion_subgraph()
    out = graph.invoke({"run_id": run_id, "config": config})
    clean_out = clean_and_materialise({"run_id": run_id, "config": config, "source_results": out["source_results"]})
    assert clean_out["data_quality"]["sources_failed"] == ["inventory"]
    assert set(clean_out["data_quality"]["sources_successful"]) == {"pos", "menu", "traffic", "staff", "emails", "reviews"}


# 2. Empty reviews (valid empty JSON list) -- customer analyst unavailable, no fabrication.
def test_empty_reviews_produces_unavailable_not_fabricated(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    shutil.copytree(DATA_DIR, tmp_path / "data")
    (tmp_path / "data" / "customer_reviews.json").write_text("[]", encoding="utf-8")
    config = _cfg(tmp_path / "data")
    run_id = "fault_reviews_" + uuid.uuid4().hex[:6]
    graph = build_ingestion_subgraph()
    out = graph.invoke({"run_id": run_id, "config": config})
    clean_out = clean_and_materialise({"run_id": run_id, "config": config, "source_results": out["source_results"]})

    reviews_result = next(r for r in out["source_results"] if r["source_name"] == "reviews")
    assert reviews_result["status"] == "success"
    assert reviews_result["accepted_row_count"] == 0

    from src.analysts.base import run_analyst
    from src.analysts.customer import SPEC

    result = run_analyst(SPEC, run_id, config, clean_out["cleaned_artifacts"], config.analysis_period,
                          config.previous_period, config.trailing_baseline_periods)
    assert result.status == "unavailable"
    assert result.findings == []


# 3. Tavily forced error -- context degrades to unavailable, disclosed, no crash.
def test_tavily_outage_degrades_gracefully(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key-that-will-fail")

    class _BoomClient:
        def __init__(self, api_key):
            pass

        def search(self, **kwargs):
            raise ConnectionError("simulated Tavily outage")

    monkeypatch.setattr("tavily.TavilyClient", _BoomClient)
    hits, status, warnings = run_local_search(["events near Saihat during 2026-01-12 to 2026-01-19"])
    assert hits == []
    assert status == "unavailable"
    assert warnings


def test_tavily_missing_key_degrades_gracefully(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    hits, status, warnings = run_local_search(["any query"])
    assert hits == []
    assert status == "unavailable"
    assert "TAVILY_API_KEY" in warnings[0]


# 4. Generated code syntax error -- policy_violation, not a crash.
def test_code_syntax_error_handled_not_crashed(tmp_path):
    req = CodeExecutionRequest(
        code="def broken(:\n  pass", input_artifacts=[], expected_output_path="result.json",
        timeout_seconds=5, allowed_imports=[], max_output_bytes=100000,
    )
    result = execute_python_code(req, tmp_path / "code", attempt=1)
    assert result["status"] == "policy_violation"


# 5. Generated code timeout -- killed cleanly within the configured limit.
def test_code_timeout_handled(tmp_path):
    req = CodeExecutionRequest(
        code="import time\ntime.sleep(10)", input_artifacts=[], expected_output_path="result.json",
        timeout_seconds=1, allowed_imports=["time"], max_output_bytes=100000,
    )
    result = execute_python_code(req, tmp_path / "code", attempt=1)
    assert result["status"] == "timeout"
    assert result["elapsed_seconds"] <= 2


# 6. Remove one optional source (inventory) -- margin/operations analysts still run.
def test_missing_optional_source_does_not_block_analyst(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    shutil.copytree(DATA_DIR, tmp_path / "data")
    (tmp_path / "data" / "inventory_weekly.xlsx").unlink()
    config = _cfg(tmp_path / "data")
    run_id = "fault_optional_" + uuid.uuid4().hex[:6]
    graph = build_ingestion_subgraph()
    out = graph.invoke({"run_id": run_id, "config": config})
    clean_out = clean_and_materialise({"run_id": run_id, "config": config, "source_results": out["source_results"]})
    assert "inventory" not in clean_out["cleaned_artifacts"]

    from src.graph.analysis_subgraph import ANALYST_SPECS
    margin_spec = ANALYST_SPECS["margin"]
    assert all(a in clean_out["cleaned_artifacts"] for a in margin_spec.required_artifacts), (
        "margin's hard requirements (pos, menu) must still be satisfiable when only "
        "the optional inventory source is missing"
    )


# 7. Cost/step cap exceeded -- controlled abort, no further LLM calls.
def test_cost_cap_triggers_controlled_abort():
    from src.graph.setup_nodes import guard_limits

    config = _cfg(DATA_DIR)
    state = {"config": config, "step_count": 0, "cost_usd": config.app_settings.limits.max_cost_usd + 1.0}
    result = guard_limits(state)
    assert result["run_status"] == "failed"
    assert result["errors"][0]["reason"] == "step/cost limit exceeded"


def test_step_cap_triggers_controlled_abort():
    from src.graph.setup_nodes import guard_limits

    config = _cfg(DATA_DIR)
    state = {"config": config, "step_count": config.app_settings.limits.max_graph_steps + 1, "cost_usd": 0.0}
    result = guard_limits(state)
    assert result["run_status"] == "failed"


# 8. PDF renderer failure -- HTML/summary remain intact, warning recorded.
def test_pdf_failure_keeps_html_and_summary(tmp_path, monkeypatch):
    import src.reporting.report_generator as rg

    config = _cfg(DATA_DIR)
    run_id = "fault_pdf_" + uuid.uuid4().hex[:6]

    # Simulate a Playwright/Chromium failure (e.g. `playwright install
    # chromium` never having been run). The renderer runs out-of-process (see
    # src/reporting/pdf_render.py -- blockbuster under `langgraph dev` blocks
    # Playwright's sync API in-process), so the fault is injected at the
    # subprocess boundary: patching a fake playwright module into *this*
    # interpreter would not reach the child at all.
    import subprocess

    def _failing_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0] if args else [], returncode=1,
            stdout="", stderr="RuntimeError: simulated renderer crash",
        )

    monkeypatch.setattr(subprocess, "run", _failing_run)

    state = {
        "config": config, "run_id": run_id, "final_findings": [], "content_ideas": [],
        "data_quality": {"sources_successful": ["pos"], "sources_partial": [], "sources_failed": [],
                          "source_summaries": [], "critical_dependencies_missing": [], "total_rows_in": 0,
                          "total_rows_dropped": 0, "total_rows_repaired": 0, "warnings": []},
        "context_bundle": {"evidence": [], "posting_windows": [], "search_status": "unavailable", "warnings": []},
        "analysis_period": config.analysis_period, "recommendation_period": config.recommendation_period,
        "run_status": "partial", "critic_results": {}, "source_results": [], "step_count": 0, "cost_usd": 0.0,
    }
    result = rg.generate_report(state)
    report = result["report"]
    assert Path(report["html_path"]).exists()
    assert report["pdf_path"] is None
    assert report["pdf_warning"] is not None
    assert report["whatsapp_summary"]
