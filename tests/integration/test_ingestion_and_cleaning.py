import shutil
import uuid
from datetime import date
from pathlib import Path

import pytest

from src.cleaning.cleaner import clean_and_materialise
from src.config.runtime_config import resolve_runtime_config
from src.graph.ingestion_subgraph import build_ingestion_subgraph

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "qahwa_saihat"


def _run_pipeline(data_dir: Path, run_id: str):
    cfg = resolve_runtime_config(
        profile_path=data_dir / "cafe_profile.json",
        data_dir=data_dir,
        app_settings_path=ROOT / "config" / "app_settings.yaml",
        source_registry_path=ROOT / "config" / "source_registry.yaml",
        target_week=date(2026, 1, 5),
        artifact_root=ROOT / "outputs" / "artifacts",
    )
    graph = build_ingestion_subgraph()
    ingest_out = graph.invoke({"run_id": run_id, "config": cfg})
    clean_out = clean_and_materialise({
        "run_id": run_id, "config": cfg, "source_results": ingest_out["source_results"],
    })
    return ingest_out, clean_out


def test_all_seven_sources_ingest_successfully(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    run_id = "test_" + uuid.uuid4().hex[:8]
    ingest_out, clean_out = _run_pipeline(DATA_DIR, run_id)
    statuses = {r["source_name"]: r["status"] for r in ingest_out["source_results"]}
    assert statuses == {
        "pos": "success", "menu": "success", "traffic": "success",
        "staff": "success", "inventory": "success", "emails": "success",
        "reviews": "success",
    }
    dq = clean_out["data_quality"]
    assert set(dq["sources_successful"]) == set(statuses.keys())
    assert dq["sources_failed"] == []


def test_double_swipe_rate_matches_documented_one_percent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    run_id = "test_" + uuid.uuid4().hex[:8]
    _, clean_out = _run_pipeline(DATA_DIR, run_id)
    pos_quality = next(s for s in clean_out["data_quality"]["source_summaries"] if s["source_name"] == "pos")
    rate = pos_quality["issue_counts"]["double_swipe_transactions"] / 38456
    assert 0.005 < rate < 0.02  # ~1% documented


def test_dead_sensor_days_rediscovered(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    run_id = "test_" + uuid.uuid4().hex[:8]
    _, clean_out = _run_pipeline(DATA_DIR, run_id)
    traffic_quality = next(s for s in clean_out["data_quality"]["source_summaries"] if s["source_name"] == "traffic")
    dead_dates = {p["date"] for p in traffic_quality["excluded_periods"]}
    assert dead_dates == {"2026-06-08", "2026-06-09", "2026-06-10"}


def test_unknown_waste_never_treated_as_zero(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    run_id = "test_" + uuid.uuid4().hex[:8]
    _, clean_out = _run_pipeline(DATA_DIR, run_id)
    inv_quality = next(s for s in clean_out["data_quality"]["source_summaries"] if s["source_name"] == "inventory")
    assert inv_quality["null_counts"]["units_wasted"] == 5


def test_corrupt_inventory_does_not_stop_other_branches(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    shutil.copytree(DATA_DIR, tmp_path / "data")
    (tmp_path / "data" / "inventory_weekly.xlsx").write_bytes(b"not an excel file")
    run_id = "test_" + uuid.uuid4().hex[:8]
    ingest_out, clean_out = _run_pipeline(tmp_path / "data", run_id)
    statuses = {r["source_name"]: r["status"] for r in ingest_out["source_results"]}
    assert statuses["inventory"] == "failed"
    assert statuses["pos"] == "success"
    assert "inventory" in clean_out["data_quality"]["sources_failed"]
    assert set(clean_out["data_quality"]["sources_successful"]) == {
        "pos", "menu", "traffic", "staff", "emails", "reviews",
    }
