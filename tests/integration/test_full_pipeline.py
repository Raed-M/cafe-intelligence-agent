"""End-to-end proof that the assembled main graph runs the full pipeline
described in implementation_plan_final.md section 6.1 against the real
Qahwa Saihat dataset: ingestion -> cleaning -> analysis -> critic -> context
-> content -> validation -> report -> HITL pause -> approve -> deliver ->
persist. All LLM call sites are monkeypatched to a single fake chat model that
inspects the system prompt to decide what to return, so this test proves the
graph *wiring* and every deterministic contract without spending real API
calls or depending on a live key.
"""
import json
import uuid
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config.runtime_config import resolve_runtime_config
from src.graph.main_graph import build_main_graph

ROOT = Path(__file__).resolve().parents[2]


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChatModel:
    def invoke(self, messages):
        system_prompt = messages[0][1] if messages and isinstance(messages[0], tuple) else ""
        user_prompt = messages[1][1] if len(messages) > 1 else ""

        if "content strategist" in system_prompt.lower():
            return _FakeMessage(self._fake_content_ideas(user_prompt))
        if "extract structured business facts" in system_prompt.lower():
            return _FakeMessage(self._fake_email_extraction())
        # Default: analyst code-generation / repair prompt.
        return _FakeMessage(self._fake_analyst_code(user_prompt))

    @staticmethod
    def _fake_analyst_code(user_prompt: str) -> str:
        ctx = json.loads(user_prompt.split("Context (artifact paths, schemas, periods):\n", 1)[-1].split("\n\nWrite one complete")[0]) \
            if "Context (artifact paths, schemas, periods):" in user_prompt else {}
        inputs = ctx.get("input_artifacts", {})
        first_artifact = next(iter(inputs), None)
        period = ctx.get("analysis_period", {})
        period_start = period.get("start", "")
        period_end = period.get("end", "")
        return f'''
import json, os
import pandas as pd
meta = json.load(open(os.environ["ANALYST_INPUTS_JSON"]))
inputs = meta["inputs"]
findings = []
if {first_artifact!r} and {first_artifact!r} in inputs:
    df = pd.read_parquet(inputs[{first_artifact!r}])
    findings.append({{
        "title": "Row count summary",
        "claim": f"Observed {{len(df)}} rows in {first_artifact}.",
        "finding_type": "descriptive",
        "metrics": {{"row_count": {{"value": int(len(df)), "unit": "rows", "numerator": None,
                                    "denominator": None, "period_start": {period_start!r}, "period_end": {period_end!r}}}}},
        "source_names": [{first_artifact!r}] if {first_artifact!r} else [],
        "sample_size": int(len(df)),
        "coverage_notes": [],
        "assumptions": [],
        "confidence": 0.6,
    }})
result = {{"status": "success", "findings": findings}}
json.dump(result, open(meta["output_path"], "w"))
'''

    @staticmethod
    def _fake_email_extraction() -> str:
        return json.dumps([{
            "email_file": "fake.txt", "sender": "fake@example.com", "date": "2026-01-01",
            "subject": "fake", "category": "noise", "entity_or_ingredient": None, "old_price": None,
            "new_price": None, "currency": None, "unit": None, "effective_date": None,
            "event_start": None, "event_end": None, "location": None, "facts": ["fake"],
            "confidence": 0.5, "evidence_text": "fake",
        }])

    @staticmethod
    def _fake_content_ideas(user_prompt: str) -> str:
        ctx_start = user_prompt.find("Context:\n") + len("Context:\n")
        ctx_end = user_prompt.find("\n\nReturn a JSON array")
        ctx = json.loads(user_prompt[ctx_start:ctx_end]) if ctx_start >= 0 and ctx_end > 0 else {}
        findings = ctx.get("approved_findings", [])
        finding_id = findings[0]["finding_id"] if findings else ""
        metric_keys = findings[0]["metric_keys"] if findings else []
        local_ids = [e["context_id"] for e in ctx.get("local_context", [])][:1]
        cal_ids = [e["context_id"] for e in ctx.get("calendar_context", [])][:1]
        windows = ctx.get("posting_windows", [])
        menu = ctx.get("menu_reference", [])
        sku = menu[0]["sku"] if menu else "ICE-001"
        sku_ar = menu[0]["item_ar"] if menu else "منتج"
        sku_en = menu[0]["item_en"] if menu else "Product"

        ideas = []
        for i in range(3):
            window = windows[i % len(windows)] if windows else {"window_id": "", "busy_metric_keys": [], "post_date": "", "start_time_local": "10:00"}
            ideas.append({
                "idea_id": f"idea-{i}", "hook_ar": f"فكرة رقم {i}", "hook_en": f"Idea number {i} for you",
                "format": ["reel", "carousel", "trend_audio"][i % 3], "product_sku": sku,
                "product_name_ar": sku_ar, "product_name_en": sku_en, "finding_id": finding_id,
                "cited_metric_keys": metric_keys, "local_context_ids": local_ids, "calendar_context_ids": cal_ids,
                "posting_window_id": window["window_id"], "timing_metric_keys": window.get("busy_metric_keys", []),
                "rationale_ar": "سياق محلي وتقويمي", "rationale_en": f"Grounded rationale variant {i}",
                "post_date": window.get("post_date", ""), "post_time_local": window.get("start_time_local", "10:00"),
                "timing_reason": "observed busy period", "inventory_suitability": "unknown",
            })
        return json.dumps(ideas)


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr("src.tools.llm_factory.get_chat_model", lambda model_name, temperature=0: _FakeChatModel())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    yield


def test_full_pipeline_reaches_human_gate_and_delivers(fake_llm):
    config = resolve_runtime_config(
        profile_path=ROOT / "data" / "qahwa_saihat" / "cafe_profile.json",
        data_dir=ROOT / "data" / "qahwa_saihat",
        app_settings_path=ROOT / "config" / "app_settings.yaml",
        source_registry_path=ROOT / "config" / "source_registry.yaml",
        target_week=date(2026, 1, 5),
        artifact_root=ROOT / "outputs" / "artifacts",
        checkpoint_db=ROOT / "outputs" / "test_evidence" / "checkpoints_test.sqlite",
        memory_db=ROOT / "outputs" / "test_evidence" / "memory_test.sqlite",
    )
    run_id = "e2e_" + uuid.uuid4().hex[:8]
    graph = build_main_graph()
    thread_config = {"configurable": {"thread_id": run_id}, "recursion_limit": 200}

    initial_state = {
        "run_id": run_id, "thread_id": run_id, "config": config,
        "analysis_period": config.analysis_period, "previous_period": config.previous_period,
        "trailing_baseline_periods": config.trailing_baseline_periods,
        "recommendation_period": config.recommendation_period,
        "critic_round": 0, "content_repair_attempts": 0,
    }
    out = graph.invoke(initial_state, config=thread_config)

    # Graph should have paused before human_gate (no human_decision recorded yet).
    snapshot = graph.get_state(thread_config)
    assert snapshot.next == ("human_gate",)
    assert out["report"]["html_path"]
    assert Path(out["report"]["html_path"]).exists()
    assert out["data_quality"]["sources_successful"]

    # Approve and resume.
    graph.update_state(thread_config, {"human_decision": "approve"})
    final_out = graph.invoke(None, config=thread_config)
    assert final_out["run_status"] == "succeeded"
    assert final_out["delivery_receipt"] is not None
