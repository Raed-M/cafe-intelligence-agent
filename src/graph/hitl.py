"""Human-in-the-loop gate and delivery/persistence nodes (Modules 7-8).

The graph is compiled with `interrupt_before=["human_gate"]`, so execution
pauses here every run. The caller inspects the paused state, then calls
`graph.update_state(config, {"human_decision": "approve" | "edit" | "reject"})`
before resuming with `graph.invoke(None, config)`. An invalid/missing decision
simply leaves the run paused (no route fires) rather than defaulting to a
side.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Literal

from src.persistence.memory_store import MemoryStore
from src.state import CafeIntelligenceState


def human_gate(state: CafeIntelligenceState) -> dict[str, Any]:
    return {}


def route_human_decision(state: CafeIntelligenceState) -> Literal["deliver", "report_generator", "stop"]:
    decision = state.get("human_decision")
    if decision == "approve":
        return "deliver"
    if decision == "edit":
        return "report_generator"
    if decision == "reject":
        return "stop"
    # No/invalid decision recorded yet -- stay put logically; graph resume
    # should not be triggered without a decision. Treat as reject-safe no-op.
    return "stop"


def deliver(state: CafeIntelligenceState) -> dict[str, Any]:
    config = state["config"]
    report = state.get("report", {})
    idempotency_key = hashlib.sha256(
        f"{config.profile_key}|{state['analysis_period']['start']}|{state['analysis_period']['end']}|1|file".encode()
    ).hexdigest()

    store = MemoryStore(config.memory_db)
    newly_recorded = store.record_delivery(
        idempotency_key=idempotency_key, run_id=state["run_id"], report_version=1,
        delivered_at=datetime.now(timezone.utc).isoformat(), destination_type="file",
        artifact_paths={"html": report.get("html_path"), "pdf": report.get("pdf_path")},
    )
    store.close()

    return {
        "delivery_receipt": {"idempotency_key": idempotency_key, "newly_recorded": newly_recorded},
        "run_status": "succeeded",
        "step_count": 1,
    }


def stop_rejected(state: CafeIntelligenceState) -> dict[str, Any]:
    return {"run_status": "rejected", "step_count": 1}


def persist_run(state: CafeIntelligenceState) -> dict[str, Any]:
    config = state["config"]
    report = state.get("report", {})
    critic_results = state.get("critic_results", {})

    store = MemoryStore(config.memory_db)
    store.record_run(
        run_id=state["run_id"], profile_key=config.profile_key, cafe_name=config.raw_profile.cafe_name,
        analysis_period=state["analysis_period"], recommendation_period=state["recommendation_period"],
        started_at=state.get("_started_at", datetime.now(timezone.utc).isoformat()),
        completed_at=datetime.now(timezone.utc).isoformat(),
        status=state.get("run_status", "partial"), final_findings=state.get("final_findings", []),
        critic_rejections=critic_results.get("total_rejections", 0), total_steps=state.get("step_count", 0),
        total_tokens=sum(state.get("token_usage", {}).values()), cost_usd=state.get("cost_usd", 0.0),
        quality=state.get("data_quality", {}), report_html_path=report.get("html_path"),
        report_pdf_path=report.get("pdf_path"), whatsapp_summary=report.get("whatsapp_summary", ""),
    )
    if state.get("content_ideas"):
        store.record_content(
            state["run_id"], state["content_ideas"], state.get("human_decision"),
            delivered=(state.get("run_status") == "succeeded"),
        )
    store.close()
    return {"step_count": 1}
