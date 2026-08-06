"""Content generation + validation nodes with a single bounded repair attempt
(plan section 5 / M19). On repeated failure, routes to the same
insufficient-evidence path used when the critic approves nothing."""
from __future__ import annotations

from typing import Any, Literal

from src.content.content_agent import generate_content_ideas
from src.persistence.memory_store import MemoryStore
from src.state import CafeIntelligenceState
from src.tools.artifact_io import read_dataframe
from src.validation.content_validator import validate_content_ideas


def content_agent_node(state: CafeIntelligenceState) -> dict[str, Any]:
    config = state["config"]
    final_findings = state.get("final_findings", [])
    if not final_findings:
        return {"content_ideas": [], "step_count": 1}

    cleaned = state.get("cleaned_artifacts", {})
    recent_memory: list[dict[str, Any]] = []
    try:
        store = MemoryStore(config.memory_db)
        recent_memory = store.recent_content(config.profile_key)
        store.close()
    except Exception:  # noqa: BLE001
        pass

    try:
        ideas = generate_content_ideas(
            final_findings=final_findings,
            context_bundle=state["context_bundle"],
            menu_artifact=cleaned.get("menu"),
            analysis_period_end=state["analysis_period"]["end"],
            model_name=config.app_settings.models.content,
            recent_memory=recent_memory,
        )
    except Exception as e:  # noqa: BLE001
        # A model-layer failure here must not crash the whole graph; the
        # validator/report path already handles an empty idea list honestly.
        return {
            "content_ideas": [], "step_count": 1,
            "errors": [{"node": "content_agent", "error": f"{type(e).__name__}: {e}"}],
        }
    return {"content_ideas": ideas, "step_count": 1}


def validate_content_node(state: CafeIntelligenceState) -> dict[str, Any]:
    config = state["config"]
    cleaned = state.get("cleaned_artifacts", {})
    ideas = state.get("content_ideas", [])

    active_skus: set[str] = set()
    stock_risk: set[str] = set()
    if "menu" in cleaned:
        menu_df = read_dataframe(cleaned["menu"])
        period_end = state["analysis_period"]["end"][:10]
        active = menu_df[(menu_df["retire_date"].isna()) | (menu_df["retire_date"].astype(str) > period_end)]
        active = active[(active["launch_date"].isna()) | (active["launch_date"].astype(str) <= period_end)]
        active_skus = set(active["sku"])
    if "inventory" in cleaned:
        inv_df = read_dataframe(cleaned["inventory"])
        risky = inv_df[inv_df.get("estimated_remaining_units").fillna(1) <= 0] if "estimated_remaining_units" in inv_df.columns else inv_df.iloc[0:0]
        stock_risk = set(risky["sku"]) if len(risky) else set()

    validation = validate_content_ideas(
        ideas, state.get("final_findings", []), state["context_bundle"], active_skus,
        config.raw_profile.opening_hours, stock_risk,
        model_name=config.app_settings.models.content_validator,
    )
    attempts = state.get("content_repair_attempts", 0)
    return {"content_validation": validation, "content_repair_attempts": attempts, "step_count": 1}


def route_after_content_validation(state: CafeIntelligenceState) -> Literal["repair", "report", "no_evidence"]:
    validation = state["content_validation"]
    if validation["valid"]:
        return "report"
    config = state["config"]
    attempts = state.get("content_repair_attempts", 0)
    if attempts < config.app_settings.limits.content_repair_attempts:
        return "repair"
    return "no_evidence"


def increment_repair_attempts(state: CafeIntelligenceState) -> dict[str, Any]:
    return {"content_repair_attempts": state.get("content_repair_attempts", 0) + 1}
