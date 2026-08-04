"""Content agent (Module 6): produces exactly three bilingual content ideas,
each required to cite an approved finding, a local context item and a
calendar/prayer context item, plus a precomputed posting window. Grounding is
enforced structurally here (what the LLM is given) and independently checked
downstream by the content validator (Module 6, second validation layer).
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Callable

from src.schemas.artifacts import ArtifactRef
from src.schemas.content import ContentIdea
from src.schemas.context import ContextBundle
from src.schemas.findings import AnalystFinding
from src.tools.artifact_io import read_dataframe

ContentGenerator = Callable[[str, dict[str, Any]], list[dict[str, Any]]]

PROMPT_PATH = Path("prompts/content_agent.md")


def _active_menu_items(menu_artifact: ArtifactRef | None, analysis_period_end: str) -> list[dict[str, Any]]:
    if not menu_artifact:
        return []
    df = read_dataframe(menu_artifact)
    active = df[(df["retire_date"].isna()) | (df["retire_date"].astype(str) > analysis_period_end[:10])]
    active = active[(active["launch_date"].isna()) | (active["launch_date"].astype(str) <= analysis_period_end[:10])]
    return active[["sku", "item_en", "item_ar", "category"]].to_dict(orient="records")


def _default_llm_generator(model_name: str) -> ContentGenerator:
    def _generate(system_prompt: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        from src.tools.llm_factory import get_chat_model

        llm = get_chat_model(model_name, temperature=0)
        user_prompt = (
            f"Context:\n{json.dumps(context, indent=2, default=str)}\n\n"
            "Return a JSON array of exactly 3 ContentIdea objects (idea_id, hook_ar, hook_en, format, "
            "product_sku, product_name_ar, product_name_en, finding_id, cited_metric_keys, "
            "local_context_ids, calendar_context_ids, posting_window_id, timing_metric_keys, "
            "rationale_ar, rationale_en, post_date, post_time_local, timing_reason, "
            "inventory_suitability). No prose outside the JSON array."
        )
        resp = llm.invoke([("system", system_prompt), ("user", user_prompt)])
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        match = re.search(r"\[.*\]", content, re.DOTALL)
        return json.loads(match.group(0) if match else content)

    return _generate


def generate_content_ideas(
    final_findings: list[AnalystFinding],
    context_bundle: ContextBundle,
    menu_artifact: ArtifactRef | None,
    analysis_period_end: str,
    model_name: str,
    recent_memory: list[dict[str, Any]] | None = None,
    generator: ContentGenerator | None = None,
) -> list[ContentIdea]:
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    context = {
        "approved_findings": [
            {"finding_id": f["finding_id"], "title": f["title"], "claim": f["claim"],
             "metric_keys": [e["result_key"] for e in f["evidence"]]}
            for f in final_findings
        ],
        "local_context": [e for e in context_bundle["evidence"] if e["kind"] in ("event", "profile")],
        "calendar_context": [e for e in context_bundle["evidence"] if e["kind"] in ("calendar", "prayer")],
        "posting_windows": context_bundle["posting_windows"],
        "menu_reference": _active_menu_items(menu_artifact, analysis_period_end),
        "recent_memory": recent_memory or [],
        "search_status": context_bundle["search_status"],
    }
    gen = generator or _default_llm_generator(model_name)
    raw_ideas = gen(system_prompt, context)

    ideas: list[ContentIdea] = []
    for raw in raw_ideas[:3]:
        ideas.append(ContentIdea(
            idea_id=raw.get("idea_id") or f"idea-{uuid.uuid4().hex[:8]}",
            hook_ar=raw.get("hook_ar", ""), hook_en=raw.get("hook_en", ""),
            format=raw.get("format", "reel"), product_sku=raw.get("product_sku", ""),
            product_name_ar=raw.get("product_name_ar", ""), product_name_en=raw.get("product_name_en", ""),
            finding_id=raw.get("finding_id", ""), cited_metric_keys=raw.get("cited_metric_keys", []),
            local_context_ids=raw.get("local_context_ids", []), calendar_context_ids=raw.get("calendar_context_ids", []),
            posting_window_id=raw.get("posting_window_id", ""), timing_metric_keys=raw.get("timing_metric_keys", []),
            rationale_ar=raw.get("rationale_ar", ""), rationale_en=raw.get("rationale_en", ""),
            post_date=raw.get("post_date", ""), post_time_local=raw.get("post_time_local", ""),
            timing_reason=raw.get("timing_reason", ""),
            inventory_suitability=raw.get("inventory_suitability", "unknown"),
        ))
    return ideas
