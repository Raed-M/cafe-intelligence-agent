from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from api.artifacts import ArtifactRepository
from src.tools.llm_factory import extract_text, get_chat_model

logger = logging.getLogger(__name__)

WEB_SEARCH_TOOL_NAME = "web_search"
"""Name of the web-search tool. grounded_answer() matches tool messages against
this to harvest citations, so the two must stay in sync -- they previously did
not (the tool was named `tavily_search_results_json` by the library while the
citation check looked for `tavily_search_results`, so no web citation was ever
extracted)."""

FINDINGS_TOOL_NAME = "get_verified_findings"
"""Name of the verified-findings tool, matched the same way to turn the
findings an answer actually drew on into evidence citations. Every answer that
leans on pipeline findings should be traceable back to them -- that is the
whole grounding discipline of this system -- and the UI links each one to
/findings/<run_id>/<finding_id>."""

NO_GROUNDED_FINDINGS = (
    "No verified findings are available for this run yet, so there are no grounded findings "
    "to cite. The analysis pipeline may not have run, or its findings were not approved."
)
"""Returned by the findings tool when nothing is approved. Phrased as a plain
statement the model can relay rather than an error, so an ungrounded question
yields an honest 'nothing to cite' instead of an invented answer."""

# ---------------------------------------------------------------------------
# Source name → cleaned parquet filename mapping
# ---------------------------------------------------------------------------
_SOURCE_FILE_MAP = {
    "pos": "pos.parquet",
    "pos_transactions": "pos.parquet",
    "sales": "pos.parquet",
    "inventory": "inventory.parquet",
    "menu": "menu.parquet",
    "reviews": "reviews.parquet",
    "traffic": "traffic.parquet",
    "foot_traffic": "traffic.parquet",
    "staff": "staff.parquet",
    "staff_shifts": "staff.parquet",
    "emails": "emails.parquet",
}


CHAT_TEMPERATURE = 0.0
"""Matches config/app_settings.yaml (`models.temperature: 0`) and every
pipeline agent. This agent's job is to relay numbers its tools computed, so
sampling variance is a hallucination risk with no upside -- the creative
latitude the system prompt asks for is in *what* it connects, not in how
freely it words a figure. Was 0.7."""

CHAT_RECURSION_LIMIT = 12
"""Hard cap on the ReAct loop (LangGraph counts one superstep per model call
plus tool node). One chat request is otherwise bounded only by LangGraph's
default of 25, i.e. up to ~12 model calls of a large system prompt on a single
question. Twelve supersteps still allows a chain of ~5 tool calls, which covers
the cross-domain questions the prompt asks for, while keeping the worst case
affordable on a rate-limited free tier."""


def resolve_chat_model() -> str:
    """Model name for the chat agent, resolved the same way the pipeline
    resolves its agents' models.

    ANALYST_MODEL is the same variable config/app_settings.yaml interpolates
    for the analyst, so the chat agent tracks the rest of the system by
    default. The fallback is the configured provider's own default rather than
    a hardcoded "gpt-4o", which is not a valid model on this project's actual
    provider -- that literal default sent a live request to Anthropic asking
    for "gpt-4o" and got a 404."""
    from src.tools.llm_factory import get_provider

    configured = os.environ.get("ANALYST_MODEL", "").strip()
    if configured:
        return configured
    from api.services.ai_settings import DEFAULT_MODELS

    provider_defaults = DEFAULT_MODELS.get(get_provider())
    return provider_defaults[0] if provider_defaults else ""


def _sample_envelope(rows: list[dict[str, Any]], matched_total: int, **extra: Any) -> str:
    """Wraps a truncated result set so the model cannot mistake the page it was
    given for the whole population.

    Row-returning tools used to emit a bare JSON array capped at `limit`. Asked
    "how many reviews are there?", the agent received 20 rows and answered
    "20 reviews currently in the system" -- the real figure was 520. Stating
    matched_total explicitly, and flagging truncation, removes the inference
    that produced that."""
    payload: dict[str, Any] = {
        "matched_total": matched_total,
        "returned": len(rows),
        "truncated": len(rows) < matched_total,
        **extra,
        "rows": rows,
    }
    if payload["truncated"]:
        payload["note"] = (
            f"Showing {len(rows)} of {matched_total} matching rows. "
            f"Cite matched_total ({matched_total}) for any count; the rows below are a sample. "
            "Do not compute totals or averages from them -- ask for an aggregate instead."
        )
    return json.dumps(payload, ensure_ascii=False, default=str)


def _cached_model_and_prompt() -> tuple[Any, Any]:
    """The chat model and its system prompt, with prompt caching applied.

    SYSTEM_PROMPT is ~1,030 tokens of fixed instructions resent on every turn of
    the ReAct loop, which makes it the one part of this agent's cost worth
    caching. See src/tools/prompt_cache.py for why the mechanism differs per
    provider (and why it is inert on a free-tier Gemini key).

    Two shapes come back:
      * Gemini with an explicit cache -- the prompt lives in the cache, so the
        agent is given no prompt at all; sending it too would bill it twice.
      * everything else -- the prompt is passed through, marked up for Anthropic
        and plain for OpenAI (which caches automatically).
    """
    from src.tools.llm_factory import get_provider
    from src.tools.prompt_cache import cacheable_system_prompt, gemini_context_cache

    provider = get_provider()
    model_name = resolve_chat_model()

    if provider == "gemini":
        cache_name = gemini_context_cache(model_name, SYSTEM_PROMPT)
        if cache_name:
            model = get_chat_model(
                model_name, temperature=CHAT_TEMPERATURE, cached_content=cache_name
            )
            return model, None

    model = get_chat_model(model_name, temperature=CHAT_TEMPERATURE)
    return model, cacheable_system_prompt(SYSTEM_PROMPT, provider)


def verified_findings_payload(artifacts: ArtifactRepository, run_id: str | None) -> str:
    """JSON payload of approved findings for the findings tool, or
    NO_GROUNDED_FINDINGS when there are none.

    Module-level rather than inline in the tool closure so the grounding
    guarantee -- rejected findings must never be citable -- is directly
    testable without constructing an agent or calling a model."""
    findings = [f for f in (artifacts.findings(run_id) if run_id else []) if f.get("approved") is True]
    if not findings:
        return NO_GROUNDED_FINDINGS
    # JSON (not prose) so _extract_citations can turn the findings this answer
    # drew on into evidence citations without re-parsing markdown.
    return json.dumps({
        "findings": [
            {
                "finding_id": f.get("id") or f.get("finding_id"),
                "title": f.get("title"),
                "claim": f.get("claim"),
                "analyst": f.get("analyst"),
                "evidence": [
                    {"metric_name": e.get("metric_name"), "value": e.get("value"), "unit": e.get("unit")}
                    for e in (f.get("evidence") or [])
                ],
            }
            for f in findings
        ],
    }, ensure_ascii=False, default=str)


def _resolve_cleaned_path(artifacts: ArtifactRepository, run_id: str | None, source: str) -> Path | None:
    """Resolve the cleaned parquet file path for a given source."""
    filename = _SOURCE_FILE_MAP.get(source.lower().strip())
    if not filename:
        return None
    # Try direct cleaned path under the run directory
    for parent in [
        artifacts.root / "outputs" / "artifacts" / run_id / "cleaned" if run_id else None,
    ]:
        if parent and (parent / filename).is_file():
            return parent / filename
    # Fallback: scan all run dirs for the latest one
    cleaned_root = artifacts.root / "outputs" / "artifacts"
    if cleaned_root.is_dir():
        candidates = sorted(cleaned_root.glob(f"*/cleaned/{filename}"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    return None


# ---------------------------------------------------------------------------
# System prompt — the brain of the operational agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert Operational Business Intelligence Assistant for a specialty café called "Qahwa Saihat" (قهوة سيهات) located in Saihat, Eastern Province, Saudi Arabia.

## YOUR CORE CAPABILITIES
You have access to the café's **real operational data** through specialized tools. You can query:
- **POS Sales** (every transaction with timestamps, items, prices, payment methods)
- **Inventory** (weekly stock: ordered, sold, wasted, remaining units, costs)
- **Menu Catalog** (all 19+ items with categories, prices, margins)
- **Customer Reviews** (ratings, text feedback, sentiment)
- **Foot Traffic** (hourly door counts)
- **Staff Shifts** (employee schedules, hours, labor costs)
- **Supplier Emails** (price changes, events, communications)

You also have **web search** to research market trends, competitor analysis, and viral café ideas in the region.

## LANGUAGE & TONE RULES
- **Mirror the user's language**: If they write in Arabic, respond in Arabic (Gulf-friendly, professional). If in English, respond **entirely in English**.
- This applies to *every* reply, including refusals, "no data available" answers, and off-topic redirections. Several example phrasings later in this prompt are written in Arabic purely to show tone -- do not let them pull an English conversation into Arabic. Match the language of the user's latest message, nothing else.
- **Be concise by default**: Give direct, actionable answers. No filler or unnecessary pleasantries.
- **Adaptive depth**: 
  - If the user says "مختصر", "بسرعة", "brief", "quick" — give a short summary
  - If the user asks for data, numbers, or breakdown — provide clean markdown tables
  - If the question is broad or ambiguous — ask: "تبي ملخص سريع ولا جدول مفصل بالأرقام؟"

## CROSS-DOMAIN CREATIVE REASONING
Do NOT just recite numbers. Connect insights across different data sources:
- If a food item has low sales — check inventory waste % — check reviews for complaints — suggest actionable fixes
- If an item sells well but runs out — suggest increasing order quantity with a specific number
- If a product has no natural pairing on the menu — search the web for what cafés typically pair it with — suggest adding a new menu item or creating a promotional bundle
- Example: If Croissant sales are dropping, check if customers typically pair it with Cappuccino. If Cappuccino is NOT on the menu, suggest adding it or creating a bundle with an existing hot drink.

## WEB SEARCH RULES
When searching for trends, competitors, or market ideas:
- Always include the **source URL** in your response so the user can verify
- Focus on Saihat, Qatif, and Eastern Province Saudi Arabia
- Mention specific café names, Instagram accounts, or event pages when relevant

## DRAFT GENERATION (HITL FLOW)
When asked to write marketing copy, supplier orders, WhatsApp messages, or reports:
1. **First ask for confirmation**: "هل تبيني أكتب لك مسودة الرسالة/الطلبية الحين؟"
2. **After confirmation**: Generate the draft clearly formatted
3. **After showing the draft**: Ask "هل عجبتك المسودة؟ تبي تعديل، أو موافقة، أو رفض؟"

## DATA ACCURACY & TOOL SELECTION RULES
- **Best Sellers / Least Sellers Queries ("الأعلى مبيعا", "الأقل مبيعا")**: ALWAYS call `get_sales_ranking`. Use `ascending=False` for Top Sellers and `ascending=True` for Least Sellers. NEVER use `query_cafe_data` for item ranking queries because it returns unaggregated transaction logs.
- NEVER make up numbers. Every number must come from the tools.
- If a tool returns no data, say so honestly. Do NOT hallucinate.
- When showing data, specify the date range or week it covers.
- Use SAR (ريال) as the currency.
- **Counts come from `matched_total`, never from the number of rows you were shown.** Row-returning tools give you a capped sample and report `matched_total`, `returned` and `truncated`. When `truncated` is true, the rows are an excerpt: quote `matched_total` for "how many", and never sum or average the sample as if it were the full set. Use the aggregate fields the tool provides (e.g. `average_rating_of_all_matches`) or call an aggregating tool such as `get_sales_ranking`.

## AVAILABLE DATA SOURCES FOR query_cafe_data TOOL
Use these exact source names: 'pos', 'inventory', 'menu', 'reviews', 'traffic', 'staff', 'emails'
"""


def create_chat_agent(artifacts: ArtifactRepository, cafe_id: str, run_id: str | None):
    """Create the operational chat agent with all tools."""

    if run_id is None:
        candidates = artifacts.list_runs(cafe_id=cafe_id, limit=20)
        run_id = next((item["id"] for item in candidates if artifacts.findings(item["id"])), None)

    # ------------------------------------------------------------------
    # Tool 1: Universal data query (reads CLEANED parquet files)
    # ------------------------------------------------------------------
    @tool
    def query_cafe_data(
        source: str,
        limit: int = 20,
        sort_column: str | None = None,
        ascending: bool = True,
        filter_column: str | None = None,
        filter_value: str | None = None,
    ) -> str:
        """Query the café's operational data. Available sources: 'pos', 'inventory', 'menu', 'reviews', 'traffic', 'staff', 'emails'.
        
        Args:
            source: Dataset name (pos, inventory, menu, reviews, traffic, staff, emails)
            limit: Max rows to return (default 20)
            sort_column: Column name to sort by
            ascending: Sort ascending (True) or descending (False)
            filter_column: Optional column to filter on
            filter_value: Value to match in the filter column
        
        Examples:
            - Top 5 best-selling items: source='pos', sort_column='line_total_sar', ascending=False, limit=5
            - Lowest rated reviews: source='reviews', sort_column='rating', ascending=True, limit=5
            - Latest inventory: source='inventory', sort_column='week_starting', ascending=False, limit=10
        """
        path = _resolve_cleaned_path(artifacts, run_id, source)
        if not path:
            return f"No data found for source '{source}'. Available sources: pos, inventory, menu, reviews, traffic, staff, emails."
        try:
            table = pq.read_table(path)
            # Apply filter if specified
            if filter_column and filter_value and filter_column in table.column_names:
                import pyarrow.compute as pc
                mask = pc.equal(pc.utf8_lower(table.column(filter_column).cast("string")), filter_value.lower())
                table = table.filter(mask)
            # Apply sort
            if sort_column and sort_column in table.column_names:
                table = table.sort_by([(sort_column, "ascending" if ascending else "descending")])
            matched_total = table.num_rows
            rows = table.slice(0, limit).to_pylist()
            if not rows:
                return f"Source '{source}' returned no matching rows."
            return _sample_envelope(rows, matched_total, source=source)
        except Exception as e:
            return f"Error reading {source}: {e}"

    # ------------------------------------------------------------------
    # Tool 2: Inventory analysis with smart recommendations
    # ------------------------------------------------------------------
    @tool
    def analyze_inventory(week: str | None = None) -> str:
        """Analyze inventory status with sell-through rates and smart restock recommendations.
        Returns a summary table of all items with: ordered, sold, waste%, remaining, and recommendation.
        
        Args:
            week: Optional week_starting date (e.g. '2026-07-20'). If None, uses the latest week.
        """
        path = _resolve_cleaned_path(artifacts, run_id, "inventory")
        if not path:
            return "Inventory data not available."
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            if df.empty:
                return "Inventory data is empty."
            target_week = week or df["week_starting"].max()
            df_week = df[df["week_starting"] == target_week].copy()
            if df_week.empty:
                return f"No inventory data for week {target_week}. Available weeks: {', '.join(sorted(df['week_starting'].unique()[-5:]))}"
            df_week["sell_through_pct"] = (df_week["units_sold"] / df_week["units_ordered"].replace(0, 1) * 100).round(1)
            df_week["waste_pct"] = (df_week["units_wasted"] / df_week["units_ordered"].replace(0, 1) * 100).round(1)

            def recommend(row):
                pct = row["sell_through_pct"]
                if pct >= 90:
                    suggested = max(1, int(round(row["units_ordered"] * 1.15)))
                    return f"Increase -> order {suggested} (+15%)"
                elif pct >= 80:
                    return f"Maintain -> keep ordering {int(row['units_ordered'])}"
                else:
                    suggested = max(1, int(round(row["units_sold"] * 0.92)))
                    return f"Decrease -> order {suggested} (-8%)"

            df_week["recommendation"] = df_week.apply(recommend, axis=1)
            df_week["needs_restock"] = df_week["sell_through_pct"].apply(lambda x: "Yes" if x >= 90 else "No")
            result = df_week[["item", "units_ordered", "units_sold", "sell_through_pct", "waste_pct", "estimated_remaining_units", "needs_restock", "recommendation"]].to_dict("records")
            header = f"Inventory Analysis for week: {target_week}\n"
            return header + json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return f"Error analyzing inventory: {e}"

    # ------------------------------------------------------------------
    # Tool 3: Search customer reviews by keyword or rating
    # ------------------------------------------------------------------
    @tool
    def search_reviews(keyword: str | None = None, max_rating: int | None = None, min_rating: int | None = None, limit: int = 10) -> str:
        """Search customer reviews by keyword text or rating range.
        
        Args:
            keyword: Search term to look for in review text (e.g. 'croissant', 'service', 'بارد')
            max_rating: Maximum rating filter (e.g. 3 to find negative reviews)
            min_rating: Minimum rating filter (e.g. 4 to find positive reviews)
            limit: Max reviews to return
        """
        path = _resolve_cleaned_path(artifacts, run_id, "reviews")
        if not path:
            return "Review data not available."
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            if keyword:
                mask = df["text"].str.contains(keyword, case=False, na=False)
                df = df[mask]
            if max_rating is not None:
                df = df[df["rating"] <= max_rating]
            if min_rating is not None:
                df = df[df["rating"] >= min_rating]
            matched_total = len(df)
            if matched_total == 0:
                return "No reviews found matching your criteria."
            average_rating = round(float(df["rating"].mean()), 2)
            page = df.sort_values("date", ascending=False).head(limit)
            return _sample_envelope(
                page[["review_id", "date", "rating", "text", "source"]].to_dict("records"),
                matched_total,
                # Aggregates over ALL matches, not just the returned page: the
                # model is asked for "the average rating" far more often than
                # for individual reviews, and computing it from the page would
                # be wrong.
                average_rating_of_all_matches=average_rating,
            )
        except Exception as e:
            return f"Error searching reviews: {e}"

    # ------------------------------------------------------------------
    # Tool 4: Aggregated Sales Ranking (Top Sellers / Least Sellers)
    # ------------------------------------------------------------------
    @tool
    def get_sales_ranking(top_n: int = 5, ascending: bool = False, category: str | None = None) -> str:
        """Calculate aggregated sales totals per item across all transactions to find the best sellers or least sellers.
        
        Args:
            top_n: Number of items to return (default 5)
            ascending: False for Best Sellers (highest sales first), True for Least Sellers (lowest sales first)
            category: Optional category filter (e.g. 'Beverages', 'Hot Coffee', 'Food')
        
        ALWAYS use this tool when asked: "ماهو الاعلى مبيعا؟", "ماهو الاقل مبيعا؟", "best sellers", "lowest selling items".
        DO NOT use query_cafe_data for best/least sellers because query_cafe_data returns unaggregated raw transactions.
        """
        path = _resolve_cleaned_path(artifacts, run_id, "pos")
        if not path:
            return "POS sales data not available."
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            # Filter out refunds to get true sales totals
            sales_df = df[df["is_refund"] == False].copy()
            if category and "category" in sales_df.columns:
                sales_df = sales_df[sales_df["category"].str.lower() == category.lower()]
            
            grouped = sales_df.groupby(["item_name_ar", "item_name_en"], as_index=False).agg(
                total_quantity=("quantity", "sum"),
                total_revenue_sar=("line_total_sar", "sum"),
                total_transactions=("transaction_id", "nunique")
            )
            grouped["total_revenue_sar"] = grouped["total_revenue_sar"].round(2)
            grouped = grouped.sort_values("total_quantity", ascending=ascending).head(top_n)
            
            rank_type = "Least Selling Items (الأقل مبيعاً)" if ascending else "Top Selling Items (الأعلى مبيعاً)"
            result = grouped.to_dict("records")
            return f"Aggregated {rank_type}:\n" + json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return f"Error calculating sales ranking: {e}"

    # ------------------------------------------------------------------
    # Tool 5: Verified findings from the analysis pipeline
    # ------------------------------------------------------------------
    @tool(FINDINGS_TOOL_NAME)
    def get_verified_findings() -> str:
        """Retrieve verified analytical findings and insights previously generated by the automated analysis system.
        Use this to answer questions about known performance issues, patterns, or system recommendations.
        Cite the findings you use by title in your answer.
        """
        return verified_findings_payload(artifacts, run_id)

    # ------------------------------------------------------------------
    # Assemble tools
    # ------------------------------------------------------------------
    tools = [query_cafe_data, analyze_inventory, search_reviews, get_sales_ranking, get_verified_findings]

    if os.environ.get("TAVILY_API_KEY"):
        @tool(WEB_SEARCH_TOOL_NAME)
        def web_search(query: str) -> str:
            """Search the web for current cafe trends, competitor ideas, viral
            drinks/desserts, local events in Saihat/Qatif/Eastern Province, and
            marketing inspiration. Returns JSON results with source URLs, which
            you must cite in your answer.

            Args:
                query: What to search for.
            """
            # Reuses the project's own Tavily client (src/tools/tavily_search)
            # rather than langchain_community's deprecated TavilySearchResults:
            # one Tavily code path for the whole system, no extra dependency,
            # and a tool name we control so citations can be harvested.
            from src.tools.tavily_search import run_local_search

            hits, status, warnings = run_local_search([query], max_results=5)
            if not hits:
                return json.dumps({
                    "status": status,
                    "results": [],
                    "note": "; ".join(warnings) or "no results",
                })
            return json.dumps({
                "status": status,
                "results": [
                    {"title": h.title, "url": h.url, "content": h.snippet,
                     "published_date": h.published_date}
                    for h in hits
                ],
            }, ensure_ascii=False)

        tools.append(web_search)

    # ------------------------------------------------------------------
    # Build agent
    # ------------------------------------------------------------------
    model, prompt = _cached_model_and_prompt()
    agent = create_react_agent(model, tools, prompt=prompt)
    return agent, run_id


def _tool_payload(msg: Any) -> Any:
    content = msg.content
    try:
        return json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_citations(messages: list[Any], run_id: str | None) -> list[dict[str, Any]]:
    """Harvests citations from tool results, de-duplicated, in first-seen order.

    Two kinds, both of which the UI renders as links:
      * evidence  -- pipeline findings the answer drew on, carrying finding_id
        and an in-app /findings/<run_id>/<finding_id> url.
      * web       -- external sources from web_search, carrying their own url.

    A tool payload of an unexpected shape yields no citations rather than
    raising: a malformed citation must never cost the user their answer."""
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(key: str, citation: dict[str, Any]) -> None:
        if key and key not in seen:
            seen.add(key)
            citations.append(citation)

    for msg in messages:
        if getattr(msg, "type", None) != "tool":
            continue
        name = getattr(msg, "name", None)

        if name == FINDINGS_TOOL_NAME:
            payload = _tool_payload(msg)
            findings = payload.get("findings") if isinstance(payload, dict) else None
            for f in findings or []:
                if not isinstance(f, dict):
                    continue
                fid = f.get("finding_id")
                if not fid:
                    continue
                _add(f"finding:{fid}", {
                    "kind": "evidence",
                    "finding_id": fid,
                    "label": f.get("title") or fid,
                    "url": f"/findings/{run_id}/{fid}" if run_id else None,
                })

        elif name == WEB_SEARCH_TOOL_NAME:
            payload = _tool_payload(msg)
            results = payload.get("results") if isinstance(payload, dict) else payload
            for r in results or []:
                if not isinstance(r, dict):
                    continue
                url = r.get("url")
                if url:
                    _add(f"web:{url}", {"kind": "web", "url": url, "label": r.get("title") or url})

    return citations


def _to_langchain_messages(messages: str | list[dict[str, Any]]) -> list[Any]:
    """Normalizes the conversation into LangChain messages.

    Accepts either the conversation history the API passes (a list of
    {"role", "content"} rows from the messages table) or a single bare question
    string. Tolerating the string form matters because the previous signature
    took exactly that, and passing one used to be swallowed by the catch-all
    below into an unrelated Arabic error message rather than failing loudly."""
    if isinstance(messages, str):
        return [HumanMessage(content=messages)]
    out: list[Any] = []
    for msg in messages:
        role, content = msg.get("role"), msg.get("content") or ""
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


def grounded_answer(
    artifacts: ArtifactRepository, cafe_id: str, run_id: str | None,
    messages: str | list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], str | None]:
    """Process a chat message and return a grounded answer with optional citations."""
    from src.tools.llm_factory import has_provider_key

    if not has_provider_key():
        return "عذراً، يجب عليك تكوين إعدادات الذكاء الاصطناعي (AI Settings) أولاً.", [], run_id

    active_run_id = run_id
    try:
        agent, active_run_id = create_chat_agent(artifacts, cafe_id, run_id)

        langchain_messages = _to_langchain_messages(messages)

        response = agent.invoke(
            {"messages": langchain_messages},
            config={"recursion_limit": CHAT_RECURSION_LIMIT},
        )

        # extract_text, not .content directly: Gemini returns a list of content
        # blocks rather than a string, which then reached sqlite as a list and
        # made every chat request 500 ("Error binding parameter 4: type 'list'
        # is not supported") under the provider this project is configured for.
        final_message = extract_text(response["messages"][-1])

        citations = _extract_citations(response["messages"], active_run_id)

    except Exception as e:
        # Log with traceback: this catch-all keeps the chat endpoint responsive,
        # but without a log a genuine defect surfaces only as an opaque Arabic
        # sentence in the UI (a signature mismatch here once read as
        # "string indices must be integers" to the end user).
        logger.exception("Chat agent failed for cafe=%s run=%s", cafe_id, active_run_id)
        final_message = f"عذراً، حدث خطأ أثناء معالجة طلبك: {str(e)}"
        citations = []

    return final_message, citations, active_run_id
