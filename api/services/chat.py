from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

from api.artifacts import ArtifactRepository
from src.tools.llm_factory import get_chat_model

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
- **Mirror the user's language**: If they write in Arabic, respond in Arabic (Gulf-friendly, professional). If in English, respond in English.
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
            rows = table.slice(0, limit).to_pylist()
            if not rows:
                return f"Source '{source}' returned no matching rows."
            return json.dumps(rows, ensure_ascii=False, default=str)
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
            df = df.sort_values("date", ascending=False).head(limit)
            if df.empty:
                return f"No reviews found matching your criteria."
            return json.dumps(df[["review_id", "date", "rating", "text", "source"]].to_dict("records"), ensure_ascii=False, default=str)
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
    @tool
    def get_verified_findings() -> str:
        """Retrieve verified analytical findings and insights previously generated by the automated analysis system.
        Use this to answer questions about known performance issues, patterns, or system recommendations.
        """
        findings = [f for f in (artifacts.findings(run_id) if run_id else []) if f.get("approved") is True]
        if not findings:
            return "No verified findings available yet. The analysis pipeline may not have run."
        lines = []
        for f in findings:
            evidences = [f"{e.get('metric_name')}: {e.get('value')} {e.get('unit') or ''}" for e in (f.get("evidence") or [])]
            ev_text = (" | Evidence: " + ", ".join(evidences)) if evidences else ""
            lines.append(f"- **{f.get('title')}**: {f.get('claim')}{ev_text}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Assemble tools
    # ------------------------------------------------------------------
    tools = [query_cafe_data, analyze_inventory, search_reviews, get_sales_ranking, get_verified_findings]

    if os.environ.get("TAVILY_API_KEY"):
        tavily = TavilySearchResults(
            max_results=5,
            include_answer=True,
            description=(
                "Search the web for current café trends, competitor ideas, viral drinks/desserts, "
                "local events in Saihat/Qatif/Eastern Province, and marketing inspiration. "
                "Always return source URLs for citation."
            ),
        )
        tools.append(tavily)

    # ------------------------------------------------------------------
    # Build agent
    # ------------------------------------------------------------------
    model_name = os.environ.get("ANALYST_MODEL", "gpt-4o")
    model = get_chat_model(model_name, temperature=0.7)

    agent = create_react_agent(model, tools, prompt=SYSTEM_PROMPT)
    return agent, run_id


def grounded_answer(
    artifacts: ArtifactRepository, cafe_id: str, run_id: str | None, messages: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]], str | None]:
    """Process a chat message and return a grounded answer with optional citations."""
    from src.tools.llm_factory import has_provider_key

    if not has_provider_key():
        return "عذراً، يجب عليك تكوين إعدادات الذكاء الاصطناعي (AI Settings) أولاً.", [], run_id

    active_run_id = run_id
    try:
        agent, active_run_id = create_chat_agent(artifacts, cafe_id, run_id)

        langchain_messages = []
        for msg in messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        response = agent.invoke({"messages": langchain_messages})
        
        # Extract the final answer
        final_message = response["messages"][-1].content
        
        # Extract web citations from tool messages if any
        citations = []
        for msg in response["messages"]:
            if hasattr(msg, "type") and msg.type == "tool" and msg.name == "tavily_search_results":
                try:
                    results = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if isinstance(results, list):
                        for r in results:
                            if isinstance(r, dict) and r.get("url"):
                                citations.append({"url": r["url"], "label": r.get("title", r["url"])})
                except (json.JSONDecodeError, TypeError):
                    pass

    except Exception as e:
        final_message = f"عذراً، حدث خطأ أثناء معالجة طلبك: {str(e)}"
        citations = []

    return final_message, citations, active_run_id
