"""Deterministic cross-source correlation hints -- no LLM involved anywhere in
this module. Three capabilities, all producing *hints* that get surfaced to
analysts/critic as "consider investigating and independently verifying this",
never a substitute for an analyst's own executed, critic-checked computation:

1. weekly_signal_deltas: builds a small joined weekly signal table across
   whichever cleaned sources are available and flags pairs of signals that
   both moved materially in the same analysis-vs-previous-period window --
   the same "build one joined table, try every other column" approach used by
   Tableau's Explain Data and Power BI's Key Influencers, done here with
   plain pandas instead of hoping a model notices the coincidence unprompted.

2. procurement_cost_scenarios: best-effort match of a standing-order email
   (quantity + rate) against a later price-change email for the same
   ingredient, computing standing_quantity x price_delta -- the exact
   calculation prompts/analysts/margin.md asks for, which a live model
   reliably skips when it's one instruction among many (proven empirically:
   zero of 29 live weeks ever attempted it, see
   outputs/test_evidence/live_llm_full_dataset_scan/README.md).

3. cross_analyst_coincidences: given one week's full candidate-finding batch
   (after all analysts finish, before the critic), flags when two findings
   from *different* analysts both report a large change for the same period
   -- surfaced to the critic's semantic-review context so it can ask whether
   a cross-domain explanation belongs in the finding, without hand-coding any
   specific storyline.

4. calendar_overlaps: Ramadan/Eid windows overlapping the analysis period --
   the critic already rejects an undisclosed causal claim spanning one of
   these (a real confound, proven against this dataset), but until an analyst
   is actually *told* the overlap exists it can only ever react to it after
   the fact via a rejection, never proactively explain a swing correctly the
   first time. Surfacing the same information to analysts closes that gap.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.schemas.artifacts import ArtifactRef
from src.schemas.findings import AnalystFinding
from src.tools.artifact_io import read_dataframe

_MATERIALITY_PCT = 10.0  # minimum abs % change to call a signal "materially moved"

_QTY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(l|kg|g|ml|units?|pcs?)\b", re.IGNORECASE)
_STANDING_ORDER_RE = re.compile(
    r"standing order|recurring order|weekly (?:delivery|order)|confirmed for.*delivery", re.IGNORECASE
)


def _pct_change(current: float, previous: float | None) -> float | None:
    if previous is None or previous == 0 or pd.isna(previous):
        return None
    return (current - previous) / abs(previous) * 100.0


def _week_bounds(period: dict[str, str]) -> tuple[str, str]:
    return period["start"][:10], period["end"][:10]


def _weekly_pos_signals(df: pd.DataFrame, start: str, end: str) -> dict[str, float] | None:
    mask = (df["business_date"] >= start) & (df["business_date"] < end)
    window = df[mask & ~df.get("is_refund", pd.Series(False, index=df.index)).fillna(False)]
    if window.empty:
        return None
    return {
        "revenue_sar": float(window["line_total_sar"].sum()),
        "baskets": float(window["transaction_id"].nunique()),
    }


def _weekly_staff_signals(df: pd.DataFrame, start: str, end: str) -> dict[str, float] | None:
    window = df[(df["date"] >= start) & (df["date"] < end)]
    if window.empty:
        return None
    return {
        "staff_hours": float(window["hours"].sum()),
        "labour_cost_sar": float(window["labour_cost_sar"].sum()),
    }


def _weekly_traffic_signals(df: pd.DataFrame, start: str, end: str) -> dict[str, float] | None:
    dead = df.get("is_dead_sensor_day", pd.Series(False, index=df.index)).fillna(False)
    window = df[(df["date"] >= start) & (df["date"] < end) & ~dead]
    if window.empty:
        return None
    return {"footfall": float(window["door_count"].sum())}


def _weekly_inventory_signals(df: pd.DataFrame, start: str, end: str) -> dict[str, float] | None:
    window = df[(df["week_starting"] >= start) & (df["week_starting"] < end)]
    if window.empty:
        return None
    return {"waste_cost_sar": float(window["known_waste_cost_sar"].fillna(0).sum())}


_SIGNAL_BUILDERS = {
    "pos": _weekly_pos_signals,
    "staff": _weekly_staff_signals,
    "traffic": _weekly_traffic_signals,
    "inventory": _weekly_inventory_signals,
}


def weekly_signal_deltas(
    cleaned_artifacts: dict[str, ArtifactRef], analysis_period: dict[str, str], previous_period: dict[str, str]
) -> list[dict[str, Any]]:
    a_start, a_end = _week_bounds(analysis_period)
    p_start, p_end = _week_bounds(previous_period)

    signals: dict[str, float] = {}
    prev_signals: dict[str, float] = {}
    for source, builder in _SIGNAL_BUILDERS.items():
        if source not in cleaned_artifacts:
            continue
        try:
            df = read_dataframe(cleaned_artifacts[source])
        except Exception:  # noqa: BLE001
            continue
        cur = builder(df, a_start, a_end)
        prev = builder(df, p_start, p_end)
        if cur:
            signals.update({f"{source}.{k}": v for k, v in cur.items()})
        if prev:
            prev_signals.update({f"{source}.{k}": v for k, v in prev.items()})

    deltas = {k: round(pct, 1) for k, v in signals.items() if (pct := _pct_change(v, prev_signals.get(k))) is not None}
    material = {k: v for k, v in deltas.items() if abs(v) >= _MATERIALITY_PCT}

    hints = []
    for key, pct in material.items():
        others = {k: v for k, v in material.items() if k != key}
        hints.append({"signal": key, "pct_change_vs_previous_period": pct, "coincident_signals": others})
    return hints


def _extract_quantity(text: str) -> tuple[float, str] | None:
    m = _QTY_RE.search(text)
    if not m:
        return None
    return float(m.group(1)), m.group(2).lower()


def procurement_cost_scenarios(emails_df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if emails_df is None or emails_df.empty or "entity_or_ingredient" not in emails_df.columns:
        return []

    df = emails_df.dropna(subset=["entity_or_ingredient"]).copy()
    if df.empty:
        return []
    df["_date_sort"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["_date_sort"])

    scenarios: list[dict[str, Any]] = []
    for ingredient, group in df.groupby(df["entity_or_ingredient"].str.lower().str.strip()):
        group = group.sort_values("_date_sort")

        baseline = None
        for _, row in group.iterrows():
            facts = row.get("facts")
            facts_list = list(facts) if facts is not None and len(facts) else []
            text = " ".join(str(x) for x in facts_list) + " " + str(row.get("evidence_text") or "")
            if _STANDING_ORDER_RE.search(text):
                qty = _extract_quantity(text)
                base_price = row.get("new_price")
                if qty and base_price is not None and not pd.isna(base_price):
                    baseline = (row, qty, float(base_price))
                    break
        if baseline is None:
            continue
        base_row, (qty, unit), base_price = baseline

        for _, row in group.iterrows():
            if row["_date_sort"] <= base_row["_date_sort"]:
                continue
            new_price, eff_date = row.get("new_price"), row.get("effective_date")
            if new_price is None or pd.isna(new_price) or eff_date is None or pd.isna(eff_date):
                continue
            delta = float(new_price) - base_price
            if abs(delta) < 1e-6:
                continue
            scenarios.append({
                "ingredient": ingredient,
                "standing_quantity": qty,
                "unit": unit,
                "old_price": base_price,
                "new_price": float(new_price),
                "price_delta_per_unit": round(delta, 4),
                "effective_date": str(eff_date)[:10],
                "estimated_weekly_cost_delta_sar": round(qty * delta, 2),
                "assumptions": [
                    "assumes the standing-order quantity is held constant after the price change",
                    "derived from supplier email text, not a recipe/BOM -- a procurement-level cost "
                    "pressure figure, not an exact per-drink/per-item cost",
                ],
                "source_emails": [str(base_row.get("email_file")), str(row.get("email_file"))],
            })
    return scenarios


def cross_analyst_coincidences(candidate_findings: list[AnalystFinding]) -> list[str]:
    by_period: dict[tuple[str, str], dict[str, list[tuple[str, float]]]] = {}
    for f in candidate_findings:
        for ev in f.get("evidence", []):
            val = ev.get("value")
            if ev.get("unit") != "%" or not isinstance(val, (int, float)) or abs(val) < _MATERIALITY_PCT:
                continue
            key = (ev.get("period_start", ""), ev.get("period_end", ""))
            by_period.setdefault(key, {}).setdefault(f["analyst_name"], []).append((f["title"], val))

    notes = []
    for (start, end), by_analyst in by_period.items():
        if len(by_analyst) < 2:
            continue
        parts = [f"{analyst}: {items[0][0]!r} ({items[0][1]:+.1f}%)" for analyst, items in by_analyst.items()]
        notes.append(
            f"Same period {start[:10]}..{end[:10]}: " + "; ".join(parts)
            + " -- consider whether these are related before treating them as independent."
        )
    return notes


def calendar_overlaps(analysis_period: dict[str, str]) -> list[str]:
    try:
        from datetime import date

        from src.context.calendar import eid_al_adha_period, eid_al_fitr_period, ramadan_period

        start = date.fromisoformat(analysis_period["start"][:10])
        end = date.fromisoformat(analysis_period["end"][:10])
    except (KeyError, ValueError):
        return []

    overlaps = []
    for year in {start.year, end.year}:
        for period_fn, name in ((ramadan_period, "Ramadan"), (eid_al_fitr_period, "Eid al-Fitr"), (eid_al_adha_period, "Eid al-Adha")):
            period = period_fn(year)
            if period and start <= period.gregorian_end and end >= period.gregorian_start and name not in overlaps:
                overlaps.append(name)
    return overlaps
