import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Define analysis period
analysis_start = datetime(2026, 3, 23, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 3, 30, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 3, 16, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 3, 23, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)

# Filter POS for analysis period (week of Mar 23-30)
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start) &
    (pos_df['timestamp'] < analysis_end) &
    (pos_df['is_refund'] == False)
].copy()

# Filter POS for previous period (week of Mar 16-23)
pos_previous = pos_df[
    (pos_df['timestamp'] >= previous_start) &
    (pos_df['timestamp'] < previous_end) &
    (pos_df['is_refund'] == False)
].copy()

# Convert inventory week_starting to datetime
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'], utc=True)

# Filter inventory for analysis week
inventory_analysis = inventory_df[
    (inventory_df['week_starting'] >= analysis_start) &
    (inventory_df['week_starting'] < analysis_end)
].copy()

# Filter inventory for previous week
inventory_previous = inventory_df[
    (inventory_df['week_starting'] >= previous_start) &
    (inventory_df['week_starting'] < previous_end)
].copy()

findings = []

# ============================================================================
# FINDING 1: Waste Cost Analysis for Analysis Week
# ============================================================================

# Only include non-null waste observations
waste_records = inventory_analysis[
    (inventory_analysis['units_wasted'].notna()) &
    (inventory_analysis['units_wasted'] > 0) &
    (inventory_analysis['known_waste_cost_sar'].notna())
].copy()

if len(waste_records) > 0:
    total_waste_cost = waste_records['known_waste_cost_sar'].sum()
    total_waste_units = waste_records['units_wasted'].sum()

    # Calculate weekly revenue for analysis period
    weekly_revenue = pos_analysis['line_total_sar'].sum()

    if weekly_revenue > 0:
        waste_pct_revenue = (total_waste_cost / weekly_revenue) * 100

        # Get waste items for coverage
        waste_items = waste_records[['sku', 'item', 'units_wasted', 'known_waste_cost_sar']].to_dict('records')

        finding_1 = {
            "title": "Quantified Waste Cost Impact (Week of Mar 23-30, 2026)",
            "claim": f"Waste cost of SAR {total_waste_cost:.2f} represents {waste_pct_revenue:.2f}% of weekly revenue (SAR {weekly_revenue:.2f}) for the analysis week. {len(waste_records)} inventory items with non-null waste observations and known waste costs were included.",
            "finding_type": "cost_analysis",
            "metrics": {
                "total_waste_cost_sar": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": round(total_waste_cost, 2),
                    "denominator": None,
                    "period_start": "2026-03-23T00:00:00+00:00",
                    "period_end": "2026-03-30T00:00:00+00:00"
                },
                "total_waste_units": {
                    "value": round(total_waste_units, 2),
                    "unit": "units",
                    "numerator": round(total_waste_units, 2),
                    "denominator": None,
                    "period_start": "2026-03-23T00:00:00+00:00",
                    "period_end": "2026-03-30T00:00:00+00:00"
                },
                "weekly_revenue_sar": {
                    "value": round(weekly_revenue, 2),
                    "unit": "SAR",
                    "numerator": round(weekly_revenue, 2),
                    "denominator": None,
                    "period_start": "2026-03-23T00:00:00+00:00",
                    "period_end": "2026-03-30T00:00:00+00:00"
                },
                "waste_as_pct_revenue": {
                    "value": round(waste_pct_revenue, 2),
                    "unit": "%",
                    "numerator": round(total_waste_cost, 2),
                    "denominator": round(weekly_revenue, 2),
                    "period_start": "2026-03-23T00:00:00+00:00",
                    "period_end": "2026-03-30T00:00:00+00:00"
                }
            },
            "source_names": ["inventory", "pos"],
            "sample_size": len(waste_records),
            "coverage_notes": [
                f"Only non-null waste observations included: {len(waste_records)} inventory records with units_wasted > 0 and known_waste_cost_sar not null",
                f"Waste items: {[f\"{r['item']} ({r['units_wasted']} units, SAR {r['known_waste_cost_sar']:.2f})\" for r in waste_items]}",
                "Null waste values excluded per architecture contract",
                "Refunds excluded from revenue calculation",
                "Period: calendar week 2026-03-23 to 2026-03-30"
            ],
            "assumptions": [
                "Unit costs in inventory.unit_cost_sar represent actual cost basis for waste valuation",
                "known_waste_cost_sar values are accurate and pre-calculated",
                "Waste observations are complete for the analysis week",
                "No dead sensors or unknown waste categories in included records"
            ],
            "confidence": 0.75
        }
        findings.append(finding_1)

# ============================================================================
# FINDING 2: Item-Level Gross Profit Analysis (Top Contributors)
# ============================================================================

# Merge POS with menu to get unit costs
pos_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'price_sar']],
    on='sku',
    how='left'
)

# Calculate item-level metrics
pos_with_cost['gross_profit_sar'] = (pos_with_cost['unit_price_sar'] - pos_with_cost['unit_cost_sar']) * pos_with_cost['quantity']
pos_with_cost['cogs_sar'] = pos_with_cost['unit_cost_sar'] * pos_with_cost['quantity']

# Group by item
item_economics = pos_with_cost.groupby(['sku', 'item_name_en']).agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'cogs_sar': 'sum',
    'gross_profit_sar': 'sum',
    'unit_cost_sar': 'first',
    'unit_price_sar': 'first'
}).reset_index()

# Calculate margin
item_economics['gross_margin_pct'] = (item_economics['gross_profit_sar'] / item_economics['line_total_sar'] * 100).fillna(0)

# Sort by gross profit contribution
item_economics = item_economics.sort_values('gross_profit_sar', ascending=False)

# Get top 3 contributors
top_3_items = item_economics.head(3)

if len(top_3_items) > 0:
    total_gp = top_3_items['gross_profit_sar'].sum()
    total_revenue = top_3_items['line_total_sar'].sum()
    total_cogs = top_3_items['cogs_sar'].sum()
    total_qty = top_3_items['quantity'].sum()

    top_items_detail = top_3_items[['sku', 'item_name_en', 'quantity', 'line_total_sar', 'cogs_sar', 'gross_profit_sar', 'gross_margin_pct']].to_dict('records')

    finding_2 = {
        "title": "Top 3 Gross Profit Contributors (Week of Mar 23-30, 2026)",
        "claim": f"The top 3 items by gross profit contribution generated SAR {total_gp:.2f} in gross profit from SAR {total_revenue:.2f} revenue ({(total_gp/total_revenue*100):.1f}% margin) across {int(total_qty)} units sold. COGS for these items was SAR {total_cogs:.2f}.",
        "finding_type": "menu_engineering",
        "metrics": {
            "top_3_gross_profit_sar": {
                "value": round(total_gp, 2),
                "unit": "SAR",
                "numerator": round(total_gp, 2),
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+00:00",
                "period_end": "2026-03-30T00:00:00+00:00"
            },
            "top_3_revenue_sar": {
                "value": round(total_revenue, 2),
                "unit": "SAR",
                "numerator": round(total_revenue, 2),
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+00:00",
                "period_end": "2026-03-30T00:00:00+00:00"
            },
            "top_3_cogs_sar": {
                "value": round(total_cogs, 2),
                "unit": "SAR",
                "numerator": round(total_cogs, 2),
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+00:00",
                "period_end": "2026-03-30T00:00:00+00:00"
            },
            "top_3_gross_margin_pct": {
                "value": round((total_gp/total_revenue*100), 2),
                "unit": "%",
                "numerator": round(total_gp, 2),
                "denominator": round(total_revenue, 2),
                "period_start": "2026-03-23T00:00:00+00:00",
                "period_end": "2026-03-30T00:00:00+00:00"
            },
            "top_3_units_sold": {
                "value": int(total_qty),
                "unit": "units",
                "numerator": int(total_qty),
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+00:00",
                "period_end": "2026-03-30T00:00:00+00:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(top_3_items),
        "coverage_notes": [
            f"Top 3 items by gross profit: {[f\"{r['item_name_en']} (SKU {r['sku']})\" for r in top_items_detail]}",
            f"Item details: {[f\"{r['item_name_en']}: {int(r['quantity'])} units, SAR {r['gross_profit_sar']:.2f} GP, {r['gross_margin_pct']:.1f}% margin\" for r in top_items_detail]}",
            "Refunds excluded from analysis",
            "Only items with non-null unit_cost_sar in menu included",
            "Period: calendar week 2026-03-23 to 2026-03-30"
        ],
        "assumptions": [
            "menu.unit_cost_sar represents actual COGS per unit",
            "menu.price_sar is the standard menu price (discounts captured in POS line_total_sar)",
            "No recipe/BOM adjustments applied; unit costs are as stated in menu",
            "Gross profit = (unit_price - unit_cost) * quantity, using POS unit_price_sar"
        ],
        "confidence": 0.85
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Supplier Price Change Impact (if any detected)
# ============================================================================

# Filter emails for price changes with effective dates
price_changes = emails_df[
    (emails_df['old_price'].notna()) &
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    # Convert effective_date to datetime
    price_changes['effective_date'] = pd.to_datetime(price_changes['effective_date'], utc=True)

    # Filter for changes effective during or before analysis period
    relevant_changes = price_changes[
        price_changes['effective_date'] <= analysis_end
    ].copy()

    if len(relevant_changes) > 0:
        # Take the most recent change per entity
        relevant_changes = relevant_changes.sort_values('effective_date', ascending=False).drop_duplicates('entity_or_ingredient')

        # Calculate price delta
        relevant_changes['price_delta'] = relevant_changes['new_price'] - relevant_changes['old_price']
        relevant_changes['pct_change'] = (relevant_changes['price_delta'] / relevant_changes['old_price'] * 100)

        # Get first change for finding
        change = relevant_changes.iloc[0]

        finding_3 = {
            "title": f"Supplier Price Change: {change['entity_or_ingredient']}",
            "claim": f"Supplier email dated {change['date']} reports price change for {change['entity_or_ingredient']} from {change['old_price']:.2f} to {change['new_price']:.2f} {change['currency']}/{change['unit']} (effective {change['effective_date'].strftime('%Y-%m-%d')}), representing a {change['pct_change']:.1f}% change. No standing order quantity or payment terms were extracted; actual procurement cost impact depends on order volume and timing.",
            "finding_type": "supplier_cost_change",
            "metrics": {
                "old_price": {
                    "value": round(change['old_price'], 4),
                    "unit": f"{change['currency']}/{change['unit']}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": None,
                    "period_end": None
                },
                "new_price": {
                    "value": round(change['new_price'], 4),
                    "unit": f"{change['currency']}/{change['unit']}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": None,
                    "period_end": None
                },
                "price_delta": {
                    "value": round(change['price_delta'], 4),
                    "unit": f"{change['currency']}/{change['unit']}",
                    "numerator": round(change['price_delta'], 4),
                    "denominator": None,
                    "period_start": None,
                    "period_end": None
                },
                "pct_change": {
                    "value": round(change['pct_change'], 2),
                    "unit": "%",
                    "numerator": round(change['price_delta'], 4),
                    "denominator": round(change['old_price'], 4),
                    "period_start": None,
                    "period_end": None
                }
            },
            "source_names": ["emails"],
            "sample_size": 1,
            "coverage_notes": [
                f"Supplier: {change['sender']}",
                f"Entity/Ingredient: {change['entity_or_ingredient']}",
                f"Email subject: {change['subject']}",
                f"Effective date: {change['effective_date'].strftime('%Y-%m-%d')}",
                "No standing order quantity extracted from email",
                "No payment terms or contract duration extracted"
            ],
            "assumptions": [
                "Email extraction confidence is as stated in emails.confidence",
                "Price change applies to future purchases only; no retroactive adjustment assumed",
                "No recipe/BOM exists to calculate per-drink impact",
                "Actual procurement cost impact requires knowledge of order volume and timing"
            ],
            "confidence": float(change['confidence']) if pd.notna(change['confidence']) else 0.6
        }
        findings.append(finding_3)

# ============================================================================
# Build output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
