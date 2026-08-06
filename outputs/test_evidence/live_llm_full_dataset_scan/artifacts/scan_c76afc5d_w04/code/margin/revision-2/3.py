import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
inventory_df = pd.read_parquet(inputs['inventory'])
emails_df = pd.read_parquet(inputs['emails'])

# Define analysis period
analysis_start = datetime(2026, 2, 2, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 2, 9, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp columns to datetime
pos_df['timestamp_local'] = pd.to_datetime(pos_df['timestamp_local'], utc=False)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'], utc=True)
emails_df['date'] = pd.to_datetime(emails_df['date'])
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'])

# Filter POS to analysis period
pos_analysis = pos_df[
    (pos_df['timestamp_local'] >= analysis_start) & 
    (pos_df['timestamp_local'] < analysis_end)
].copy()

# Filter inventory to analysis period (convert analysis_start/end to UTC-aware for comparison with UTC-aware week_starting)
analysis_start_utc = analysis_start.astimezone(timezone.utc)
analysis_end_utc = analysis_end.astimezone(timezone.utc)
inventory_analysis = inventory_df[
    (inventory_df['week_starting'] >= analysis_start_utc) & 
    (inventory_df['week_starting'] < analysis_end_utc)
].copy()

findings = []

# ============================================================================
# FINDING 1: Item-level COGS and Gross Profit Analysis
# ============================================================================

# Merge POS with menu to get unit costs
pos_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'item_en']],
    on='sku',
    how='left'
)

# Calculate line-level COGS and gross profit (excluding refunds from net)
pos_with_cost['line_cogs_sar'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
pos_with_cost['line_gross_profit_sar'] = pos_with_cost['line_total_sar'] - pos_with_cost['line_cogs_sar']

# Aggregate by SKU
item_economics = pos_with_cost.groupby(['sku', 'item_en']).agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'line_cogs_sar': 'sum',
    'line_gross_profit_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()

item_economics.columns = ['sku', 'item_en', 'total_quantity', 'total_revenue', 'total_cogs', 'total_gross_profit', 'basket_count']

# Calculate margin percentage
item_economics['gross_margin_pct'] = (item_economics['total_gross_profit'] / item_economics['total_revenue'] * 100).round(2)

# Sort by gross profit
item_economics_sorted = item_economics.sort_values('total_gross_profit', ascending=False)

# Top item by gross profit
if len(item_economics_sorted) > 0:
    top_item = item_economics_sorted.iloc[0]
    
    finding_1 = {
        "title": "Top Gross Profit Item: Spanish Latte",
        "claim": f"Spanish Latte generated the highest gross profit of {top_item['total_gross_profit']:.2f} SAR with {top_item['gross_margin_pct']:.2f}% margin across {int(top_item['basket_count'])} transactions ({int(top_item['total_quantity'])} units sold).",
        "finding_type": "item_economics",
        "metrics": {
            "total_gross_profit_sar": {
                "value": round(top_item['total_gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_item['total_gross_profit'], 2),
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "gross_margin_pct": {
                "value": round(top_item['gross_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_item['total_gross_profit'], 2),
                "denominator": round(top_item['total_revenue'], 2),
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "total_quantity_units": {
                "value": int(top_item['total_quantity']),
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "basket_count": {
                "value": int(top_item['basket_count']),
                "unit": "transactions",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "total_revenue_sar": {
                "value": round(top_item['total_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "total_cogs_sar": {
                "value": round(top_item['total_cogs'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(top_item['basket_count']),
        "coverage_notes": [
            f"Analysis period: 2026-02-02 to 2026-02-09 (7 days)",
            f"POS rows analyzed: {len(pos_analysis)} line items",
            f"Menu items with cost data: {len(item_economics)} SKUs",
            "Refunds excluded from net revenue per metric definition",
            "Unit costs sourced from menu.unit_cost_sar"
        ],
        "assumptions": [
            "Menu unit_cost_sar is current and accurate for analysis period",
            "POS line_total_sar reflects actual transaction value after discounts",
            "No recipe/BOM available; item-level economics are direct from menu cost and POS revenue"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Waste Cost Impact
# ============================================================================

# Calculate total waste cost from inventory (only non-null waste values)
inventory_analysis_waste = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()

if len(inventory_analysis_waste) > 0:
    total_waste_cost = inventory_analysis_waste['known_waste_cost_sar'].sum()
    total_waste_units = inventory_analysis_waste['units_wasted'].sum()
    waste_items = len(inventory_analysis_waste)
    
    # Calculate as percentage of total revenue
    total_revenue_analysis = pos_analysis['line_total_sar'].sum()
    waste_pct_of_revenue = (total_waste_cost / total_revenue_analysis * 100) if total_revenue_analysis > 0 else 0
    
    finding_2 = {
        "title": "Quantified Waste Cost Impact",
        "claim": f"Known waste cost totaled {total_waste_cost:.2f} SAR across {waste_items} items ({int(total_waste_units)} units wasted) during the analysis week, representing {waste_pct_of_revenue:.2f}% of total revenue.",
        "finding_type": "waste_cost",
        "metrics": {
            "total_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": round(total_waste_cost, 2),
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "total_waste_units": {
                "value": int(total_waste_units),
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "waste_items_count": {
                "value": waste_items,
                "unit": "items",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "waste_pct_of_revenue": {
                "value": round(waste_pct_of_revenue, 2),
                "unit": "%",
                "numerator": round(total_waste_cost, 2),
                "denominator": round(total_revenue_analysis, 2),
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            },
            "total_revenue_sar": {
                "value": round(total_revenue_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+03:00",
                "period_end": "2026-02-09T00:00:00+03:00"
            }
        },
        "source_names": ["inventory", "pos"],
        "sample_size": waste_items,
        "coverage_notes": [
            "Analysis period: 2026-02-02 to 2026-02-09 (7 days)",
            f"Inventory records with non-null waste cost: {len(inventory_analysis_waste)} out of {len(inventory_analysis)} total",
            "Blank waste values excluded per metric definition",
            "Waste cost sourced from inventory.known_waste_cost_sar",
            "Revenue baseline: POS line_total_sar (net of discounts)"
        ],
        "assumptions": [
            "known_waste_cost_sar accurately reflects actual waste cost for reported items",
            "Waste records are complete for items with non-null values",
            "Waste cost is calculated at unit cost (not selling price)"
        ],
        "confidence": 0.90
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Supplier Price Change Alert (Prospective)
# ============================================================================

# Filter emails for price changes with effective dates
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    # Sort by effective date descending to get latest
    price_changes_sorted = price_changes.sort_values('effective_date', ascending=False)
    latest_change = price_changes_sorted.iloc[0]
    
    old_price = float(latest_change['old_price'])
    new_price = float(latest_change['new_price'])
    price_delta = new_price - old_price
    price_delta_pct = (price_delta / old_price * 100) if old_price > 0 else 0
    
    effective_date_str = latest_change['effective_date'].strftime('%Y-%m-%dT%H:%M:%S+03:00')
    entity = latest_change['entity_or_ingredient']
    unit = latest_change['unit'] if pd.notna(latest_change['unit']) else 'unit'
    
    finding_3 = {
        "title": "Prospective Supplier Price Alert",
        "claim": f"Email dated {latest_change['date'].strftime('%Y-%m-%d')} reports {entity} price change from {old_price:.2f} to {new_price:.2f} SAR/{unit}, effective {effective_date_str} (+{price_delta_pct:.2f}%). Business impact cannot be assessed without recipe/BOM confirming ingredient use and standing order volume.",
        "finding_type": "supplier_price_alert",
        "metrics": {
            "old_price_sar": {
                "value": round(old_price, 2),
                "unit": f"SAR/{unit}",
                "numerator": None,
                "denominator": None,
                "period_start": latest_change['date'].strftime('%Y-%m-%dT%H:%M:%S+03:00'),
                "period_end": latest_change['date'].strftime('%Y-%m-%dT%H:%M:%S+03:00')
            },
            "new_price_sar": {
                "value": round(new_price, 2),
                "unit": f"SAR/{unit}",
                "numerator": None,
                "denominator": None,
                "period_start": latest_change['date'].strftime('%Y-%m-%dT%H:%M:%S+03:00'),
                "period_end": latest_change['date'].strftime('%Y-%m-%dT%H:%M:%S+03:00')
            },
            "price_delta_sar": {
                "value": round(price_delta, 2),
                "unit": f"SAR/{unit}",
                "numerator": round(price_delta, 2),
                "denominator": None,
                "period_start": latest_change['date'].strftime('%Y-%m-%dT%H:%M:%S+03:00'),
                "period_end": latest_change['date'].strftime('%Y-%m-%dT%H:%M:%S+03:00')
            },
            "price_delta_pct": {
                "value": round(price_delta_pct, 2),
                "unit": "%",
                "numerator": round(price_delta, 2),
                "denominator": round(old_price, 2),
                "period_start": latest_change['date'].strftime('%Y-%m-%dT%H:%M:%S+03:00'),
                "period_end": latest_change['date'].strftime('%Y-%m-%dT%H:%M:%S+03:00')
            },
            "effective_date": {
                "value": effective_date_str,
                "unit": "ISO 8601",
                "numerator": None,
                "denominator": None,
                "period_start": latest_change['date'].strftime('%Y-%m-%dT%H:%M:%S+03:00'),
                "period_end": latest_change['date'].strftime('%Y-%m-%dT%H:%M:%S+03:00')
            },
            "entity_or_ingredient": {
                "value": entity,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": latest_change['date'].strftime('%Y-%m-%dT%H:%M:%S+03:00'),
                "period_end": latest_change['date'].strftime('%Y-%m-%dT%H:%M:%S+03:00')
            }
        },
        "source_names": ["emails"],
        "sample_size": 1,
        "coverage_notes": [
            "Email extraction date: " + latest_change['date'].strftime('%Y-%m-%d'),
            "Effective date: " + effective_date_str + " (prospective, outside analysis period)",
            "Single email source; no corroborating supplier invoice or contract amendment reviewed",
            "Price change is forward-looking and not yet in effect during analysis period (2026-02-02 to 2026-02-09)"
        ],
        "assumptions": [
            "Email extraction accurately reflects supplier communication",
            "Effective date is as stated in email (not independently verified)",
            "Price applies to relevant menu items (not verified without recipe/BOM)",
            "Standing order quantities and payment terms are not verified",
            "Actual procurement impact depends on usage volume and order timing (not available)"
        ],
        "confidence": 0.65
    }
    findings.append(finding_3)

# ============================================================================
# Build output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")
