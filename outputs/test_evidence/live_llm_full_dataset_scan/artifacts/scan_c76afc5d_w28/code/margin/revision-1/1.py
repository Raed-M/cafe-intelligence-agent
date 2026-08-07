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
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Define analysis periods
analysis_start = "2026-07-20T00:00:00+03:00"
analysis_end = "2026-07-27T00:00:00+03:00"
previous_start = "2026-07-13T00:00:00+03:00"
previous_end = "2026-07-20T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)
previous_start_dt = pd.to_datetime(previous_start)
previous_end_dt = pd.to_datetime(previous_end)

# Parse POS timestamp
pos_df['timestamp_dt'] = pd.to_datetime(pos_df['timestamp'])

# Filter POS for analysis period
pos_analysis = pos_df[(pos_df['timestamp_dt'] >= analysis_start_dt) & 
                       (pos_df['timestamp_dt'] < analysis_end_dt)].copy()

# Filter POS for previous period
pos_previous = pos_df[(pos_df['timestamp_dt'] >= previous_start_dt) & 
                       (pos_df['timestamp_dt'] < previous_end_dt)].copy()

# Parse inventory week_starting
inventory_df['week_starting_dt'] = pd.to_datetime(inventory_df['week_starting'])

# Filter inventory for analysis week
inventory_analysis = inventory_df[
    (inventory_df['week_starting_dt'] >= analysis_start_dt) & 
    (inventory_df['week_starting_dt'] < analysis_end_dt)
].copy()

# Filter inventory for previous week
inventory_previous = inventory_df[
    (inventory_df['week_starting_dt'] >= previous_start_dt) & 
    (inventory_df['week_starting_dt'] < previous_end_dt)
].copy()

findings = []

# ============================================================================
# FINDING 1: Item-level COGS and Gross Profit Analysis (Analysis Period)
# ============================================================================

# Merge POS with menu to get unit costs
pos_analysis_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'price_sar']], 
    on='sku', 
    how='left'
)

# Calculate item-level metrics (excluding refunds)
pos_analysis_no_refund = pos_analysis_with_cost[~pos_analysis_with_cost['is_refund']].copy()

# Calculate COGS and gross profit per line
pos_analysis_no_refund['cogs_sar'] = pos_analysis_no_refund['quantity'] * pos_analysis_no_refund['unit_cost_sar']
pos_analysis_no_refund['gross_profit_sar'] = pos_analysis_no_refund['line_total_sar'] - pos_analysis_no_refund['cogs_sar']
pos_analysis_no_refund['gross_margin_pct'] = (pos_analysis_no_refund['gross_profit_sar'] / pos_analysis_no_refund['line_total_sar'] * 100).fillna(0)

# Aggregate by SKU
sku_metrics = pos_analysis_no_refund.groupby('sku').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'cogs_sar': 'sum',
    'gross_profit_sar': 'sum',
    'item_name_en': 'first',
    'category': 'first'
}).reset_index()

sku_metrics['gross_margin_pct'] = (sku_metrics['gross_profit_sar'] / sku_metrics['line_total_sar'] * 100).round(2)
sku_metrics = sku_metrics.sort_values('gross_profit_sar', ascending=False)

# Total metrics
total_revenue = pos_analysis_no_refund['line_total_sar'].sum()
total_cogs = pos_analysis_no_refund['cogs_sar'].sum()
total_gross_profit = total_revenue - total_cogs
total_gross_margin = (total_gross_profit / total_revenue * 100) if total_revenue > 0 else 0

# Count unique transactions
unique_transactions = pos_analysis_no_refund['transaction_id'].nunique()

finding_1 = {
    "title": "Item-Level Gross Profit and Margin Analysis (Analysis Week)",
    "claim": f"During the analysis week (2026-07-20 to 2026-07-27), total gross profit was {total_gross_profit:.2f} SAR across {unique_transactions} transactions with {total_gross_margin:.2f}% gross margin. Top performer: {sku_metrics.iloc[0]['item_name_en']} with {sku_metrics.iloc[0]['gross_profit_sar']:.2f} SAR profit.",
    "finding_type": "margin_analysis",
    "metrics": {
        "total_revenue_sar": {
            "value": round(total_revenue, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "total_cogs_sar": {
            "value": round(total_cogs, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "total_gross_profit_sar": {
            "value": round(total_gross_profit, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "gross_margin_pct": {
            "value": round(total_gross_margin, 2),
            "unit": "%",
            "numerator": round(total_gross_profit, 2),
            "denominator": round(total_revenue, 2),
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "top_item_name": {
            "value": sku_metrics.iloc[0]['item_name_en'],
            "unit": None,
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "top_item_profit_sar": {
            "value": round(sku_metrics.iloc[0]['gross_profit_sar'], 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    },
    "source_names": ["pos", "menu"],
    "sample_size": len(pos_analysis_no_refund),
    "coverage_notes": [
        f"Analysis period: 2026-07-20 to 2026-07-27",
        f"Refunds excluded from calculations",
        f"Unit costs sourced from menu.parquet",
        f"Unique transactions: {unique_transactions}",
        f"SKUs with cost data: {sku_metrics.shape[0]}"
    ],
    "assumptions": [
        "Menu unit_cost_sar is current and applicable to all sales in period",
        "Line totals are net of discounts",
        "No recipe/BOM adjustments applied"
    ],
    "confidence": 0.95
}

findings.append(finding_1)

# ============================================================================
# FINDING 2: Waste Cost Impact (Inventory Data)
# ============================================================================

# Calculate waste cost for analysis week
waste_analysis = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()

if len(waste_analysis) > 0:
    total_waste_cost_analysis = waste_analysis['known_waste_cost_sar'].sum()
    total_units_wasted_analysis = waste_analysis['units_wasted'].sum()
    
    # Compare to previous week
    waste_previous = inventory_previous[inventory_previous['known_waste_cost_sar'].notna()].copy()
    total_waste_cost_previous = waste_previous['known_waste_cost_sar'].sum() if len(waste_previous) > 0 else 0
    total_units_wasted_previous = waste_previous['units_wasted'].sum() if len(waste_previous) > 0 else 0
    
    waste_change = total_waste_cost_analysis - total_waste_cost_previous
    waste_change_pct = (waste_change / total_waste_cost_previous * 100) if total_waste_cost_previous > 0 else 0
    
    finding_2 = {
        "title": "Quantified Waste Cost Impact",
        "claim": f"Known waste cost in analysis week was {total_waste_cost_analysis:.2f} SAR ({total_units_wasted_analysis:.0f} units), representing a {waste_change_pct:+.1f}% change from previous week ({total_waste_cost_previous:.2f} SAR).",
        "finding_type": "waste_cost_analysis",
        "metrics": {
            "waste_cost_sar_analysis": {
                "value": round(total_waste_cost_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "units_wasted_analysis": {
                "value": round(total_units_wasted_analysis, 0),
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "waste_cost_sar_previous": {
                "value": round(total_waste_cost_previous, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": previous_start,
                "period_end": previous_end
            },
            "waste_cost_change_sar": {
                "value": round(waste_change, 2),
                "unit": "SAR",
                "numerator": round(waste_change, 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "waste_cost_change_pct": {
                "value": round(waste_change_pct, 2),
                "unit": "%",
                "numerator": round(waste_change, 2),
                "denominator": round(total_waste_cost_previous, 2) if total_waste_cost_previous > 0 else None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(waste_analysis),
        "coverage_notes": [
            f"Only non-null known_waste_cost_sar values included",
            f"Analysis week items with waste data: {len(waste_analysis)}",
            f"Previous week items with waste data: {len(waste_previous)}",
            f"Blank waste values excluded per methodology"
        ],
        "assumptions": [
            "known_waste_cost_sar reflects actual waste cost",
            "Waste cost calculation methodology consistent across periods"
        ],
        "confidence": 0.90
    }
    
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Supplier Price Changes from Email Evidence
# ============================================================================

# Parse email dates
emails_df['date_dt'] = pd.to_datetime(emails_df['date'])
emails_df['effective_date_dt'] = pd.to_datetime(emails_df['effective_date'])

# Filter for price changes with both old and new prices
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['old_price'] != 0)
].copy()

if len(price_changes) > 0:
    # Calculate percentage change
    price_changes['price_change_pct'] = (
        (price_changes['new_price'] - price_changes['old_price']) / 
        price_changes['old_price'] * 100
    ).round(2)
    
    # Sort by effective date
    price_changes = price_changes.sort_values('effective_date_dt', ascending=False)
    
    # Get most recent price change
    most_recent = price_changes.iloc[0]
    
    ingredient_name = most_recent['entity_or_ingredient']
    old_price = most_recent['old_price']
    new_price = most_recent['new_price']
    price_change_pct = most_recent['price_change_pct']
    unit = most_recent['unit']
    effective_date = most_recent['effective_date']
    
    # Check if this ingredient affects any menu items (no recipe/BOM, so cannot calculate exact impact)
    finding_3 = {
        "title": "Supplier Price Change Detection",
        "claim": f"Email evidence identifies {len(price_changes)} supplier price change(s). Most recent: {ingredient_name} price changed {price_change_pct:+.2f}% (from {old_price} to {new_price} {unit}) effective {effective_date}. No recipe/BOM available to calculate exact menu item cost impact.",
        "finding_type": "supplier_cost_change",
        "metrics": {
            "total_price_changes_identified": {
                "value": len(price_changes),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "most_recent_ingredient": {
                "value": ingredient_name,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "most_recent_old_price": {
                "value": round(old_price, 2),
                "unit": most_recent['currency'],
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "most_recent_new_price": {
                "value": round(new_price, 2),
                "unit": most_recent['currency'],
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "most_recent_price_change_pct": {
                "value": price_change_pct,
                "unit": "%",
                "numerator": round(new_price - old_price, 2),
                "denominator": round(old_price, 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "most_recent_unit_of_measure": {
                "value": unit,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "most_recent_effective_date": {
                "value": effective_date,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["emails"],
        "sample_size": len(price_changes),
        "coverage_notes": [
            f"Total price changes with both old and new prices: {len(price_changes)}",
            f"Email extraction confidence varies by record",
            f"No recipe/BOM data available to map ingredient costs to menu items"
        ],
        "assumptions": [
            "Email extraction accurately captured supplier price communications",
            "Effective dates are as stated in supplier communications",
            "Price changes apply to future purchases only (no retroactive adjustment assumed)"
        ],
        "confidence": 0.85
    }
    
    findings.append(finding_3)

# ============================================================================
# Build output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write to output path
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)