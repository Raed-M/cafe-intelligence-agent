import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

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

# Define analysis periods
analysis_start = "2026-06-29"
analysis_end = "2026-07-06"
previous_start = "2026-06-22"
previous_end = "2026-06-29"

# Convert timestamp columns to datetime
pos_df['timestamp_local'] = pd.to_datetime(pos_df['timestamp_local'])
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])

# Filter POS for analysis period (exclude refunds for revenue, but include for quantity tracking)
pos_analysis = pos_df[
    (pos_df['calendar_date'] >= analysis_start) & 
    (pos_df['calendar_date'] < analysis_end)
].copy()

pos_previous = pos_df[
    (pos_df['calendar_date'] >= previous_start) & 
    (pos_df['calendar_date'] < previous_end)
].copy()

# Calculate item-level economics for analysis period
# Group by SKU and item_name to get totals
item_economics = pos_analysis[pos_analysis['is_refund'] == False].groupby('sku').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'item_name_en': 'first',
    'category': 'first'
}).reset_index()

# Merge with menu to get unit costs
item_economics = item_economics.merge(
    menu_df[['sku', 'unit_cost_sar', 'price_sar']],
    on='sku',
    how='left'
)

# Calculate COGS and gross profit
item_economics['cogs_sar'] = item_economics['quantity'] * item_economics['unit_cost_sar']
item_economics['gross_profit_sar'] = item_economics['line_total_sar'] - item_economics['cogs_sar']
item_economics['gross_margin_pct'] = (item_economics['gross_profit_sar'] / item_economics['line_total_sar'] * 100).round(2)

# Sort by gross profit descending
item_economics_sorted = item_economics.sort_values('gross_profit_sar', ascending=False)

# Get top 5 items by gross profit
top_5_items = item_economics_sorted.head(5)

# Calculate totals for top 5
top_5_total_revenue = top_5_items['line_total_sar'].sum()
top_5_total_gp = top_5_items['gross_profit_sar'].sum()

# Calculate total revenue and GP across all items in analysis period
total_revenue_analysis = item_economics['line_total_sar'].sum()
total_gp_analysis = item_economics['gross_profit_sar'].sum()

# Waste analysis for analysis period
inventory_analysis = inventory_df[
    (inventory_df['week_starting'] >= analysis_start) & 
    (inventory_df['week_starting'] < analysis_end)
].copy()

# Only count non-null waste costs
waste_cost_analysis = inventory_analysis['known_waste_cost_sar'].sum()
waste_items_count = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].shape[0]

# Supplier price changes from emails
supplier_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['category'] == 'supplier_price_change')
].copy()

# Calculate percentage changes
supplier_changes['percentage_change'] = (
    (supplier_changes['new_price'] - supplier_changes['old_price']) / 
    supplier_changes['old_price'] * 100
).round(2)

# Sort by percentage change descending
supplier_changes_sorted = supplier_changes.sort_values('percentage_change', ascending=False)

# Build findings
findings = []

# Finding 1: Top 5 items by gross profit
if len(top_5_items) > 0:
    finding_1 = {
        "title": "Top 5 Items by Gross Profit (Analysis Week)",
        "claim": f"Five items generated {top_5_total_gp:.1f} SAR in gross profit during the analysis week (2026-06-29 to 2026-07-06).",
        "finding_type": "item_economics",
        "metrics": {
            "top_5_gross_profit_sar": {
                "value": round(top_5_total_gp, 1),
                "unit": "SAR",
                "numerator": round(top_5_total_gp, 1),
                "denominator": None,
                "period_start": "2026-06-29T00:00:00+03:00",
                "period_end": "2026-07-06T00:00:00+03:00"
            },
            "top_5_revenue_sar": {
                "value": round(top_5_total_revenue, 1),
                "unit": "SAR",
                "numerator": round(top_5_total_revenue, 1),
                "denominator": None,
                "period_start": "2026-06-29T00:00:00+03:00",
                "period_end": "2026-07-06T00:00:00+03:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(top_5_items),
        "coverage_notes": [
            "Analysis period: 2026-06-29 to 2026-07-06",
            "Refunds excluded from revenue and quantity calculations",
            "Unit costs from menu_items.unit_cost_sar applied uniformly",
            "No recipe/BOM available; ingredient-level variance not captured"
        ],
        "assumptions": [
            "Unit cost from menu_items.unit_cost_sar applied uniformly to all sales",
            "No recipe/BOM available; actual ingredient costs may vary by supplier or preparation method",
            "Discount amounts already deducted in line_total_sar"
        ],
        "confidence": 0.95
    }
    
    # Add individual item metrics
    for idx, row in top_5_items.iterrows():
        item_key = f"item_{row['sku']}_gp_sar"
        finding_1['metrics'][item_key] = {
            "value": round(row['gross_profit_sar'], 1),
            "unit": "SAR",
            "numerator": round(row['gross_profit_sar'], 1),
            "denominator": None,
            "period_start": "2026-06-29T00:00:00+03:00",
            "period_end": "2026-07-06T00:00:00+03:00"
        }
        item_margin_key = f"item_{row['sku']}_margin_pct"
        finding_1['metrics'][item_margin_key] = {
            "value": round(row['gross_margin_pct'], 2),
            "unit": "%",
            "numerator": round(row['gross_profit_sar'], 1),
            "denominator": round(row['line_total_sar'], 1),
            "period_start": "2026-06-29T00:00:00+03:00",
            "period_end": "2026-07-06T00:00:00+03:00"
        }
    
    findings.append(finding_1)

# Finding 2: Waste cost impact
if waste_cost_analysis > 0 and waste_items_count > 0:
    finding_2 = {
        "title": "Quantified Waste Cost (Analysis Week)",
        "claim": f"Known waste cost totaled {waste_cost_analysis:.2f} SAR across {waste_items_count} items during the analysis week.",
        "finding_type": "waste_cost",
        "metrics": {
            "waste_cost_sar": {
                "value": round(waste_cost_analysis, 2),
                "unit": "SAR",
                "numerator": round(waste_cost_analysis, 2),
                "denominator": None,
                "period_start": "2026-06-29T00:00:00+03:00",
                "period_end": "2026-07-06T00:00:00+03:00"
            },
            "waste_items_count": {
                "value": waste_items_count,
                "unit": "count",
                "numerator": waste_items_count,
                "denominator": None,
                "period_start": "2026-06-29T00:00:00+03:00",
                "period_end": "2026-07-06T00:00:00+03:00"
            }
        },
        "source_names": ["inventory"],
        "sample_size": waste_items_count,
        "coverage_notes": [
            "Only non-null waste cost observations included (known_waste_cost_sar)",
            "Blank waste values treated as unknown, not zero",
            "Analysis period: 2026-06-29 to 2026-07-06"
        ],
        "assumptions": [
            "known_waste_cost_sar field represents actual quantified waste cost",
            "Missing waste values indicate unknown waste, not zero waste"
        ],
        "confidence": 0.90
    }
    findings.append(finding_2)

# Finding 3: Supplier price changes
if len(supplier_changes_sorted) > 0:
    top_supplier_change = supplier_changes_sorted.iloc[0]
    
    finding_3 = {
        "title": "Supplier Price Changes Detected",
        "claim": f"Supplier price for {top_supplier_change['entity_or_ingredient']} increased from {top_supplier_change['old_price']} to {top_supplier_change['new_price']} {top_supplier_change['currency']}/{top_supplier_change['unit']}, representing a {top_supplier_change['percentage_change']:.2f}% increase.",
        "finding_type": "supplier_price_change",
        "metrics": {
            "ingredient": {
                "value": str(top_supplier_change['entity_or_ingredient']),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-29T00:00:00+03:00",
                "period_end": "2026-07-06T00:00:00+03:00"
            },
            "old_price": {
                "value": round(top_supplier_change['old_price'], 2),
                "unit": f"{top_supplier_change['currency']}/{top_supplier_change['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-29T00:00:00+03:00",
                "period_end": "2026-07-06T00:00:00+03:00"
            },
            "new_price": {
                "value": round(top_supplier_change['new_price'], 2),
                "unit": f"{top_supplier_change['currency']}/{top_supplier_change['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-29T00:00:00+03:00",
                "period_end": "2026-07-06T00:00:00+03:00"
            },
            "percentage_change": {
                "value": round(top_supplier_change['percentage_change'], 2),
                "unit": "%",
                "numerator": round(top_supplier_change['new_price'] - top_supplier_change['old_price'], 2),
                "denominator": round(top_supplier_change['old_price'], 2),
                "period_start": "2026-06-29T00:00:00+03:00",
                "period_end": "2026-07-06T00:00:00+03:00"
            }
        },
        "source_names": ["emails"],
        "sample_size": 1,
        "coverage_notes": [
            "Supplier price change extracted from email evidence",
            "No recipe/BOM available; per-drink impact not calculated",
            "Effective date from email: " + str(top_supplier_change['effective_date']),
            "Evidence confidence: " + str(top_supplier_change['confidence'])
        ],
        "assumptions": [
            "Price change applies to specified ingredient only",
            "No recipe/BOM available; actual menu item cost impact cannot be quantified",
            "Effective date represents when price change took effect",
            "Standing order volume and payment terms assumed unchanged"
        ],
        "confidence": float(top_supplier_change['confidence']) if pd.notna(top_supplier_change['confidence']) else 0.85
    }
    findings.append(finding_3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write to output path
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)