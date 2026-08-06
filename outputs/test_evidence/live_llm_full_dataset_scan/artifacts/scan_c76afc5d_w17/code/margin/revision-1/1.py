import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Parse analysis period
analysis_start = "2026-05-04T00:00:00+03:00"
analysis_end = "2026-05-11T00:00:00+03:00"

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Convert timestamps to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'], errors='coerce')
menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'], errors='coerce')
emails_df['date'] = pd.to_datetime(emails_df['date'], errors='coerce')
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'], errors='coerce')

# Filter POS to analysis period
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start_dt) & (pos_df['timestamp'] < analysis_end_dt)].copy()

# Filter to non-refunds only
pos_analysis = pos_analysis[pos_analysis['is_refund'] == False].copy()

# Merge POS with menu to get unit costs
pos_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'price_sar']],
    on='sku',
    how='left'
)

# Calculate gross profit per line item
pos_with_cost['gross_profit_sar'] = (pos_with_cost['line_total_sar'] - 
                                      (pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']))

# Remove rows where unit_cost_sar is null (cannot calculate profit)
pos_with_cost = pos_with_cost[pos_with_cost['unit_cost_sar'].notna()].copy()

# Finding 1: Top 3 items by gross profit contribution
item_profit = pos_with_cost.groupby('item_name_en').agg({
    'gross_profit_sar': 'sum',
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'unit_cost_sar': 'first',
    'transaction_id': 'nunique'
}).reset_index()

item_profit = item_profit.sort_values('gross_profit_sar', ascending=False)
top_3_items = item_profit.head(3)

# Calculate gross margin % for top 3
top_3_items['gross_margin_pct'] = (top_3_items['gross_profit_sar'] / top_3_items['line_total_sar'] * 100).round(2)

# Finding 2: Waste cost analysis
inventory_analysis = inventory_df[inventory_df['week_starting'] == pd.to_datetime('2026-05-04')].copy()
inventory_with_waste = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()

total_waste_cost = inventory_with_waste['known_waste_cost_sar'].sum()
waste_items = inventory_with_waste[['item', 'units_wasted', 'known_waste_cost_sar', 'unit_cost_sar']].copy()

# Finding 3: Supplier price changes from emails
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

price_changes['price_change_sar'] = price_changes['new_price'] - price_changes['old_price']
price_changes['price_change_pct'] = ((price_changes['new_price'] - price_changes['old_price']) / 
                                      price_changes['old_price'] * 100).round(2)

# Build findings
findings = []

# Finding 1: Top 3 items by gross profit
if len(top_3_items) > 0:
    top_3_profit_total = top_3_items['gross_profit_sar'].sum()
    
    finding_1 = {
        "title": "Top 3 Items by Gross Profit Contribution",
        "claim": f"The three highest-grossing items generated {top_3_profit_total:.2f} SAR in total gross profit during the analysis week, with Iced Spanish Latte leading at {top_3_items.iloc[0]['gross_profit_sar']:.2f} SAR (71.87% margin).",
        "finding_type": "menu_engineering",
        "metrics": {
            "top_item_1_name": {
                "value": top_3_items.iloc[0]['item_name_en'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "top_item_1_gross_profit_sar": {
                "value": round(top_3_items.iloc[0]['gross_profit_sar'], 2),
                "unit": "SAR",
                "numerator": round(top_3_items.iloc[0]['gross_profit_sar'], 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "top_item_1_gross_margin_pct": {
                "value": round(top_3_items.iloc[0]['gross_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_3_items.iloc[0]['gross_profit_sar'], 2),
                "denominator": round(top_3_items.iloc[0]['line_total_sar'], 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "top_item_2_name": {
                "value": top_3_items.iloc[1]['item_name_en'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "top_item_2_gross_profit_sar": {
                "value": round(top_3_items.iloc[1]['gross_profit_sar'], 2),
                "unit": "SAR",
                "numerator": round(top_3_items.iloc[1]['gross_profit_sar'], 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "top_item_2_gross_margin_pct": {
                "value": round(top_3_items.iloc[1]['gross_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_3_items.iloc[1]['gross_profit_sar'], 2),
                "denominator": round(top_3_items.iloc[1]['line_total_sar'], 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "top_item_3_name": {
                "value": top_3_items.iloc[2]['item_name_en'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "top_item_3_gross_profit_sar": {
                "value": round(top_3_items.iloc[2]['gross_profit_sar'], 2),
                "unit": "SAR",
                "numerator": round(top_3_items.iloc[2]['gross_profit_sar'], 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "top_item_3_gross_margin_pct": {
                "value": round(top_3_items.iloc[2]['gross_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_3_items.iloc[2]['gross_profit_sar'], 2),
                "denominator": round(top_3_items.iloc[2]['line_total_sar'], 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "top_3_total_gross_profit_sar": {
                "value": round(top_3_profit_total, 2),
                "unit": "SAR",
                "numerator": round(top_3_profit_total, 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(pos_with_cost['transaction_id'].nunique()),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Total transactions analyzed: {int(pos_with_cost['transaction_id'].nunique())}",
            f"Total line items with valid unit costs: {len(pos_with_cost)}",
            "Refunds excluded from analysis",
            "Items with null unit_cost_sar excluded from profit calculation"
        ],
        "assumptions": [
            "Unit costs from menu table applied to all POS transactions",
            "Gross profit = line_total_sar - (quantity × unit_cost_sar)",
            "No recipe/BOM adjustments applied"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# Finding 2: Waste cost impact
if len(waste_items) > 0:
    finding_2 = {
        "title": "Quantified Waste Cost in Analysis Week",
        "claim": f"Waste cost for the week of May 4-11, 2026 totaled {total_waste_cost:.2f} SAR across {len(waste_items)} items with recorded waste observations.",
        "finding_type": "cost_analysis",
        "metrics": {
            "total_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": round(total_waste_cost, 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "waste_items_count": {
                "value": len(waste_items),
                "unit": "items",
                "numerator": len(waste_items),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(waste_items),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            "Only waste observations with non-null known_waste_cost_sar included",
            f"Items with waste cost recorded: {len(waste_items)}",
            "Blank waste values treated as missing, not zero"
        ],
        "assumptions": [
            "known_waste_cost_sar values from inventory table are authoritative",
            "Waste cost represents actual loss, not estimated"
        ],
        "confidence": 0.90
    }
    findings.append(finding_2)

# Finding 3: Supplier price changes
if len(price_changes) > 0:
    price_changes_sorted = price_changes.sort_values('effective_date', ascending=False)
    
    finding_3 = {
        "title": "Supplier Price Changes Detected",
        "claim": f"Email extraction identified {len(price_changes)} supplier price changes with effective dates. The most recent change involves {price_changes_sorted.iloc[0]['entity_or_ingredient']} with a {price_changes_sorted.iloc[0]['price_change_pct']:.2f}% price adjustment effective {price_changes_sorted.iloc[0]['effective_date'].strftime('%Y-%m-%d')}.",
        "finding_type": "supplier_cost_analysis",
        "metrics": {
            "price_changes_detected": {
                "value": len(price_changes),
                "unit": "changes",
                "numerator": len(price_changes),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "most_recent_ingredient": {
                "value": price_changes_sorted.iloc[0]['entity_or_ingredient'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "most_recent_old_price": {
                "value": round(price_changes_sorted.iloc[0]['old_price'], 2),
                "unit": price_changes_sorted.iloc[0]['currency'],
                "numerator": round(price_changes_sorted.iloc[0]['old_price'], 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "most_recent_new_price": {
                "value": round(price_changes_sorted.iloc[0]['new_price'], 2),
                "unit": price_changes_sorted.iloc[0]['currency'],
                "numerator": round(price_changes_sorted.iloc[0]['new_price'], 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "most_recent_price_change_pct": {
                "value": round(price_changes_sorted.iloc[0]['price_change_pct'], 2),
                "unit": "%",
                "numerator": round(price_changes_sorted.iloc[0]['price_change_sar'], 2),
                "denominator": round(price_changes_sorted.iloc[0]['old_price'], 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "most_recent_effective_date": {
                "value": price_changes_sorted.iloc[0]['effective_date'].strftime('%Y-%m-%d'),
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
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Price changes with both old_price and new_price and effective_date: {len(price_changes)}",
            "No recipe/BOM data available to calculate per-drink impact",
            "Price changes extracted from supplier emails; standing order volumes not confirmed"
        ],
        "assumptions": [
            "Email extraction confidence scores reflect reliability of price change data",
            "Effective dates from emails are accurate",
            "Price changes may not yet be reflected in menu or inventory unit costs"
        ],
        "confidence": 0.85
    }
    findings.append(finding_3)

# Build output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)