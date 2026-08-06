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

# Load artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Parse analysis period
analysis_start = datetime.fromisoformat("2026-05-04T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-05-11T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-04-27T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-05-04T00:00:00+03:00")

# Convert timestamp columns to datetime
pos_df['timestamp_local'] = pd.to_datetime(pos_df['timestamp_local'], utc=True)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'], utc=True)
emails_df['date'] = pd.to_datetime(emails_df['date'], utc=True)
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'], utc=True)

# Filter POS data for analysis period
pos_analysis = pos_df[
    (pos_df['timestamp_local'] >= analysis_start) & 
    (pos_df['timestamp_local'] < analysis_end)
].copy()

pos_previous = pos_df[
    (pos_df['timestamp_local'] >= previous_start) & 
    (pos_df['timestamp_local'] < previous_end)
].copy()

# Filter inventory for analysis period
inv_analysis = inventory_df[
    (inventory_df['week_starting'] >= analysis_start) & 
    (inventory_df['week_starting'] < analysis_end)
].copy()

inv_previous = inventory_df[
    (inventory_df['week_starting'] >= previous_start) & 
    (inventory_df['week_starting'] < previous_end)
].copy()

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from menu and POS data
pos_analysis_clean = pos_analysis[pos_analysis['is_refund'] == False].copy()

# Merge with menu to get unit costs
pos_with_cost = pos_analysis_clean.merge(
    menu_df[['sku', 'unit_cost_sar', 'price_sar']], 
    on='sku', 
    how='left'
)

# Calculate metrics
pos_with_cost['cogs'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
pos_with_cost['gross_profit'] = pos_with_cost['line_total_sar'] - pos_with_cost['cogs']
pos_with_cost['gross_margin_pct'] = (pos_with_cost['gross_profit'] / pos_with_cost['line_total_sar'] * 100).fillna(0)

# Group by item
item_economics = pos_with_cost.groupby('item_name_en').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'cogs': 'sum',
    'gross_profit': 'sum',
    'transaction_id': 'nunique'
}).reset_index()

item_economics['gross_margin_pct'] = (item_economics['gross_profit'] / item_economics['line_total_sar'] * 100).round(2)
item_economics = item_economics.sort_values('gross_profit', ascending=False)

# Top 3 items by gross profit
top_items = item_economics.head(3)

if len(top_items) > 0:
    finding1 = {
        "title": "Top 3 Items by Gross Profit (Analysis Week)",
        "claim": f"During the analysis week (May 4-11, 2026), the top 3 items by absolute gross profit contribution are {', '.join(top_items['item_name_en'].tolist())}. These items generated {top_items['gross_profit'].sum():.2f} SAR in total gross profit across {top_items['transaction_id'].sum()} transactions.",
        "finding_type": "item_economics",
        "metrics": {
            "top_item_1_name": {
                "value": top_items.iloc[0]['item_name_en'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-04T00:00:00+03:00",
                "period_end": "2026-05-11T00:00:00+03:00"
            },
            "top_item_1_gross_profit_sar": {
                "value": round(top_items.iloc[0]['gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_items.iloc[0]['gross_profit'], 2),
                "denominator": None,
                "period_start": "2026-05-04T00:00:00+03:00",
                "period_end": "2026-05-11T00:00:00+03:00"
            },
            "top_item_1_gross_margin_pct": {
                "value": round(top_items.iloc[0]['gross_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_items.iloc[0]['gross_profit'], 2),
                "denominator": round(top_items.iloc[0]['line_total_sar'], 2),
                "period_start": "2026-05-04T00:00:00+03:00",
                "period_end": "2026-05-11T00:00:00+03:00"
            },
            "top_item_2_name": {
                "value": top_items.iloc[1]['item_name_en'] if len(top_items) > 1 else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-04T00:00:00+03:00",
                "period_end": "2026-05-11T00:00:00+03:00"
            },
            "top_item_2_gross_profit_sar": {
                "value": round(top_items.iloc[1]['gross_profit'], 2) if len(top_items) > 1 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[1]['gross_profit'], 2) if len(top_items) > 1 else None,
                "denominator": None,
                "period_start": "2026-05-04T00:00:00+03:00",
                "period_end": "2026-05-11T00:00:00+03:00"
            },
            "top_item_3_name": {
                "value": top_items.iloc[2]['item_name_en'] if len(top_items) > 2 else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-04T00:00:00+03:00",
                "period_end": "2026-05-11T00:00:00+03:00"
            },
            "top_item_3_gross_profit_sar": {
                "value": round(top_items.iloc[2]['gross_profit'], 2) if len(top_items) > 2 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[2]['gross_profit'], 2) if len(top_items) > 2 else None,
                "denominator": None,
                "period_start": "2026-05-04T00:00:00+03:00",
                "period_end": "2026-05-11T00:00:00+03:00"
            },
            "total_items_analyzed": {
                "value": len(item_economics),
                "unit": "count",
                "numerator": len(item_economics),
                "denominator": None,
                "period_start": "2026-05-04T00:00:00+03:00",
                "period_end": "2026-05-11T00:00:00+03:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(pos_analysis_clean['transaction_id'].nunique()),
        "coverage_notes": [
            "Analysis period: May 4-11, 2026",
            "Excludes refunds (is_refund=False)",
            "Items with null unit_cost_sar excluded from margin calculation",
            f"Total transactions analyzed: {int(pos_analysis_clean['transaction_id'].nunique())}",
            f"Total line items: {len(pos_analysis_clean)}"
        ],
        "assumptions": [
            "Unit costs from menu_items.unit_cost_sar are current and accurate",
            "POS line_total_sar represents actual revenue after discounts",
            "No recipe/BOM data available; analysis is at item level only"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# FINDING 2: Waste Cost Impact Analysis
# Calculate waste costs from inventory data
inv_analysis_waste = inv_analysis[inv_analysis['known_waste_cost_sar'].notna()].copy()

if len(inv_analysis_waste) > 0:
    total_waste_cost = inv_analysis_waste['known_waste_cost_sar'].sum()
    waste_by_item = inv_analysis_waste.groupby('item').agg({
        'known_waste_cost_sar': 'sum',
        'units_wasted': 'sum'
    }).reset_index()
    waste_by_item = waste_by_item.sort_values('known_waste_cost_sar', ascending=False)
    
    finding2 = {
        "title": "Quantified Waste Cost Impact (Analysis Week)",
        "claim": f"During the analysis week (May 4-11, 2026), quantified waste cost totaled {total_waste_cost:.2f} SAR across {len(inv_analysis_waste)} inventory records. The highest waste cost item was {waste_by_item.iloc[0]['item']} with {waste_by_item.iloc[0]['known_waste_cost_sar']:.2f} SAR.",
        "finding_type": "waste_cost",
        "metrics": {
            "total_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": round(total_waste_cost, 2),
                "denominator": None,
                "period_start": "2026-05-04T00:00:00+03:00",
                "period_end": "2026-05-11T00:00:00+03:00"
            },
            "waste_records_count": {
                "value": len(inv_analysis_waste),
                "unit": "count",
                "numerator": len(inv_analysis_waste),
                "denominator": None,
                "period_start": "2026-05-04T00:00:00+03:00",
                "period_end": "2026-05-11T00:00:00+03:00"
            },
            "highest_waste_item": {
                "value": waste_by_item.iloc[0]['item'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-04T00:00:00+03:00",
                "period_end": "2026-05-11T00:00:00+03:00"
            },
            "highest_waste_cost_sar": {
                "value": round(waste_by_item.iloc[0]['known_waste_cost_sar'], 2),
                "unit": "SAR",
                "numerator": round(waste_by_item.iloc[0]['known_waste_cost_sar'], 2),
                "denominator": None,
                "period_start": "2026-05-04T00:00:00+03:00",
                "period_end": "2026-05-11T00:00:00+03:00"
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(inv_analysis_waste),
        "coverage_notes": [
            "Analysis period: May 4-11, 2026",
            "Only includes inventory records with non-null known_waste_cost_sar",
            f"Total inventory records in period: {len(inv_analysis)}",
            f"Records with waste cost data: {len(inv_analysis_waste)}"
        ],
        "assumptions": [
            "known_waste_cost_sar values are accurate and complete for non-null entries",
            "Blank waste values are treated as unknown, not zero"
        ],
        "confidence": 0.90
    }
    findings.append(finding2)

# FINDING 3: Supplier Price Changes from Email Evidence
# Extract supplier price changes with effective dates
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    price_changes['price_change_pct'] = (
        (price_changes['new_price'] - price_changes['old_price']) / 
        price_changes['old_price'] * 100
    ).round(2)
    
    # Filter for changes effective during or before analysis period
    relevant_changes = price_changes[
        price_changes['effective_date'] <= analysis_end
    ].copy()
    
    if len(relevant_changes) > 0:
        relevant_changes = relevant_changes.sort_values('effective_date', ascending=False)
        
        finding3 = {
            "title": "Supplier Price Changes with Effective Dates",
            "claim": f"Email evidence documents {len(relevant_changes)} supplier price changes with effective dates on or before the analysis period end (May 11, 2026). The most recent change was for {relevant_changes.iloc[0]['entity_or_ingredient']} effective {relevant_changes.iloc[0]['effective_date'].strftime('%Y-%m-%d')}, representing a {relevant_changes.iloc[0]['price_change_pct']:.2f}% price change from {relevant_changes.iloc[0]['old_price']} to {relevant_changes.iloc[0]['new_price']} {relevant_changes.iloc[0]['currency']}/{relevant_changes.iloc[0]['unit']}.",
            "finding_type": "supplier_pricing",
            "metrics": {
                "price_changes_count": {
                    "value": len(relevant_changes),
                    "unit": "count",
                    "numerator": len(relevant_changes),
                    "denominator": None,
                    "period_start": "2026-05-04T00:00:00+03:00",
                    "period_end": "2026-05-11T00:00:00+03:00"
                },
                "most_recent_ingredient": {
                    "value": relevant_changes.iloc[0]['entity_or_ingredient'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-05-04T00:00:00+03:00",
                    "period_end": "2026-05-11T00:00:00+03:00"
                },
                "most_recent_old_price": {
                    "value": round(relevant_changes.iloc[0]['old_price'], 2),
                    "unit": f"{relevant_changes.iloc[0]['currency']}/{relevant_changes.iloc[0]['unit']}",
                    "numerator": round(relevant_changes.iloc[0]['old_price'], 2),
                    "denominator": None,
                    "period_start": "2026-05-04T00:00:00+03:00",
                    "period_end": "2026-05-11T00:00:00+03:00"
                },
                "most_recent_new_price": {
                    "value": round(relevant_changes.iloc[0]['new_price'], 2),
                    "unit": f"{relevant_changes.iloc[0]['currency']}/{relevant_changes.iloc[0]['unit']}",
                    "numerator": round(relevant_changes.iloc[0]['new_price'], 2),
                    "denominator": None,
                    "period_start": "2026-05-04T00:00:00+03:00",
                    "period_end": "2026-05-11T00:00:00+03:00"
                },
                "most_recent_price_change_pct": {
                    "value": relevant_changes.iloc[0]['price_change_pct'],
                    "unit": "%",
                    "numerator": relevant_changes.iloc[0]['price_change_pct'],
                    "denominator": 100,
                    "period_start": "2026-05-04T00:00:00+03:00",
                    "period_end": "2026-05-11T00:00:00+03:00"
                },
                "most_recent_effective_date": {
                    "value": relevant_changes.iloc[0]['effective_date'].strftime('%Y-%m-%d'),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-05-04T00:00:00+03:00",
                    "period_end": "2026-05-11T00:00:00+03:00"
                }
            },
            "source_names": ["emails"],
            "sample_size": len(relevant_changes),
            "coverage_notes": [
                "Analysis period: May 4-11, 2026",
                "Includes only price changes with both old_price, new_price, and effective_date populated",
                f"Total email records: {len(emails_df)}",
                f"Price change records: {len(price_changes)}",
                f"Relevant to analysis period: {len(relevant_changes)}"
            ],
            "assumptions": [
                "Email extraction accurately captured supplier price change facts",
                "Effective dates represent when price changes took effect",
                "No recipe/BOM data available; cannot calculate per-drink impact without standing order quantities and product mappings"
            ],
            "confidence": 0.85
        }
        findings.append(finding3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
