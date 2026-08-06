import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any

# Load environment
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Load artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Parse timestamps
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
emails_df['date'] = pd.to_datetime(emails_df['date'])
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'])

# Define analysis period
analysis_start = pd.to_datetime('2026-02-09T00:00:00+03:00')
analysis_end = pd.to_datetime('2026-02-16T00:00:00+03:00')
previous_start = pd.to_datetime('2026-02-02T00:00:00+03:00')
previous_end = pd.to_datetime('2026-02-09T00:00:00+03:00')

# Filter POS for analysis period (exclude refunds for revenue, but keep for analysis)
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Filter inventory for analysis week
inv_analysis = inventory_df[inventory_df['week_starting'] == pd.to_datetime('2026-02-09')]
inv_previous = inventory_df[inventory_df['week_starting'] == pd.to_datetime('2026-02-02')]

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from menu and POS
pos_sales = pos_analysis[~pos_analysis['is_refund']].copy()
pos_sales['revenue_sar'] = pos_sales['line_total_sar']
pos_sales['quantity_sold'] = pos_sales['quantity']

# Merge with menu to get unit costs
pos_with_cost = pos_sales.merge(menu_df[['sku', 'unit_cost_sar', 'item_en']], on='sku', how='left')

# Calculate COGS and gross profit by item
pos_with_cost['cogs_sar'] = pos_with_cost['quantity_sold'] * pos_with_cost['unit_cost_sar']
pos_with_cost['gross_profit_sar'] = pos_with_cost['revenue_sar'] - pos_with_cost['cogs_sar']
pos_with_cost['gross_margin_pct'] = (pos_with_cost['gross_profit_sar'] / pos_with_cost['revenue_sar'] * 100).fillna(0)

# Aggregate by item
item_economics = pos_with_cost.groupby('item_en').agg({
    'quantity_sold': 'sum',
    'revenue_sar': 'sum',
    'cogs_sar': 'sum',
    'gross_profit_sar': 'sum',
    'unit_price_sar': 'first',
    'unit_cost_sar': 'first'
}).reset_index()

item_economics['gross_margin_pct'] = (item_economics['gross_profit_sar'] / item_economics['revenue_sar'] * 100).round(2)
item_economics = item_economics.sort_values('gross_profit_sar', ascending=False)

# Top 3 items by gross profit
top_items = item_economics.head(3)

if len(top_items) > 0:
    finding1 = {
        "title": "Top 3 Items by Gross Profit (Analysis Week)",
        "claim": f"The top 3 items by gross profit contribution in week of {analysis_start.date()} are: {', '.join(top_items['item_en'].tolist())} with combined gross profit of {top_items['gross_profit_sar'].sum():.2f} SAR",
        "finding_type": "item_economics",
        "metrics": {
            "top_item_1_name": {
                "value": top_items.iloc[0]['item_en'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_gross_profit_sar": {
                "value": round(top_items.iloc[0]['gross_profit_sar'], 2),
                "unit": "SAR",
                "numerator": round(top_items.iloc[0]['gross_profit_sar'], 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_quantity": {
                "value": int(top_items.iloc[0]['quantity_sold']),
                "unit": "units",
                "numerator": int(top_items.iloc[0]['quantity_sold']),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_margin_pct": {
                "value": round(top_items.iloc[0]['gross_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_items.iloc[0]['gross_margin_pct'], 2),
                "denominator": 100,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_2_name": {
                "value": top_items.iloc[1]['item_en'] if len(top_items) > 1 else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_2_gross_profit_sar": {
                "value": round(top_items.iloc[1]['gross_profit_sar'], 2) if len(top_items) > 1 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[1]['gross_profit_sar'], 2) if len(top_items) > 1 else None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_3_name": {
                "value": top_items.iloc[2]['item_en'] if len(top_items) > 2 else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_3_gross_profit_sar": {
                "value": round(top_items.iloc[2]['gross_profit_sar'], 2) if len(top_items) > 2 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[2]['gross_profit_sar'], 2) if len(top_items) > 2 else None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_gross_profit_all_items": {
                "value": round(item_economics['gross_profit_sar'].sum(), 2),
                "unit": "SAR",
                "numerator": round(item_economics['gross_profit_sar'].sum(), 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(pos_sales),
        "coverage_notes": [
            f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
            f"Total transactions analyzed: {pos_sales['transaction_id'].nunique()}",
            f"Total line items: {len(pos_sales)}",
            "Refunds excluded from revenue calculations",
            "Unit costs sourced from menu.parquet",
            "COGS calculated as quantity × unit_cost_sar"
        ],
        "assumptions": [
            "Menu unit_cost_sar values are current and accurate for the analysis period",
            "No recipe/BOM adjustments applied",
            "Waste costs not included in item-level COGS (tracked separately in inventory)"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# FINDING 2: Waste Cost Impact Analysis
# Calculate waste costs from inventory data
inv_analysis_with_waste = inv_analysis[inv_analysis['known_waste_cost_sar'].notna()].copy()

if len(inv_analysis_with_waste) > 0:
    total_waste_cost = inv_analysis_with_waste['known_waste_cost_sar'].sum()
    waste_items = inv_analysis_with_waste[['item', 'units_wasted', 'known_waste_cost_sar']].copy()
    waste_items = waste_items.sort_values('known_waste_cost_sar', ascending=False)
    
    finding2 = {
        "title": "Quantified Waste Cost Impact (Analysis Week)",
        "claim": f"Known waste cost in week of {analysis_start.date()} totals {total_waste_cost:.2f} SAR across {len(waste_items)} items with waste observations",
        "finding_type": "waste_cost",
        "metrics": {
            "total_known_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": round(total_waste_cost, 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "items_with_waste_observations": {
                "value": len(waste_items),
                "unit": "count",
                "numerator": len(waste_items),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "highest_waste_item": {
                "value": waste_items.iloc[0]['item'] if len(waste_items) > 0 else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "highest_waste_cost_sar": {
                "value": round(waste_items.iloc[0]['known_waste_cost_sar'], 2) if len(waste_items) > 0 else None,
                "unit": "SAR",
                "numerator": round(waste_items.iloc[0]['known_waste_cost_sar'], 2) if len(waste_items) > 0 else None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "waste_as_pct_of_gross_profit": {
                "value": round((total_waste_cost / item_economics['gross_profit_sar'].sum() * 100), 2) if item_economics['gross_profit_sar'].sum() > 0 else 0,
                "unit": "%",
                "numerator": round(total_waste_cost, 2),
                "denominator": round(item_economics['gross_profit_sar'].sum(), 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(waste_items),
        "coverage_notes": [
            f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
            "Only non-null waste cost observations included",
            "Blank waste values treated as unknown, not zero",
            f"Items with waste data: {len(waste_items)} out of {len(inv_analysis)} total inventory records"
        ],
        "assumptions": [
            "known_waste_cost_sar values are accurate and complete for reported waste",
            "Waste cost represents actual loss to business"
        ],
        "confidence": 0.85
    }
    findings.append(finding2)

# FINDING 3: Supplier Price Changes and Procurement Impact
# Analyze emails for price changes
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    price_changes['price_change_pct'] = ((price_changes['new_price'] - price_changes['old_price']) / price_changes['old_price'] * 100).round(2)
    price_changes = price_changes.sort_values('price_change_pct', ascending=False)
    
    # Get the most significant price change
    top_change = price_changes.iloc[0]
    
    finding3 = {
        "title": "Supplier Price Change Detection",
        "claim": f"Email evidence shows {top_change['entity_or_ingredient']} price change from {top_change['old_price']} to {top_change['new_price']} {top_change['currency']}/{top_change['unit']}, effective {top_change['effective_date'].date()}, representing a {top_change['price_change_pct']:.2f}% change",
        "finding_type": "supplier_price_change",
        "metrics": {
            "ingredient_name": {
                "value": top_change['entity_or_ingredient'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "old_price": {
                "value": round(top_change['old_price'], 2),
                "unit": f"{top_change['currency']}/{top_change['unit']}",
                "numerator": round(top_change['old_price'], 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "new_price": {
                "value": round(top_change['new_price'], 2),
                "unit": f"{top_change['currency']}/{top_change['unit']}",
                "numerator": round(top_change['new_price'], 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "price_change_pct": {
                "value": top_change['price_change_pct'],
                "unit": "%",
                "numerator": top_change['price_change_pct'],
                "denominator": 100,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "effective_date": {
                "value": top_change['effective_date'].isoformat(),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "sender": {
                "value": top_change['sender'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["emails"],
        "sample_size": len(price_changes),
        "coverage_notes": [
            f"Total price change events detected: {len(price_changes)}",
            "Only price changes with both old and new prices included",
            "Effective dates extracted from email evidence",
            "No standing order quantities available to calculate procurement cost impact"
        ],
        "assumptions": [
            "Email extraction accurately captured price change facts",
            "Price changes apply to ingredients/suppliers mentioned",
            "No recipe/BOM data available to calculate per-drink impact",
            "Standing order volumes and payment terms unknown"
        ],
        "confidence": 0.75
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
