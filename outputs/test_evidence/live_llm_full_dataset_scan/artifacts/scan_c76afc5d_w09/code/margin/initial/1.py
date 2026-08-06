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

# Define analysis period
analysis_start = datetime(2026, 3, 9, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 3, 16, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 3, 2, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 3, 9, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Convert inventory week_starting to datetime
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Get inventory for analysis week
analysis_week = pd.Timestamp('2026-03-09', tz=timezone.utc)
inventory_analysis = inventory_df[inventory_df['week_starting'] == analysis_week].copy()

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from POS and menu
pos_with_menu = pos_analysis.merge(menu_df[['sku', 'unit_cost_sar', 'price_sar']], on='sku', how='left')

# Filter out refunds for revenue calculation
pos_sales = pos_with_menu[~pos_with_menu['is_refund']].copy()

# Group by item to calculate totals
item_economics = pos_sales.groupby(['sku', 'item_name']).agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'unit_cost_sar': 'first',
    'price_sar': 'first',
    'transaction_id': 'nunique'
}).reset_index()

item_economics.columns = ['sku', 'item_name', 'total_quantity', 'total_revenue', 'unit_cost_sar', 'menu_price_sar', 'basket_count']

# Calculate COGS and gross profit
item_economics['total_cogs'] = item_economics['total_quantity'] * item_economics['unit_cost_sar']
item_economics['gross_profit'] = item_economics['total_revenue'] - item_economics['total_cogs']
item_economics['gross_margin_pct'] = (item_economics['gross_profit'] / item_economics['total_revenue'] * 100).round(2)

# Sort by gross profit contribution
item_economics_sorted = item_economics.sort_values('gross_profit', ascending=False)

# Get top 3 items by gross profit
top_items = item_economics_sorted.head(3)

if len(top_items) > 0:
    finding1 = {
        "title": "Top 3 Items by Gross Profit Contribution (Week of 2026-03-09)",
        "claim": f"The top 3 items by gross profit contribution in the analysis week are {', '.join(top_items['item_name'].values)}, collectively generating {top_items['gross_profit'].sum():.2f} SAR in gross profit.",
        "finding_type": "item_economics",
        "metrics": {
            "top_item_1_name": {
                "value": top_items.iloc[0]['item_name'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            },
            "top_item_1_gross_profit_sar": {
                "value": round(top_items.iloc[0]['gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_items.iloc[0]['total_revenue'], 2),
                "denominator": round(top_items.iloc[0]['total_cogs'], 2),
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            },
            "top_item_1_gross_margin_pct": {
                "value": top_items.iloc[0]['gross_margin_pct'],
                "unit": "%",
                "numerator": round(top_items.iloc[0]['gross_profit'], 2),
                "denominator": round(top_items.iloc[0]['total_revenue'], 2),
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            },
            "top_item_1_quantity": {
                "value": int(top_items.iloc[0]['total_quantity']),
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            },
            "top_item_2_name": {
                "value": top_items.iloc[1]['item_name'] if len(top_items) > 1 else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            },
            "top_item_2_gross_profit_sar": {
                "value": round(top_items.iloc[1]['gross_profit'], 2) if len(top_items) > 1 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[1]['total_revenue'], 2) if len(top_items) > 1 else None,
                "denominator": round(top_items.iloc[1]['total_cogs'], 2) if len(top_items) > 1 else None,
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            },
            "top_item_3_name": {
                "value": top_items.iloc[2]['item_name'] if len(top_items) > 2 else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            },
            "top_item_3_gross_profit_sar": {
                "value": round(top_items.iloc[2]['gross_profit'], 2) if len(top_items) > 2 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[2]['total_revenue'], 2) if len(top_items) > 2 else None,
                "denominator": round(top_items.iloc[2]['total_cogs'], 2) if len(top_items) > 2 else None,
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            },
            "total_top_3_gross_profit_sar": {
                "value": round(top_items['gross_profit'].sum(), 2),
                "unit": "SAR",
                "numerator": round(top_items['total_revenue'].sum(), 2),
                "denominator": round(top_items['total_cogs'].sum(), 2),
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(pos_sales),
        "coverage_notes": [
            "Analysis covers POS transactions from 2026-03-09 to 2026-03-16",
            "Excludes refund transactions (is_refund=True)",
            "Unit costs sourced from menu.parquet",
            "Revenue calculated from line_total_sar (includes discounts)"
        ],
        "assumptions": [
            "Menu unit_cost_sar is current and applicable to analysis period",
            "POS line_total_sar accurately reflects actual revenue after discounts",
            "No recipe/BOM data available; analysis is at item level only"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# FINDING 2: Waste Cost Analysis
# Calculate waste costs from inventory data
inventory_analysis_with_waste = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()

if len(inventory_analysis_with_waste) > 0:
    total_waste_cost = inventory_analysis_with_waste['known_waste_cost_sar'].sum()
    waste_items = inventory_analysis_with_waste[['sku', 'item', 'units_wasted', 'known_waste_cost_sar']].copy()
    waste_items = waste_items.sort_values('known_waste_cost_sar', ascending=False)
    
    finding2 = {
        "title": "Quantified Waste Cost (Week of 2026-03-09)",
        "claim": f"Known waste cost for the week of 2026-03-09 totals {total_waste_cost:.2f} SAR across {len(waste_items)} items with recorded waste observations.",
        "finding_type": "waste_analysis",
        "metrics": {
            "total_known_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            },
            "items_with_waste_observations": {
                "value": len(waste_items),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            },
            "highest_waste_item_name": {
                "value": waste_items.iloc[0]['item'] if len(waste_items) > 0 else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            },
            "highest_waste_cost_sar": {
                "value": round(waste_items.iloc[0]['known_waste_cost_sar'], 2) if len(waste_items) > 0 else None,
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            },
            "highest_waste_units": {
                "value": int(waste_items.iloc[0]['units_wasted']) if len(waste_items) > 0 else None,
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(waste_items),
        "coverage_notes": [
            "Analysis covers inventory week starting 2026-03-09",
            "Only includes items with non-null known_waste_cost_sar values",
            "Blank waste values are excluded per data quality rules",
            f"Total inventory items in week: {len(inventory_analysis)}, items with waste cost: {len(waste_items)}"
        ],
        "assumptions": [
            "known_waste_cost_sar accurately reflects actual waste cost",
            "Waste observations are complete for items with non-null values"
        ],
        "confidence": 0.90
    }
    findings.append(finding2)

# FINDING 3: Supplier Price Changes and Procurement Impact
# Look for price changes in emails
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    price_changes['effective_date'] = pd.to_datetime(price_changes['effective_date'])
    price_changes['price_change_pct'] = ((price_changes['new_price'] - price_changes['old_price']) / price_changes['old_price'] * 100).round(2)
    price_changes = price_changes.sort_values('price_change_pct', ascending=False)
    
    # Get the most significant price change
    top_change = price_changes.iloc[0]
    
    finding3 = {
        "title": "Supplier Price Change Detection",
        "claim": f"Email evidence shows {top_change['entity_or_ingredient']} price change from {top_change['old_price']} to {top_change['new_price']} {top_change['currency']}/{top_change['unit']}, effective {top_change['effective_date']}, representing a {top_change['price_change_pct']}% change.",
        "finding_type": "supplier_pricing",
        "metrics": {
            "ingredient_or_entity": {
                "value": top_change['entity_or_ingredient'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            },
            "old_price": {
                "value": round(top_change['old_price'], 2),
                "unit": f"{top_change['currency']}/{top_change['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            },
            "new_price": {
                "value": round(top_change['new_price'], 2),
                "unit": f"{top_change['currency']}/{top_change['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            },
            "price_change_pct": {
                "value": top_change['price_change_pct'],
                "unit": "%",
                "numerator": round(top_change['new_price'] - top_change['old_price'], 2),
                "denominator": round(top_change['old_price'], 2),
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            },
            "effective_date": {
                "value": str(top_change['effective_date']),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            },
            "sender": {
                "value": top_change['sender'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+00:00",
                "period_end": "2026-03-16T00:00:00+00:00"
            }
        },
        "source_names": ["emails"],
        "sample_size": len(price_changes),
        "coverage_notes": [
            f"Total price change records in emails: {len(price_changes)}",
            "Analysis shows most significant price change by percentage",
            "Only includes records with both old_price and new_price and effective_date"
        ],
        "assumptions": [
            "Email extraction accurately captured supplier price information",
            "Effective date represents when price change takes effect",
            "No recipe/BOM data available; cannot calculate per-drink impact without standing order quantities"
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
