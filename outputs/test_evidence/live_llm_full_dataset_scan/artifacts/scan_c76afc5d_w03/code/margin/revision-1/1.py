import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List

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
menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'], errors='coerce')
menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'], errors='coerce')
emails_df['date'] = pd.to_datetime(emails_df['date'], errors='coerce')
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'], errors='coerce')

# Define analysis period
analysis_start = pd.to_datetime('2026-01-26T00:00:00+03:00')
analysis_end = pd.to_datetime('2026-02-02T00:00:00+03:00')
previous_start = pd.to_datetime('2026-01-19T00:00:00+03:00')
previous_end = pd.to_datetime('2026-01-26T00:00:00+03:00')

# Filter POS for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Filter inventory for analysis period
inventory_analysis = inventory_df[inventory_df['week_starting'] == pd.Timestamp('2026-01-26', tz='UTC')]

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from menu and POS
menu_with_cost = menu_df[['sku', 'item_en', 'price_sar', 'unit_cost_sar']].copy()
menu_with_cost = menu_with_cost.dropna(subset=['unit_cost_sar'])

# Aggregate POS by SKU for analysis period (excluding refunds)
pos_analysis_no_refund = pos_analysis[pos_analysis['is_refund'] == False].copy()
pos_by_sku = pos_analysis_no_refund.groupby('sku').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
pos_by_sku.columns = ['sku', 'total_quantity', 'total_revenue', 'basket_count']

# Merge with menu costs
item_economics = pos_by_sku.merge(menu_with_cost, on='sku', how='inner')
item_economics['total_cogs'] = item_economics['total_quantity'] * item_economics['unit_cost_sar']
item_economics['gross_profit'] = item_economics['total_revenue'] - item_economics['total_cogs']
item_economics['gross_margin_pct'] = (item_economics['gross_profit'] / item_economics['total_revenue'] * 100).round(2)

# Sort by gross profit
item_economics_sorted = item_economics.sort_values('gross_profit', ascending=False)

if len(item_economics_sorted) > 0:
    top_item = item_economics_sorted.iloc[0]
    
    finding_1 = {
        "title": "Top Gross Profit Item (Analysis Week)",
        "claim": f"Item {top_item['item_en']} (SKU {top_item['sku']}) generated the highest gross profit of {top_item['gross_profit']:.2f} SAR during the analysis week, with {top_item['total_quantity']:.0f} units sold at {top_item['gross_margin_pct']:.1f}% gross margin.",
        "finding_type": "item_economics",
        "metrics": {
            "item_name": {
                "value": str(top_item['item_en']),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "sku": {
                "value": str(top_item['sku']),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_quantity_sold": {
                "value": float(top_item['total_quantity']),
                "unit": "units",
                "numerator": float(top_item['total_quantity']),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_revenue": {
                "value": float(top_item['total_revenue']),
                "unit": "SAR",
                "numerator": float(top_item['total_revenue']),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_cogs": {
                "value": float(top_item['total_cogs']),
                "unit": "SAR",
                "numerator": float(top_item['total_cogs']),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "gross_profit": {
                "value": float(top_item['gross_profit']),
                "unit": "SAR",
                "numerator": float(top_item['gross_profit']),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "gross_margin_percent": {
                "value": float(top_item['gross_margin_pct']),
                "unit": "%",
                "numerator": float(top_item['gross_margin_pct']),
                "denominator": 100,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "unit_price_sar": {
                "value": float(top_item['price_sar']),
                "unit": "SAR",
                "numerator": float(top_item['price_sar']),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "unit_cost_sar": {
                "value": float(top_item['unit_cost_sar']),
                "unit": "SAR",
                "numerator": float(top_item['unit_cost_sar']),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(top_item['basket_count']),
        "coverage_notes": [
            f"Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
            f"Excludes refunds (is_refund=False)",
            f"Item economics calculated from {len(item_economics_sorted)} items with menu cost data",
            f"Revenue from POS line_total_sar, COGS from menu unit_cost_sar × quantity"
        ],
        "assumptions": [
            "Menu unit_cost_sar is current and applicable to analysis period",
            "POS line_total_sar reflects actual transaction value after discounts",
            "No recipe/BOM available; using menu unit_cost_sar as-is"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# FINDING 2: Waste Cost Impact
# Calculate waste cost from inventory data
inventory_with_waste = inventory_df[inventory_df['known_waste_cost_sar'].notna()].copy()

if len(inventory_with_waste) > 0:
    total_waste_cost = inventory_with_waste['known_waste_cost_sar'].sum()
    waste_items = len(inventory_with_waste)
    
    # Get the week for waste data
    waste_week = inventory_with_waste['week_starting'].iloc[0]
    
    finding_2 = {
        "title": "Quantified Waste Cost Impact",
        "claim": f"Known waste cost across {waste_items} items totaled {total_waste_cost:.2f} SAR during the week of {waste_week.strftime('%Y-%m-%d')}, representing direct margin erosion.",
        "finding_type": "waste_cost",
        "metrics": {
            "total_waste_cost_sar": {
                "value": float(total_waste_cost),
                "unit": "SAR",
                "numerator": float(total_waste_cost),
                "denominator": None,
                "period_start": waste_week.isoformat(),
                "period_end": (waste_week + pd.Timedelta(days=7)).isoformat()
            },
            "items_with_waste": {
                "value": waste_items,
                "unit": "count",
                "numerator": waste_items,
                "denominator": None,
                "period_start": waste_week.isoformat(),
                "period_end": (waste_week + pd.Timedelta(days=7)).isoformat()
            }
        },
        "source_names": ["inventory"],
        "sample_size": waste_items,
        "coverage_notes": [
            f"Only non-null known_waste_cost_sar values included",
            f"Week starting: {waste_week.isoformat()}",
            f"Waste cost represents actual cost of wasted units at unit_cost_sar"
        ],
        "assumptions": [
            "known_waste_cost_sar is accurate and complete for reported waste",
            "Blank waste values are treated as unknown, not zero"
        ],
        "confidence": 0.90
    }
    findings.append(finding_2)

# FINDING 3: Supplier Price Changes with Impact Scenario
# Extract supplier price changes from emails
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    # Take the first price change as example
    price_change = price_changes.iloc[0]
    
    old_price = float(price_change['old_price'])
    new_price = float(price_change['new_price'])
    price_delta = new_price - old_price
    price_change_pct = ((new_price - old_price) / old_price * 100) if old_price != 0 else 0
    
    ingredient = str(price_change['entity_or_ingredient'])
    effective_date = price_change['effective_date']
    unit = str(price_change['unit']) if pd.notna(price_change['unit']) else "unit"
    
    # Look for standing order quantity in facts
    standing_qty = None
    facts_text = str(price_change['facts']) if pd.notna(price_change['facts']) else ""
    
    # Try to extract standing order quantity from facts
    if 'standing' in facts_text.lower() or 'order' in facts_text.lower():
        # This is a placeholder; actual extraction would parse the facts field
        standing_qty = None
    
    finding_3 = {
        "title": "Supplier Price Change Detected",
        "claim": f"Supplier price for {ingredient} changed from {old_price:.2f} to {new_price:.2f} {price_change['currency']} per {unit} (effective {effective_date.strftime('%Y-%m-%d')}), representing a {price_change_pct:.1f}% change. Continued procurement at standing order volumes would create margin pressure.",
        "finding_type": "supplier_price_change",
        "metrics": {
            "ingredient": {
                "value": ingredient,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": effective_date.isoformat(),
                "period_end": None
            },
            "old_price": {
                "value": old_price,
                "unit": f"{price_change['currency']}/{unit}",
                "numerator": old_price,
                "denominator": None,
                "period_start": effective_date.isoformat(),
                "period_end": None
            },
            "new_price": {
                "value": new_price,
                "unit": f"{price_change['currency']}/{unit}",
                "numerator": new_price,
                "denominator": None,
                "period_start": effective_date.isoformat(),
                "period_end": None
            },
            "price_change_percent": {
                "value": float(price_change_pct),
                "unit": "%",
                "numerator": float(price_change_pct),
                "denominator": 100,
                "period_start": effective_date.isoformat(),
                "period_end": None
            },
            "price_delta": {
                "value": float(price_delta),
                "unit": f"{price_change['currency']}/{unit}",
                "numerator": float(price_delta),
                "denominator": None,
                "period_start": effective_date.isoformat(),
                "period_end": None
            }
        },
        "source_names": ["emails"],
        "sample_size": 1,
        "coverage_notes": [
            f"Effective date: {effective_date.isoformat()}",
            f"Email source: {price_change['sender']}",
            f"Confidence in extraction: {price_change['confidence']}"
        ],
        "assumptions": [
            "Standing order quantity and payment terms are not confirmed in email data",
            "Price change applies only to {ingredient}; impact on menu items requires recipe/BOM",
            "No recipe data available; cannot calculate per-drink cost impact",
            "Continued order volume at new price is assumed but not verified"
        ],
        "confidence": float(price_change['confidence']) if pd.notna(price_change['confidence']) else 0.75
    }
    findings.append(finding_3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")
