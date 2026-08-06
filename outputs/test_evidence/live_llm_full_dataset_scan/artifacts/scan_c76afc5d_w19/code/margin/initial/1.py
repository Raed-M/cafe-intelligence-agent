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
analysis_start = "2026-05-18"
analysis_end = "2026-05-25"
previous_start = "2026-05-11"
previous_end = "2026-05-18"

# Convert timestamp columns to datetime
pos_df['timestamp_local'] = pd.to_datetime(pos_df['timestamp_local'])
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
emails_df['date'] = pd.to_datetime(emails_df['date'])
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'])

# Filter POS for analysis period
analysis_pos = pos_df[
    (pos_df['calendar_date'] >= analysis_start) & 
    (pos_df['calendar_date'] < analysis_end)
].copy()

previous_pos = pos_df[
    (pos_df['calendar_date'] >= previous_start) & 
    (pos_df['calendar_date'] < previous_end)
].copy()

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from menu and POS data
menu_analysis = menu_df[['sku', 'item_en', 'price_sar', 'unit_cost_sar']].copy()

# Aggregate POS by SKU for analysis period
sku_sales = analysis_pos.groupby('sku').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
sku_sales.columns = ['sku', 'total_quantity', 'total_revenue', 'basket_count']

# Merge with menu to get unit costs
sku_economics = sku_sales.merge(menu_analysis, on='sku', how='left')

# Calculate COGS and gross profit
sku_economics['total_cogs'] = sku_economics['total_quantity'] * sku_economics['unit_cost_sar']
sku_economics['gross_profit'] = sku_economics['total_revenue'] - sku_economics['total_cogs']
sku_economics['gross_margin_pct'] = (sku_economics['gross_profit'] / sku_economics['total_revenue'] * 100).round(2)

# Sort by gross profit to find top contributor
sku_economics_sorted = sku_economics.sort_values('gross_profit', ascending=False)

# Get top item by gross profit
if len(sku_economics_sorted) > 0:
    top_item = sku_economics_sorted.iloc[0]
    
    finding1 = {
        "title": "Top Gross Profit Contributor - Item Economics",
        "claim": f"{top_item['item_en']} generated the highest gross profit of {top_item['gross_profit']:.2f} SAR during the analysis week, with {top_item['total_quantity']:.0f} units sold at {top_item['gross_margin_pct']:.1f}% gross margin.",
        "finding_type": "item_economics",
        "metrics": {
            "item_name": {
                "value": top_item['item_en'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "total_revenue": {
                "value": round(top_item['total_revenue'], 2),
                "unit": "SAR",
                "numerator": round(top_item['total_revenue'], 2),
                "denominator": None,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "total_cogs": {
                "value": round(top_item['total_cogs'], 2),
                "unit": "SAR",
                "numerator": round(top_item['total_cogs'], 2),
                "denominator": None,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "gross_profit": {
                "value": round(top_item['gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_item['gross_profit'], 2),
                "denominator": None,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "gross_margin_pct": {
                "value": round(top_item['gross_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_item['gross_profit'], 2),
                "denominator": round(top_item['total_revenue'], 2),
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "units_sold": {
                "value": int(top_item['total_quantity']),
                "unit": "units",
                "numerator": int(top_item['total_quantity']),
                "denominator": None,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "baskets": {
                "value": int(top_item['basket_count']),
                "unit": "transactions",
                "numerator": int(top_item['basket_count']),
                "denominator": None,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(top_item['basket_count']),
        "coverage_notes": [
            f"Analysis period: 2026-05-18 to 2026-05-25",
            f"Item SKU: {top_item['sku']}",
            f"Unit cost from menu: {top_item['unit_cost_sar']} SAR",
            f"Menu price: {top_item['price_sar']} SAR"
        ],
        "assumptions": [
            "Unit cost from menu_items.unit_cost_sar applied uniformly to all units sold",
            "No recipe/BOM available; cost is item-level only",
            "Refunds included in net calculations per POS line_total_sar"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# FINDING 2: Waste Cost Analysis
# Filter inventory for analysis week
analysis_week = pd.to_datetime("2026-05-18")
analysis_inventory = inventory_df[inventory_df['week_starting'] == analysis_week].copy()

# Calculate total waste cost (only non-null values)
waste_data = analysis_inventory[analysis_inventory['known_waste_cost_sar'].notna()].copy()

if len(waste_data) > 0:
    total_waste_cost = waste_data['known_waste_cost_sar'].sum()
    total_units_wasted = waste_data['units_wasted'].sum()
    
    # Get items with waste
    waste_items = waste_data[['sku', 'item', 'units_wasted', 'known_waste_cost_sar']].copy()
    waste_items = waste_items.sort_values('known_waste_cost_sar', ascending=False)
    
    if len(waste_items) > 0:
        top_waste_item = waste_items.iloc[0]
        
        finding2 = {
            "title": "Quantified Waste Cost - Week of 2026-05-18",
            "claim": f"Identified {len(waste_items)} items with quantified waste during the analysis week. {top_waste_item['item']} had the highest waste cost of {top_waste_item['known_waste_cost_sar']:.2f} SAR from {top_waste_item['units_wasted']:.0f} wasted units.",
            "finding_type": "waste_cost",
            "metrics": {
                "total_waste_cost": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": round(total_waste_cost, 2),
                    "denominator": None,
                    "period_start": "2026-05-18T00:00:00+03:00",
                    "period_end": "2026-05-25T00:00:00+03:00"
                },
                "total_units_wasted": {
                    "value": int(total_units_wasted),
                    "unit": "units",
                    "numerator": int(total_units_wasted),
                    "denominator": None,
                    "period_start": "2026-05-18T00:00:00+03:00",
                    "period_end": "2026-05-25T00:00:00+03:00"
                },
                "items_with_waste": {
                    "value": len(waste_items),
                    "unit": "count",
                    "numerator": len(waste_items),
                    "denominator": None,
                    "period_start": "2026-05-18T00:00:00+03:00",
                    "period_end": "2026-05-25T00:00:00+03:00"
                },
                "top_waste_item": {
                    "value": top_waste_item['item'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-05-18T00:00:00+03:00",
                    "period_end": "2026-05-25T00:00:00+03:00"
                },
                "top_waste_cost": {
                    "value": round(top_waste_item['known_waste_cost_sar'], 2),
                    "unit": "SAR",
                    "numerator": round(top_waste_item['known_waste_cost_sar'], 2),
                    "denominator": None,
                    "period_start": "2026-05-18T00:00:00+03:00",
                    "period_end": "2026-05-25T00:00:00+03:00"
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(waste_items),
            "coverage_notes": [
                "Only non-null waste_cost_sar values included",
                "Week starting: 2026-05-18",
                "Blank waste values treated as unknown, not zero"
            ],
            "assumptions": [
                "known_waste_cost_sar from inventory reflects actual waste cost",
                "Waste cost calculation methodology from source system is accurate"
            ],
            "confidence": 0.85
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
    # Calculate percentage change
    price_changes['pct_change'] = ((price_changes['new_price'] - price_changes['old_price']) / price_changes['old_price'] * 100).round(2)
    
    # Sort by absolute percentage change
    price_changes['abs_pct_change'] = price_changes['pct_change'].abs()
    price_changes_sorted = price_changes.sort_values('abs_pct_change', ascending=False)
    
    top_change = price_changes_sorted.iloc[0]
    
    # Check if this is a price increase or decrease
    direction = "increase" if top_change['new_price'] > top_change['old_price'] else "decrease"
    
    finding3 = {
        "title": "Supplier Price Change - Procurement Cost Pressure",
        "claim": f"Supplier email dated {top_change['date'].strftime('%Y-%m-%d')} indicates a {direction} in {top_change['entity_or_ingredient']} price from {top_change['old_price']} to {top_change['new_price']} {top_change['currency']} per {top_change['unit']}, effective {top_change['effective_date'].strftime('%Y-%m-%d')}. This represents a {abs(top_change['pct_change']):.1f}% {direction}.",
        "finding_type": "supplier_price_change",
        "metrics": {
            "ingredient": {
                "value": top_change['entity_or_ingredient'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "old_price": {
                "value": round(top_change['old_price'], 2),
                "unit": top_change['currency'],
                "numerator": round(top_change['old_price'], 2),
                "denominator": None,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "new_price": {
                "value": round(top_change['new_price'], 2),
                "unit": top_change['currency'],
                "numerator": round(top_change['new_price'], 2),
                "denominator": None,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "price_change_pct": {
                "value": round(top_change['pct_change'], 2),
                "unit": "%",
                "numerator": round(top_change['new_price'] - top_change['old_price'], 2),
                "denominator": round(top_change['old_price'], 2),
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "unit": {
                "value": top_change['unit'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            },
            "effective_date": {
                "value": top_change['effective_date'].strftime('%Y-%m-%d'),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-05-18T00:00:00+03:00",
                "period_end": "2026-05-25T00:00:00+03:00"
            }
        },
        "source_names": ["emails"],
        "sample_size": 1,
        "coverage_notes": [
            f"Email date: {top_change['date'].strftime('%Y-%m-%d')}",
            f"Sender: {top_change['sender']}",
            f"Confidence level from extraction: {top_change['confidence']}"
        ],
        "assumptions": [
            "No recipe/BOM available; cannot calculate per-drink impact without ingredient quantities",
            "Standing order quantities and payment terms not confirmed; procurement cost scenario would require additional supplier communication",
            "Price change applies only to the named ingredient; cross-product impact unknown"
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
