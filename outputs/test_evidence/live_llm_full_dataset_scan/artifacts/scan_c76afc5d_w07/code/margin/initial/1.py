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
analysis_start = datetime(2026, 2, 23, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 3, 2, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 2, 16, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 2, 23, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Filter inventory for analysis week
inventory_analysis = inventory_df[inventory_df['week_starting'] == '2026-02-23'].copy()
inventory_previous = inventory_df[inventory_df['week_starting'] == '2026-02-16'].copy()

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from POS and menu data
pos_with_menu = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'price_sar']],
    on='sku',
    how='left'
)

# Filter out refunds for revenue calculation
pos_sales = pos_with_menu[~pos_with_menu['is_refund']].copy()

# Calculate metrics by item
item_economics = pos_sales.groupby(['sku', 'item_name']).agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'unit_cost_sar': 'first',
    'price_sar': 'first',
    'transaction_id': 'nunique'
}).reset_index()

item_economics.columns = ['sku', 'item_name', 'total_quantity', 'total_revenue', 'unit_cost_sar', 'menu_price', 'basket_count']

# Calculate COGS and gross profit
item_economics['total_cogs'] = item_economics['total_quantity'] * item_economics['unit_cost_sar']
item_economics['gross_profit'] = item_economics['total_revenue'] - item_economics['total_cogs']
item_economics['gross_margin_pct'] = (item_economics['gross_profit'] / item_economics['total_revenue'] * 100).round(2)

# Sort by gross profit
item_economics_sorted = item_economics.sort_values('gross_profit', ascending=False)

# Top 3 items by gross profit
top_items = item_economics_sorted.head(3)

if len(top_items) > 0:
    top_item = top_items.iloc[0]
    finding1 = {
        "title": "Top Gross Profit Item - Analysis Period",
        "claim": f"Item '{top_item['item_name']}' (SKU: {top_item['sku']}) generated the highest gross profit of {top_item['gross_profit']:.2f} SAR during the analysis period (Feb 23 - Mar 2, 2026), with {int(top_item['total_quantity'])} units sold across {int(top_item['basket_count'])} transactions.",
        "finding_type": "item_economics",
        "metrics": {
            "item_name": {
                "value": top_item['item_name'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-23T00:00:00+00:00",
                "period_end": "2026-03-02T00:00:00+00:00"
            },
            "sku": {
                "value": top_item['sku'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-23T00:00:00+00:00",
                "period_end": "2026-03-02T00:00:00+00:00"
            },
            "total_quantity_sold": {
                "value": int(top_item['total_quantity']),
                "unit": "units",
                "numerator": int(top_item['total_quantity']),
                "denominator": None,
                "period_start": "2026-02-23T00:00:00+00:00",
                "period_end": "2026-03-02T00:00:00+00:00"
            },
            "total_revenue": {
                "value": round(top_item['total_revenue'], 2),
                "unit": "SAR",
                "numerator": round(top_item['total_revenue'], 2),
                "denominator": None,
                "period_start": "2026-02-23T00:00:00+00:00",
                "period_end": "2026-03-02T00:00:00+00:00"
            },
            "total_cogs": {
                "value": round(top_item['total_cogs'], 2),
                "unit": "SAR",
                "numerator": round(top_item['total_cogs'], 2),
                "denominator": None,
                "period_start": "2026-02-23T00:00:00+00:00",
                "period_end": "2026-03-02T00:00:00+00:00"
            },
            "gross_profit": {
                "value": round(top_item['gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_item['gross_profit'], 2),
                "denominator": None,
                "period_start": "2026-02-23T00:00:00+00:00",
                "period_end": "2026-03-02T00:00:00+00:00"
            },
            "gross_margin_pct": {
                "value": top_item['gross_margin_pct'],
                "unit": "%",
                "numerator": round(top_item['gross_profit'], 2),
                "denominator": round(top_item['total_revenue'], 2),
                "period_start": "2026-02-23T00:00:00+00:00",
                "period_end": "2026-03-02T00:00:00+00:00"
            },
            "unit_cost_sar": {
                "value": round(top_item['unit_cost_sar'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-23T00:00:00+00:00",
                "period_end": "2026-03-02T00:00:00+00:00"
            },
            "basket_count": {
                "value": int(top_item['basket_count']),
                "unit": "transactions",
                "numerator": int(top_item['basket_count']),
                "denominator": None,
                "period_start": "2026-02-23T00:00:00+00:00",
                "period_end": "2026-03-02T00:00:00+00:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(pos_sales.shape[0]),
        "coverage_notes": [
            "Analysis period: 2026-02-23 to 2026-03-02",
            "Includes all non-refund POS transactions",
            "Unit costs sourced from menu.parquet",
            "COGS calculated as quantity × unit_cost_sar",
            "Gross profit = revenue - COGS"
        ],
        "assumptions": [
            "Menu unit_cost_sar is current and applicable to all sales in period",
            "No recipe/BOM data available; using menu unit costs as-is",
            "Refunds excluded from revenue calculations"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# FINDING 2: Waste Cost Impact Analysis
# Calculate waste costs from inventory data
inventory_analysis_with_waste = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()

if len(inventory_analysis_with_waste) > 0:
    total_waste_cost = inventory_analysis_with_waste['known_waste_cost_sar'].sum()
    waste_items = inventory_analysis_with_waste[['sku', 'item', 'units_wasted', 'known_waste_cost_sar']].copy()
    waste_items = waste_items.sort_values('known_waste_cost_sar', ascending=False)
    
    if len(waste_items) > 0:
        top_waste_item = waste_items.iloc[0]
        finding2 = {
            "title": "Highest Waste Cost Item - Analysis Week",
            "claim": f"Item '{top_waste_item['item']}' (SKU: {top_waste_item['sku']}) incurred the highest waste cost of {top_waste_item['known_waste_cost_sar']:.2f} SAR during week of Feb 23, 2026, with {int(top_waste_item['units_wasted'])} units wasted.",
            "finding_type": "waste_analysis",
            "metrics": {
                "item_name": {
                    "value": top_waste_item['item'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-23T00:00:00+00:00",
                    "period_end": "2026-03-02T00:00:00+00:00"
                },
                "sku": {
                    "value": top_waste_item['sku'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-23T00:00:00+00:00",
                    "period_end": "2026-03-02T00:00:00+00:00"
                },
                "units_wasted": {
                    "value": int(top_waste_item['units_wasted']),
                    "unit": "units",
                    "numerator": int(top_waste_item['units_wasted']),
                    "denominator": None,
                    "period_start": "2026-02-23T00:00:00+00:00",
                    "period_end": "2026-03-02T00:00:00+00:00"
                },
                "known_waste_cost_sar": {
                    "value": round(top_waste_item['known_waste_cost_sar'], 2),
                    "unit": "SAR",
                    "numerator": round(top_waste_item['known_waste_cost_sar'], 2),
                    "denominator": None,
                    "period_start": "2026-02-23T00:00:00+00:00",
                    "period_end": "2026-03-02T00:00:00+00:00"
                },
                "total_waste_cost_all_items": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": round(total_waste_cost, 2),
                    "denominator": None,
                    "period_start": "2026-02-23T00:00:00+00:00",
                    "period_end": "2026-03-02T00:00:00+00:00"
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(inventory_analysis_with_waste),
            "coverage_notes": [
                "Analysis week: Feb 23, 2026",
                "Only items with non-null known_waste_cost_sar included",
                f"Total items with waste data: {len(inventory_analysis_with_waste)}",
                "Waste cost sourced from inventory.known_waste_cost_sar"
            ],
            "assumptions": [
                "known_waste_cost_sar represents actual waste cost incurred",
                "Blank waste values treated as unknown, not zero"
            ],
            "confidence": 0.90
        }
        findings.append(finding2)

# FINDING 3: Supplier Price Change Impact Analysis
# Check for price changes in emails
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    # Calculate percentage change
    price_changes['price_change_pct'] = (
        ((price_changes['new_price'] - price_changes['old_price']) / price_changes['old_price'] * 100)
    ).round(2)
    
    # Sort by absolute price change
    price_changes['abs_price_change'] = abs(price_changes['new_price'] - price_changes['old_price'])
    price_changes_sorted = price_changes.sort_values('abs_price_change', ascending=False)
    
    if len(price_changes_sorted) > 0:
        top_change = price_changes_sorted.iloc[0]
        
        # Try to find related inventory/menu items
        entity = str(top_change['entity_or_ingredient']).lower()
        
        finding3 = {
            "title": "Supplier Price Change - Potential Cost Impact",
            "claim": f"Supplier price change detected for '{top_change['entity_or_ingredient']}': {top_change['old_price']:.2f} {top_change['currency']}/{top_change['unit']} → {top_change['new_price']:.2f} {top_change['currency']}/{top_change['unit']} ({top_change['price_change_pct']:+.2f}%), effective {top_change['effective_date']}. Impact on menu items requires standing order volume and payment terms confirmation.",
            "finding_type": "supplier_price_change",
            "metrics": {
                "entity_or_ingredient": {
                    "value": top_change['entity_or_ingredient'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-23T00:00:00+00:00",
                    "period_end": "2026-03-02T00:00:00+00:00"
                },
                "old_price": {
                    "value": round(top_change['old_price'], 2),
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": round(top_change['old_price'], 2),
                    "denominator": None,
                    "period_start": "2026-02-23T00:00:00+00:00",
                    "period_end": "2026-03-02T00:00:00+00:00"
                },
                "new_price": {
                    "value": round(top_change['new_price'], 2),
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": round(top_change['new_price'], 2),
                    "denominator": None,
                    "period_start": "2026-02-23T00:00:00+00:00",
                    "period_end": "2026-03-02T00:00:00+00:00"
                },
                "price_change_pct": {
                    "value": top_change['price_change_pct'],
                    "unit": "%",
                    "numerator": round(top_change['new_price'] - top_change['old_price'], 2),
                    "denominator": round(top_change['old_price'], 2),
                    "period_start": "2026-02-23T00:00:00+00:00",
                    "period_end": "2026-03-02T00:00:00+00:00"
                },
                "effective_date": {
                    "value": str(top_change['effective_date']),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-23T00:00:00+00:00",
                    "period_end": "2026-03-02T00:00:00+00:00"
                },
                "currency": {
                    "value": top_change['currency'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-23T00:00:00+00:00",
                    "period_end": "2026-03-02T00:00:00+00:00"
                },
                "unit": {
                    "value": top_change['unit'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-23T00:00:00+00:00",
                    "period_end": "2026-03-02T00:00:00+00:00"
                }
            },
            "source_names": ["emails"],
            "sample_size": len(price_changes),
            "coverage_notes": [
                f"Total supplier price changes detected: {len(price_changes)}",
                "Only changes with old_price, new_price, and effective_date included",
                "No recipe/BOM data available to calculate per-drink impact",
                "Standing order quantities not confirmed in email extraction"
            ],
            "assumptions": [
                "Price change applies to future purchases only",
                "No information on standing order volumes or payment terms",
                "Actual margin impact depends on order volume and payment timing",
                "No per-drink impact calculated without recipe data"
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
    json.dump(output, f, indent=2)
