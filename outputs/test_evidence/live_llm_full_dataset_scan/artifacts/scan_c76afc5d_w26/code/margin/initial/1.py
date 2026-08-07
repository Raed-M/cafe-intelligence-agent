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
analysis_start = pd.to_datetime('2026-07-06T00:00:00+03:00')
analysis_end = pd.to_datetime('2026-07-13T00:00:00+03:00')
previous_start = pd.to_datetime('2026-06-29T00:00:00+03:00')
previous_end = pd.to_datetime('2026-07-06T00:00:00+03:00')

# Filter POS for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Filter inventory for analysis week
inv_analysis = inventory_df[inventory_df['week_starting'] == pd.to_datetime('2026-07-06')]
inv_previous = inventory_df[inventory_df['week_starting'] == pd.to_datetime('2026-06-29')]

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from menu and POS
menu_with_cost = menu_df[['sku', 'item_en', 'price_sar', 'unit_cost_sar']].copy()

# Aggregate POS by SKU for analysis period
pos_by_sku = pos_analysis[pos_analysis['is_refund'] == False].groupby('sku').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
pos_by_sku.columns = ['sku', 'total_quantity', 'total_revenue', 'basket_count']

# Merge with menu costs
item_economics = pos_by_sku.merge(menu_with_cost, on='sku', how='left')

# Calculate COGS and gross profit
item_economics['cogs'] = item_economics['total_quantity'] * item_economics['unit_cost_sar']
item_economics['gross_profit'] = item_economics['total_revenue'] - item_economics['cogs']
item_economics['gross_margin_pct'] = (item_economics['gross_profit'] / item_economics['total_revenue'] * 100).round(2)

# Filter for items with meaningful sales
item_economics_filtered = item_economics[item_economics['total_quantity'] > 0].copy()
item_economics_filtered = item_economics_filtered.sort_values('gross_profit', ascending=False)

if len(item_economics_filtered) > 0:
    top_item = item_economics_filtered.iloc[0]
    
    finding1 = {
        "title": "Top Gross Profit Item - Week of 2026-07-06",
        "claim": f"{top_item['item_en']} generated the highest gross profit of {top_item['gross_profit']:.2f} SAR with {top_item['gross_margin_pct']:.1f}% margin",
        "finding_type": "item_economics",
        "metrics": {
            "item_name": {"value": top_item['item_en'], "unit": None, "numerator": None, "denominator": None, "period_start": "2026-07-06T00:00:00+03:00", "period_end": "2026-07-13T00:00:00+03:00"},
            "total_revenue": {"value": round(top_item['total_revenue'], 2), "unit": "SAR", "numerator": round(top_item['total_revenue'], 2), "denominator": None, "period_start": "2026-07-06T00:00:00+03:00", "period_end": "2026-07-13T00:00:00+03:00"},
            "total_cogs": {"value": round(top_item['cogs'], 2), "unit": "SAR", "numerator": round(top_item['cogs'], 2), "denominator": None, "period_start": "2026-07-06T00:00:00+03:00", "period_end": "2026-07-13T00:00:00+03:00"},
            "gross_profit": {"value": round(top_item['gross_profit'], 2), "unit": "SAR", "numerator": round(top_item['gross_profit'], 2), "denominator": None, "period_start": "2026-07-06T00:00:00+03:00", "period_end": "2026-07-13T00:00:00+03:00"},
            "gross_margin_pct": {"value": top_item['gross_margin_pct'], "unit": "%", "numerator": top_item['gross_margin_pct'], "denominator": 100, "period_start": "2026-07-06T00:00:00+03:00", "period_end": "2026-07-13T00:00:00+03:00"},
            "quantity_sold": {"value": int(top_item['total_quantity']), "unit": "units", "numerator": int(top_item['total_quantity']), "denominator": None, "period_start": "2026-07-06T00:00:00+03:00", "period_end": "2026-07-13T00:00:00+03:00"},
            "unit_cost": {"value": top_item['unit_cost_sar'], "unit": "SAR", "numerator": top_item['unit_cost_sar'], "denominator": None, "period_start": "2026-07-06T00:00:00+03:00", "period_end": "2026-07-13T00:00:00+03:00"}
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(top_item['basket_count']),
        "coverage_notes": [
            f"Analysis period: 2026-07-06 to 2026-07-13",
            f"Includes {len(item_economics_filtered)} items with sales",
            f"Excludes refunds (is_refund=False)",
            f"Unit costs from menu.unit_cost_sar"
        ],
        "assumptions": [
            "Menu unit_cost_sar is accurate and current",
            "No recipe/BOM adjustments applied",
            "Quantity from POS line items",
            "Revenue calculated as sum of line_total_sar"
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
    
    if len(waste_items) > 0:
        top_waste_item = waste_items.iloc[0]
        
        finding2 = {
            "title": "Highest Waste Cost Item - Week of 2026-07-06",
            "claim": f"{top_waste_item['item']} incurred {top_waste_item['known_waste_cost_sar']:.2f} SAR in waste cost from {int(top_waste_item['units_wasted'])} units wasted",
            "finding_type": "waste_cost",
            "metrics": {
                "item_name": {"value": top_waste_item['item'], "unit": None, "numerator": None, "denominator": None, "period_start": "2026-07-06T00:00:00+03:00", "period_end": "2026-07-13T00:00:00+03:00"},
                "units_wasted": {"value": int(top_waste_item['units_wasted']), "unit": "units", "numerator": int(top_waste_item['units_wasted']), "denominator": None, "period_start": "2026-07-06T00:00:00+03:00", "period_end": "2026-07-13T00:00:00+03:00"},
                "waste_cost_sar": {"value": round(top_waste_item['known_waste_cost_sar'], 2), "unit": "SAR", "numerator": round(top_waste_item['known_waste_cost_sar'], 2), "denominator": None, "period_start": "2026-07-06T00:00:00+03:00", "period_end": "2026-07-13T00:00:00+03:00"},
                "total_waste_cost_week": {"value": round(total_waste_cost, 2), "unit": "SAR", "numerator": round(total_waste_cost, 2), "denominator": None, "period_start": "2026-07-06T00:00:00+03:00", "period_end": "2026-07-13T00:00:00+03:00"}
            },
            "source_names": ["inventory"],
            "sample_size": len(inv_analysis_with_waste),
            "coverage_notes": [
                f"Analysis period: 2026-07-06 to 2026-07-13",
                f"Only items with non-null known_waste_cost_sar included",
                f"Total items with waste data: {len(inv_analysis_with_waste)}",
                "Waste cost calculated from inventory.known_waste_cost_sar"
            ],
            "assumptions": [
                "known_waste_cost_sar reflects actual waste cost",
                "Null waste values treated as unknown, not zero",
                "Waste cost = units_wasted × unit_cost_sar"
            ],
            "confidence": 0.90
        }
        findings.append(finding2)

# FINDING 3: Supplier Price Changes and Impact
# Analyze supplier emails for price changes
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
    
    # Try to find related menu items
    entity_lower = str(top_change['entity_or_ingredient']).lower()
    related_items = menu_df[menu_df['item_en'].str.lower().str.contains(entity_lower, na=False)]
    
    finding3 = {
        "title": f"Supplier Price Change - {top_change['entity_or_ingredient']}",
        "claim": f"{top_change['entity_or_ingredient']} price changed from {top_change['old_price']} to {top_change['new_price']} {top_change['currency']} per {top_change['unit']} (effective {top_change['effective_date'].strftime('%Y-%m-%d')}), a {top_change['price_change_pct']:.1f}% change",
        "finding_type": "supplier_price_change",
        "metrics": {
            "ingredient": {"value": top_change['entity_or_ingredient'], "unit": None, "numerator": None, "denominator": None, "period_start": top_change['effective_date'].isoformat(), "period_end": None},
            "old_price": {"value": top_change['old_price'], "unit": f"{top_change['currency']}/{top_change['unit']}", "numerator": top_change['old_price'], "denominator": None, "period_start": None, "period_end": None},
            "new_price": {"value": top_change['new_price'], "unit": f"{top_change['currency']}/{top_change['unit']}", "numerator": top_change['new_price'], "denominator": None, "period_start": top_change['effective_date'].isoformat(), "period_end": None},
            "price_change_pct": {"value": top_change['price_change_pct'], "unit": "%", "numerator": top_change['price_change_pct'], "denominator": 100, "period_start": top_change['effective_date'].isoformat(), "period_end": None},
            "effective_date": {"value": top_change['effective_date'].strftime('%Y-%m-%d'), "unit": None, "numerator": None, "denominator": None, "period_start": top_change['effective_date'].isoformat(), "period_end": None}
        },
        "source_names": ["emails"],
        "sample_size": len(price_changes),
        "coverage_notes": [
            f"Total supplier price changes identified: {len(price_changes)}",
            f"Related menu items found: {len(related_items)}",
            "Price changes extracted from supplier emails",
            "No recipe/BOM data available to calculate per-drink impact"
        ],
        "assumptions": [
            "Email extraction confidence reflects data quality",
            "Price change applies to specified unit and currency",
            "No recipe/BOM exists to calculate exact menu item cost impact",
            "Related items identified by name matching only"
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

print(f"Analysis complete. {len(findings)} findings generated.")
