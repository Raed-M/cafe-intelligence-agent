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

# Define analysis periods
analysis_period = {
    "start": "2026-07-06T00:00:00+03:00",
    "end": "2026-07-13T00:00:00+03:00"
}
previous_period = {
    "start": "2026-06-29T00:00:00+03:00",
    "end": "2026-07-06T00:00:00+03:00"
}

# Convert to datetime for filtering
analysis_start = pd.to_datetime(analysis_period["start"])
analysis_end = pd.to_datetime(analysis_period["end"])
previous_start = pd.to_datetime(previous_period["start"])
previous_end = pd.to_datetime(previous_period["end"])

# Filter POS data for analysis period
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Filter inventory data for analysis period
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
inv_analysis = inventory_df[inventory_df['week_starting'] == pd.to_datetime('2026-07-06')].copy()
inv_previous = inventory_df[inventory_df['week_starting'] == pd.to_datetime('2026-06-29')].copy()

findings = []

# Finding 1: Item-level COGS and Gross Profit Analysis
if len(pos_analysis) > 0 and len(menu_df) > 0:
    # Merge POS with menu to get unit costs
    pos_with_cost = pos_analysis.merge(menu_df[['sku', 'unit_cost_sar']], on='sku', how='left')
    
    # Calculate COGS and gross profit for each line item
    pos_with_cost['cogs_sar'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
    pos_with_cost['gross_profit_sar'] = pos_with_cost['line_total_sar'] - pos_with_cost['cogs_sar']
    
    # Filter out refunds for net calculations
    pos_sales = pos_with_cost[~pos_with_cost['is_refund']].copy()
    
    # Aggregate by item
    item_economics = pos_sales.groupby('item_name_en').agg({
        'quantity': 'sum',
        'line_total_sar': 'sum',
        'cogs_sar': 'sum',
        'gross_profit_sar': 'sum',
        'transaction_id': 'nunique'
    }).reset_index()
    
    item_economics['gross_margin_pct'] = (item_economics['gross_profit_sar'] / item_economics['line_total_sar'] * 100).round(2)
    item_economics = item_economics.sort_values('gross_profit_sar', ascending=False)
    
    # Top 3 items by gross profit
    top_items = item_economics.head(3)
    
    if len(top_items) > 0:
        finding_1 = {
            "title": "Top 3 Items by Gross Profit (Analysis Week)",
            "claim": f"During the analysis week (2026-07-06 to 2026-07-13), the top 3 items by gross profit contribution are: {', '.join(top_items['item_name_en'].tolist())}. These items generated {top_items['gross_profit_sar'].sum():.2f} SAR in total gross profit across {top_items['transaction_id'].sum()} transactions.",
            "finding_type": "item_economics",
            "metrics": {
                "top_item_1_name": {
                    "value": top_items.iloc[0]['item_name_en'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_period["start"],
                    "period_end": analysis_period["end"]
                },
                "top_item_1_gross_profit_sar": {
                    "value": round(top_items.iloc[0]['gross_profit_sar'], 2),
                    "unit": "SAR",
                    "numerator": round(top_items.iloc[0]['gross_profit_sar'], 2),
                    "denominator": None,
                    "period_start": analysis_period["start"],
                    "period_end": analysis_period["end"]
                },
                "top_item_1_quantity_sold": {
                    "value": int(top_items.iloc[0]['quantity']),
                    "unit": "units",
                    "numerator": int(top_items.iloc[0]['quantity']),
                    "denominator": None,
                    "period_start": analysis_period["start"],
                    "period_end": analysis_period["end"]
                },
                "top_item_1_gross_margin_pct": {
                    "value": round(top_items.iloc[0]['gross_margin_pct'], 2),
                    "unit": "%",
                    "numerator": round(top_items.iloc[0]['gross_margin_pct'], 2),
                    "denominator": 100,
                    "period_start": analysis_period["start"],
                    "period_end": analysis_period["end"]
                },
                "top_item_2_name": {
                    "value": top_items.iloc[1]['item_name_en'] if len(top_items) > 1 else None,
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_period["start"],
                    "period_end": analysis_period["end"]
                },
                "top_item_2_gross_profit_sar": {
                    "value": round(top_items.iloc[1]['gross_profit_sar'], 2) if len(top_items) > 1 else None,
                    "unit": "SAR",
                    "numerator": round(top_items.iloc[1]['gross_profit_sar'], 2) if len(top_items) > 1 else None,
                    "denominator": None,
                    "period_start": analysis_period["start"],
                    "period_end": analysis_period["end"]
                },
                "top_item_3_name": {
                    "value": top_items.iloc[2]['item_name_en'] if len(top_items) > 2 else None,
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_period["start"],
                    "period_end": analysis_period["end"]
                },
                "top_item_3_gross_profit_sar": {
                    "value": round(top_items.iloc[2]['gross_profit_sar'], 2) if len(top_items) > 2 else None,
                    "unit": "SAR",
                    "numerator": round(top_items.iloc[2]['gross_profit_sar'], 2) if len(top_items) > 2 else None,
                    "denominator": None,
                    "period_start": analysis_period["start"],
                    "period_end": analysis_period["end"]
                },
                "total_gross_profit_top_3_sar": {
                    "value": round(top_items['gross_profit_sar'].sum(), 2),
                    "unit": "SAR",
                    "numerator": round(top_items['gross_profit_sar'].sum(), 2),
                    "denominator": None,
                    "period_start": analysis_period["start"],
                    "period_end": analysis_period["end"]
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": int(pos_sales['transaction_id'].nunique()),
            "coverage_notes": [
                f"Analysis period: 2026-07-06 to 2026-07-13",
                f"Total POS transactions in period: {int(pos_sales['transaction_id'].nunique())}",
                f"Total line items analyzed: {len(pos_sales)}",
                f"Items with menu cost data: {pos_with_cost['unit_cost_sar'].notna().sum()} out of {len(pos_sales)}",
                "Refunds excluded from net calculations"
            ],
            "assumptions": [
                "Menu unit_cost_sar values are current and applicable to analysis period",
                "POS line_total_sar includes all discounts and is accurate",
                "Quantity represents actual units sold"
            ],
            "confidence": 0.95
        }
        findings.append(finding_1)

# Finding 2: Waste Cost Analysis
if len(inv_analysis) > 0:
    # Calculate waste costs for items with non-null waste values
    inv_with_waste = inv_analysis[inv_analysis['units_wasted'].notna() & (inv_analysis['units_wasted'] > 0)].copy()
    
    if len(inv_with_waste) > 0:
        inv_with_waste['waste_cost_sar'] = inv_with_waste['units_wasted'] * inv_with_waste['unit_cost_sar']
        
        total_waste_units = inv_with_waste['units_wasted'].sum()
        total_waste_cost = inv_with_waste['waste_cost_sar'].sum()
        
        # Compare with previous period
        inv_prev_with_waste = inv_previous[inv_previous['units_wasted'].notna() & (inv_previous['units_wasted'] > 0)].copy()
        prev_waste_cost = 0
        if len(inv_prev_with_waste) > 0:
            inv_prev_with_waste['waste_cost_sar'] = inv_prev_with_waste['units_wasted'] * inv_prev_with_waste['unit_cost_sar']
            prev_waste_cost = inv_prev_with_waste['waste_cost_sar'].sum()
        
        waste_change = total_waste_cost - prev_waste_cost
        
        finding_2 = {
            "title": "Quantified Waste Cost (Analysis Week)",
            "claim": f"During the analysis week (2026-07-06 to 2026-07-13), documented waste totaled {total_waste_units:.0f} units with a known waste cost of {total_waste_cost:.2f} SAR. This represents a change of {waste_change:.2f} SAR compared to the previous week.",
            "finding_type": "waste_cost",
            "metrics": {
                "total_waste_units": {
                    "value": round(total_waste_units, 2),
                    "unit": "units",
                    "numerator": round(total_waste_units, 2),
                    "denominator": None,
                    "period_start": analysis_period["start"],
                    "period_end": analysis_period["end"]
                },
                "total_waste_cost_sar": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": round(total_waste_cost, 2),
                    "denominator": None,
                    "period_start": analysis_period["start"],
                    "period_end": analysis_period["end"]
                },
                "previous_week_waste_cost_sar": {
                    "value": round(prev_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": round(prev_waste_cost, 2),
                    "denominator": None,
                    "period_start": previous_period["start"],
                    "period_end": previous_period["end"]
                },
                "waste_cost_change_sar": {
                    "value": round(waste_change, 2),
                    "unit": "SAR",
                    "numerator": round(waste_change, 2),
                    "denominator": None,
                    "period_start": analysis_period["start"],
                    "period_end": analysis_period["end"]
                },
                "items_with_waste": {
                    "value": len(inv_with_waste),
                    "unit": "count",
                    "numerator": len(inv_with_waste),
                    "denominator": None,
                    "period_start": analysis_period["start"],
                    "period_end": analysis_period["end"]
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(inv_with_waste),
            "coverage_notes": [
                f"Analysis period: 2026-07-06 to 2026-07-13",
                f"Items with documented waste: {len(inv_with_waste)} out of {len(inv_analysis)} inventory records",
                "Only non-null waste values included in calculation",
                "Waste cost calculated as units_wasted × unit_cost_sar"
            ],
            "assumptions": [
                "Unit costs in inventory records are accurate for waste valuation",
                "Waste units are accurately recorded",
                "Blank waste values represent zero waste (not missing data)"
            ],
            "confidence": 0.85
        }
        findings.append(finding_2)

# Finding 3: Supplier Price Changes from Emails
if len(emails_df) > 0:
    # Filter for price change emails with old and new prices
    price_changes = emails_df[
        (emails_df['old_price'].notna()) & 
        (emails_df['new_price'].notna()) &
        (emails_df['category'] == 'supplier_price_change')
    ].copy()
    
    if len(price_changes) > 0:
        price_changes['date'] = pd.to_datetime(price_changes['date'])
        price_changes['effective_date'] = pd.to_datetime(price_changes['effective_date'])
        
        # Calculate percentage change
        price_changes['price_change_pct'] = ((price_changes['new_price'] - price_changes['old_price']) / price_changes['old_price'] * 100).round(2)
        
        # Get the most recent price change
        recent_change = price_changes.sort_values('effective_date', ascending=False).iloc[0]
        
        finding_3 = {
            "title": "Supplier Price Change Detected",
            "claim": f"Email evidence indicates a price change for {recent_change['entity_or_ingredient']}: from {recent_change['old_price']:.2f} to {recent_change['new_price']:.2f} {recent_change['currency']} per {recent_change['unit']}, effective {recent_change['effective_date'].strftime('%Y-%m-%d')}. This represents a {recent_change['price_change_pct']:.2f}% change.",
            "finding_type": "supplier_price_change",
            "metrics": {
                "ingredient": {
                    "value": recent_change['entity_or_ingredient'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_period["start"],
                    "period_end": analysis_period["end"]
                },
                "old_price": {
                    "value": round(recent_change['old_price'], 2),
                    "unit": recent_change['currency'],
                    "numerator": round(recent_change['old_price'], 2),
                    "denominator": None,
                    "period_start": analysis_period["start"],
                    "period_end": analysis_period["end"]
                },
                "new_price": {
                    "value": round(recent_change['new_price'], 2),
                    "unit": recent_change['currency'],
                    "numerator": round(recent_change['new_price'], 2),
                    "denominator": None,
                    "period_start": analysis_period["start"],
                    "period_end": analysis_period["end"]
                },
                "price_change_pct": {
                    "value": round(recent_change['price_change_pct'], 2),
                    "unit": "%",
                    "numerator": round(recent_change['price_change_pct'], 2),
                    "denominator": 100,
                    "period_start": analysis_period["start"],
                    "period_end": analysis_period["end"]
                },
                "unit": {
                    "value": recent_change['unit'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_period["start"],
                    "period_end": analysis_period["end"]
                },
                "effective_date": {
                    "value": recent_change['effective_date'].strftime('%Y-%m-%d'),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_period["start"],
                    "period_end": analysis_period["end"]
                }
            },
            "source_names": ["emails"],
            "sample_size": len(price_changes),
            "coverage_notes": [
                f"Analysis period: 2026-07-06 to 2026-07-13",
                f"Total supplier price change emails found: {len(price_changes)}",
                "Most recent price change reported",
                "Price changes extracted from supplier emails with confidence scores"
            ],
            "assumptions": [
                "Email extraction accurately captured price change details",
                "Effective date represents when price change takes effect",
                "Price change applies to the specified ingredient/entity only",
                "No recipe/BOM data available to calculate per-drink impact"
            ],
            "confidence": 0.80
        }
        findings.append(finding_3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write to output file
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. {len(findings)} findings generated.")
