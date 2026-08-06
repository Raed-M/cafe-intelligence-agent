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

# Read input artifacts
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
analysis_start = pd.to_datetime("2026-01-12T00:00:00+03:00")
analysis_end = pd.to_datetime("2026-01-19T00:00:00+03:00")
previous_start = pd.to_datetime("2026-01-05T00:00:00+03:00")
previous_end = pd.to_datetime("2026-01-12T00:00:00+03:00")

# Filter POS data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Filter inventory for analysis week
inv_analysis = inventory_df[inventory_df['week_starting'] == pd.Timestamp('2026-01-12')]
inv_previous = inventory_df[inventory_df['week_starting'] == pd.Timestamp('2026-01-05')]

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from menu and POS data
if len(pos_analysis) > 0 and len(menu_df) > 0:
    # Merge POS with menu to get unit costs
    pos_with_cost = pos_analysis.merge(
        menu_df[['sku', 'unit_cost_sar', 'price_sar']],
        on='sku',
        how='left'
    )
    
    # Filter out refunds for revenue calculation
    pos_sales = pos_with_cost[~pos_with_cost['is_refund']].copy()
    
    # Calculate metrics by item
    item_metrics = pos_sales.groupby('item_name').agg({
        'quantity': 'sum',
        'line_total_sar': 'sum',
        'unit_cost_sar': 'first',
        'price_sar': 'first',
        'transaction_id': 'nunique'
    }).reset_index()
    
    item_metrics.columns = ['item_name', 'total_quantity', 'total_revenue', 'unit_cost', 'menu_price', 'basket_count']
    
    # Calculate COGS and gross profit
    item_metrics['total_cogs'] = item_metrics['total_quantity'] * item_metrics['unit_cost']
    item_metrics['gross_profit'] = item_metrics['total_revenue'] - item_metrics['total_cogs']
    item_metrics['gross_margin_pct'] = (item_metrics['gross_profit'] / item_metrics['total_revenue'] * 100).round(2)
    
    # Sort by gross profit contribution
    item_metrics_sorted = item_metrics.sort_values('gross_profit', ascending=False)
    
    # Top 3 items by gross profit
    top_items = item_metrics_sorted.head(3)
    
    if len(top_items) > 0:
        finding_1 = {
            "title": "Top 3 Items by Gross Profit Contribution (Week of Jan 12-19, 2026)",
            "claim": f"The top 3 items by gross profit contribution are {', '.join(top_items['item_name'].values[:3])} with combined gross profit of {top_items['gross_profit'].sum():.2f} SAR",
            "finding_type": "item_economics",
            "metrics": {
                "top_item_1_name": {
                    "value": top_items.iloc[0]['item_name'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "top_item_1_gross_profit_sar": {
                    "value": round(top_items.iloc[0]['gross_profit'], 2),
                    "unit": "SAR",
                    "numerator": round(top_items.iloc[0]['gross_profit'], 2),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "top_item_1_quantity": {
                    "value": int(top_items.iloc[0]['total_quantity']),
                    "unit": "units",
                    "numerator": int(top_items.iloc[0]['total_quantity']),
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
                    "value": top_items.iloc[1]['item_name'] if len(top_items) > 1 else None,
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "top_item_2_gross_profit_sar": {
                    "value": round(top_items.iloc[1]['gross_profit'], 2) if len(top_items) > 1 else None,
                    "unit": "SAR",
                    "numerator": round(top_items.iloc[1]['gross_profit'], 2) if len(top_items) > 1 else None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "top_item_3_name": {
                    "value": top_items.iloc[2]['item_name'] if len(top_items) > 2 else None,
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "top_item_3_gross_profit_sar": {
                    "value": round(top_items.iloc[2]['gross_profit'], 2) if len(top_items) > 2 else None,
                    "unit": "SAR",
                    "numerator": round(top_items.iloc[2]['gross_profit'], 2) if len(top_items) > 2 else None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "total_items_analyzed": {
                    "value": len(item_metrics),
                    "unit": "items",
                    "numerator": len(item_metrics),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": len(pos_sales),
            "coverage_notes": [
                f"Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
                f"Total POS transactions analyzed: {len(pos_sales)}",
                f"Unique items: {len(item_metrics)}",
                "Refunds excluded from revenue calculation",
                "Unit costs sourced from menu.parquet"
            ],
            "assumptions": [
                "Menu unit_cost_sar represents actual COGS per unit",
                "POS line_total_sar is accurate revenue after discounts",
                "No recipe/BOM adjustments applied",
                "Waste costs not included in this item-level analysis"
            ],
            "confidence": 0.95
        }
        findings.append(finding_1)

# FINDING 2: Waste Cost Impact Analysis
# Calculate known waste costs from inventory data
if len(inv_analysis) > 0:
    inv_analysis_with_waste = inv_analysis[inv_analysis['known_waste_cost_sar'].notna()].copy()
    
    if len(inv_analysis_with_waste) > 0:
        total_waste_cost = inv_analysis_with_waste['known_waste_cost_sar'].sum()
        waste_items = inv_analysis_with_waste[['item', 'units_wasted', 'known_waste_cost_sar']].copy()
        waste_items = waste_items.sort_values('known_waste_cost_sar', ascending=False)
        
        finding_2 = {
            "title": "Quantified Waste Cost Impact (Week of Jan 12-19, 2026)",
            "claim": f"Total known waste cost for the analysis week is {total_waste_cost:.2f} SAR across {len(waste_items)} items with waste observations",
            "finding_type": "waste_cost",
            "metrics": {
                "total_waste_cost_sar": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": round(total_waste_cost, 2),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "items_with_waste": {
                    "value": len(waste_items),
                    "unit": "items",
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
                "highest_waste_units": {
                    "value": int(waste_items.iloc[0]['units_wasted']) if len(waste_items) > 0 else None,
                    "unit": "units",
                    "numerator": int(waste_items.iloc[0]['units_wasted']) if len(waste_items) > 0 else None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(waste_items),
            "coverage_notes": [
                f"Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
                f"Only non-null waste_cost_sar values included: {len(waste_items)} items",
                "Blank waste values excluded per methodology",
                f"Total inventory records for period: {len(inv_analysis)}"
            ],
            "assumptions": [
                "known_waste_cost_sar from inventory.parquet is accurate",
                "Waste cost represents actual loss value",
                "No adjustments for partial waste observations"
            ],
            "confidence": 0.90
        }
        findings.append(finding_2)

# FINDING 3: Supplier Price Changes and Procurement Impact
# Analyze supplier emails for price changes
if len(emails_df) > 0:
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
        
        # Get top price change
        top_change = price_changes_sorted.iloc[0]
        
        finding_3 = {
            "title": "Supplier Price Changes Detected",
            "claim": f"Supplier price change detected for {top_change['entity_or_ingredient']}: {top_change['old_price']:.2f} {top_change['currency']}/{top_change['unit']} → {top_change['new_price']:.2f} {top_change['currency']}/{top_change['unit']} ({top_change['price_change_pct']:+.2f}%), effective {top_change['effective_date'].strftime('%Y-%m-%d')}",
            "finding_type": "supplier_price_change",
            "metrics": {
                "ingredient": {
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
                    "value": round(top_change['price_change_pct'], 2),
                    "unit": "%",
                    "numerator": round(top_change['price_change_pct'], 2),
                    "denominator": 100,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "effective_date": {
                    "value": top_change['effective_date'].strftime('%Y-%m-%d'),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "total_price_changes_detected": {
                    "value": len(price_changes),
                    "unit": "changes",
                    "numerator": len(price_changes),
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["emails"],
            "sample_size": len(price_changes),
            "coverage_notes": [
                f"Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
                f"Total price changes with old/new prices: {len(price_changes)}",
                "Only changes with both old and new prices included",
                "No recipe/BOM data available to calculate per-drink impact",
                "Standing order quantities not available in email data"
            ],
            "assumptions": [
                "Email extraction accurately captured old and new prices",
                "Effective dates are accurate",
                "Price changes apply to supplier invoices as stated",
                "No standing order volume data available to calculate procurement cost impact"
            ],
            "confidence": 0.85
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
