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

# Define analysis period
analysis_start = datetime.fromisoformat("2026-03-16T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-03-23T00:00:00+03:00")

# Convert timestamp columns to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
emails_df['date'] = pd.to_datetime(emails_df['date'])
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'])

# Filter POS data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()

# Filter inventory for analysis period (week starting 2026-03-16)
inventory_analysis = inventory_df[inventory_df['week_starting'] == pd.Timestamp('2026-03-16')].copy()

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item-level economics from POS and menu data

# Merge POS with menu to get unit costs
pos_with_cost = pos_analysis.merge(menu_df[['sku', 'unit_cost_sar']], on='sku', how='left')

# Calculate line-level COGS and gross profit
pos_with_cost['line_cogs_sar'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
pos_with_cost['line_gross_profit_sar'] = pos_with_cost['line_total_sar'] - pos_with_cost['line_cogs_sar']

# Exclude refunds from analysis
pos_sales = pos_with_cost[pos_with_cost['is_refund'] == False].copy()

# Group by item to get total metrics
item_economics = pos_sales.groupby(['sku', 'item_name_en']).agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'line_cogs_sar': 'sum',
    'line_gross_profit_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()

item_economics.columns = ['sku', 'item_name', 'total_quantity', 'total_revenue', 'total_cogs', 'total_gross_profit', 'basket_count']

# Calculate margin percentage
item_economics['margin_pct'] = (item_economics['total_gross_profit'] / item_economics['total_revenue'] * 100).round(2)

# Sort by gross profit and get top 5
top_5_items = item_economics.nlargest(5, 'total_gross_profit')

# Verify data quality
top_5_valid = top_5_items[top_5_items['total_cogs'].notna() & (top_5_items['total_cogs'] > 0)]

if len(top_5_valid) > 0:
    finding_1 = {
        "title": "Top 5 Items by Gross Profit (Analysis Week)",
        "claim": f"The top 5 items by gross profit in the analysis period (2026-03-16 to 2026-03-23) generated SAR {top_5_valid['total_gross_profit'].sum():.2f} in total gross profit from {top_5_valid['total_quantity'].sum():.0f} units sold across {top_5_valid['basket_count'].sum():.0f} baskets.",
        "finding_type": "item_economics",
        "metrics": {},
        "source_names": ["pos", "menu"],
        "sample_size": int(top_5_valid['basket_count'].sum()),
        "coverage_notes": [
            f"Analysis period: 2026-03-16 to 2026-03-23",
            f"POS rows analyzed: {len(pos_sales)}",
            f"Items with valid unit costs: {len(top_5_valid)}",
            "Refunds excluded from calculations",
            "Unit costs sourced from menu.parquet"
        ],
        "assumptions": [
            "Menu unit_cost_sar is accurate and current for analysis period",
            "POS line_total_sar reflects actual revenue after discounts",
            "No recipe/BOM data available; using menu unit costs as-is"
        ],
        "confidence": 0.95
    }
    
    # Add individual item metrics
    for idx, row in top_5_valid.iterrows():
        rank = idx + 1
        finding_1["metrics"][f"top_item_{rank}_name"] = {
            "value": row['item_name'],
            "unit": None,
            "numerator": None,
            "denominator": None,
            "period_start": "2026-03-16T00:00:00+03:00",
            "period_end": "2026-03-23T00:00:00+03:00"
        }
        finding_1["metrics"][f"top_item_{rank}_gross_profit"] = {
            "value": round(row['total_gross_profit'], 2),
            "unit": "SAR",
            "numerator": round(row['total_gross_profit'], 2),
            "denominator": None,
            "period_start": "2026-03-16T00:00:00+03:00",
            "period_end": "2026-03-23T00:00:00+03:00"
        }
        finding_1["metrics"][f"top_item_{rank}_quantity"] = {
            "value": int(row['total_quantity']),
            "unit": "units",
            "numerator": int(row['total_quantity']),
            "denominator": None,
            "period_start": "2026-03-16T00:00:00+03:00",
            "period_end": "2026-03-23T00:00:00+03:00"
        }
        finding_1["metrics"][f"top_item_{rank}_margin_pct"] = {
            "value": round(row['margin_pct'], 2),
            "unit": "%",
            "numerator": round(row['total_gross_profit'], 2),
            "denominator": round(row['total_revenue'], 2),
            "period_start": "2026-03-16T00:00:00+03:00",
            "period_end": "2026-03-23T00:00:00+03:00"
        }
    
    finding_1["metrics"]["total_top_5_gross_profit"] = {
        "value": round(top_5_valid['total_gross_profit'].sum(), 2),
        "unit": "SAR",
        "numerator": round(top_5_valid['total_gross_profit'].sum(), 2),
        "denominator": None,
        "period_start": "2026-03-16T00:00:00+03:00",
        "period_end": "2026-03-23T00:00:00+03:00"
    }
    finding_1["metrics"]["total_top_5_quantity"] = {
        "value": int(top_5_valid['total_quantity'].sum()),
        "unit": "units",
        "numerator": int(top_5_valid['total_quantity'].sum()),
        "denominator": None,
        "period_start": "2026-03-16T00:00:00+03:00",
        "period_end": "2026-03-23T00:00:00+03:00"
    }
    
    findings.append(finding_1)

# FINDING 2: Waste Cost Impact Analysis
# Calculate known waste costs from inventory data

waste_analysis = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna() & (inventory_analysis['known_waste_cost_sar'] > 0)].copy()

if len(waste_analysis) > 0:
    total_waste_cost = waste_analysis['known_waste_cost_sar'].sum()
    total_waste_units = waste_analysis['units_wasted'].sum()
    
    finding_2 = {
        "title": "Quantified Waste Cost (Week of 2026-03-16)",
        "claim": f"Known waste in the week of 2026-03-16 totaled SAR {total_waste_cost:.2f} across {int(total_waste_units)} units, representing measurable cost leakage from {len(waste_analysis)} items.",
        "finding_type": "waste_cost",
        "metrics": {
            "total_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": round(total_waste_cost, 2),
                "denominator": None,
                "period_start": "2026-03-16T00:00:00+03:00",
                "period_end": "2026-03-23T00:00:00+03:00"
            },
            "total_waste_units": {
                "value": int(total_waste_units),
                "unit": "units",
                "numerator": int(total_waste_units),
                "denominator": None,
                "period_start": "2026-03-16T00:00:00+03:00",
                "period_end": "2026-03-23T00:00:00+03:00"
            },
            "items_with_waste": {
                "value": len(waste_analysis),
                "unit": "count",
                "numerator": len(waste_analysis),
                "denominator": None,
                "period_start": "2026-03-16T00:00:00+03:00",
                "period_end": "2026-03-23T00:00:00+03:00"
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(waste_analysis),
        "coverage_notes": [
            "Analysis period: week starting 2026-03-16",
            "Only non-null waste_cost_sar values included",
            "Blank waste values treated as unknown, not zero",
            f"Items with measurable waste: {len(waste_analysis)} out of {len(inventory_analysis)} total items"
        ],
        "assumptions": [
            "known_waste_cost_sar reflects actual cost of wasted inventory",
            "Waste data is complete for items with non-null values"
        ],
        "confidence": 0.90
    }
    
    findings.append(finding_2)

# FINDING 3: Supplier Price Changes and Procurement Impact
# Detect dated supplier price changes from emails

price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) & 
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    # Calculate percentage change
    price_changes['pct_change'] = ((price_changes['new_price'] - price_changes['old_price']) / price_changes['old_price'] * 100).round(2)
    
    # Filter for significant changes (>5%)
    significant_changes = price_changes[abs(price_changes['pct_change']) > 5].copy()
    
    if len(significant_changes) > 0:
        # Get the most recent significant change
        latest_change = significant_changes.sort_values('effective_date', ascending=False).iloc[0]
        
        finding_3 = {
            "title": "Supplier Price Change Alert",
            "claim": f"Supplier email dated {latest_change['date'].strftime('%Y-%m-%d')} documents a {latest_change['pct_change']:.1f}% price change for {latest_change['entity_or_ingredient']} (from {latest_change['old_price']:.2f} to {latest_change['new_price']:.2f} {latest_change['currency']}/{latest_change['unit']}), effective {latest_change['effective_date'].strftime('%Y-%m-%d')}. This represents a material cost pressure requiring menu or procurement review.",
            "finding_type": "supplier_price_change",
            "metrics": {
                "ingredient": {
                    "value": latest_change['entity_or_ingredient'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": latest_change['effective_date'].isoformat(),
                    "period_end": None
                },
                "old_price": {
                    "value": round(latest_change['old_price'], 2),
                    "unit": f"{latest_change['currency']}/{latest_change['unit']}",
                    "numerator": round(latest_change['old_price'], 2),
                    "denominator": None,
                    "period_start": latest_change['date'].isoformat(),
                    "period_end": latest_change['date'].isoformat()
                },
                "new_price": {
                    "value": round(latest_change['new_price'], 2),
                    "unit": f"{latest_change['currency']}/{latest_change['unit']}",
                    "numerator": round(latest_change['new_price'], 2),
                    "denominator": None,
                    "period_start": latest_change['effective_date'].isoformat(),
                    "period_end": None
                },
                "price_change_pct": {
                    "value": round(latest_change['pct_change'], 2),
                    "unit": "%",
                    "numerator": round(latest_change['pct_change'], 2),
                    "denominator": 100,
                    "period_start": latest_change['effective_date'].isoformat(),
                    "period_end": None
                },
                "email_date": {
                    "value": latest_change['date'].strftime('%Y-%m-%d'),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": latest_change['date'].isoformat(),
                    "period_end": latest_change['date'].isoformat()
                },
                "effective_date": {
                    "value": latest_change['effective_date'].strftime('%Y-%m-%d'),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": latest_change['effective_date'].isoformat(),
                    "period_end": None
                }
            },
            "source_names": ["emails"],
            "sample_size": 1,
            "coverage_notes": [
                f"Email source: {latest_change['sender']}",
                f"Subject: {latest_change['subject']}",
                "Price change extracted from supplier communication",
                "No standing order quantity data available for procurement scenario calculation"
            ],
            "assumptions": [
                "Email extraction accurately captured old and new prices",
                "Effective date represents when price change takes effect",
                "Price applies to the specified ingredient/entity only"
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

print(f"Analysis complete. {len(findings)} findings generated.")
