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
analysis_start = datetime(2026, 6, 29, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 7, 6, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 6, 22, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 6, 29, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Prepare findings list
findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from menu and POS data
menu_analysis = menu_df[['sku', 'item_en', 'price_sar', 'unit_cost_sar', 'category']].copy()

# Join with POS data for analysis period
pos_with_menu = pos_analysis.merge(menu_analysis, on='sku', how='left')

# Filter out refunds for revenue calculation
pos_sales = pos_with_menu[pos_with_menu['is_refund'] == False].copy()

# Calculate metrics by item
item_metrics = pos_sales.groupby('sku').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'unit_price_sar': 'first',
    'unit_cost_sar': 'first',
    'item_name_en': 'first'
}).reset_index()

item_metrics.columns = ['sku', 'total_quantity', 'total_revenue', 'unit_price', 'unit_cost', 'item_name']

# Add category from menu
item_metrics = item_metrics.merge(menu_analysis[['sku', 'category']], on='sku', how='left')

# Calculate COGS and gross profit
item_metrics['total_cogs'] = item_metrics['total_quantity'] * item_metrics['unit_cost']
item_metrics['gross_profit'] = item_metrics['total_revenue'] - item_metrics['total_cogs']
item_metrics['gross_margin_pct'] = (item_metrics['gross_profit'] / item_metrics['total_revenue'] * 100).round(2)

# Sort by gross profit
item_metrics_sorted = item_metrics.sort_values('gross_profit', ascending=False)

# Get top 3 items by gross profit
top_items = item_metrics_sorted.head(3)

if len(top_items) > 0:
    finding1 = {
        "title": "Top 3 Items by Gross Profit (Analysis Period)",
        "claim": f"The top 3 items by gross profit contribution during {analysis_start.date()} to {analysis_end.date()} are {', '.join(top_items['item_name'].values)} with combined gross profit of {top_items['gross_profit'].sum():.2f} SAR",
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
            "top_item_1_gross_profit": {
                "value": round(top_items.iloc[0]['gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_items.iloc[0]['total_revenue'], 2),
                "denominator": round(top_items.iloc[0]['total_cogs'], 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_margin_pct": {
                "value": top_items.iloc[0]['gross_margin_pct'],
                "unit": "%",
                "numerator": None,
                "denominator": None,
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
            "top_item_2_gross_profit": {
                "value": round(top_items.iloc[1]['gross_profit'], 2) if len(top_items) > 1 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[1]['total_revenue'], 2) if len(top_items) > 1 else None,
                "denominator": round(top_items.iloc[1]['total_cogs'], 2) if len(top_items) > 1 else None,
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
            "top_item_3_gross_profit": {
                "value": round(top_items.iloc[2]['gross_profit'], 2) if len(top_items) > 2 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[2]['total_revenue'], 2) if len(top_items) > 2 else None,
                "denominator": round(top_items.iloc[2]['total_cogs'], 2) if len(top_items) > 2 else None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(pos_sales),
        "coverage_notes": [
            "Analysis period: 2026-06-29 to 2026-07-06",
            "Excludes refunds (is_refund=False)",
            "Unit costs from menu.unit_cost_sar",
            "Revenue from POS line_total_sar"
        ],
        "assumptions": [
            "Menu unit_cost_sar is current and applicable to analysis period",
            "POS line_total_sar accurately reflects revenue after discounts",
            "No recipe/BOM data available; using menu unit costs as-is"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# FINDING 2: Waste Cost Analysis
# Filter inventory for analysis period week
inventory_analysis = inventory_df[inventory_df['week_starting'] == '2026-06-29'].copy()

if len(inventory_analysis) > 0:
    # Calculate total waste cost
    waste_items = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()
    
    if len(waste_items) > 0:
        total_waste_cost = waste_items['known_waste_cost_sar'].sum()
        total_units_wasted = waste_items['units_wasted'].sum()
        
        finding2 = {
            "title": "Quantified Waste Cost (Week of 2026-06-29)",
            "claim": f"Known waste cost for the week of 2026-06-29 totals {total_waste_cost:.2f} SAR across {int(total_units_wasted)} units wasted",
            "finding_type": "waste_analysis",
            "metrics": {
                "total_waste_cost_sar": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-06-29T00:00:00+03:00",
                    "period_end": "2026-07-06T00:00:00+03:00"
                },
                "total_units_wasted": {
                    "value": int(total_units_wasted),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-06-29T00:00:00+03:00",
                    "period_end": "2026-07-06T00:00:00+03:00"
                },
                "waste_items_count": {
                    "value": len(waste_items),
                    "unit": "items",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-06-29T00:00:00+03:00",
                    "period_end": "2026-07-06T00:00:00+03:00"
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(waste_items),
            "coverage_notes": [
                "Only includes items with non-null known_waste_cost_sar",
                "Week starting 2026-06-29",
                f"Items with waste data: {len(waste_items)} out of {len(inventory_analysis)} total items"
            ],
            "assumptions": [
                "known_waste_cost_sar values are accurate and complete for reported waste",
                "Blank waste values are treated as unknown, not zero"
            ],
            "confidence": 0.85
        }
        findings.append(finding2)

# FINDING 3: Supplier Price Changes from Email Evidence
# Filter emails for price changes with effective dates
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) & 
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    # Get the most recent price change
    price_changes['effective_date'] = pd.to_datetime(price_changes['effective_date'])
    price_changes_sorted = price_changes.sort_values('effective_date', ascending=False)
    
    recent_change = price_changes_sorted.iloc[0]
    
    # Calculate percentage change
    old_price = float(recent_change['old_price'])
    new_price = float(recent_change['new_price'])
    pct_change = ((new_price - old_price) / old_price * 100) if old_price != 0 else 0
    
    finding3 = {
        "title": "Recent Supplier Price Change Detected",
        "claim": f"Supplier price change for {recent_change['entity_or_ingredient']}: {old_price} → {new_price} {recent_change['currency']}/{recent_change['unit']} (effective {recent_change['effective_date']}), representing a {pct_change:.1f}% change",
        "finding_type": "supplier_pricing",
        "metrics": {
            "ingredient": {
                "value": recent_change['entity_or_ingredient'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": str(recent_change['effective_date']),
                "period_end": str(recent_change['effective_date'])
            },
            "old_price": {
                "value": old_price,
                "unit": f"{recent_change['currency']}/{recent_change['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": str(recent_change['effective_date']),
                "period_end": str(recent_change['effective_date'])
            },
            "new_price": {
                "value": new_price,
                "unit": f"{recent_change['currency']}/{recent_change['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": str(recent_change['effective_date']),
                "period_end": str(recent_change['effective_date'])
            },
            "percentage_change": {
                "value": round(pct_change, 2),
                "unit": "%",
                "numerator": round(new_price - old_price, 2),
                "denominator": old_price,
                "period_start": str(recent_change['effective_date']),
                "period_end": str(recent_change['effective_date'])
            }
        },
        "source_names": ["emails"],
        "sample_size": 1,
        "coverage_notes": [
            f"Most recent price change from {len(price_changes)} total price changes in email data",
            "Effective date: " + str(recent_change['effective_date']),
            "Sender: " + str(recent_change['sender'])
        ],
        "assumptions": [
            "Email extraction accurately captured old_price, new_price, and effective_date",
            "Price change applies to the specified ingredient/entity only",
            "No recipe/BOM data available; impact on menu items cannot be calculated without additional information"
        ],
        "confidence": 0.90
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
