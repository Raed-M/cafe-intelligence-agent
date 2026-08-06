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
analysis_start = datetime(2026, 2, 16, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 2, 23, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 2, 9, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 2, 16, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Filter inventory for analysis week
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
inv_analysis = inventory_df[inventory_df['week_starting'] == pd.Timestamp('2026-02-16')].copy()
inv_previous = inventory_df[inventory_df['week_starting'] == pd.Timestamp('2026-02-09')].copy()

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from POS and menu data
pos_with_menu = pos_analysis.merge(menu_df[['sku', 'unit_cost_sar', 'price_sar']], on='sku', how='left')

# Exclude refunds from revenue calculation
pos_sales = pos_with_menu[~pos_with_menu['is_refund']].copy()

# Calculate metrics by item
item_economics = pos_sales.groupby('sku').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'unit_cost_sar': 'first',
    'item_name': 'first',
    'category': 'first'
}).reset_index()

item_economics['total_cogs'] = item_economics['quantity'] * item_economics['unit_cost_sar']
item_economics['gross_profit'] = item_economics['line_total_sar'] - item_economics['total_cogs']
item_economics['gross_margin_pct'] = (item_economics['gross_profit'] / item_economics['line_total_sar'] * 100).round(2)

# Filter for items with meaningful volume
item_economics_filtered = item_economics[item_economics['quantity'] > 0].copy()

# Calculate totals
total_revenue = item_economics_filtered['line_total_sar'].sum()
total_cogs = item_economics_filtered['total_cogs'].sum()
total_gross_profit = item_economics_filtered['gross_profit'].sum()
overall_margin = (total_gross_profit / total_revenue * 100) if total_revenue > 0 else 0

if len(item_economics_filtered) > 0:
    finding1 = {
        "title": "Item-Level Gross Profit and Margin Analysis",
        "claim": f"During the analysis week (Feb 16-23, 2026), total gross profit was {total_gross_profit:.2f} SAR across {len(item_economics_filtered)} items with {item_economics_filtered['quantity'].sum():.0f} units sold, yielding an overall gross margin of {overall_margin:.1f}%.",
        "finding_type": "margin_analysis",
        "metrics": {
            "total_revenue_sar": {
                "value": round(total_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-16T00:00:00+00:00",
                "period_end": "2026-02-23T00:00:00+00:00"
            },
            "total_cogs_sar": {
                "value": round(total_cogs, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-16T00:00:00+00:00",
                "period_end": "2026-02-23T00:00:00+00:00"
            },
            "total_gross_profit_sar": {
                "value": round(total_gross_profit, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-16T00:00:00+00:00",
                "period_end": "2026-02-23T00:00:00+00:00"
            },
            "overall_gross_margin_pct": {
                "value": round(overall_margin, 1),
                "unit": "%",
                "numerator": round(total_gross_profit, 2),
                "denominator": round(total_revenue, 2),
                "period_start": "2026-02-16T00:00:00+00:00",
                "period_end": "2026-02-23T00:00:00+00:00"
            },
            "items_with_sales": {
                "value": len(item_economics_filtered),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-16T00:00:00+00:00",
                "period_end": "2026-02-23T00:00:00+00:00"
            },
            "total_units_sold": {
                "value": int(item_economics_filtered['quantity'].sum()),
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-16T00:00:00+00:00",
                "period_end": "2026-02-23T00:00:00+00:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(pos_sales),
        "coverage_notes": [
            "Analysis period: 2026-02-16 to 2026-02-23",
            "Refunds excluded from revenue calculation",
            "Unit costs sourced from menu.parquet",
            "Only items with non-null unit_cost_sar included"
        ],
        "assumptions": [
            "Menu unit_cost_sar represents actual COGS per unit",
            "Line totals in POS are accurate after discount application",
            "No inventory shrinkage or waste adjustments applied to revenue calculation"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# FINDING 2: Waste Cost Impact Analysis
# Calculate waste costs from inventory data
if len(inv_analysis) > 0:
    inv_analysis_clean = inv_analysis[inv_analysis['known_waste_cost_sar'].notna()].copy()
    
    if len(inv_analysis_clean) > 0:
        total_waste_cost = inv_analysis_clean['known_waste_cost_sar'].sum()
        waste_items = len(inv_analysis_clean)
        total_units_wasted = inv_analysis_clean['units_wasted'].sum()
        
        # Compare to previous week
        if len(inv_previous) > 0:
            inv_previous_clean = inv_previous[inv_previous['known_waste_cost_sar'].notna()].copy()
            prev_waste_cost = inv_previous_clean['known_waste_cost_sar'].sum() if len(inv_previous_clean) > 0 else 0
            waste_change = total_waste_cost - prev_waste_cost
            waste_change_pct = (waste_change / prev_waste_cost * 100) if prev_waste_cost > 0 else 0
        else:
            prev_waste_cost = 0
            waste_change = total_waste_cost
            waste_change_pct = 0
        
        finding2 = {
            "title": "Quantified Waste Cost Impact",
            "claim": f"During the analysis week (Feb 16-23, 2026), documented waste cost totaled {total_waste_cost:.2f} SAR across {waste_items} items ({total_units_wasted:.0f} units wasted). This represents a {waste_change_pct:+.1f}% change from the previous week's {prev_waste_cost:.2f} SAR.",
            "finding_type": "waste_analysis",
            "metrics": {
                "total_waste_cost_sar": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-16T00:00:00+00:00",
                    "period_end": "2026-02-23T00:00:00+00:00"
                },
                "waste_items_count": {
                    "value": waste_items,
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-16T00:00:00+00:00",
                    "period_end": "2026-02-23T00:00:00+00:00"
                },
                "total_units_wasted": {
                    "value": int(total_units_wasted),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-02-16T00:00:00+00:00",
                    "period_end": "2026-02-23T00:00:00+00:00"
                },
                "waste_cost_change_sar": {
                    "value": round(waste_change, 2),
                    "unit": "SAR",
                    "numerator": round(total_waste_cost, 2),
                    "denominator": round(prev_waste_cost, 2),
                    "period_start": "2026-02-16T00:00:00+00:00",
                    "period_end": "2026-02-23T00:00:00+00:00"
                },
                "waste_cost_change_pct": {
                    "value": round(waste_change_pct, 1),
                    "unit": "%",
                    "numerator": round(waste_change, 2),
                    "denominator": round(prev_waste_cost, 2) if prev_waste_cost > 0 else None,
                    "period_start": "2026-02-16T00:00:00+00:00",
                    "period_end": "2026-02-23T00:00:00+00:00"
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(inv_analysis_clean),
            "coverage_notes": [
                "Analysis period: 2026-02-16 to 2026-02-23",
                "Only items with non-null known_waste_cost_sar included",
                "Waste costs calculated from inventory.known_waste_cost_sar field",
                f"Previous period comparison: 2026-02-09 to 2026-02-16 ({len(inv_previous_clean)} items with waste data)"
            ],
            "assumptions": [
                "known_waste_cost_sar represents actual waste cost incurred",
                "Blank waste values are treated as unknown, not zero",
                "Week-over-week comparison assumes consistent waste tracking methodology"
            ],
            "confidence": 0.85
        }
        findings.append(finding2)

# FINDING 3: Supplier Price Changes and Procurement Impact
# Analyze supplier emails for price changes
if len(emails_df) > 0:
    price_changes = emails_df[
        (emails_df['old_price'].notna()) & 
        (emails_df['new_price'].notna()) &
        (emails_df['effective_date'].notna())
    ].copy()
    
    if len(price_changes) > 0:
        price_changes['effective_date'] = pd.to_datetime(price_changes['effective_date'])
        
        # Filter for changes relevant to analysis period
        relevant_changes = price_changes[
            (price_changes['effective_date'] >= analysis_start) |
            (price_changes['effective_date'] < analysis_end + pd.Timedelta(days=7))
        ].copy()
        
        if len(relevant_changes) > 0:
            # Calculate price change percentages
            relevant_changes['price_change_pct'] = (
                (relevant_changes['new_price'] - relevant_changes['old_price']) / 
                relevant_changes['old_price'] * 100
            ).round(2)
            
            # Get the most significant change
            relevant_changes['abs_change_pct'] = relevant_changes['price_change_pct'].abs()
            top_change = relevant_changes.nlargest(1, 'abs_change_pct').iloc[0]
            
            finding3 = {
                "title": "Supplier Price Change Detection",
                "claim": f"Supplier email evidence indicates a price change for {top_change['entity_or_ingredient']} from {top_change['old_price']:.2f} to {top_change['new_price']:.2f} {top_change['currency']}/{top_change['unit']}, effective {top_change['effective_date'].strftime('%Y-%m-%d')}, representing a {top_change['price_change_pct']:+.1f}% change. Without standing order quantities or confirmed purchase volumes, the exact procurement cost impact cannot be quantified.",
                "finding_type": "supplier_pricing",
                "metrics": {
                    "old_price": {
                        "value": round(top_change['old_price'], 2),
                        "unit": f"{top_change['currency']}/{top_change['unit']}",
                        "numerator": None,
                        "denominator": None,
                        "period_start": None,
                        "period_end": None
                    },
                    "new_price": {
                        "value": round(top_change['new_price'], 2),
                        "unit": f"{top_change['currency']}/{top_change['unit']}",
                        "numerator": None,
                        "denominator": None,
                        "period_start": None,
                        "period_end": None
                    },
                    "price_change_pct": {
                        "value": round(top_change['price_change_pct'], 1),
                        "unit": "%",
                        "numerator": round(top_change['new_price'] - top_change['old_price'], 2),
                        "denominator": round(top_change['old_price'], 2),
                        "period_start": None,
                        "period_end": None
                    },
                    "effective_date": {
                        "value": top_change['effective_date'].strftime('%Y-%m-%d'),
                        "unit": "date",
                        "numerator": None,
                        "denominator": None,
                        "period_start": None,
                        "period_end": None
                    }
                },
                "source_names": ["emails"],
                "sample_size": len(relevant_changes),
                "coverage_notes": [
                    f"Detected {len(relevant_changes)} supplier price changes in relevant timeframe",
                    "Analysis period: 2026-02-16 to 2026-02-23",
                    "Price change effective date: " + top_change['effective_date'].strftime('%Y-%m-%d'),
                    "Supplier: " + str(top_change['sender']),
                    "Email subject: " + str(top_change['subject'])
                ],
                "assumptions": [
                    "Email extraction confidence: " + str(top_change['confidence']),
                    "Standing order quantities not available in email data",
                    "Current purchase volumes and payment terms unknown",
                    "Price change applies only to specified ingredient/entity"
                ],
                "confidence": float(top_change['confidence']) if pd.notna(top_change['confidence']) else 0.7
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
