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
analysis_start = datetime(2026, 3, 23, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 3, 30, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 3, 16, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 3, 23, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Filter inventory for analysis week
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
inv_analysis = inventory_df[inventory_df['week_starting'] == pd.Timestamp('2026-03-23')]

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from menu and POS
menu_df_copy = menu_df.copy()
menu_df_copy['sku'] = menu_df_copy['sku'].astype(str)
pos_analysis['sku'] = pos_analysis['sku'].astype(str)

# Merge POS with menu to get unit costs
pos_with_cost = pos_analysis.merge(menu_df_copy[['sku', 'unit_cost_sar', 'price_sar']], on='sku', how='left')

# Filter out refunds for revenue calculation
pos_sales = pos_with_cost[pos_with_cost['is_refund'] == False].copy()

# Calculate metrics
total_revenue = pos_sales['line_total_sar'].sum()
total_quantity = pos_sales['quantity'].sum()

# Calculate COGS (unit_cost × quantity)
pos_sales['cogs'] = pos_sales['unit_cost_sar'] * pos_sales['quantity']
total_cogs = pos_sales['cogs'].sum()

# Calculate gross profit
gross_profit = total_revenue - total_cogs
gross_margin_pct = (gross_profit / total_revenue * 100) if total_revenue > 0 else 0

# Count unique transactions
unique_transactions = pos_sales['transaction_id'].nunique()

if total_revenue > 0:
    findings.append({
        "title": "Item-Level Gross Profit and Margin Analysis",
        "claim": f"During the analysis week (Mar 23-30, 2026), the cafe generated SAR {total_revenue:.2f} in revenue across {unique_transactions} transactions with a gross profit of SAR {gross_profit:.2f} and gross margin of {gross_margin_pct:.1f}%.",
        "finding_type": "margin_analysis",
        "metrics": {
            "total_revenue_sar": {
                "value": round(total_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+00:00",
                "period_end": "2026-03-30T00:00:00+00:00"
            },
            "total_cogs_sar": {
                "value": round(total_cogs, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+00:00",
                "period_end": "2026-03-30T00:00:00+00:00"
            },
            "gross_profit_sar": {
                "value": round(gross_profit, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+00:00",
                "period_end": "2026-03-30T00:00:00+00:00"
            },
            "gross_margin_pct": {
                "value": round(gross_margin_pct, 1),
                "unit": "%",
                "numerator": round(gross_profit, 2),
                "denominator": round(total_revenue, 2),
                "period_start": "2026-03-23T00:00:00+00:00",
                "period_end": "2026-03-30T00:00:00+00:00"
            },
            "unique_transactions": {
                "value": unique_transactions,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+00:00",
                "period_end": "2026-03-30T00:00:00+00:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(pos_sales),
        "coverage_notes": [
            "Analysis includes all non-refund POS transactions during 2026-03-23 to 2026-03-30",
            "COGS calculated using menu_items.unit_cost_sar × realized POS quantities",
            "Refunds excluded from revenue and profit calculations per metric definitions"
        ],
        "assumptions": [
            "Menu unit_cost_sar values are current and applicable to analysis period",
            "POS line_total_sar reflects actual revenue after discounts",
            "No inventory adjustments or waste costs included in this item-level analysis"
        ],
        "confidence": 0.95
    })

# FINDING 2: Waste Cost Impact
# Calculate known waste costs from inventory
inv_analysis['sku'] = inv_analysis['sku'].astype(str)
inv_with_menu = inv_analysis.merge(menu_df_copy[['sku', 'unit_cost_sar']], on='sku', how='left')

# Only include rows with non-null waste values
waste_data = inv_with_menu[inv_with_menu['units_wasted'].notna() & (inv_with_menu['units_wasted'] > 0)].copy()

if len(waste_data) > 0:
    # Calculate waste cost
    waste_data['waste_cost'] = waste_data['units_wasted'] * waste_data['unit_cost_sar']
    total_waste_cost = waste_data['waste_cost'].sum()
    total_units_wasted = waste_data['units_wasted'].sum()
    
    # Calculate as percentage of revenue
    waste_cost_pct = (total_waste_cost / total_revenue * 100) if total_revenue > 0 else 0
    
    findings.append({
        "title": "Quantified Waste Cost Impact",
        "claim": f"During the week of Mar 23-30, 2026, documented waste totaled {total_units_wasted:.0f} units with a cost impact of SAR {total_waste_cost:.2f}, representing {waste_cost_pct:.2f}% of weekly revenue.",
        "finding_type": "waste_analysis",
        "metrics": {
            "total_units_wasted": {
                "value": round(total_units_wasted, 0),
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+00:00",
                "period_end": "2026-03-30T00:00:00+00:00"
            },
            "total_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+00:00",
                "period_end": "2026-03-30T00:00:00+00:00"
            },
            "waste_cost_pct_of_revenue": {
                "value": round(waste_cost_pct, 2),
                "unit": "%",
                "numerator": round(total_waste_cost, 2),
                "denominator": round(total_revenue, 2),
                "period_start": "2026-03-23T00:00:00+00:00",
                "period_end": "2026-03-30T00:00:00+00:00"
            }
        },
        "source_names": ["inventory", "menu"],
        "sample_size": len(waste_data),
        "coverage_notes": [
            "Only non-null waste observations included in calculation",
            "Waste cost calculated using inventory unit_cost_sar × units_wasted",
            "Data covers inventory records for week starting 2026-03-23"
        ],
        "assumptions": [
            "Unit costs in inventory match menu unit_cost_sar",
            "Waste units represent actual spoilage/disposal",
            "No recovery value assumed for wasted items"
        ],
        "confidence": 0.90
    })

# FINDING 3: Supplier Price Changes and Procurement Impact
# Extract supplier price changes from emails
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    price_changes['effective_date'] = pd.to_datetime(price_changes['effective_date'])
    
    # Filter for changes effective during or before analysis period
    relevant_changes = price_changes[price_changes['effective_date'] <= analysis_end].copy()
    
    if len(relevant_changes) > 0:
        # Calculate price deltas
        relevant_changes['price_delta'] = relevant_changes['new_price'] - relevant_changes['old_price']
        relevant_changes['price_delta_pct'] = (relevant_changes['price_delta'] / relevant_changes['old_price'] * 100)
        
        # Find the most significant change
        relevant_changes['abs_delta_pct'] = relevant_changes['price_delta_pct'].abs()
        top_change = relevant_changes.loc[relevant_changes['abs_delta_pct'].idxmax()]
        
        findings.append({
            "title": "Supplier Price Change Detection",
            "claim": f"Email evidence shows {top_change['entity_or_ingredient']} price change from {top_change['old_price']} to {top_change['new_price']} {top_change['currency']}/{top_change['unit']}, effective {top_change['effective_date'].strftime('%Y-%m-%d')}, representing a {top_change['price_delta_pct']:.1f}% change.",
            "finding_type": "supplier_pricing",
            "metrics": {
                "ingredient": {
                    "value": str(top_change['entity_or_ingredient']),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-23T00:00:00+00:00",
                    "period_end": "2026-03-30T00:00:00+00:00"
                },
                "old_price": {
                    "value": round(float(top_change['old_price']), 2),
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-23T00:00:00+00:00",
                    "period_end": "2026-03-30T00:00:00+00:00"
                },
                "new_price": {
                    "value": round(float(top_change['new_price']), 2),
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-23T00:00:00+00:00",
                    "period_end": "2026-03-30T00:00:00+00:00"
                },
                "price_delta_pct": {
                    "value": round(float(top_change['price_delta_pct']), 1),
                    "unit": "%",
                    "numerator": round(float(top_change['price_delta']), 2),
                    "denominator": round(float(top_change['old_price']), 2),
                    "period_start": "2026-03-23T00:00:00+00:00",
                    "period_end": "2026-03-30T00:00:00+00:00"
                },
                "effective_date": {
                    "value": top_change['effective_date'].strftime('%Y-%m-%d'),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-23T00:00:00+00:00",
                    "period_end": "2026-03-30T00:00:00+00:00"
                }
            },
            "source_names": ["emails"],
            "sample_size": len(relevant_changes),
            "coverage_notes": [
                f"Identified {len(relevant_changes)} supplier price changes effective on or before analysis period",
                "Price changes extracted from supplier emails with confidence scores",
                "No recipe/BOM data available to calculate per-drink impact"
            ],
            "assumptions": [
                "Email extraction accurately captured supplier price communications",
                "Effective dates represent actual implementation dates",
                "Price changes apply to future procurement; impact on current period depends on inventory timing"
            ],
            "confidence": 0.85
        })

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. {len(findings)} findings generated.")
