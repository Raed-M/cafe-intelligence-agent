import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Define analysis periods
analysis_start = "2026-04-06T00:00:00+03:00"
analysis_end = "2026-04-13T00:00:00+03:00"
previous_start = "2026-03-30T00:00:00+03:00"
previous_end = "2026-04-06T00:00:00+03:00"

# Parse timestamps
pos_df['timestamp_local'] = pd.to_datetime(pos_df['timestamp_local'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
emails_df['date'] = pd.to_datetime(emails_df['date'])
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'])

analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)
previous_start_dt = pd.to_datetime(previous_start)
previous_end_dt = pd.to_datetime(previous_end)

# Filter POS for analysis period
pos_analysis = pos_df[
    (pos_df['timestamp_local'] >= analysis_start_dt) & 
    (pos_df['timestamp_local'] < analysis_end_dt)
].copy()

pos_previous = pos_df[
    (pos_df['timestamp_local'] >= previous_start_dt) & 
    (pos_df['timestamp_local'] < previous_end_dt)
].copy()

# Filter inventory for analysis week
inventory_analysis = inventory_df[
    inventory_df['week_starting'] == pd.to_datetime('2026-04-06')
].copy()

inventory_previous = inventory_df[
    inventory_df['week_starting'] == pd.to_datetime('2026-03-30')
].copy()

findings = []

# ============================================================================
# FINDING 1: Item-level COGS and Gross Profit Analysis (Analysis Period)
# ============================================================================

# Merge POS with menu to get unit costs
pos_analysis_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'price_sar']],
    on='sku',
    how='left'
)

# Calculate item-level metrics (excluding refunds)
pos_analysis_no_refund = pos_analysis_with_cost[pos_analysis_with_cost['is_refund'] == False].copy()

# Group by SKU to calculate totals
item_metrics = pos_analysis_no_refund.groupby('sku').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'unit_cost_sar': 'first',
    'item_name_en': 'first',
    'category': 'first'
}).reset_index()

item_metrics.columns = ['sku', 'total_quantity', 'total_revenue', 'unit_cost_sar', 'item_name', 'category']

# Calculate COGS and gross profit
item_metrics['total_cogs'] = item_metrics['total_quantity'] * item_metrics['unit_cost_sar']
item_metrics['gross_profit'] = item_metrics['total_revenue'] - item_metrics['total_cogs']
item_metrics['gross_margin_pct'] = (item_metrics['gross_profit'] / item_metrics['total_revenue'] * 100).round(2)

# Filter for items with valid cost data
item_metrics_valid = item_metrics[item_metrics['unit_cost_sar'].notna()].copy()

if len(item_metrics_valid) > 0:
    # Top 3 by gross profit
    top_profit_items = item_metrics_valid.nlargest(3, 'gross_profit')
    
    total_revenue_analysis = pos_analysis_no_refund['line_total_sar'].sum()
    total_cogs_analysis = item_metrics_valid['total_cogs'].sum()
    total_gross_profit_analysis = total_revenue_analysis - total_cogs_analysis
    overall_margin_pct = (total_gross_profit_analysis / total_revenue_analysis * 100) if total_revenue_analysis > 0 else 0
    
    finding_1 = {
        "title": "Item-Level Gross Profit and Margin Analysis (Analysis Period)",
        "claim": f"During the analysis period (2026-04-06 to 2026-04-13), total gross profit across {len(item_metrics_valid)} items with valid cost data was {total_gross_profit_analysis:.2f} SAR with an overall gross margin of {overall_margin_pct:.2f}%. Top contributor: {top_profit_items.iloc[0]['item_name']} with {top_profit_items.iloc[0]['gross_profit']:.2f} SAR gross profit.",
        "finding_type": "margin_analysis",
        "metrics": {
            "total_revenue_sar": {
                "value": round(total_revenue_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_cogs_sar": {
                "value": round(total_cogs_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_gross_profit_sar": {
                "value": round(total_gross_profit_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "overall_gross_margin_pct": {
                "value": round(overall_margin_pct, 2),
                "unit": "%",
                "numerator": round(total_gross_profit_analysis, 2),
                "denominator": round(total_revenue_analysis, 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "items_with_valid_cost_count": {
                "value": len(item_metrics_valid),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "top_profit_item_name": {
                "value": top_profit_items.iloc[0]['item_name'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "top_profit_item_gross_profit_sar": {
                "value": round(top_profit_items.iloc[0]['gross_profit'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(pos_analysis_no_refund),
        "coverage_notes": [
            "Analysis period: 2026-04-06 to 2026-04-13",
            f"POS transactions analyzed: {len(pos_analysis_no_refund)} line items",
            f"Items with valid unit cost data: {len(item_metrics_valid)} of {len(item_metrics)} menu items",
            "Refunds excluded from revenue and COGS calculations",
            "Unit costs sourced from menu.parquet"
        ],
        "assumptions": [
            "Unit cost from menu applies uniformly to all sales in period",
            "No recipe/BOM data available; per-drink ingredient impact not calculated",
            "Blank waste values treated as unknown, not zero"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Waste Cost Impact (Inventory Data)
# ============================================================================

# Calculate waste cost for analysis period inventory
if len(inventory_analysis) > 0:
    waste_with_cost = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()
    
    if len(waste_with_cost) > 0:
        total_waste_cost = waste_with_cost['known_waste_cost_sar'].sum()
        total_units_wasted = waste_with_cost['units_wasted'].sum()
        
        finding_2 = {
            "title": "Quantified Waste Cost (Analysis Period)",
            "claim": f"During the week of 2026-04-06, {len(waste_with_cost)} items with recorded waste had a total known waste cost of {total_waste_cost:.2f} SAR, representing {total_units_wasted:.0f} units wasted.",
            "finding_type": "waste_analysis",
            "metrics": {
                "total_known_waste_cost_sar": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "total_units_wasted": {
                    "value": round(total_units_wasted, 2),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "items_with_waste_cost_recorded": {
                    "value": len(waste_with_cost),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(inventory_analysis),
            "coverage_notes": [
                "Analysis period: week starting 2026-04-06",
                f"Inventory records with waste data: {len(inventory_analysis)}",
                f"Records with non-null known_waste_cost_sar: {len(waste_with_cost)}",
                "Only non-null waste cost values included in total"
            ],
            "assumptions": [
                "Blank waste cost values treated as unknown, not zero",
                "known_waste_cost_sar represents actual cost of wasted units"
            ],
            "confidence": 0.90
        }
        findings.append(finding_2)

# ============================================================================
# FINDING 3: Supplier Price Changes from Email Evidence
# ============================================================================

# Filter emails for price changes within or near analysis period
emails_relevant = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(emails_relevant) > 0:
    # Calculate price change percentage
    emails_relevant['price_change_pct'] = (
        ((emails_relevant['new_price'] - emails_relevant['old_price']) / emails_relevant['old_price'] * 100)
    ).round(2)
    
    # Sort by effective date
    emails_relevant = emails_relevant.sort_values('effective_date')
    
    # Take first price change as finding
    if len(emails_relevant) > 0:
        first_change = emails_relevant.iloc[0]
        
        finding_3 = {
            "title": "Supplier Price Change Detected",
            "claim": f"Email evidence from {first_change['sender']} dated {first_change['date'].strftime('%Y-%m-%d')} documents a price change for {first_change['entity_or_ingredient']}: {first_change['old_price']} {first_change['currency']}/{first_change['unit']} → {first_change['new_price']} {first_change['currency']}/{first_change['unit']} (effective {first_change['effective_date'].strftime('%Y-%m-%d')}), representing a {first_change['price_change_pct']:.2f}% change.",
            "finding_type": "supplier_price_change",
            "metrics": {
                "entity_or_ingredient": {
                    "value": first_change['entity_or_ingredient'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": first_change['effective_date'].isoformat(),
                    "period_end": first_change['effective_date'].isoformat()
                },
                "old_price": {
                    "value": round(first_change['old_price'], 4),
                    "unit": f"{first_change['currency']}/{first_change['unit']}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": first_change['effective_date'].isoformat(),
                    "period_end": first_change['effective_date'].isoformat()
                },
                "new_price": {
                    "value": round(first_change['new_price'], 4),
                    "unit": f"{first_change['currency']}/{first_change['unit']}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": first_change['effective_date'].isoformat(),
                    "period_end": first_change['effective_date'].isoformat()
                },
                "price_change_pct": {
                    "value": first_change['price_change_pct'],
                    "unit": "%",
                    "numerator": round(first_change['new_price'] - first_change['old_price'], 4),
                    "denominator": round(first_change['old_price'], 4),
                    "period_start": first_change['effective_date'].isoformat(),
                    "period_end": first_change['effective_date'].isoformat()
                },
                "effective_date": {
                    "value": first_change['effective_date'].strftime('%Y-%m-%d'),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": first_change['effective_date'].isoformat(),
                    "period_end": first_change['effective_date'].isoformat()
                },
                "sender": {
                    "value": first_change['sender'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": first_change['effective_date'].isoformat(),
                    "period_end": first_change['effective_date'].isoformat()
                }
            },
            "source_names": ["emails"],
            "sample_size": len(emails_relevant),
            "coverage_notes": [
                f"Total emails with price change data: {len(emails_relevant)}",
                f"Email date: {first_change['date'].strftime('%Y-%m-%d')}",
                f"Effective date: {first_change['effective_date'].strftime('%Y-%m-%d')}",
                "No recipe/BOM available; per-product cost impact not calculated",
                "Standing order quantities not confirmed in email evidence"
            ],
            "assumptions": [
                "Email extraction confidence: {:.2f}".format(first_change['confidence']),
                "Price change applies to specified ingredient only",
                "No assumption made about standing order volumes or payment terms without explicit evidence"
            ],
            "confidence": first_change['confidence']
        }
        findings.append(finding_3)

# ============================================================================
# Write output
# ============================================================================

result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")