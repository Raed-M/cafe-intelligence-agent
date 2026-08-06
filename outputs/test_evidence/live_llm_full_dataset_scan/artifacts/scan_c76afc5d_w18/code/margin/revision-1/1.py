import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Parse analysis periods
analysis_start = "2026-05-11T00:00:00+03:00"
analysis_end = "2026-05-18T00:00:00+03:00"
previous_start = "2026-05-04T00:00:00+03:00"
previous_end = "2026-05-11T00:00:00+03:00"

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Convert timestamp columns to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'], errors='coerce')
menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'], errors='coerce')
emails_df['date'] = pd.to_datetime(emails_df['date'], errors='coerce')
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'], errors='coerce')

# Parse analysis period boundaries
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)
previous_start_dt = pd.to_datetime(previous_start)
previous_end_dt = pd.to_datetime(previous_end)

findings = []

# ============================================================================
# FINDING 1: Item-level gross profit analysis for analysis period
# ============================================================================

# Filter POS for analysis period, exclude refunds
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start_dt) &
    (pos_df['timestamp'] < analysis_end_dt) &
    (pos_df['is_refund'] == False)
].copy()

# Merge with menu to get unit costs
pos_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'item_en']],
    on='sku',
    how='left'
)

# Calculate item-level metrics
item_metrics = pos_with_cost.groupby('sku').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'unit_cost_sar': 'first',
    'item_en': 'first'
}).reset_index()

item_metrics.columns = ['sku', 'units_sold', 'revenue_sar', 'unit_cost_sar', 'item_name']

# Calculate COGS and gross profit
item_metrics['cogs_sar'] = item_metrics['units_sold'] * item_metrics['unit_cost_sar']
item_metrics['gross_profit_sar'] = item_metrics['revenue_sar'] - item_metrics['cogs_sar']
item_metrics['gross_margin_pct'] = (item_metrics['gross_profit_sar'] / item_metrics['revenue_sar'] * 100).round(1)

# Filter for items with valid unit costs and positive revenue
item_metrics_valid = item_metrics[
    (item_metrics['unit_cost_sar'].notna()) &
    (item_metrics['revenue_sar'] > 0)
].copy()

# Sort by gross profit
item_metrics_valid = item_metrics_valid.sort_values('gross_profit_sar', ascending=False)

if len(item_metrics_valid) > 0:
    top_item = item_metrics_valid.iloc[0]
    
    finding_1 = {
        "title": "Highest Gross Profit Item in Analysis Period",
        "claim": f"{top_item['item_name']} generated {top_item['gross_profit_sar']:.1f} SAR gross profit ({top_item['gross_margin_pct']:.1f}% margin) from {int(top_item['units_sold'])} units sold at {top_item['revenue_sar']:.1f} SAR revenue during 2026-05-11 to 2026-05-18.",
        "finding_type": "item_economics",
        "metrics": {
            "units_sold": {
                "value": int(top_item['units_sold']),
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "revenue_sar": {
                "value": round(top_item['revenue_sar'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "unit_cost_sar": {
                "value": round(top_item['unit_cost_sar'], 2),
                "unit": "SAR/unit",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "cogs_sar": {
                "value": round(top_item['cogs_sar'], 2),
                "unit": "SAR",
                "numerator": int(top_item['units_sold']),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "gross_profit_sar": {
                "value": round(top_item['gross_profit_sar'], 2),
                "unit": "SAR",
                "numerator": round(top_item['revenue_sar'], 2),
                "denominator": round(top_item['cogs_sar'], 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "gross_margin_pct": {
                "value": round(top_item['gross_margin_pct'], 1),
                "unit": "%",
                "numerator": round(top_item['gross_profit_sar'], 2),
                "denominator": round(top_item['revenue_sar'], 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(pos_analysis),
        "coverage_notes": [
            f"Analysis period: 2026-05-11 to 2026-05-18 (7 days)",
            f"POS transactions analyzed: {len(pos_analysis)} line items",
            f"Refunds excluded from analysis",
            f"Unit cost sourced from menu.parquet; no recipe/BOM available",
            f"Item {top_item['item_name']} (SKU: {top_item['sku']}) ranked #1 by gross profit among {len(item_metrics_valid)} items with valid costs"
        ],
        "assumptions": [
            "Menu unit_cost_sar is current and accurate for the analysis period",
            "No recipe/BOM available; unit cost treated as declared in menu",
            "Line totals in POS are accurate and reflect actual revenue",
            "Discounts are already deducted in line_total_sar"
        ],
        "confidence": 0.85
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Supplier price change impact analysis
# ============================================================================

# Filter emails for price changes with valid old/new prices and effective dates
price_changes = emails_df[
    (emails_df['old_price'].notna()) &
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna()) &
    (emails_df['category'] == 'price_change')
].copy()

if len(price_changes) > 0:
    # Calculate price deltas
    price_changes['price_delta'] = price_changes['new_price'] - price_changes['old_price']
    price_changes['pct_change'] = (price_changes['price_delta'] / price_changes['old_price'] * 100).round(2)
    
    # Filter for changes that occurred before or during analysis period
    price_changes_relevant = price_changes[
        price_changes['effective_date'] <= analysis_end_dt
    ].copy()
    
    if len(price_changes_relevant) > 0:
        # Sort by absolute price delta
        price_changes_relevant = price_changes_relevant.sort_values('price_delta', ascending=False, key=abs)
        
        top_change = price_changes_relevant.iloc[0]
        
        finding_2 = {
            "title": "Significant Supplier Price Change Detected",
            "claim": f"Supplier price for {top_change['entity_or_ingredient']} changed from {top_change['old_price']:.2f} {top_change['currency']}/{top_change['unit']} to {top_change['new_price']:.2f} {top_change['currency']}/{top_change['unit']} (change: {top_change['pct_change']:.2f}%), effective {top_change['effective_date'].strftime('%Y-%m-%d')}. This represents a {abs(top_change['price_delta']):.2f} {top_change['currency']} per-unit cost shift.",
            "finding_type": "supplier_cost_change",
            "metrics": {
                "old_price": {
                    "value": round(top_change['old_price'], 2),
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "new_price": {
                    "value": round(top_change['new_price'], 2),
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "price_delta": {
                    "value": round(top_change['price_delta'], 2),
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "percent_change": {
                    "value": round(top_change['pct_change'], 2),
                    "unit": "%",
                    "numerator": round(top_change['price_delta'], 2),
                    "denominator": round(top_change['old_price'], 2),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "effective_date": {
                    "value": top_change['effective_date'].strftime('%Y-%m-%d'),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": ["emails"],
            "sample_size": len(price_changes_relevant),
            "coverage_notes": [
                f"Email extraction identified {len(price_changes)} price change events",
                f"Filtered to {len(price_changes_relevant)} changes with effective dates on or before analysis period end",
                f"Ingredient/entity: {top_change['entity_or_ingredient']}",
                f"Confidence in extraction: {top_change['confidence']}"
            ],
            "assumptions": [
                "Email extraction accurately captured supplier price change facts",
                "Effective date represents when new price applies",
                "No recipe/BOM available; cannot calculate per-drink cost impact without standing order volumes"
            ],
            "confidence": float(top_change['confidence']) if isinstance(top_change['confidence'], (int, float)) else 0.7
        }
        findings.append(finding_2)

# ============================================================================
# FINDING 3: Waste cost analysis for analysis period
# ============================================================================

# Filter inventory for analysis period
inventory_analysis = inventory_df[
    (inventory_df['week_starting'] >= analysis_start_dt) &
    (inventory_df['week_starting'] < analysis_end_dt)
].copy()

# Calculate total waste cost (only non-null values)
inventory_analysis['waste_cost_sar'] = inventory_analysis['known_waste_cost_sar'].fillna(0)
total_waste_cost = inventory_analysis['waste_cost_sar'].sum()
total_units_wasted = inventory_analysis['units_wasted'].fillna(0).sum()

if total_waste_cost > 0:
    waste_items = inventory_analysis[inventory_analysis['waste_cost_sar'] > 0].copy()
    waste_items = waste_items.sort_values('waste_cost_sar', ascending=False)
    
    if len(waste_items) > 0:
        top_waste = waste_items.iloc[0]
        
        finding_3 = {
            "title": "Highest Waste Cost Item in Analysis Period",
            "claim": f"{top_waste['item']} (SKU: {top_waste['sku']}) incurred {top_waste['waste_cost_sar']:.2f} SAR in waste cost from {int(top_waste['units_wasted'])} units wasted during 2026-05-11 to 2026-05-18. Total waste cost across all items: {total_waste_cost:.2f} SAR.",
            "finding_type": "waste_cost",
            "metrics": {
                "units_wasted": {
                    "value": int(top_waste['units_wasted']),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "waste_cost_sar": {
                    "value": round(top_waste['waste_cost_sar'], 2),
                    "unit": "SAR",
                    "numerator": int(top_waste['units_wasted']),
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "unit_cost_sar": {
                    "value": round(top_waste['unit_cost_sar'], 2),
                    "unit": "SAR/unit",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "total_waste_cost_all_items": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "total_units_wasted_all_items": {
                    "value": int(total_units_wasted),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(inventory_analysis),
            "coverage_notes": [
                f"Inventory records for week starting {top_waste['week_starting'].strftime('%Y-%m-%d')}",
                f"Only non-null waste cost values included in calculation",
                f"Total items with waste data: {len(inventory_analysis)}",
                f"Items with non-zero waste cost: {len(waste_items)}"
            ],
            "assumptions": [
                "known_waste_cost_sar in inventory reflects actual waste cost",
                "Blank waste values treated as zero (not missing)",
                "Unit cost from inventory matches actual cost at time of waste"
            ],
            "confidence": 0.9
        }
        findings.append(finding_3)

# ============================================================================
# Construct output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings generated.")
