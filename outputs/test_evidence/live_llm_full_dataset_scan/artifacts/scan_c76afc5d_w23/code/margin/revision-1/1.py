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
analysis_start = "2026-06-15"
analysis_end = "2026-06-22"
previous_start = "2026-06-08"
previous_end = "2026-06-15"

# Convert timestamp columns to datetime
pos_df['timestamp_local'] = pd.to_datetime(pos_df['timestamp_local'])
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
emails_df['date'] = pd.to_datetime(emails_df['date'])
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'])

findings = []

# ============================================================================
# FINDING 1: Item-level COGS and Gross Profit for Analysis Period
# ============================================================================

# Filter POS for analysis period (exclude refunds)
analysis_pos = pos_df[
    (pos_df['calendar_date'] >= analysis_start) &
    (pos_df['calendar_date'] < analysis_end) &
    (pos_df['is_refund'] == False)
].copy()

# Merge with menu to get unit costs
analysis_with_cost = analysis_pos.merge(
    menu_df[['sku', 'unit_cost_sar', 'item_en']],
    on='sku',
    how='left'
)

# Calculate item-level metrics
analysis_with_cost['cogs'] = analysis_with_cost['quantity'] * analysis_with_cost['unit_cost_sar']
analysis_with_cost['gross_profit'] = analysis_with_cost['line_total_sar'] - analysis_with_cost['cogs']
analysis_with_cost['gross_margin_pct'] = (
    analysis_with_cost['gross_profit'] / analysis_with_cost['line_total_sar'] * 100
).fillna(0)

# Aggregate by item
item_economics = analysis_with_cost.groupby('sku').agg({
    'item_en': 'first',
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'cogs': 'sum',
    'gross_profit': 'sum',
    'unit_cost_sar': 'first'
}).reset_index()

item_economics['gross_margin_pct'] = (
    item_economics['gross_profit'] / item_economics['line_total_sar'] * 100
).fillna(0)

# Find top 3 by revenue
top_items = item_economics.nlargest(3, 'line_total_sar')

if len(top_items) > 0:
    finding_1 = {
        "title": "Top 3 Items by Revenue: Item-Level COGS and Gross Profit (Analysis Period)",
        "claim": f"During {analysis_start} to {analysis_end}, the top 3 revenue-generating items show distinct margin profiles. Item economics calculated from menu unit costs and realized POS quantities.",
        "finding_type": "item_economics",
        "metrics": {},
        "source_names": ["pos", "menu"],
        "sample_size": len(analysis_pos),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Refunds excluded (is_refund=False)",
            f"Unit costs sourced from menu.unit_cost_sar",
            f"COGS = quantity × unit_cost_sar",
            f"Gross profit = line_total_sar - COGS",
            f"Top 3 items by revenue shown"
        ],
        "assumptions": [
            "Menu unit_cost_sar is current and applicable to all sales in period",
            "No recipe/BOM available; per-unit cost is as stated in menu",
            "Discount amounts are already reflected in line_total_sar"
        ],
        "confidence": 0.95
    }
    
    for idx, row in top_items.iterrows():
        prefix = f"item_{row['sku']}"
        finding_1['metrics'][f"{prefix}_name"] = {
            "value": row['item_en'],
            "unit": None,
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        }
        finding_1['metrics'][f"{prefix}_quantity_sold"] = {
            "value": float(row['quantity']),
            "unit": "units",
            "numerator": float(row['quantity']),
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        }
        finding_1['metrics'][f"{prefix}_revenue"] = {
            "value": float(row['line_total_sar']),
            "unit": "SAR",
            "numerator": float(row['line_total_sar']),
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        }
        finding_1['metrics'][f"{prefix}_cogs"] = {
            "value": float(row['cogs']),
            "unit": "SAR",
            "numerator": float(row['cogs']),
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        }
        finding_1['metrics'][f"{prefix}_gross_profit"] = {
            "value": float(row['gross_profit']),
            "unit": "SAR",
            "numerator": float(row['gross_profit']),
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        }
        finding_1['metrics'][f"{prefix}_gross_margin_pct"] = {
            "value": float(row['gross_margin_pct']),
            "unit": "%",
            "numerator": float(row['gross_margin_pct']),
            "denominator": 100.0,
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Supplier Price Changes with Effective Dates
# ============================================================================

# Filter emails for price changes with valid dates
price_changes = emails_df[
    (emails_df['old_price'].notna()) &
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    # Sort by effective date descending to get most recent
    price_changes = price_changes.sort_values('effective_date', ascending=False)
    
    # Calculate percentage change
    price_changes['pct_change'] = (
        (price_changes['new_price'] - price_changes['old_price']) / 
        price_changes['old_price'] * 100
    )
    
    # Get top 3 by absolute percentage change
    price_changes['abs_pct_change'] = price_changes['pct_change'].abs()
    top_changes = price_changes.nlargest(3, 'abs_pct_change')
    
    finding_2 = {
        "title": "Detected Supplier Price Changes: Magnitude and Effective Dates",
        "claim": f"Email extraction identified {len(price_changes)} supplier price changes across the analysis and trailing periods. Top 3 by magnitude shown with old/new prices, units, percentage change, and effective dates.",
        "finding_type": "supplier_pricing",
        "metrics": {},
        "source_names": ["emails"],
        "sample_size": len(price_changes),
        "coverage_notes": [
            f"Total price changes detected: {len(price_changes)}",
            f"Filtered for: old_price, new_price, and effective_date all non-null",
            f"Top 3 by absolute percentage change shown",
            f"Effective dates span analysis and trailing baseline periods"
        ],
        "assumptions": [
            "Email extraction confidence scores reflect reliability of entity/price identification",
            "Effective dates are as stated in supplier communications",
            "Currency is consistent (SAR assumed for all)"
        ],
        "confidence": 0.85
    }
    
    for idx, row in top_changes.iterrows():
        change_idx = list(top_changes.index).index(idx)
        prefix = f"price_change_{change_idx + 1}"
        
        finding_2['metrics'][f"{prefix}_ingredient"] = {
            "value": str(row['entity_or_ingredient']),
            "unit": None,
            "numerator": None,
            "denominator": None,
            "period_start": str(row['effective_date'].date()),
            "period_end": str(row['effective_date'].date())
        }
        finding_2['metrics'][f"{prefix}_old_price"] = {
            "value": float(row['old_price']),
            "unit": f"{row['currency']} per {row['unit']}",
            "numerator": float(row['old_price']),
            "denominator": None,
            "period_start": str(row['effective_date'].date()),
            "period_end": str(row['effective_date'].date())
        }
        finding_2['metrics'][f"{prefix}_new_price"] = {
            "value": float(row['new_price']),
            "unit": f"{row['currency']} per {row['unit']}",
            "numerator": float(row['new_price']),
            "denominator": None,
            "period_start": str(row['effective_date'].date()),
            "period_end": str(row['effective_date'].date())
        }
        finding_2['metrics'][f"{prefix}_pct_change"] = {
            "value": float(row['pct_change']),
            "unit": "%",
            "numerator": float(row['pct_change']),
            "denominator": 100.0,
            "period_start": str(row['effective_date'].date()),
            "period_end": str(row['effective_date'].date())
        }
        finding_2['metrics'][f"{prefix}_effective_date"] = {
            "value": str(row['effective_date'].date()),
            "unit": None,
            "numerator": None,
            "denominator": None,
            "period_start": str(row['effective_date'].date()),
            "period_end": str(row['effective_date'].date())
        }
        finding_2['metrics'][f"{prefix}_confidence"] = {
            "value": float(row['confidence']) if pd.notna(row['confidence']) else None,
            "unit": None,
            "numerator": float(row['confidence']) if pd.notna(row['confidence']) else None,
            "denominator": 1.0,
            "period_start": str(row['effective_date'].date()),
            "period_end": str(row['effective_date'].date())
        }
    
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Quantified Waste Cost (Non-Null Observations Only)
# ============================================================================

# Filter inventory for analysis period with non-null waste cost
inventory_analysis = inventory_df[
    (inventory_df['week_starting'] >= analysis_start) &
    (inventory_df['week_starting'] < analysis_end) &
    (inventory_df['known_waste_cost_sar'].notna())
].copy()

if len(inventory_analysis) > 0:
    total_waste_cost = inventory_analysis['known_waste_cost_sar'].sum()
    total_units_wasted = inventory_analysis['units_wasted'].sum()
    
    finding_3 = {
        "title": "Quantified Waste Cost: Analysis Period",
        "claim": f"During {analysis_start} to {analysis_end}, inventory records show {len(inventory_analysis)} items with non-null waste cost observations, totaling {total_waste_cost:.2f} SAR in known waste.",
        "finding_type": "waste_cost",
        "metrics": {
            "total_waste_cost_sar": {
                "value": float(total_waste_cost),
                "unit": "SAR",
                "numerator": float(total_waste_cost),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_units_wasted": {
                "value": float(total_units_wasted),
                "unit": "units",
                "numerator": float(total_units_wasted),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "items_with_waste_data": {
                "value": len(inventory_analysis),
                "unit": "count",
                "numerator": len(inventory_analysis),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(inventory_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Only items with non-null known_waste_cost_sar included",
            f"Blank waste values excluded per methodology",
            f"{len(inventory_analysis)} items with waste cost observations"
        ],
        "assumptions": [
            "known_waste_cost_sar reflects actual waste cost as recorded",
            "Waste cost is calculated from units_wasted × unit_cost_sar or equivalent",
            "No waste cost imputation applied to null values"
        ],
        "confidence": 0.90
    }
    
    findings.append(finding_3)

# ============================================================================
# Build output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write to output path
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")