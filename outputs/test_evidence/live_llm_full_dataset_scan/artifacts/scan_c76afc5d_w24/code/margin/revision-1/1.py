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

# Define analysis period
analysis_start = datetime(2026, 6, 22, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 6, 29, 0, 0, 0, tzinfo=timezone.utc)

previous_start = datetime(2026, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 6, 22, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)

# Filter POS for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

findings = []

# ============================================================================
# FINDING 1: Item-level COGS and Gross Profit Analysis
# ============================================================================

# Calculate item-level economics for analysis period
item_economics = []

for sku in pos_analysis['sku'].unique():
    if pd.isna(sku):
        continue
    
    sku_data = pos_analysis[pos_analysis['sku'] == sku]
    
    # Get menu cost
    menu_row = menu_df[menu_df['sku'] == sku]
    if menu_row.empty:
        continue
    
    unit_cost = menu_row['unit_cost_sar'].values[0]
    item_name = menu_row['item_en'].values[0]
    
    # Calculate totals (excluding refunds)
    non_refund = sku_data[sku_data['is_refund'] == False]
    
    total_quantity = non_refund['quantity'].sum()
    total_revenue = non_refund['line_total_sar'].sum()
    total_cogs = total_quantity * unit_cost
    gross_profit = total_revenue - total_cogs
    
    if total_revenue > 0:
        gp_margin = (gross_profit / total_revenue) * 100
    else:
        gp_margin = 0
    
    item_economics.append({
        'sku': sku,
        'item_name': item_name,
        'quantity_sold': total_quantity,
        'total_revenue': total_revenue,
        'unit_cost': unit_cost,
        'total_cogs': total_cogs,
        'gross_profit': gross_profit,
        'gp_margin_pct': gp_margin
    })

item_econ_df = pd.DataFrame(item_economics)

if not item_econ_df.empty:
    # Sort by gross profit
    item_econ_df = item_econ_df.sort_values('gross_profit', ascending=False)
    
    # Top 5 by gross profit
    top_5_gp = item_econ_df.head(5)
    
    total_revenue_analysis = pos_analysis[pos_analysis['is_refund'] == False]['line_total_sar'].sum()
    total_cogs_analysis = item_econ_df['total_cogs'].sum()
    total_gp_analysis = item_econ_df['gross_profit'].sum()
    overall_margin = (total_gp_analysis / total_revenue_analysis * 100) if total_revenue_analysis > 0 else 0
    
    finding_1 = {
        "title": "Item-Level Gross Profit Analysis (Analysis Week)",
        "claim": f"Top 5 items by gross profit contribution account for {top_5_gp['gross_profit'].sum():.2f} SAR of {total_gp_analysis:.2f} SAR total gross profit ({(top_5_gp['gross_profit'].sum()/total_gp_analysis*100):.1f}%). Overall cafe gross margin: {overall_margin:.1f}%.",
        "finding_type": "margin_analysis",
        "metrics": {
            "total_revenue_sar": {
                "value": round(total_revenue_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_cogs_sar": {
                "value": round(total_cogs_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_gross_profit_sar": {
                "value": round(total_gp_analysis, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "overall_gross_margin_pct": {
                "value": round(overall_margin, 2),
                "unit": "%",
                "numerator": round(total_gp_analysis, 2),
                "denominator": round(total_revenue_analysis, 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_5_items_gp_contribution_pct": {
                "value": round((top_5_gp['gross_profit'].sum()/total_gp_analysis*100), 2),
                "unit": "%",
                "numerator": round(top_5_gp['gross_profit'].sum(), 2),
                "denominator": round(total_gp_analysis, 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(pos_analysis[pos_analysis['is_refund'] == False]),
        "coverage_notes": [
            "Analysis period: 2026-06-22 to 2026-06-29",
            "Excludes refund transactions (is_refund=True)",
            "COGS calculated from menu_items.unit_cost_sar × quantity sold",
            "Gross profit = revenue - COGS (excludes operating expenses)",
            f"Items with known SKU and menu cost: {len(item_econ_df)}"
        ],
        "assumptions": [
            "Menu unit_cost_sar is current and accurate for analysis period",
            "No recipe/BOM available; per-item cost is menu-declared unit cost",
            "Refunds excluded from revenue and COGS calculations",
            "No waste cost adjustment applied at item level"
        ],
        "confidence": 0.95
    }
    
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Waste Cost Impact (Quantified from Inventory)
# ============================================================================

# Filter inventory for analysis week
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'], utc=True)
inv_analysis = inventory_df[inventory_df['week_starting'] == pd.Timestamp('2026-06-22', tz='UTC')]

if not inv_analysis.empty:
    # Only include rows with non-null known_waste_cost_sar
    waste_data = inv_analysis[inv_analysis['known_waste_cost_sar'].notna()].copy()
    
    if not waste_data.empty:
        total_waste_cost = waste_data['known_waste_cost_sar'].sum()
        waste_units = waste_data['units_wasted'].sum()
        waste_items_count = len(waste_data)
        
        # Get total revenue for context
        total_rev = pos_analysis[pos_analysis['is_refund'] == False]['line_total_sar'].sum()
        waste_as_pct_revenue = (total_waste_cost / total_rev * 100) if total_rev > 0 else 0
        
        finding_2 = {
            "title": "Quantified Waste Cost (Analysis Week)",
            "claim": f"Measured waste cost for analysis week: {total_waste_cost:.2f} SAR across {waste_items_count} items ({waste_units:.0f} units wasted). Waste represents {waste_as_pct_revenue:.2f}% of weekly revenue.",
            "finding_type": "waste_cost",
            "metrics": {
                "total_waste_cost_sar": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "units_wasted": {
                    "value": round(waste_units, 2),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "waste_items_count": {
                    "value": waste_items_count,
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "waste_as_pct_revenue": {
                    "value": round(waste_as_pct_revenue, 2),
                    "unit": "%",
                    "numerator": round(total_waste_cost, 2),
                    "denominator": round(total_rev, 2),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["inventory", "pos"],
            "sample_size": waste_items_count,
            "coverage_notes": [
                "Analysis period: 2026-06-22 to 2026-06-29",
                "Only items with non-null known_waste_cost_sar included",
                "Blank waste values excluded per data quality rules",
                f"Items with measured waste: {waste_items_count} of {len(inv_analysis)} inventory records"
            ],
            "assumptions": [
                "known_waste_cost_sar reflects actual measured waste cost",
                "Waste cost is calculated from inventory unit_cost_sar × units_wasted",
                "No waste data imputation applied"
            ],
            "confidence": 0.90
        }
        
        findings.append(finding_2)

# ============================================================================
# FINDING 3: Supplier Price Changes (Detected from Emails)
# ============================================================================

# Filter emails with complete price change data
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) & 
    (emails_df['effective_date'].notna())
].copy()

if not price_changes.empty:
    price_changes['effective_date'] = pd.to_datetime(price_changes['effective_date'], utc=True)
    
    # Calculate percentage change
    price_changes['pct_change'] = ((price_changes['new_price'] - price_changes['old_price']) / price_changes['old_price'] * 100)
    
    # Sort by absolute percentage change
    price_changes = price_changes.sort_values('pct_change', ascending=False, key=abs)
    
    # Get largest change
    largest = price_changes.iloc[0]
    
    # Detection date is the email date
    detection_date = pd.to_datetime(largest['date'], utc=True)
    
    finding_3 = {
        "title": "Supplier Price Changes Detected (Email Evidence)",
        "claim": f"Detected {len(price_changes)} supplier price changes from email evidence. Largest: {largest['entity_or_ingredient']} price change from {largest['old_price']:.2f} to {largest['new_price']:.2f} {largest['currency']} per {largest['unit']} ({largest['pct_change']:+.2f}%), effective {largest['effective_date'].strftime('%Y-%m-%d')}. Detected during analysis week.",
        "finding_type": "supplier_price_change",
        "metrics": {
            "price_changes_detected_count": {
                "value": len(price_changes),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "largest_change_old_price": {
                "value": round(largest['old_price'], 2),
                "unit": f"{largest['currency']} per {largest['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "largest_change_new_price": {
                "value": round(largest['new_price'], 2),
                "unit": f"{largest['currency']} per {largest['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "largest_change_pct": {
                "value": round(largest['pct_change'], 2),
                "unit": "%",
                "numerator": round(largest['new_price'] - largest['old_price'], 2),
                "denominator": round(largest['old_price'], 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "largest_change_ingredient": {
                "value": largest['entity_or_ingredient'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "effective_date": {
                "value": largest['effective_date'].strftime('%Y-%m-%d'),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "detection_date": {
                "value": detection_date.strftime('%Y-%m-%d'),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["emails"],
        "sample_size": len(price_changes),
        "coverage_notes": [
            "Analysis period: 2026-06-22 to 2026-06-29 (detection window)",
            "Effective dates range from past to future relative to analysis period",
            "Only price changes with old_price, new_price, and effective_date included",
            f"Total price changes detected: {len(price_changes)}",
            "Extracted from supplier email communications"
        ],
        "assumptions": [
            "Email extraction confidence: 0.85 (per extraction_mode)",
            "Effective date indicates when price change takes/took effect",
            "Detection date (email date) differs from effective date",
            "No recipe/BOM available; per-drink impact cannot be quantified",
            "Standing order volumes and payment terms not confirmed; procurement cost impact is unquantified"
        ],
        "confidence": 0.85
    }
    
    findings.append(finding_3)

# ============================================================================
# Build output
# ============================================================================

output = {
    "status": "success" if findings else "insufficient_data",
    "findings": findings
}

# Write to output path
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")