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

# Parse analysis period
analysis_start = "2026-05-11T00:00:00+03:00"
analysis_end = "2026-05-18T00:00:00+03:00"

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Convert timestamps to comparable format
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Remove timezone info from analysis_start_dt and analysis_end_dt for comparison with tz-naive data
analysis_start_dt_naive = analysis_start_dt.tz_localize(None)
analysis_end_dt_naive = analysis_end_dt.tz_localize(None)

# Filter POS to analysis period, exclude refunds
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start_dt) &
    (pos_df['timestamp'] < analysis_end_dt) &
    (pos_df['is_refund'] == False)
].copy()

# Prepare result structure
findings = []

# ============================================================================
# FINDING 1: Item-level gross profit and margin (highest contributor)
# ============================================================================

# Aggregate POS by SKU for analysis period
sku_revenue = pos_analysis.groupby('sku').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'item_name_en': 'first'
}).reset_index()
sku_revenue.columns = ['sku', 'units_sold', 'revenue_sar', 'item_name']

# Merge with menu to get unit costs
sku_revenue = sku_revenue.merge(
    menu_df[['sku', 'unit_cost_sar']],
    on='sku',
    how='left'
)

# Calculate COGS and gross profit
sku_revenue['cogs_sar'] = sku_revenue['units_sold'] * sku_revenue['unit_cost_sar']
sku_revenue['gross_profit_sar'] = sku_revenue['revenue_sar'] - sku_revenue['cogs_sar']
sku_revenue['gross_margin_pct'] = (sku_revenue['gross_profit_sar'] / sku_revenue['revenue_sar'] * 100).round(1)

# Find highest gross profit item with valid unit cost
valid_items = sku_revenue[sku_revenue['unit_cost_sar'].notna()].copy()
if len(valid_items) > 0:
    top_item = valid_items.loc[valid_items['gross_profit_sar'].idxmax()]
    
    finding_1 = {
        "title": "Highest Gross Profit Item (Analysis Period)",
        "claim": f"{top_item['item_name']} generated an estimated {top_item['gross_profit_sar']:.2f} SAR gross profit (estimated {top_item['gross_margin_pct']:.1f}% margin) from {int(top_item['units_sold'])} units sold at {top_item['revenue_sar']:.2f} SAR revenue during 2026-05-11 to 2026-05-18, based on menu-declared unit cost of {top_item['unit_cost_sar']:.1f} SAR (no recipe/BOM verification available).",
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
                "value": round(top_item['unit_cost_sar'], 1),
                "unit": "SAR/unit",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "cogs_sar": {
                "value": round(top_item['cogs_sar'], 2),
                "unit": "SAR",
                "numerator": round(top_item['units_sold'], 0),
                "denominator": round(top_item['unit_cost_sar'], 1),
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
        "source_names": ["pos.parquet", "menu.parquet"],
        "sample_size": int(top_item['units_sold']),
        "coverage_notes": [
            f"Analysis period: 2026-05-11 to 2026-05-18",
            f"Refunds excluded from POS",
            f"Menu unit_cost_sar used as proxy for COGS; no recipe/BOM available",
            f"Item {top_item['sku']} had {int(top_item['units_sold'])} transactions in period"
        ],
        "assumptions": [
            "Menu unit_cost_sar is current and accurate for the analysis period",
            "No recipe/BOM available; unit cost treated as declared in menu",
            "Unit cost applies uniformly across all units sold in period",
            "No waste or portion control drift between menu cost and actual COGS"
        ],
        "confidence": 0.70
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Supplier price change with temporal alignment
# ============================================================================

# Filter emails for price changes with valid old/new prices
price_changes = emails_df[
    (emails_df['old_price'].notna()) &
    (emails_df['new_price'].notna()) &
    (emails_df['old_price'] > 0) &
    (emails_df['new_price'] > 0)
].copy()

if len(price_changes) > 0:
    # Take the most recent price change
    price_changes['date'] = pd.to_datetime(price_changes['date'])
    latest_change = price_changes.sort_values('date').iloc[-1]
    
    old_price = float(latest_change['old_price'])
    new_price = float(latest_change['new_price'])
    price_delta = new_price - old_price
    pct_change = (price_delta / old_price * 100) if old_price > 0 else 0
    
    # Use effective_date if available, otherwise use email date
    if pd.notna(latest_change['effective_date']):
        eff_date = pd.to_datetime(latest_change['effective_date'])
    else:
        eff_date = latest_change['date']
    
    eff_date_iso = eff_date.isoformat()
    
    finding_2 = {
        "title": "Supplier Price Change Detected",
        "claim": f"Supplier price change for {latest_change['entity_or_ingredient']}: {old_price:.2f} SAR/{latest_change['unit']} → {new_price:.2f} SAR/{latest_change['unit']} (delta: {price_delta:.2f} SAR, {pct_change:.2f}% change), effective {eff_date.strftime('%Y-%m-%d')}. Without recipe/BOM and standing order volumes, per-unit cost impact on menu items cannot be calculated.",
        "finding_type": "supplier_cost_change",
        "metrics": {
            "old_price_sar": {
                "value": round(old_price, 2),
                "unit": f"SAR/{latest_change['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": eff_date_iso,
                "period_end": eff_date_iso
            },
            "new_price_sar": {
                "value": round(new_price, 2),
                "unit": f"SAR/{latest_change['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": eff_date_iso,
                "period_end": eff_date_iso
            },
            "price_delta_sar": {
                "value": round(price_delta, 2),
                "unit": f"SAR/{latest_change['unit']}",
                "numerator": round(new_price, 2),
                "denominator": round(old_price, 2),
                "period_start": eff_date_iso,
                "period_end": eff_date_iso
            },
            "percent_change": {
                "value": round(pct_change, 2),
                "unit": "%",
                "numerator": round(price_delta, 2),
                "denominator": round(old_price, 2),
                "period_start": eff_date_iso,
                "period_end": eff_date_iso
            }
        },
        "source_names": ["emails.parquet"],
        "sample_size": None,
        "coverage_notes": [
            f"Supplier: {latest_change['sender']}",
            f"Email date: {latest_change['date'].strftime('%Y-%m-%d')}",
            f"Effective date: {eff_date.strftime('%Y-%m-%d')}",
            f"Ingredient/entity: {latest_change['entity_or_ingredient']}",
            f"Confidence in extraction: {latest_change['confidence']}"
        ],
        "assumptions": [
            "Price change extracted from supplier email is accurate",
            "Effective date reflects when new price takes effect",
            "No recipe/BOM available; cannot calculate per-drink cost impact",
            "Standing order volumes and payment terms unknown; procurement scenario not calculated"
        ],
        "confidence": 0.75
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Waste cost analysis (highest waste cost item with ranking)
# ============================================================================

# Filter inventory to analysis period
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
# Remove timezone info for comparison
inventory_week_naive = inventory_df['week_starting'].dt.tz_localize(None)
inv_analysis = inventory_df[
    (inventory_week_naive >= analysis_start_dt_naive) &
    (inventory_week_naive < analysis_end_dt_naive)
].copy()

# Calculate waste cost for items with non-null waste and unit cost
inv_analysis['waste_cost_sar'] = inv_analysis['units_wasted'] * inv_analysis['unit_cost_sar']
inv_analysis = inv_analysis[inv_analysis['waste_cost_sar'].notna()].copy()

# Filter to items with non-zero waste cost
waste_items = inv_analysis[inv_analysis['waste_cost_sar'] > 0].copy()

if len(waste_items) > 0:
    # Rank by waste cost
    waste_items = waste_items.sort_values('waste_cost_sar', ascending=False)
    top_waste = waste_items.iloc[0]
    
    # Calculate totals for coverage
    total_waste_cost = waste_items['waste_cost_sar'].sum()
    total_units_wasted = waste_items['units_wasted'].sum()
    
    # Build comparative evidence
    waste_ranking = waste_items[['item', 'units_wasted', 'unit_cost_sar', 'waste_cost_sar']].copy()
    waste_ranking = waste_ranking.sort_values('waste_cost_sar', ascending=False)
    
    # Create ranking list for coverage notes
    ranking_list = []
    for idx, row in waste_ranking.iterrows():
        ranking_list.append({
            'item': row['item'],
            'waste_cost_sar': round(row['waste_cost_sar'], 2)
        })
    
    finding_3 = {
        "title": "Highest Waste Cost Item (Analysis Period)",
        "claim": f"{top_waste['item']} had the highest waste cost of {top_waste['waste_cost_sar']:.2f} SAR from {int(top_waste['units_wasted'])} units wasted at {top_waste['unit_cost_sar']:.1f} SAR/unit during 2026-05-11 to 2026-05-18. Total waste cost across {len(waste_items)} items: {total_waste_cost:.2f} SAR ({int(total_units_wasted)} units).",
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
            "unit_cost_sar": {
                "value": round(top_waste['unit_cost_sar'], 1),
                "unit": "SAR/unit",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "waste_cost_sar": {
                "value": round(top_waste['waste_cost_sar'], 2),
                "unit": "SAR",
                "numerator": int(top_waste['units_wasted']),
                "denominator": round(top_waste['unit_cost_sar'], 1),
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
            },
            "items_with_waste": {
                "value": len(waste_items),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["inventory.parquet"],
        "sample_size": int(total_units_wasted),
        "coverage_notes": [
            f"Analysis period: 2026-05-11 to 2026-05-18",
            f"Items with non-zero waste cost: {len(waste_items)}",
            f"Waste cost ranking (all items): {ranking_list}",
            f"Unit costs sourced from inventory.parquet for the analysis period",
            f"Refunds and dead sensors not explicitly filtered from inventory; assume inventory records are clean"
        ],
        "assumptions": [
            "Unit cost from inventory (SAR/unit) represents the COGS applicable to units wasted during this period",
            "Waste records in inventory are accurate and exclude non-waste adjustments",
            "Unit cost is uniform across all wasted units for each item",
            "No recipe/BOM available; waste valuation uses inventory unit cost as proxy for COGS"
        ],
        "confidence": 0.75
    }
    findings.append(finding_3)

# ============================================================================
# Construct final output
# ============================================================================

result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write result to output path
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")
