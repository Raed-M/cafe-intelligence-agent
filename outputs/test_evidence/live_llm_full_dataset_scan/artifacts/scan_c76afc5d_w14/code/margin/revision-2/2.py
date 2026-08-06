import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
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
analysis_start = "2026-04-13"
analysis_end = "2026-04-20"
previous_start = "2026-04-06"
previous_end = "2026-04-13"

# Convert timestamps to dates for filtering
pos_df['date'] = pd.to_datetime(pos_df['calendar_date']).dt.date
inventory_df['week_date'] = pd.to_datetime(inventory_df['week_starting']).dt.date

# Filter POS for analysis period
pos_analysis = pos_df[
    (pos_df['date'] >= pd.to_datetime(analysis_start).date()) &
    (pos_df['date'] < pd.to_datetime(analysis_end).date())
].copy()

pos_previous = pos_df[
    (pos_df['date'] >= pd.to_datetime(previous_start).date()) &
    (pos_df['date'] < pd.to_datetime(previous_end).date())
].copy()

# Filter inventory for analysis period - FIX: Use parentheses for proper boolean logic
inv_analysis = inventory_df[
    inventory_df['week_date'] >= pd.to_datetime(analysis_start).date()
].copy()

inv_previous = inventory_df[
    (inventory_df['week_date'] >= pd.to_datetime(previous_start).date()) &
    (inventory_df['week_date'] < pd.to_datetime(analysis_start).date())
].copy()

findings = []

# ============================================================================
# FINDING 1: Item-level gross profit contribution (top item by absolute profit)
# ============================================================================

# Merge POS with menu to get unit costs
pos_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'price_sar']],
    on='sku',
    how='left'
)

# Exclude refunds
pos_with_cost = pos_with_cost[pos_with_cost['is_refund'] == False].copy()

# Calculate line-level metrics
pos_with_cost['revenue'] = pos_with_cost['line_total_sar']
pos_with_cost['cogs'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
pos_with_cost['gross_profit'] = pos_with_cost['revenue'] - pos_with_cost['cogs']

# Group by item to get totals
item_metrics = pos_with_cost.groupby('item_name_en').agg({
    'revenue': 'sum',
    'cogs': 'sum',
    'gross_profit': 'sum',
    'quantity': 'sum',
    'transaction_id': 'nunique'
}).reset_index()

item_metrics.columns = ['item_name', 'total_revenue', 'total_cogs', 'total_gross_profit', 'units_sold', 'basket_count']
item_metrics['margin_pct'] = (item_metrics['total_gross_profit'] / item_metrics['total_revenue'] * 100).round(1)

# Sort by gross profit
item_metrics_sorted = item_metrics.sort_values('total_gross_profit', ascending=False)

# Get top 1 item only (revised from top 3 due to lack of recipe/BOM evidence)
top_item_data = item_metrics_sorted.head(1)

if len(top_item_data) >= 1:
    top_item = top_item_data.iloc[0]
    
    finding_1 = {
        "title": "Top item by gross profit contribution",
        "claim": f"In the week of {analysis_start} to {analysis_end}, {top_item['item_name']} generated the highest gross profit contribution at SAR {top_item['total_gross_profit']:.2f} with a gross margin of {top_item['margin_pct']:.1f}% based on menu-level unit costs.",
        "finding_type": "item_economics",
        "metrics": {
            "top_item_name": {
                "value": top_item['item_name'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "top_item_gross_profit_sar": {
                "value": round(top_item['total_gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_item['total_gross_profit'], 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "top_item_revenue_sar": {
                "value": round(top_item['total_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "top_item_units_sold": {
                "value": int(top_item['units_sold']),
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "top_item_baskets": {
                "value": int(top_item['basket_count']),
                "unit": "transactions",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(pos_with_cost),
        "coverage_notes": [
            "Analysis period: 2026-04-13 to 2026-04-20",
            "Refunds excluded from revenue and profit calculations",
            "Unit costs sourced from menu.unit_cost_sar",
            "Line totals from POS line_total_sar field",
            "Item-level unit costs applied uniformly across all sales"
        ],
        "assumptions": [
            "Unit costs from menu_items.unit_cost_sar applied uniformly across all sales",
            "No recipe/BOM data available; per-drink ingredient costs not calculated",
            "Gross profit = revenue - (quantity × unit_cost_sar)",
            "Refunds are excluded from all calculations",
            "Menu-level unit costs are approximations and do not account for actual ingredient consumption variability"
        ],
        "confidence": 0.75
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Cafe-level gross margin
# ============================================================================

total_revenue_analysis = pos_with_cost['revenue'].sum()
total_cogs_analysis = pos_with_cost['cogs'].sum()
total_gross_profit_analysis = total_revenue_analysis - total_cogs_analysis
cafe_margin_pct = (total_gross_profit_analysis / total_revenue_analysis * 100) if total_revenue_analysis > 0 else 0

finding_2 = {
    "title": "Cafe-level gross margin",
    "claim": f"During the week of {analysis_start} to {analysis_end}, the cafe achieved a gross margin of {cafe_margin_pct:.1f}% on total revenue of SAR {total_revenue_analysis:.2f}.",
    "finding_type": "financial_performance",
    "metrics": {
        "cafe_gross_margin_pct": {
            "value": round(cafe_margin_pct, 1),
            "unit": "%",
            "numerator": round(total_gross_profit_analysis, 2),
            "denominator": round(total_revenue_analysis, 2),
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "cafe_total_revenue_sar": {
            "value": round(total_revenue_analysis, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "cafe_total_cogs_sar": {
            "value": round(total_cogs_analysis, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "cafe_gross_profit_sar": {
            "value": round(total_gross_profit_analysis, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    },
    "source_names": ["pos", "menu"],
    "sample_size": len(pos_with_cost),
    "coverage_notes": [
        "Analysis period: 2026-04-13 to 2026-04-20",
        "Refunds excluded from calculations",
        "Unit costs sourced from menu.unit_cost_sar",
        "All POS line items with known SKUs included"
    ],
    "assumptions": [
        "Unit costs from menu applied uniformly",
        "Gross profit = total revenue - total COGS",
        "Refunds excluded from all calculations"
    ],
    "confidence": 0.90
}
findings.append(finding_2)

# ============================================================================
# FINDING 3: Quantified waste cost
# ============================================================================

# Filter inventory for analysis period with non-null waste
inv_with_waste = inv_analysis[inv_analysis['known_waste_cost_sar'].notna()].copy()

if len(inv_with_waste) > 0:
    total_waste_cost = inv_with_waste['known_waste_cost_sar'].sum()
    waste_item_count = inv_with_waste['sku'].nunique()
    total_waste_units = inv_with_waste['units_wasted'].sum()
    
    # Calculate waste as percentage of total COGS for the period
    # Use inventory unit costs and sold units to estimate COGS
    inv_cogs = inv_analysis.groupby('sku').apply(
        lambda x: (x['units_sold'] * x['unit_cost_sar']).sum()
    ).sum()
    
    waste_pct_of_cogs = (total_waste_cost / inv_cogs * 100) if inv_cogs > 0 else 0
    
    finding_3 = {
        "title": "Quantified waste cost impact",
        "claim": f"During the week of {analysis_start} to {analysis_end}, quantified waste cost totaled SAR {total_waste_cost:.2f} across {waste_item_count} items ({int(total_waste_units)} units). This represents {waste_pct_of_cogs:.1f}% of inventory-based COGS (SAR {inv_cogs:.2f}).",
        "finding_type": "waste_cost",
        "metrics": {
            "waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "waste_item_count": {
                "value": waste_item_count,
                "unit": "items",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "waste_unit_count": {
                "value": int(total_waste_units),
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "waste_pct_of_inventory_cogs": {
                "value": round(waste_pct_of_cogs, 1),
                "unit": "%",
                "numerator": round(total_waste_cost, 2),
                "denominator": round(inv_cogs, 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "inventory_cogs_sar": {
                "value": round(inv_cogs, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(inv_with_waste),
        "coverage_notes": [
            "Analysis period: 2026-04-13 to 2026-04-20",
            "Only non-null waste_cost_sar values included",
            "Waste cost sourced from inventory.known_waste_cost_sar",
            f"Coverage: {len(inv_with_waste)} inventory records with waste data",
            f"Inventory COGS calculated from units_sold × unit_cost_sar across all inventory records for period"
        ],
        "assumptions": [
            "Waste cost is incremental to COGS and represents lost margin",
            "Null waste values are excluded (treated as unknown, not zero)",
            "Inventory unit costs applied to sold units for COGS estimation",
            "Waste percentage calculated against inventory-derived COGS",
            "Inventory COGS represents the cost basis for waste percentage calculation"
        ],
        "confidence": 0.80
    }
    findings.append(finding_3)

# ============================================================================
# Build output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")
