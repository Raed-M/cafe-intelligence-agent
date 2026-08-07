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

# Load artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Define analysis period
analysis_start = datetime(2026, 7, 20, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 7, 27, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 7, 13, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 7, 20, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Prepare findings list
findings = []

# ============================================================================
# FINDING 1: Item-level COGS and Gross Profit Analysis
# ============================================================================

# Merge POS with menu to get unit costs
pos_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'price_sar']],
    on='sku',
    how='left'
)

# Calculate metrics for items with known costs
pos_with_cost['cogs'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
pos_with_cost['gross_profit'] = pos_with_cost['line_total_sar'] - pos_with_cost['cogs']

# Exclude refunds from analysis
pos_with_cost_no_refunds = pos_with_cost[~pos_with_cost['is_refund']].copy()

# Group by item to get totals
item_economics = pos_with_cost_no_refunds.groupby('item_name_en').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'cogs': 'sum',
    'gross_profit': 'sum',
    'transaction_id': 'nunique'
}).reset_index()

item_economics.columns = ['item_name', 'total_quantity', 'total_revenue', 'total_cogs', 'total_gross_profit', 'basket_count']
item_economics['gross_margin_pct'] = (item_economics['total_gross_profit'] / item_economics['total_revenue'] * 100).round(2)

# Find top 3 items by gross profit contribution
top_items = item_economics.nlargest(3, 'total_gross_profit')

if len(top_items) > 0:
    finding_1 = {
        "title": "Top 3 Items by Gross Profit Contribution (Analysis Period)",
        "claim": f"During {analysis_start.date()} to {analysis_end.date()}, the top 3 items by gross profit contribution generated {top_items['total_gross_profit'].sum():.2f} SAR in total gross profit, representing {(top_items['total_gross_profit'].sum() / item_economics['total_gross_profit'].sum() * 100):.1f}% of total item-level gross profit.",
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
                "value": round(top_items.iloc[0]['total_gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_items.iloc[0]['total_gross_profit'], 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_margin_pct": {
                "value": round(top_items.iloc[0]['gross_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_items.iloc[0]['gross_margin_pct'], 2),
                "denominator": 100,
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
                "value": round(top_items.iloc[1]['total_gross_profit'], 2) if len(top_items) > 1 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[1]['total_gross_profit'], 2) if len(top_items) > 1 else None,
                "denominator": None,
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
                "value": round(top_items.iloc[2]['total_gross_profit'], 2) if len(top_items) > 2 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[2]['total_gross_profit'], 2) if len(top_items) > 2 else None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_items_analyzed": {
                "value": len(item_economics),
                "unit": "count",
                "numerator": len(item_economics),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(pos_with_cost_no_refunds.shape[0]),
        "coverage_notes": [
            f"Analysis covers {pos_with_cost_no_refunds.shape[0]} non-refund POS line items from {analysis_start.date()} to {analysis_end.date()}",
            f"Menu unit costs available for {pos_with_cost_no_refunds['unit_cost_sar'].notna().sum()} of {pos_with_cost_no_refunds.shape[0]} items ({pos_with_cost_no_refunds['unit_cost_sar'].notna().sum() / pos_with_cost_no_refunds.shape[0] * 100:.1f}%)",
            f"Refunds excluded from analysis (is_refund=True)"
        ],
        "assumptions": [
            "Menu unit_cost_sar represents actual COGS per unit",
            "Line totals are net of discounts",
            "No recipe/BOM adjustments applied"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Waste Cost Analysis
# ============================================================================

# Filter inventory for analysis week
inventory_analysis = inventory_df[inventory_df['week_starting'] == '2026-07-20'].copy()

# Calculate total waste cost
waste_with_cost = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()

if len(waste_with_cost) > 0:
    total_waste_cost = waste_with_cost['known_waste_cost_sar'].sum()
    total_units_wasted = waste_with_cost['units_wasted'].sum()
    
    finding_2 = {
        "title": "Quantified Waste Cost (Week of 2026-07-20)",
        "claim": f"During the week of 2026-07-20, {int(total_units_wasted)} units were wasted across {len(waste_with_cost)} items with known waste costs, totaling {total_waste_cost:.2f} SAR in waste cost.",
        "finding_type": "waste_analysis",
        "metrics": {
            "total_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": round(total_waste_cost, 2),
                "denominator": None,
                "period_start": "2026-07-20T00:00:00+03:00",
                "period_end": "2026-07-27T00:00:00+03:00"
            },
            "total_units_wasted": {
                "value": int(total_units_wasted),
                "unit": "units",
                "numerator": int(total_units_wasted),
                "denominator": None,
                "period_start": "2026-07-20T00:00:00+03:00",
                "period_end": "2026-07-27T00:00:00+03:00"
            },
            "items_with_waste": {
                "value": len(waste_with_cost),
                "unit": "count",
                "numerator": len(waste_with_cost),
                "denominator": None,
                "period_start": "2026-07-20T00:00:00+03:00",
                "period_end": "2026-07-27T00:00:00+03:00"
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(waste_with_cost),
        "coverage_notes": [
            f"Only {len(waste_with_cost)} of {len(inventory_analysis)} inventory items have non-null known_waste_cost_sar values",
            "Blank waste values are excluded per methodology",
            "Week starting 2026-07-20 only"
        ],
        "assumptions": [
            "known_waste_cost_sar represents actual waste cost",
            "Waste items are independent observations"
        ],
        "confidence": 0.85
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Supplier Price Changes from Email Evidence
# ============================================================================

# Filter emails for price changes with effective dates
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    # Convert dates
    price_changes['effective_date'] = pd.to_datetime(price_changes['effective_date'])
    
    # Calculate percentage change
    price_changes['pct_change'] = (
        (price_changes['new_price'] - price_changes['old_price']) / 
        price_changes['old_price'] * 100
    ).round(2)
    
    # Sort by effective date
    price_changes = price_changes.sort_values('effective_date')
    
    # Get first price change as example
    first_change = price_changes.iloc[0]
    
    finding_3 = {
        "title": "Supplier Price Change Evidence",
        "claim": f"Email evidence documents a price change for {first_change['entity_or_ingredient']}: from {first_change['old_price']} to {first_change['new_price']} {first_change['currency']} per {first_change['unit']}, effective {first_change['effective_date'].date()}. This represents a {first_change['pct_change']:.1f}% change.",
        "finding_type": "supplier_pricing",
        "metrics": {
            "ingredient_name": {
                "value": str(first_change['entity_or_ingredient']),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": first_change['effective_date'].isoformat(),
                "period_end": None
            },
            "old_price": {
                "value": float(first_change['old_price']),
                "unit": first_change['currency'],
                "numerator": float(first_change['old_price']),
                "denominator": None,
                "period_start": first_change['effective_date'].isoformat(),
                "period_end": None
            },
            "new_price": {
                "value": float(first_change['new_price']),
                "unit": first_change['currency'],
                "numerator": float(first_change['new_price']),
                "denominator": None,
                "period_start": first_change['effective_date'].isoformat(),
                "period_end": None
            },
            "price_change_pct": {
                "value": round(first_change['pct_change'], 2),
                "unit": "%",
                "numerator": round(first_change['pct_change'], 2),
                "denominator": 100,
                "period_start": first_change['effective_date'].isoformat(),
                "period_end": None
            },
            "unit_of_measure": {
                "value": str(first_change['unit']),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": first_change['effective_date'].isoformat(),
                "period_end": None
            },
            "total_price_changes_identified": {
                "value": len(price_changes),
                "unit": "count",
                "numerator": len(price_changes),
                "denominator": None,
                "period_start": price_changes['effective_date'].min().isoformat(),
                "period_end": price_changes['effective_date'].max().isoformat()
            }
        },
        "source_names": ["emails"],
        "sample_size": len(price_changes),
        "coverage_notes": [
            f"Email extraction identified {len(price_changes)} price changes with old/new prices and effective dates",
            "Price changes span from {price_changes['effective_date'].min().date()} to {price_changes['effective_date'].max().date()}",
            "No standing order quantities found in email facts to calculate procurement cost scenarios"
        ],
        "assumptions": [
            "Email extraction confidence scores are reliable",
            "Effective dates represent actual implementation dates",
            "Price changes apply to supplier invoices going forward"
        ],
        "confidence": 0.80
    }
    findings.append(finding_3)

# ============================================================================
# Prepare output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings generated.")
