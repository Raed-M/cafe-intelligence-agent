import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

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
analysis_start = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 6, 8, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 5, 25, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period (non-refunds only for revenue/COGS)
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start) & 
    (pos_df['timestamp'] < analysis_end) &
    (pos_df['is_refund'] == False)
].copy()

pos_previous = pos_df[
    (pos_df['timestamp'] >= previous_start) & 
    (pos_df['timestamp'] < previous_end) &
    (pos_df['is_refund'] == False)
].copy()

# Inventory for week starting 2026-06-01
inventory_analysis = inventory_df[inventory_df['week_starting'] == '2026-06-01'].copy()
inventory_previous = inventory_df[inventory_df['week_starting'] == '2026-05-25'].copy()

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

# Calculate item-level economics
pos_with_cost['cogs'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
pos_with_cost['gross_profit'] = pos_with_cost['line_total_sar'] - pos_with_cost['cogs']
pos_with_cost['gross_margin_pct'] = (
    (pos_with_cost['gross_profit'] / pos_with_cost['line_total_sar'] * 100)
    .fillna(0)
)

# Aggregate by item
item_economics = pos_with_cost.groupby('item_name_en').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'cogs': 'sum',
    'gross_profit': 'sum',
    'transaction_id': 'nunique'
}).reset_index()

item_economics.columns = ['item_name', 'total_quantity', 'total_revenue', 'total_cogs', 'total_gross_profit', 'basket_count']
item_economics['gross_margin_pct'] = (
    (item_economics['total_gross_profit'] / item_economics['total_revenue'] * 100)
    .fillna(0)
)

# Sort by gross profit contribution
item_economics = item_economics.sort_values('total_gross_profit', ascending=False)

# Top 3 items by gross profit
top_items = item_economics.head(3)

if len(top_items) > 0:
    finding_1 = {
        "title": "Top 3 Items by Gross Profit Contribution (Week of 2026-06-01)",
        "claim": f"The top 3 items by gross profit contribution are {', '.join(top_items['item_name'].tolist())}, collectively generating {top_items['total_gross_profit'].sum():.2f} SAR in gross profit from {top_items['total_quantity'].sum():.0f} units sold across {top_items['basket_count'].sum():.0f} baskets.",
        "finding_type": "item_economics",
        "metrics": {
            "top_item_1_name": {
                "value": top_items.iloc[0]['item_name'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-06-01T00:00:00Z",
                "period_end": "2026-06-08T00:00:00Z"
            },
            "top_item_1_gross_profit": {
                "value": round(top_items.iloc[0]['total_gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_items.iloc[0]['total_gross_profit'], 2),
                "denominator": None,
                "period_start": "2026-06-01T00:00:00Z",
                "period_end": "2026-06-08T00:00:00Z"
            },
            "top_item_1_quantity": {
                "value": int(top_items.iloc[0]['total_quantity']),
                "unit": "units",
                "numerator": int(top_items.iloc[0]['total_quantity']),
                "denominator": None,
                "period_start": "2026-06-01T00:00:00Z",
                "period_end": "2026-06-08T00:00:00Z"
            },
            "top_item_1_margin_pct": {
                "value": round(top_items.iloc[0]['gross_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_items.iloc[0]['gross_margin_pct'], 2),
                "denominator": 100,
                "period_start": "2026-06-01T00:00:00Z",
                "period_end": "2026-06-08T00:00:00Z"
            },
            "top_3_combined_gross_profit": {
                "value": round(top_items['total_gross_profit'].sum(), 2),
                "unit": "SAR",
                "numerator": round(top_items['total_gross_profit'].sum(), 2),
                "denominator": None,
                "period_start": "2026-06-01T00:00:00Z",
                "period_end": "2026-06-08T00:00:00Z"
            },
            "top_3_combined_baskets": {
                "value": int(top_items['basket_count'].sum()),
                "unit": "baskets",
                "numerator": int(top_items['basket_count'].sum()),
                "denominator": None,
                "period_start": "2026-06-01T00:00:00Z",
                "period_end": "2026-06-08T00:00:00Z"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(pos_analysis),
        "coverage_notes": [
            "Analysis period: 2026-06-01 to 2026-06-08",
            "Excludes refunds (is_refund=False)",
            "Unit costs sourced from menu.parquet",
            "COGS = quantity × unit_cost_sar",
            "Gross profit = line_total_sar - COGS",
            f"Total items analyzed: {len(item_economics)}"
        ],
        "assumptions": [
            "Menu unit_cost_sar is current and applicable to all transactions in period",
            "No recipe/BOM adjustments applied",
            "Waste costs not included in item-level COGS (tracked separately in inventory)"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Waste Cost Impact
# ============================================================================

# Calculate waste costs for analysis period
waste_cost_analysis = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()

if len(waste_cost_analysis) > 0:
    total_waste_cost = waste_cost_analysis['known_waste_cost_sar'].sum()
    total_waste_units = waste_cost_analysis['units_wasted'].sum()
    
    waste_items = waste_cost_analysis[['item', 'units_wasted', 'known_waste_cost_sar']].copy()
    waste_items = waste_items.sort_values('known_waste_cost_sar', ascending=False)
    
    finding_2 = {
        "title": "Quantified Waste Cost Impact (Week of 2026-06-01)",
        "claim": f"Known waste cost for the week of 2026-06-01 totals {total_waste_cost:.2f} SAR across {int(total_waste_units)} units wasted, representing direct margin erosion.",
        "finding_type": "waste_economics",
        "metrics": {
            "total_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": round(total_waste_cost, 2),
                "denominator": None,
                "period_start": "2026-06-01T00:00:00Z",
                "period_end": "2026-06-08T00:00:00Z"
            },
            "total_waste_units": {
                "value": int(total_waste_units),
                "unit": "units",
                "numerator": int(total_waste_units),
                "denominator": None,
                "period_start": "2026-06-01T00:00:00Z",
                "period_end": "2026-06-08T00:00:00Z"
            },
            "waste_cost_per_unit": {
                "value": round(total_waste_cost / total_waste_units, 2) if total_waste_units > 0 else 0,
                "unit": "SAR/unit",
                "numerator": round(total_waste_cost, 2),
                "denominator": int(total_waste_units),
                "period_start": "2026-06-01T00:00:00Z",
                "period_end": "2026-06-08T00:00:00Z"
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(waste_cost_analysis),
        "coverage_notes": [
            "Analysis period: week starting 2026-06-01",
            "Only non-null known_waste_cost_sar values included",
            f"Items with waste data: {len(waste_cost_analysis)}",
            "Waste cost represents COGS of discarded inventory"
        ],
        "assumptions": [
            "known_waste_cost_sar reflects actual unit cost of wasted items",
            "Waste is not recovered or donated (full cost impact)"
        ],
        "confidence": 0.90
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Supplier Price Changes from Email Evidence
# ============================================================================

# Filter emails for price changes with effective dates
price_change_emails = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_change_emails) > 0:
    price_change_emails['effective_date'] = pd.to_datetime(price_change_emails['effective_date'])
    
    # Filter for changes effective during or before analysis period
    price_change_emails = price_change_emails[
        price_change_emails['effective_date'] <= analysis_end
    ].copy()
    
    if len(price_change_emails) > 0:
        # Calculate price change metrics
        price_change_emails['price_delta'] = price_change_emails['new_price'] - price_change_emails['old_price']
        price_change_emails['pct_change'] = (
            (price_change_emails['price_delta'] / price_change_emails['old_price'] * 100)
            .fillna(0)
        )
        
        # Sort by absolute price delta
        price_change_emails = price_change_emails.sort_values('price_delta', ascending=False, key=abs)
        
        top_change = price_change_emails.iloc[0]
        
        finding_3 = {
            "title": "Supplier Price Change: Significant Cost Pressure Detected",
            "claim": f"Supplier email evidence shows {top_change['entity_or_ingredient']} price change from {top_change['old_price']:.2f} to {top_change['new_price']:.2f} {top_change['currency']}/{top_change['unit']}, effective {top_change['effective_date'].strftime('%Y-%m-%d')}, representing a {top_change['pct_change']:.1f}% change. This is a documented supplier fact; impact on menu item COGS depends on order volume and payment terms.",
            "finding_type": "supplier_price_change",
            "metrics": {
                "ingredient": {
                    "value": str(top_change['entity_or_ingredient']),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-06-01T00:00:00Z",
                    "period_end": "2026-06-08T00:00:00Z"
                },
                "old_price": {
                    "value": round(top_change['old_price'], 2),
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": round(top_change['old_price'], 2),
                    "denominator": None,
                    "period_start": "2026-06-01T00:00:00Z",
                    "period_end": "2026-06-08T00:00:00Z"
                },
                "new_price": {
                    "value": round(top_change['new_price'], 2),
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": round(top_change['new_price'], 2),
                    "denominator": None,
                    "period_start": "2026-06-01T00:00:00Z",
                    "period_end": "2026-06-08T00:00:00Z"
                },
                "price_delta": {
                    "value": round(top_change['price_delta'], 2),
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": round(top_change['price_delta'], 2),
                    "denominator": None,
                    "period_start": "2026-06-01T00:00:00Z",
                    "period_end": "2026-06-08T00:00:00Z"
                },
                "pct_change": {
                    "value": round(top_change['pct_change'], 2),
                    "unit": "%",
                    "numerator": round(top_change['pct_change'], 2),
                    "denominator": 100,
                    "period_start": "2026-06-01T00:00:00Z",
                    "period_end": "2026-06-08T00:00:00Z"
                },
                "effective_date": {
                    "value": top_change['effective_date'].strftime('%Y-%m-%d'),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-06-01T00:00:00Z",
                    "period_end": "2026-06-08T00:00:00Z"
                }
            },
            "source_names": ["emails"],
            "sample_size": len(price_change_emails),
            "coverage_notes": [
                f"Total price change emails with effective dates: {len(price_change_emails)}",
                "Showing highest absolute price delta",
                "Email extraction confidence: " + str(top_change['confidence']),
                "No recipe/BOM data available to calculate per-drink impact"
            ],
            "assumptions": [
                "Email extraction accurately reflects supplier communication",
                "Price change applies to future orders; current menu costs may not yet reflect change",
                "Standing order volume and payment terms are unknown; actual margin impact requires procurement data"
            ],
            "confidence": 0.85
        }
        findings.append(finding_3)

# ============================================================================
# Build output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
