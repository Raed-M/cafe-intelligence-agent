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
analysis_start = datetime(2026, 1, 5, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 1, 12, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2025, 12, 29, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 1, 5, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Convert inventory week_starting to datetime
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Get inventory for analysis week
analysis_week = pd.Timestamp('2026-01-05', tz=timezone.utc)
inventory_analysis = inventory_df[inventory_df['week_starting'] == analysis_week].copy()

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from POS and menu
pos_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'price_sar']],
    on='sku',
    how='left'
)

# Filter out refunds for revenue calculation
pos_sales = pos_with_cost[~pos_with_cost['is_refund']].copy()

# Calculate metrics by item
item_economics = pos_sales.groupby(['sku', 'item_name']).agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'unit_cost_sar': 'first',
    'price_sar': 'first',
    'transaction_id': 'nunique'
}).reset_index()

item_economics.columns = ['sku', 'item_name', 'total_quantity', 'total_revenue', 'unit_cost_sar', 'price_sar', 'basket_count']

# Calculate COGS and gross profit
item_economics['total_cogs'] = item_economics['total_quantity'] * item_economics['unit_cost_sar']
item_economics['gross_profit'] = item_economics['total_revenue'] - item_economics['total_cogs']
item_economics['gross_margin_pct'] = (item_economics['gross_profit'] / item_economics['total_revenue'] * 100).round(2)

# Sort by gross profit
item_economics_sorted = item_economics.sort_values('gross_profit', ascending=False)

if len(item_economics_sorted) > 0:
    top_item = item_economics_sorted.iloc[0]
    
    finding_1 = {
        "title": "Top Gross Profit Item - Week of 2026-01-05",
        "claim": f"{top_item['item_name']} generated the highest gross profit of {top_item['gross_profit']:.2f} SAR with {top_item['gross_margin_pct']:.1f}% margin",
        "finding_type": "item_economics",
        "metrics": {
            "item_name": {
                "value": top_item['item_name'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-05T00:00:00+00:00",
                "period_end": "2026-01-12T00:00:00+00:00"
            },
            "total_revenue": {
                "value": round(top_item['total_revenue'], 2),
                "unit": "SAR",
                "numerator": round(top_item['total_revenue'], 2),
                "denominator": None,
                "period_start": "2026-01-05T00:00:00+00:00",
                "period_end": "2026-01-12T00:00:00+00:00"
            },
            "total_cogs": {
                "value": round(top_item['total_cogs'], 2),
                "unit": "SAR",
                "numerator": round(top_item['total_cogs'], 2),
                "denominator": None,
                "period_start": "2026-01-05T00:00:00+00:00",
                "period_end": "2026-01-12T00:00:00+00:00"
            },
            "gross_profit": {
                "value": round(top_item['gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_item['gross_profit'], 2),
                "denominator": None,
                "period_start": "2026-01-05T00:00:00+00:00",
                "period_end": "2026-01-12T00:00:00+00:00"
            },
            "gross_margin_pct": {
                "value": round(top_item['gross_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_item['gross_profit'], 2),
                "denominator": round(top_item['total_revenue'], 2),
                "period_start": "2026-01-05T00:00:00+00:00",
                "period_end": "2026-01-12T00:00:00+00:00"
            },
            "units_sold": {
                "value": int(top_item['total_quantity']),
                "unit": "units",
                "numerator": int(top_item['total_quantity']),
                "denominator": None,
                "period_start": "2026-01-05T00:00:00+00:00",
                "period_end": "2026-01-12T00:00:00+00:00"
            },
            "baskets": {
                "value": int(top_item['basket_count']),
                "unit": "transactions",
                "numerator": int(top_item['basket_count']),
                "denominator": None,
                "period_start": "2026-01-05T00:00:00+00:00",
                "period_end": "2026-01-12T00:00:00+00:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(top_item['basket_count']),
        "coverage_notes": [
            "Analysis period: 2026-01-05 to 2026-01-12",
            "Excludes refunds from revenue calculation",
            "Unit costs from menu.unit_cost_sar",
            "Revenue from POS line_total_sar"
        ],
        "assumptions": [
            "Menu unit_cost_sar is current and applicable to all sales in period",
            "POS line_total_sar accurately reflects actual revenue",
            "No recipe/BOM available; using menu-level unit costs only"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# FINDING 2: Waste Cost Impact
# Calculate waste cost for analysis week
waste_analysis = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()

if len(waste_analysis) > 0:
    total_waste_cost = waste_analysis['known_waste_cost_sar'].sum()
    total_waste_units = waste_analysis['units_wasted'].sum()
    
    waste_items = waste_analysis[['sku', 'item', 'units_wasted', 'known_waste_cost_sar']].copy()
    waste_items = waste_items[waste_items['known_waste_cost_sar'] > 0].sort_values('known_waste_cost_sar', ascending=False)
    
    if len(waste_items) > 0:
        top_waste = waste_items.iloc[0]
        
        finding_2 = {
            "title": "Highest Waste Cost Item - Week of 2026-01-05",
            "claim": f"{top_waste['item']} incurred {top_waste['known_waste_cost_sar']:.2f} SAR in waste cost from {int(top_waste['units_wasted'])} units wasted",
            "finding_type": "waste_cost",
            "metrics": {
                "item_name": {
                    "value": top_waste['item'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-05T00:00:00+00:00",
                    "period_end": "2026-01-12T00:00:00+00:00"
                },
                "waste_cost_sar": {
                    "value": round(top_waste['known_waste_cost_sar'], 2),
                    "unit": "SAR",
                    "numerator": round(top_waste['known_waste_cost_sar'], 2),
                    "denominator": None,
                    "period_start": "2026-01-05T00:00:00+00:00",
                    "period_end": "2026-01-12T00:00:00+00:00"
                },
                "units_wasted": {
                    "value": int(top_waste['units_wasted']),
                    "unit": "units",
                    "numerator": int(top_waste['units_wasted']),
                    "denominator": None,
                    "period_start": "2026-01-05T00:00:00+00:00",
                    "period_end": "2026-01-12T00:00:00+00:00"
                },
                "total_waste_cost_week": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": round(total_waste_cost, 2),
                    "denominator": None,
                    "period_start": "2026-01-05T00:00:00+00:00",
                    "period_end": "2026-01-12T00:00:00+00:00"
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(waste_items),
            "coverage_notes": [
                "Analysis period: 2026-01-05 to 2026-01-12",
                "Only non-null waste cost observations included",
                "Waste cost from inventory.known_waste_cost_sar",
                f"Total items with waste data: {len(waste_items)}"
            ],
            "assumptions": [
                "known_waste_cost_sar accurately reflects actual waste cost",
                "Waste units and costs are recorded at point of disposal"
            ],
            "confidence": 0.90
        }
        findings.append(finding_2)

# FINDING 3: Supplier Price Changes from Emails
# Filter for price changes with effective dates
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    price_changes['effective_date'] = pd.to_datetime(price_changes['effective_date'])
    
    # Ensure effective_date is timezone-naive for comparison with analysis_end
    if price_changes['effective_date'].dt.tz is not None:
        price_changes['effective_date'] = price_changes['effective_date'].dt.tz_localize(None)
    
    # Convert analysis_end to naive datetime for comparison
    analysis_end_naive = analysis_end.replace(tzinfo=None)
    
    # Filter for changes effective during or before analysis period
    price_changes = price_changes[price_changes['effective_date'] <= analysis_end_naive].copy()
    
    if len(price_changes) > 0:
        # Calculate percentage change
        price_changes['price_change_pct'] = (
            (price_changes['new_price'] - price_changes['old_price']) / 
            price_changes['old_price'] * 100
        ).round(2)
        
        # Sort by absolute percentage change
        price_changes['abs_change_pct'] = price_changes['price_change_pct'].abs()
        price_changes_sorted = price_changes.sort_values('abs_change_pct', ascending=False)
        
        top_change = price_changes_sorted.iloc[0]
        
        finding_3 = {
            "title": "Significant Supplier Price Change - " + top_change['entity_or_ingredient'],
            "claim": f"{top_change['entity_or_ingredient']} price changed from {top_change['old_price']:.2f} to {top_change['new_price']:.2f} {top_change['currency']} per {top_change['unit']} (effective {top_change['effective_date'].strftime('%Y-%m-%d')}), a {top_change['price_change_pct']:.1f}% change",
            "finding_type": "supplier_price_change",
            "metrics": {
                "ingredient": {
                    "value": top_change['entity_or_ingredient'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-05T00:00:00+00:00",
                    "period_end": "2026-01-12T00:00:00+00:00"
                },
                "old_price": {
                    "value": round(top_change['old_price'], 2),
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": round(top_change['old_price'], 2),
                    "denominator": None,
                    "period_start": "2026-01-05T00:00:00+00:00",
                    "period_end": "2026-01-12T00:00:00+00:00"
                },
                "new_price": {
                    "value": round(top_change['new_price'], 2),
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": round(top_change['new_price'], 2),
                    "denominator": None,
                    "period_start": "2026-01-05T00:00:00+00:00",
                    "period_end": "2026-01-12T00:00:00+00:00"
                },
                "price_change_pct": {
                    "value": round(top_change['price_change_pct'], 2),
                    "unit": "%",
                    "numerator": round(top_change['new_price'] - top_change['old_price'], 2),
                    "denominator": round(top_change['old_price'], 2),
                    "period_start": "2026-01-05T00:00:00+00:00",
                    "period_end": "2026-01-12T00:00:00+00:00"
                },
                "effective_date": {
                    "value": top_change['effective_date'].strftime('%Y-%m-%d'),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-05T00:00:00+00:00",
                    "period_end": "2026-01-12T00:00:00+00:00"
                }
            },
            "source_names": ["emails"],
            "sample_size": len(price_changes),
            "coverage_notes": [
                "Analysis period: 2026-01-05 to 2026-01-12",
                f"Total price changes identified: {len(price_changes)}",
                "Only changes with old_price, new_price, and effective_date included",
                "No recipe/BOM available; impact on menu items cannot be calculated without ingredient quantities"
            ],
            "assumptions": [
                "Email extraction accurately captured supplier price information",
                "Effective date represents when price change takes effect",
                "Price change applies to specified unit and currency"
            ],
            "confidence": 0.85
        }
        findings.append(finding_3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
