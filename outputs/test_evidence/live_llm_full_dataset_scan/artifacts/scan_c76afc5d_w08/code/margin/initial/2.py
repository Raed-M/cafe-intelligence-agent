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
analysis_start = datetime(2026, 3, 2, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 3, 9, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 2, 23, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 3, 2, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Filter inventory for analysis week
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
inv_analysis = inventory_df[inventory_df['week_starting'] == pd.Timestamp('2026-03-02')]

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from menu and POS
menu_df_clean = menu_df.dropna(subset=['unit_cost_sar'])

# Merge POS with menu to get unit costs
pos_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'price_sar']],
    on='sku',
    how='left'
)

# Filter out refunds for revenue calculation
pos_sales = pos_with_cost[~pos_with_cost['is_refund']].copy()

# Calculate metrics
pos_sales['cogs'] = pos_sales['quantity'] * pos_sales['unit_cost_sar']
pos_sales['gross_profit'] = pos_sales['line_total_sar'] - pos_sales['cogs']

# Group by item
item_economics = pos_sales.groupby('item_name_en').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'cogs': 'sum',
    'gross_profit': 'sum',
    'transaction_id': 'nunique'
}).reset_index()

item_economics['gross_margin_pct'] = (item_economics['gross_profit'] / item_economics['line_total_sar'] * 100).round(2)
item_economics = item_economics.sort_values('gross_profit', ascending=False)

# Top 3 items by gross profit
top_items = item_economics.head(3)

if len(top_items) > 0:
    top_item = top_items.iloc[0]
    finding1 = {
        "title": "Top Gross Profit Item (Week of 2026-03-02)",
        "claim": f"{top_item['item_name_en']} generated the highest gross profit of {top_item['gross_profit']:.2f} SAR with {top_item['gross_margin_pct']:.1f}% margin from {int(top_item['quantity'])} units sold.",
        "finding_type": "item_economics",
        "metrics": {
            "item_name": {
                "value": top_item['item_name_en'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00Z",
                "period_end": "2026-03-09T00:00:00Z"
            },
            "gross_profit_sar": {
                "value": round(top_item['gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_item['gross_profit'], 2),
                "denominator": None,
                "period_start": "2026-03-02T00:00:00Z",
                "period_end": "2026-03-09T00:00:00Z"
            },
            "gross_margin_pct": {
                "value": round(top_item['gross_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_item['gross_margin_pct'], 2),
                "denominator": 100,
                "period_start": "2026-03-02T00:00:00Z",
                "period_end": "2026-03-09T00:00:00Z"
            },
            "units_sold": {
                "value": int(top_item['quantity']),
                "unit": "units",
                "numerator": int(top_item['quantity']),
                "denominator": None,
                "period_start": "2026-03-02T00:00:00Z",
                "period_end": "2026-03-09T00:00:00Z"
            },
            "revenue_sar": {
                "value": round(top_item['line_total_sar'], 2),
                "unit": "SAR",
                "numerator": round(top_item['line_total_sar'], 2),
                "denominator": None,
                "period_start": "2026-03-02T00:00:00Z",
                "period_end": "2026-03-09T00:00:00Z"
            },
            "cogs_sar": {
                "value": round(top_item['cogs'], 2),
                "unit": "SAR",
                "numerator": round(top_item['cogs'], 2),
                "denominator": None,
                "period_start": "2026-03-02T00:00:00Z",
                "period_end": "2026-03-09T00:00:00Z"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(top_item['transaction_id']),
        "coverage_notes": [
            "Analysis period: 2026-03-02 to 2026-03-09",
            "Excludes refunds (is_refund=False)",
            "Unit costs from menu.unit_cost_sar",
            "Revenue from POS line_total_sar"
        ],
        "assumptions": [
            "Menu unit_cost_sar is current and accurate",
            "POS line_total_sar reflects actual revenue after discounts",
            "No recipe/BOM adjustments applied"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# FINDING 2: Waste Cost Impact
# Calculate waste cost from inventory
inv_with_waste = inv_analysis[inv_analysis['known_waste_cost_sar'].notna()].copy()

if len(inv_with_waste) > 0:
    total_waste_cost = inv_with_waste['known_waste_cost_sar'].sum()
    waste_items = inv_with_waste[['item', 'units_wasted', 'known_waste_cost_sar']].copy()
    waste_items = waste_items.sort_values('known_waste_cost_sar', ascending=False)
    
    if len(waste_items) > 0:
        top_waste = waste_items.iloc[0]
        finding2 = {
            "title": "Highest Waste Cost Item (Week of 2026-03-02)",
            "claim": f"{top_waste['item']} incurred {top_waste['known_waste_cost_sar']:.2f} SAR in waste cost from {int(top_waste['units_wasted'])} wasted units.",
            "finding_type": "waste_analysis",
            "metrics": {
                "item_name": {
                    "value": top_waste['item'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-02T00:00:00Z",
                    "period_end": "2026-03-09T00:00:00Z"
                },
                "waste_cost_sar": {
                    "value": round(top_waste['known_waste_cost_sar'], 2),
                    "unit": "SAR",
                    "numerator": round(top_waste['known_waste_cost_sar'], 2),
                    "denominator": None,
                    "period_start": "2026-03-02T00:00:00Z",
                    "period_end": "2026-03-09T00:00:00Z"
                },
                "units_wasted": {
                    "value": int(top_waste['units_wasted']),
                    "unit": "units",
                    "numerator": int(top_waste['units_wasted']),
                    "denominator": None,
                    "period_start": "2026-03-02T00:00:00Z",
                    "period_end": "2026-03-09T00:00:00Z"
                },
                "total_waste_cost_sar": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": round(total_waste_cost, 2),
                    "denominator": None,
                    "period_start": "2026-03-02T00:00:00Z",
                    "period_end": "2026-03-09T00:00:00Z"
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(inv_with_waste),
            "coverage_notes": [
                "Analysis period: 2026-03-02 to 2026-03-09",
                "Only includes items with non-null known_waste_cost_sar",
                "Waste cost from inventory.known_waste_cost_sar"
            ],
            "assumptions": [
                "known_waste_cost_sar accurately reflects waste value",
                "Waste data is complete for reported items"
            ],
            "confidence": 0.85
        }
        findings.append(finding2)

# FINDING 3: Supplier Price Changes
# Extract supplier price changes from emails
emails_clean = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(emails_clean) > 0:
    emails_clean['old_price'] = pd.to_numeric(emails_clean['old_price'], errors='coerce')
    emails_clean['new_price'] = pd.to_numeric(emails_clean['new_price'], errors='coerce')
    emails_clean['effective_date'] = pd.to_datetime(emails_clean['effective_date'])
    
    # Remove rows with NaN after conversion
    emails_clean = emails_clean.dropna(subset=['old_price', 'new_price'])
    
    # Calculate price change percentage
    emails_clean['price_change_pct'] = ((emails_clean['new_price'] - emails_clean['old_price']) / emails_clean['old_price'] * 100).round(2)
    
    # Convert effective_date to timezone-naive UTC for comparison
    emails_clean['effective_date'] = emails_clean['effective_date'].dt.tz_localize(None)
    analysis_end_naive = analysis_end.replace(tzinfo=None)
    
    # Filter for changes during or before analysis period
    emails_clean = emails_clean[emails_clean['effective_date'] <= analysis_end_naive]
    emails_clean = emails_clean.sort_values('price_change_pct', ascending=False)
    
    if len(emails_clean) > 0:
        top_change = emails_clean.iloc[0]
        finding3 = {
            "title": "Largest Supplier Price Change",
            "claim": f"{top_change['entity_or_ingredient']} price changed by {top_change['price_change_pct']:.1f}% (from {top_change['old_price']:.2f} to {top_change['new_price']:.2f} {top_change['currency']}/{top_change['unit']}) effective {top_change['effective_date'].strftime('%Y-%m-%d')}.",
            "finding_type": "supplier_pricing",
            "metrics": {
                "ingredient": {
                    "value": top_change['entity_or_ingredient'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-02T00:00:00Z",
                    "period_end": "2026-03-09T00:00:00Z"
                },
                "old_price": {
                    "value": round(top_change['old_price'], 2),
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": round(top_change['old_price'], 2),
                    "denominator": None,
                    "period_start": "2026-03-02T00:00:00Z",
                    "period_end": "2026-03-09T00:00:00Z"
                },
                "new_price": {
                    "value": round(top_change['new_price'], 2),
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": round(top_change['new_price'], 2),
                    "denominator": None,
                    "period_start": "2026-03-02T00:00:00Z",
                    "period_end": "2026-03-09T00:00:00Z"
                },
                "price_change_pct": {
                    "value": round(top_change['price_change_pct'], 2),
                    "unit": "%",
                    "numerator": round(top_change['price_change_pct'], 2),
                    "denominator": 100,
                    "period_start": "2026-03-02T00:00:00Z",
                    "period_end": "2026-03-09T00:00:00Z"
                },
                "effective_date": {
                    "value": top_change['effective_date'].strftime('%Y-%m-%d'),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-03-02T00:00:00Z",
                    "period_end": "2026-03-09T00:00:00Z"
                }
            },
            "source_names": ["emails"],
            "sample_size": len(emails_clean),
            "coverage_notes": [
                "Analysis period: 2026-03-02 to 2026-03-09",
                "Includes supplier price changes with effective dates on or before analysis end",
                "Price changes extracted from supplier emails"
            ],
            "assumptions": [
                "Email extraction accurately captured price change facts",
                "Effective dates are accurate",
                "Price changes apply to relevant menu items (not verified without recipe/BOM)"
            ],
            "confidence": 0.80
        }
        findings.append(finding3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
