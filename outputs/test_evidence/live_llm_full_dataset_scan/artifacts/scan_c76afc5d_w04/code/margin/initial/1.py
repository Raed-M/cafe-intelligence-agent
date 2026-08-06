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
analysis_start = datetime(2026, 2, 2, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 2, 9, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 1, 26, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 2, 2, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Convert inventory week_starting to datetime
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Get inventory for analysis week
analysis_week = pd.Timestamp('2026-02-02', tz=timezone.utc)
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

item_economics.columns = ['sku', 'item_name', 'total_quantity', 'total_revenue', 'unit_cost_sar', 'menu_price', 'basket_count']

# Calculate COGS and gross profit
item_economics['total_cogs'] = item_economics['total_quantity'] * item_economics['unit_cost_sar']
item_economics['gross_profit'] = item_economics['total_revenue'] - item_economics['total_cogs']
item_economics['gross_margin_pct'] = (item_economics['gross_profit'] / item_economics['total_revenue'] * 100).round(2)

# Sort by gross profit
item_economics_sorted = item_economics.sort_values('gross_profit', ascending=False)

if len(item_economics_sorted) > 0:
    top_item = item_economics_sorted.iloc[0]
    
    finding_1 = {
        "title": "Top Gross Profit Item - Week of Feb 2-9, 2026",
        "claim": f"{top_item['item_name']} generated the highest gross profit of {top_item['gross_profit']:.2f} SAR with {top_item['gross_margin_pct']:.1f}% margin across {int(top_item['basket_count'])} transactions.",
        "finding_type": "item_economics",
        "metrics": {
            "item_name": {
                "value": top_item['item_name'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+00:00",
                "period_end": "2026-02-09T00:00:00+00:00"
            },
            "total_revenue": {
                "value": round(top_item['total_revenue'], 2),
                "unit": "SAR",
                "numerator": round(top_item['total_revenue'], 2),
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+00:00",
                "period_end": "2026-02-09T00:00:00+00:00"
            },
            "total_cogs": {
                "value": round(top_item['total_cogs'], 2),
                "unit": "SAR",
                "numerator": round(top_item['total_cogs'], 2),
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+00:00",
                "period_end": "2026-02-09T00:00:00+00:00"
            },
            "gross_profit": {
                "value": round(top_item['gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_item['gross_profit'], 2),
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+00:00",
                "period_end": "2026-02-09T00:00:00+00:00"
            },
            "gross_margin_pct": {
                "value": round(top_item['gross_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_item['gross_profit'], 2),
                "denominator": round(top_item['total_revenue'], 2),
                "period_start": "2026-02-02T00:00:00+00:00",
                "period_end": "2026-02-09T00:00:00+00:00"
            },
            "total_quantity": {
                "value": int(top_item['total_quantity']),
                "unit": "units",
                "numerator": int(top_item['total_quantity']),
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+00:00",
                "period_end": "2026-02-09T00:00:00+00:00"
            },
            "basket_count": {
                "value": int(top_item['basket_count']),
                "unit": "transactions",
                "numerator": int(top_item['basket_count']),
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+00:00",
                "period_end": "2026-02-09T00:00:00+00:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(pos_sales[pos_sales['sku'] == top_item['sku']].shape[0]),
        "coverage_notes": [
            "Analysis period: 2026-02-02 to 2026-02-09",
            "Excludes refunds (is_refund=False)",
            "Unit costs from menu.unit_cost_sar",
            "Revenue from POS line_total_sar",
            f"Total items analyzed: {len(item_economics_sorted)}"
        ],
        "assumptions": [
            "Menu unit_cost_sar is current and applicable to analysis period",
            "POS line_total_sar reflects actual revenue after discounts",
            "No recipe/BOM adjustments applied"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# FINDING 2: Waste Cost Impact Analysis
# Calculate waste costs from inventory data
inventory_analysis['waste_cost_total'] = inventory_analysis['known_waste_cost_sar'].fillna(0)

waste_by_item = inventory_analysis[inventory_analysis['waste_cost_total'] > 0].copy()
waste_by_item = waste_by_item.sort_values('waste_cost_total', ascending=False)

if len(waste_by_item) > 0:
    top_waste_item = waste_by_item.iloc[0]
    
    finding_2 = {
        "title": "Highest Waste Cost Item - Week of Feb 2-9, 2026",
        "claim": f"{top_waste_item['item']} incurred {top_waste_item['waste_cost_total']:.2f} SAR in waste cost with {int(top_waste_item['units_wasted'])} units wasted.",
        "finding_type": "waste_analysis",
        "metrics": {
            "item_name": {
                "value": top_waste_item['item'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+00:00",
                "period_end": "2026-02-09T00:00:00+00:00"
            },
            "waste_cost_sar": {
                "value": round(top_waste_item['waste_cost_total'], 2),
                "unit": "SAR",
                "numerator": round(top_waste_item['waste_cost_total'], 2),
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+00:00",
                "period_end": "2026-02-09T00:00:00+00:00"
            },
            "units_wasted": {
                "value": int(top_waste_item['units_wasted']),
                "unit": "units",
                "numerator": int(top_waste_item['units_wasted']),
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+00:00",
                "period_end": "2026-02-09T00:00:00+00:00"
            },
            "units_sold": {
                "value": int(top_waste_item['units_sold']),
                "unit": "units",
                "numerator": int(top_waste_item['units_sold']),
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+00:00",
                "period_end": "2026-02-09T00:00:00+00:00"
            },
            "waste_rate_pct": {
                "value": round(top_waste_item['units_wasted'] / (top_waste_item['units_wasted'] + top_waste_item['units_sold']) * 100, 2) if (top_waste_item['units_wasted'] + top_waste_item['units_sold']) > 0 else 0,
                "unit": "%",
                "numerator": int(top_waste_item['units_wasted']),
                "denominator": int(top_waste_item['units_wasted'] + top_waste_item['units_sold']),
                "period_start": "2026-02-02T00:00:00+00:00",
                "period_end": "2026-02-09T00:00:00+00:00"
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(waste_by_item),
        "coverage_notes": [
            "Analysis period: 2026-02-02 to 2026-02-09",
            "Only items with non-null known_waste_cost_sar included",
            f"Total items with waste: {len(waste_by_item)}",
            "Waste cost from inventory.known_waste_cost_sar"
        ],
        "assumptions": [
            "known_waste_cost_sar reflects actual waste cost",
            "Waste data is complete for the analysis period"
        ],
        "confidence": 0.90
    }
    findings.append(finding_2)

# FINDING 3: Supplier Price Changes from Email Evidence
# Filter emails with price changes
price_change_emails = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['old_price'] != emails_df['new_price'])
].copy()

if len(price_change_emails) > 0:
    # Get the most recent price change
    price_change_emails['date'] = pd.to_datetime(price_change_emails['date'])
    price_change_emails = price_change_emails.sort_values('date', ascending=False)
    
    latest_change = price_change_emails.iloc[0]
    
    # Calculate percentage change
    old_price = float(latest_change['old_price'])
    new_price = float(latest_change['new_price'])
    pct_change = ((new_price - old_price) / old_price * 100) if old_price != 0 else 0
    
    finding_3 = {
        "title": "Supplier Price Change - Latest Update",
        "claim": f"{latest_change['entity_or_ingredient']} price changed from {old_price} to {new_price} {latest_change['currency']} per {latest_change['unit']} (effective {latest_change['effective_date']}) - a {pct_change:.1f}% change.",
        "finding_type": "supplier_pricing",
        "metrics": {
            "entity_or_ingredient": {
                "value": latest_change['entity_or_ingredient'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+00:00",
                "period_end": "2026-02-09T00:00:00+00:00"
            },
            "old_price": {
                "value": round(old_price, 2),
                "unit": latest_change['currency'],
                "numerator": round(old_price, 2),
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+00:00",
                "period_end": "2026-02-09T00:00:00+00:00"
            },
            "new_price": {
                "value": round(new_price, 2),
                "unit": latest_change['currency'],
                "numerator": round(new_price, 2),
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+00:00",
                "period_end": "2026-02-09T00:00:00+00:00"
            },
            "price_change_pct": {
                "value": round(pct_change, 2),
                "unit": "%",
                "numerator": round(new_price - old_price, 2),
                "denominator": round(old_price, 2),
                "period_start": "2026-02-02T00:00:00+00:00",
                "period_end": "2026-02-09T00:00:00+00:00"
            },
            "unit": {
                "value": latest_change['unit'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+00:00",
                "period_end": "2026-02-09T00:00:00+00:00"
            },
            "effective_date": {
                "value": str(latest_change['effective_date']),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-02-02T00:00:00+00:00",
                "period_end": "2026-02-09T00:00:00+00:00"
            }
        },
        "source_names": ["emails"],
        "sample_size": len(price_change_emails),
        "coverage_notes": [
            "Analysis period: 2026-02-02 to 2026-02-09",
            f"Total supplier price changes identified: {len(price_change_emails)}",
            "Latest change shown",
            "Email extraction confidence varies by source"
        ],
        "assumptions": [
            "Email extraction accurately captured price and unit information",
            "Effective date reflects actual implementation date",
            "Price applies to relevant menu items (not verified without recipe/BOM)"
        ],
        "confidence": float(latest_change['confidence']) if pd.notna(latest_change['confidence']) else 0.75
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
