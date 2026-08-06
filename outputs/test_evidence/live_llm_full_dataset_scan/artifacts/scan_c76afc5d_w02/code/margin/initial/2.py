import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any

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
analysis_start = datetime(2026, 1, 19, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 1, 26, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp columns to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
emails_df['date'] = pd.to_datetime(emails_df['date'])
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'])

# Ensure all datetime comparisons use UTC timezone-aware datetimes
# Convert naive datetimes to UTC-aware if needed
if pos_df['timestamp'].dt.tz is None:
    pos_df['timestamp'] = pos_df['timestamp'].dt.tz_localize('UTC')
else:
    pos_df['timestamp'] = pos_df['timestamp'].dt.tz_convert('UTC')

if inventory_df['week_starting'].dt.tz is None:
    inventory_df['week_starting'] = inventory_df['week_starting'].dt.tz_localize('UTC')
else:
    inventory_df['week_starting'] = inventory_df['week_starting'].dt.tz_convert('UTC')

if emails_df['effective_date'].dt.tz is None:
    emails_df['effective_date'] = emails_df['effective_date'].dt.tz_localize('UTC')
else:
    emails_df['effective_date'] = emails_df['effective_date'].dt.tz_convert('UTC')

# Filter POS data for analysis period
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start) & 
    (pos_df['timestamp'] < analysis_end) &
    (pos_df['is_refund'] == False)
].copy()

# Filter inventory for analysis period
inventory_analysis = inventory_df[
    inventory_df['week_starting'] == pd.Timestamp('2026-01-19', tz='UTC')
].copy()

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from POS and menu data
item_economics = pos_analysis.groupby('sku').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'discount_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()

item_economics.columns = ['sku', 'total_quantity', 'gross_revenue', 'total_discount', 'basket_count']

# Merge with menu to get unit costs
item_economics = item_economics.merge(
    menu_df[['sku', 'item_en', 'unit_cost_sar', 'price_sar']],
    on='sku',
    how='left'
)

# Calculate net revenue and COGS
item_economics['net_revenue'] = item_economics['gross_revenue'] - item_economics['total_discount']
item_economics['total_cogs'] = item_economics['total_quantity'] * item_economics['unit_cost_sar']
item_economics['gross_profit'] = item_economics['net_revenue'] - item_economics['total_cogs']
item_economics['gross_margin_pct'] = (
    item_economics['gross_profit'] / item_economics['net_revenue'] * 100
).round(2)

# Filter for items with significant volume
significant_items = item_economics[item_economics['total_quantity'] >= 10].copy()
significant_items = significant_items.sort_values('gross_profit', ascending=False)

if len(significant_items) > 0:
    top_item = significant_items.iloc[0]
    
    finding_1 = {
        "title": "Top Profit-Contributing Item (Analysis Week)",
        "claim": f"{top_item['item_en']} generated the highest gross profit of {top_item['gross_profit']:.2f} SAR with {top_item['gross_margin_pct']:.1f}% margin from {int(top_item['total_quantity'])} units sold across {int(top_item['basket_count'])} transactions.",
        "finding_type": "item_economics",
        "metrics": {
            "item_name": {
                "value": top_item['item_en'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-19T00:00:00+00:00",
                "period_end": "2026-01-26T00:00:00+00:00"
            },
            "total_quantity_sold": {
                "value": int(top_item['total_quantity']),
                "unit": "units",
                "numerator": int(top_item['total_quantity']),
                "denominator": None,
                "period_start": "2026-01-19T00:00:00+00:00",
                "period_end": "2026-01-26T00:00:00+00:00"
            },
            "net_revenue": {
                "value": round(top_item['net_revenue'], 2),
                "unit": "SAR",
                "numerator": round(top_item['gross_revenue'], 2),
                "denominator": round(top_item['total_discount'], 2),
                "period_start": "2026-01-19T00:00:00+00:00",
                "period_end": "2026-01-26T00:00:00+00:00"
            },
            "total_cogs": {
                "value": round(top_item['total_cogs'], 2),
                "unit": "SAR",
                "numerator": int(top_item['total_quantity']),
                "denominator": round(top_item['unit_cost_sar'], 2),
                "period_start": "2026-01-19T00:00:00+00:00",
                "period_end": "2026-01-26T00:00:00+00:00"
            },
            "gross_profit": {
                "value": round(top_item['gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_item['net_revenue'], 2),
                "denominator": round(top_item['total_cogs'], 2),
                "period_start": "2026-01-19T00:00:00+00:00",
                "period_end": "2026-01-26T00:00:00+00:00"
            },
            "gross_margin_percentage": {
                "value": round(top_item['gross_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_item['gross_profit'], 2),
                "denominator": round(top_item['net_revenue'], 2),
                "period_start": "2026-01-19T00:00:00+00:00",
                "period_end": "2026-01-26T00:00:00+00:00"
            },
            "basket_count": {
                "value": int(top_item['basket_count']),
                "unit": "transactions",
                "numerator": int(top_item['basket_count']),
                "denominator": None,
                "period_start": "2026-01-19T00:00:00+00:00",
                "period_end": "2026-01-26T00:00:00+00:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(top_item['basket_count']),
        "coverage_notes": [
            "Analysis covers POS transactions from 2026-01-19 to 2026-01-26",
            "Excludes refunds (is_refund=False)",
            "Unit costs sourced from menu.parquet",
            "Net revenue calculated as gross_revenue - discount_sar"
        ],
        "assumptions": [
            "Menu unit_cost_sar is current and accurate for the analysis period",
            "All POS line items have matching SKU in menu",
            "Discount amounts are correctly recorded in POS"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# FINDING 2: Waste Cost Impact Analysis
# Calculate waste costs from inventory data
waste_analysis = inventory_analysis[inventory_analysis['units_wasted'].notna()].copy()
waste_analysis['waste_cost'] = waste_analysis['units_wasted'] * waste_analysis['unit_cost_sar']

if len(waste_analysis) > 0:
    total_waste_units = waste_analysis['units_wasted'].sum()
    total_waste_cost = waste_analysis['waste_cost'].sum()
    
    # Calculate as percentage of total COGS for that week
    total_cogs_week = (waste_analysis['units_sold'] * waste_analysis['unit_cost_sar']).sum()
    waste_pct_of_cogs = (total_waste_cost / (total_cogs_week + total_waste_cost) * 100) if (total_cogs_week + total_waste_cost) > 0 else 0
    
    finding_2 = {
        "title": "Quantified Waste Cost Impact (Week of 2026-01-19)",
        "claim": f"Documented waste totaled {int(total_waste_units)} units costing {total_waste_cost:.2f} SAR, representing {waste_pct_of_cogs:.2f}% of total product cost (COGS + waste) for items with recorded waste.",
        "finding_type": "waste_cost",
        "metrics": {
            "total_waste_units": {
                "value": int(total_waste_units),
                "unit": "units",
                "numerator": int(total_waste_units),
                "denominator": None,
                "period_start": "2026-01-19T00:00:00+00:00",
                "period_end": "2026-01-26T00:00:00+00:00"
            },
            "total_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": int(total_waste_units),
                "denominator": round(waste_analysis['unit_cost_sar'].mean(), 2),
                "period_start": "2026-01-19T00:00:00+00:00",
                "period_end": "2026-01-26T00:00:00+00:00"
            },
            "waste_as_pct_of_total_product_cost": {
                "value": round(waste_pct_of_cogs, 2),
                "unit": "%",
                "numerator": round(total_waste_cost, 2),
                "denominator": round(total_cogs_week + total_waste_cost, 2),
                "period_start": "2026-01-19T00:00:00+00:00",
                "period_end": "2026-01-26T00:00:00+00:00"
            },
            "items_with_waste_recorded": {
                "value": len(waste_analysis),
                "unit": "SKUs",
                "numerator": len(waste_analysis),
                "denominator": None,
                "period_start": "2026-01-19T00:00:00+00:00",
                "period_end": "2026-01-26T00:00:00+00:00"
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(waste_analysis),
        "coverage_notes": [
            "Analysis covers inventory week starting 2026-01-19",
            "Only includes items with non-null units_wasted values",
            f"{len(waste_analysis)} SKUs had documented waste",
            "Waste cost calculated as units_wasted × unit_cost_sar"
        ],
        "assumptions": [
            "Unit costs in inventory are accurate for waste valuation",
            "Waste units are correctly recorded",
            "Null waste values represent items with no waste (not missing data)"
        ],
        "confidence": 0.85
    }
    findings.append(finding_2)

# FINDING 3: Supplier Price Changes and Procurement Impact
# Identify price changes from emails
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    # Filter for changes effective during or before analysis period
    price_changes = price_changes[
        price_changes['effective_date'] <= analysis_end
    ].copy()
    
    if len(price_changes) > 0:
        # Calculate percentage change
        price_changes['pct_change'] = (
            (price_changes['new_price'] - price_changes['old_price']) / 
            price_changes['old_price'] * 100
        ).round(2)
        
        # Sort by absolute percentage change
        price_changes['abs_pct_change'] = price_changes['pct_change'].abs()
        price_changes = price_changes.sort_values('abs_pct_change', ascending=False)
        
        top_change = price_changes.iloc[0]
        
        finding_3 = {
            "title": "Significant Supplier Price Change Detected",
            "claim": f"Supplier price for {top_change['entity_or_ingredient']} changed from {top_change['old_price']:.2f} to {top_change['new_price']:.2f} {top_change['currency']} per {top_change['unit']} (effective {top_change['effective_date'].strftime('%Y-%m-%d')}), representing a {top_change['pct_change']:.2f}% change. This is a documented supplier communication with confidence {top_change['confidence']}.",
            "finding_type": "supplier_price_change",
            "metrics": {
                "ingredient_or_entity": {
                    "value": top_change['entity_or_ingredient'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-19T00:00:00+00:00",
                    "period_end": "2026-01-26T00:00:00+00:00"
                },
                "old_price": {
                    "value": round(top_change['old_price'], 2),
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": round(top_change['old_price'], 2),
                    "denominator": None,
                    "period_start": "2026-01-19T00:00:00+00:00",
                    "period_end": "2026-01-26T00:00:00+00:00"
                },
                "new_price": {
                    "value": round(top_change['new_price'], 2),
                    "unit": f"{top_change['currency']}/{top_change['unit']}",
                    "numerator": round(top_change['new_price'], 2),
                    "denominator": None,
                    "period_start": "2026-01-19T00:00:00+00:00",
                    "period_end": "2026-01-26T00:00:00+00:00"
                },
                "percentage_change": {
                    "value": round(top_change['pct_change'], 2),
                    "unit": "%",
                    "numerator": round(top_change['new_price'] - top_change['old_price'], 2),
                    "denominator": round(top_change['old_price'], 2),
                    "period_start": "2026-01-19T00:00:00+00:00",
                    "period_end": "2026-01-26T00:00:00+00:00"
                },
                "effective_date": {
                    "value": top_change['effective_date'].strftime('%Y-%m-%d'),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-19T00:00:00+00:00",
                    "period_end": "2026-01-26T00:00:00+00:00"
                },
                "extraction_confidence": {
                    "value": round(top_change['confidence'], 2),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-01-19T00:00:00+00:00",
                    "period_end": "2026-01-26T00:00:00+00:00"
                }
            },
            "source_names": ["emails"],
            "sample_size": 1,
            "coverage_notes": [
                "Price change extracted from supplier email",
                "Effective date is from email extraction",
                "No recipe/BOM available to calculate per-drink impact",
                "Price change is documented supplier communication, not estimated"
            ],
            "assumptions": [
                "Email extraction confidence score is accurate",
                "Price change applies to the stated ingredient/entity only",
                "Standing order quantities and payment terms are not confirmed in available data"
            ],
            "confidence": round(top_change['confidence'], 2)
        }
        findings.append(finding_3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
