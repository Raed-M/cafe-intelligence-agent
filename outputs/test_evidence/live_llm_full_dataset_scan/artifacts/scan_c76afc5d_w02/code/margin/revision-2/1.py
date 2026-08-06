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

# Load artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Define analysis period
analysis_start = "2026-01-19"
analysis_end = "2026-01-26"

# Convert timestamp columns to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
emails_df['date'] = pd.to_datetime(emails_df['date'])
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'])

# Filter POS data for analysis period
pos_analysis = pos_df[
    (pos_df['calendar_date'] >= analysis_start) & 
    (pos_df['calendar_date'] < analysis_end)
].copy()

# Filter inventory for analysis period
inventory_analysis = inventory_df[
    (inventory_df['week_starting'] >= analysis_start) & 
    (inventory_df['week_starting'] < analysis_end)
].copy()

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from POS and menu data

# Merge POS with menu to get unit costs
pos_with_costs = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'price_sar']],
    on='sku',
    how='left'
)

# Filter out refunds for revenue calculation
pos_sales = pos_with_costs[pos_with_costs['is_refund'] == False].copy()

# Calculate metrics by item
item_economics = pos_sales.groupby(['sku', 'item_name']).agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'unit_cost_sar': 'first',
    'transaction_id': 'nunique'
}).reset_index()

item_economics.columns = ['sku', 'item_name', 'total_quantity_sold', 'gross_revenue', 'unit_cost_sar', 'basket_count']

# Calculate COGS and gross profit
item_economics['total_cogs'] = item_economics['total_quantity_sold'] * item_economics['unit_cost_sar']
item_economics['gross_profit'] = item_economics['gross_revenue'] - item_economics['total_cogs']
item_economics['gross_margin_pct'] = (item_economics['gross_profit'] / item_economics['gross_revenue'] * 100).round(2)

# Find top contributor by gross profit
top_item = item_economics.loc[item_economics['gross_profit'].idxmax()]

if len(item_economics) > 0 and top_item['gross_profit'] > 0:
    finding1 = {
        "title": "Top Gross Profit Contributor - Item Economics",
        "claim": f"{top_item['item_name']} generated {top_item['gross_profit']:.2f} SAR gross profit from {int(top_item['total_quantity_sold'])} units sold across {int(top_item['basket_count'])} transactions, with {top_item['gross_margin_pct']:.2f}% gross margin.",
        "finding_type": "item_economics",
        "metrics": {
            "item_name": {
                "value": top_item['item_name'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_quantity_sold": {
                "value": int(top_item['total_quantity_sold']),
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "gross_revenue": {
                "value": round(top_item['gross_revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_cogs": {
                "value": round(top_item['total_cogs'], 2),
                "unit": "SAR",
                "numerator": round(top_item['unit_cost_sar'], 2),
                "denominator": int(top_item['total_quantity_sold']),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "gross_profit": {
                "value": round(top_item['gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_item['gross_revenue'], 2),
                "denominator": round(top_item['total_cogs'], 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "gross_margin_pct": {
                "value": round(top_item['gross_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_item['gross_profit'], 2),
                "denominator": round(top_item['gross_revenue'], 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "basket_count": {
                "value": int(top_item['basket_count']),
                "unit": "transactions",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(item_economics),
        "coverage_notes": [
            f"Analysis covers {len(pos_sales)} POS line items from {pos_sales['transaction_id'].nunique()} transactions during {analysis_start} to {analysis_end}",
            "Refunds excluded from revenue calculation",
            "Unit costs sourced from menu.parquet",
            f"Top item identified from {len(item_economics)} distinct SKUs with sales"
        ],
        "assumptions": [
            "Menu unit_cost_sar represents actual COGS per unit",
            "POS line_total_sar is accurate and consistent",
            "No recipe/BOM adjustments applied"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# FINDING 2: Waste Cost Impact Analysis
# Calculate waste cost for items with recorded waste

waste_items = inventory_analysis[inventory_analysis['units_wasted'].notna() & (inventory_analysis['units_wasted'] > 0)].copy()

if len(waste_items) > 0:
    waste_items['waste_cost_sar'] = waste_items['units_wasted'] * waste_items['unit_cost_sar']
    
    total_waste_units = waste_items['units_wasted'].sum()
    total_waste_cost = waste_items['waste_cost_sar'].sum()
    total_cogs_waste_items = waste_items['units_sold'] * waste_items['unit_cost_sar']
    total_cogs_waste_items = total_cogs_waste_items.sum()
    
    # Calculate waste as percentage of total product cost (COGS + waste)
    total_product_cost = total_cogs_waste_items + total_waste_cost
    waste_pct = (total_waste_cost / total_product_cost * 100) if total_product_cost > 0 else 0
    
    finding2 = {
        "title": "Waste Cost Impact - Items with Recorded Waste",
        "claim": f"Waste cost of {total_waste_cost:.2f} SAR from {int(total_waste_units)} units across {len(waste_items)} SKUs with recorded waste represents {waste_pct:.2f}% of total product cost (COGS + waste) for these waste-bearing items only during the week of {analysis_start} to {analysis_end}. This percentage does not represent the cafe's overall waste impact, as it excludes SKUs with zero recorded waste and items without waste tracking.",
        "finding_type": "waste_analysis",
        "metrics": {
            "total_waste_units": {
                "value": int(total_waste_units),
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": round(total_waste_cost, 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_cogs_waste_items": {
                "value": round(total_cogs_waste_items, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_product_cost": {
                "value": round(total_product_cost, 2),
                "unit": "SAR",
                "numerator": round(total_cogs_waste_items, 2),
                "denominator": round(total_waste_cost, 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "waste_as_pct_of_total_product_cost": {
                "value": round(waste_pct, 2),
                "unit": "%",
                "numerator": round(total_waste_cost, 2),
                "denominator": round(total_product_cost, 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "num_skus_with_waste": {
                "value": len(waste_items),
                "unit": "SKUs",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(waste_items),
        "coverage_notes": [
            f"Analysis covers {len(waste_items)} SKUs with recorded waste during {analysis_start} to {analysis_end}",
            "Only items with non-null and positive units_wasted values included",
            "Unit costs sourced from inventory.parquet",
            "Waste cost calculated as units_wasted × unit_cost_sar"
        ],
        "assumptions": [
            "Null waste values represent items with no waste tracking or zero waste (not missing data)",
            "Unit costs in inventory match actual product cost",
            "Waste units are accurately recorded"
        ],
        "confidence": 0.85
    }
    findings.append(finding2)

# FINDING 3: Supplier Price Changes from Email Evidence
# Extract and analyze supplier price changes

price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    # Calculate percentage change
    price_changes['price_change_pct'] = (
        ((price_changes['new_price'] - price_changes['old_price']) / price_changes['old_price'] * 100)
    ).round(2)
    
    # Filter for changes effective during or after analysis period
    relevant_changes = price_changes[
        price_changes['effective_date'] >= pd.to_datetime(analysis_start)
    ].copy()
    
    if len(relevant_changes) > 0:
        # Get the most significant change
        relevant_changes['abs_pct_change'] = relevant_changes['price_change_pct'].abs()
        top_change = relevant_changes.loc[relevant_changes['abs_pct_change'].idxmax()]
        
        finding3 = {
            "title": "Supplier Price Change - Procurement Cost Pressure",
            "claim": f"{top_change['entity_or_ingredient']} price changed from {top_change['old_price']:.2f} to {top_change['new_price']:.2f} {top_change['currency']} per {top_change['unit']} (effective {top_change['effective_date'].strftime('%Y-%m-%d')}), representing a {top_change['price_change_pct']:.2f}% change. This is a dated supplier price change with no standing order quantity data available; actual procurement cost impact depends on order volume and payment terms.",
            "finding_type": "supplier_pricing",
            "metrics": {
                "entity_or_ingredient": {
                    "value": top_change['entity_or_ingredient'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "old_price": {
                    "value": round(top_change['old_price'], 2),
                    "unit": f"{top_change['currency']} per {top_change['unit']}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": None,
                    "period_end": None
                },
                "new_price": {
                    "value": round(top_change['new_price'], 2),
                    "unit": f"{top_change['currency']} per {top_change['unit']}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": None,
                    "period_end": None
                },
                "price_change_pct": {
                    "value": round(top_change['price_change_pct'], 2),
                    "unit": "%",
                    "numerator": round(top_change['new_price'] - top_change['old_price'], 2),
                    "denominator": round(top_change['old_price'], 2),
                    "period_start": None,
                    "period_end": None
                },
                "effective_date": {
                    "value": top_change['effective_date'].strftime('%Y-%m-%d'),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": None,
                    "period_end": None
                },
                "currency": {
                    "value": top_change['currency'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": None,
                    "period_end": None
                },
                "unit": {
                    "value": top_change['unit'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": None,
                    "period_end": None
                }
            },
            "source_names": ["emails"],
            "sample_size": len(relevant_changes),
            "coverage_notes": [
                f"Analysis identified {len(relevant_changes)} supplier price changes effective on or after {analysis_start}",
                "Top change selected by absolute percentage magnitude",
                "No standing order quantities available in email data",
                "No recipe/BOM data available to calculate per-drink impact"
            ],
            "assumptions": [
                "Email extraction accurately captured old_price, new_price, and effective_date",
                "Price change applies to cafe's procurement (not verified against actual orders)",
                "No standing order volume or payment terms data available",
                "Actual cost impact requires order volume and timing information not present in current data"
            ],
            "confidence": 0.70
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

print(f"Analysis complete. {len(findings)} findings generated.")
