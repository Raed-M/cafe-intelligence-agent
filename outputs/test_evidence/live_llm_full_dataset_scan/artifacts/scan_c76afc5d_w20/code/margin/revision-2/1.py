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

# Define analysis periods
analysis_start = "2026-05-25T00:00:00+03:00"
analysis_end = "2026-06-01T00:00:00+03:00"
previous_start = "2026-05-18T00:00:00+03:00"
previous_end = "2026-05-25T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)
previous_start_dt = pd.to_datetime(previous_start)
previous_end_dt = pd.to_datetime(previous_end)

# Convert POS timestamp to datetime
pos_df['timestamp_dt'] = pd.to_datetime(pos_df['timestamp'])

# Filter POS data for analysis period
pos_analysis = pos_df[(pos_df['timestamp_dt'] >= analysis_start_dt) & 
                      (pos_df['timestamp_dt'] < analysis_end_dt)].copy()

# Filter inventory data for analysis week
inventory_df['week_starting_dt'] = pd.to_datetime(inventory_df['week_starting'])
inventory_analysis = inventory_df[inventory_df['week_starting_dt'] == pd.to_datetime('2026-05-25')].copy()

# Initialize findings list
findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item-level economics from POS and menu
pos_analysis['line_total_net'] = pos_analysis['line_total_sar']
pos_analysis['revenue_net'] = pos_analysis['quantity'] * pos_analysis['unit_price_sar'] - pos_analysis['discount_sar']

# Merge with menu to get unit costs
pos_with_cost = pos_analysis.merge(menu_df[['sku', 'unit_cost_sar']], on='sku', how='left')

# Calculate COGS and gross profit
pos_with_cost['cogs'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
pos_with_cost['gross_profit'] = pos_with_cost['revenue_net'] - pos_with_cost['cogs']
pos_with_cost['gross_margin_pct'] = (pos_with_cost['gross_profit'] / pos_with_cost['revenue_net'] * 100).fillna(0)

# Aggregate by item
item_economics = pos_with_cost.groupby(['sku', 'item_name_en']).agg({
    'quantity': 'sum',
    'revenue_net': 'sum',
    'cogs': 'sum',
    'gross_profit': 'sum',
    'transaction_id': 'nunique'
}).reset_index()

item_economics['gross_margin_pct'] = (item_economics['gross_profit'] / item_economics['revenue_net'] * 100).round(2)
item_economics = item_economics.sort_values('gross_profit', ascending=False)

# Find top performer by gross profit
if len(item_economics) > 0:
    top_item = item_economics.iloc[0]
    
    finding_1 = {
        "title": "Top Gross Profit Item (Analysis Week)",
        "claim": f"Item {top_item['item_name_en']} (SKU: {top_item['sku']}) generated the highest gross profit of {top_item['gross_profit']:.2f} SAR with {int(top_item['quantity'])} units sold and {top_item['gross_margin_pct']:.1f}% gross margin.",
        "finding_type": "item_economics",
        "metrics": {
            "item_name": {
                "value": top_item['item_name_en'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "sku": {
                "value": top_item['sku'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "gross_profit_sar": {
                "value": round(top_item['gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_item['gross_profit'], 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "units_sold": {
                "value": int(top_item['quantity']),
                "unit": "units",
                "numerator": int(top_item['quantity']),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "gross_margin_pct": {
                "value": round(top_item['gross_margin_pct'], 1),
                "unit": "%",
                "numerator": round(top_item['gross_margin_pct'], 1),
                "denominator": 100,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "revenue_net_sar": {
                "value": round(top_item['revenue_net'], 2),
                "unit": "SAR",
                "numerator": round(top_item['revenue_net'], 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "cogs_sar": {
                "value": round(top_item['cogs'], 2),
                "unit": "SAR",
                "numerator": round(top_item['cogs'], 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "baskets": {
                "value": int(top_item['transaction_id']),
                "unit": "transactions",
                "numerator": int(top_item['transaction_id']),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(pos_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"POS records in analysis period: {len(pos_analysis)}",
            f"Items with complete cost data: {len(item_economics)}",
            "Refunds included in net calculations per metric definition",
            "Unit costs sourced from menu.unit_cost_sar"
        ],
        "assumptions": [
            "Menu unit_cost_sar values are current and accurate for the analysis period",
            "POS line_total_sar reflects actual transaction amounts after discounts",
            "No recipe/BOM adjustments applied; unit costs are as-stated in menu"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# FINDING 2: Known Waste Cost Analysis
# Filter inventory for items with known waste cost
waste_items = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna() & 
                                 (inventory_analysis['known_waste_cost_sar'] > 0)].copy()

if len(waste_items) > 0:
    waste_items = waste_items.sort_values('known_waste_cost_sar', ascending=False)
    
    # Create waste cost summary
    total_waste_cost = waste_items['known_waste_cost_sar'].sum()
    waste_count = len(waste_items)
    
    # Get all waste items for comparison
    waste_summary = waste_items[['sku', 'item', 'units_wasted', 'known_waste_cost_sar']].copy()
    waste_summary = waste_summary.sort_values('known_waste_cost_sar', ascending=False)
    
    # Build waste cost list for metrics
    waste_metrics = {}
    for idx, (_, row) in enumerate(waste_summary.iterrows()):
        waste_metrics[f"item_{idx+1}_name"] = {
            "value": row['item'],
            "unit": None,
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        }
        waste_metrics[f"item_{idx+1}_sku"] = {
            "value": row['sku'],
            "unit": None,
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        }
        waste_metrics[f"item_{idx+1}_units_wasted"] = {
            "value": int(row['units_wasted']) if pd.notna(row['units_wasted']) else 0,
            "unit": "units",
            "numerator": int(row['units_wasted']) if pd.notna(row['units_wasted']) else 0,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        }
        waste_metrics[f"item_{idx+1}_waste_cost_sar"] = {
            "value": round(row['known_waste_cost_sar'], 2),
            "unit": "SAR",
            "numerator": round(row['known_waste_cost_sar'], 2),
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    
    waste_metrics["total_waste_cost_sar"] = {
        "value": round(total_waste_cost, 2),
        "unit": "SAR",
        "numerator": round(total_waste_cost, 2),
        "denominator": None,
        "period_start": analysis_start,
        "period_end": analysis_end
    }
    waste_metrics["items_with_known_waste"] = {
        "value": waste_count,
        "unit": "count",
        "numerator": waste_count,
        "denominator": None,
        "period_start": analysis_start,
        "period_end": analysis_end
    }
    
    # Build waste items list for claim
    waste_items_list = []
    for _, row in waste_summary.iterrows():
        waste_items_list.append(f"{row['item']} ({int(row['units_wasted'])} units, {row['known_waste_cost_sar']:.2f} SAR)")
    
    finding_2 = {
        "title": "Known Waste Cost Breakdown (Analysis Week)",
        "claim": f"Total known waste cost across {waste_count} items: {total_waste_cost:.2f} SAR. Items with waste: {'; '.join(waste_items_list)}",
        "finding_type": "waste_cost",
        "metrics": waste_metrics,
        "source_names": ["inventory"],
        "sample_size": waste_count,
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Week starting: 2026-05-25",
            f"Items with known waste cost (non-null, > 0): {waste_count}",
            "Blank waste values excluded per data quality rules",
            "Waste costs calculated from inventory.known_waste_cost_sar"
        ],
        "assumptions": [
            "Waste costs are accurate and based on unit_cost_sar from inventory",
            "Only non-null waste cost values included in analysis",
            "Waste cost = units_wasted × unit_cost_sar"
        ],
        "confidence": 0.90
    }
    findings.append(finding_2)

# FINDING 3: Supplier Price Changes from Email Evidence
# Filter emails for price changes with valid dates
price_change_emails = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_change_emails) > 0:
    # Convert dates
    price_change_emails['effective_date_dt'] = pd.to_datetime(price_change_emails['effective_date'])
    price_change_emails['email_date_dt'] = pd.to_datetime(price_change_emails['date'])
    
    # Calculate percentage change
    price_change_emails['pct_change'] = (
        (price_change_emails['new_price'] - price_change_emails['old_price']) / 
        price_change_emails['old_price'] * 100
    ).round(2)
    
    # Sort by absolute percentage change
    price_change_emails['abs_pct_change'] = price_change_emails['pct_change'].abs()
    price_change_emails = price_change_emails.sort_values('abs_pct_change', ascending=False)
    
    # Get top price change
    if len(price_change_emails) > 0:
        top_change = price_change_emails.iloc[0]
        
        # Build metrics for price change
        price_metrics = {
            "supplier": {
                "value": top_change['sender'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "ingredient": {
                "value": top_change['entity_or_ingredient'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "old_price": {
                "value": round(top_change['old_price'], 2),
                "unit": top_change['currency'],
                "numerator": round(top_change['old_price'], 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "new_price": {
                "value": round(top_change['new_price'], 2),
                "unit": top_change['currency'],
                "numerator": round(top_change['new_price'], 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "unit": {
                "value": top_change['unit'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "percentage_change": {
                "value": top_change['pct_change'],
                "unit": "%",
                "numerator": top_change['pct_change'],
                "denominator": 100,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "effective_date": {
                "value": str(top_change['effective_date']),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "email_date": {
                "value": str(top_change['date']),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        }
        
        direction = "increase" if top_change['pct_change'] > 0 else "decrease"
        
        finding_3 = {
            "title": "Largest Supplier Price Change (Email Evidence)",
            "claim": f"Supplier {top_change['sender']} announced a {direction} in {top_change['entity_or_ingredient']} price from {top_change['old_price']:.2f} to {top_change['new_price']:.2f} {top_change['currency']} per {top_change['unit']}, effective {top_change['effective_date']} ({top_change['pct_change']:.1f}% change). Email date: {top_change['date']}.",
            "finding_type": "supplier_price_change",
            "metrics": price_metrics,
            "source_names": ["emails"],
            "sample_size": len(price_change_emails),
            "coverage_notes": [
                f"Analysis period: {analysis_start} to {analysis_end}",
                f"Total price change emails with complete data: {len(price_change_emails)}",
                "Price changes extracted from supplier emails",
                "Effective dates may fall outside analysis period; impact assessment requires standing order volumes and payment terms"
            ],
            "assumptions": [
                "Email extraction confidence and facts are accurate",
                "Price change applies to the stated ingredient/entity only",
                "No recipe/BOM data available; per-drink cost impact cannot be calculated without ingredient quantities",
                "Standing order volumes and payment terms are not confirmed; procurement cost scenario would require additional evidence"
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

print(f"Analysis complete. {len(findings)} findings generated.")
print(f"Output written to {output_path}")
