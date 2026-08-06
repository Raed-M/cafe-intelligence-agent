import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# Load input paths from environment
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read input artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Define analysis period
analysis_start = datetime.fromisoformat("2026-01-12T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-01-19T00:00:00+03:00")

# Convert timestamp columns to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
emails_df['date'] = pd.to_datetime(emails_df['date'])
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'], errors='coerce')

# Filter data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)]
inventory_analysis = inventory_df[(inventory_df['week_starting'] >= analysis_start) & (inventory_df['week_starting'] < analysis_end)]

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item-level economics from POS and menu data
pos_with_menu = pos_analysis.merge(menu_df[['sku', 'unit_cost_sar', 'price_sar']], on='sku', how='left')

# Filter out refunds for revenue calculation
pos_sales = pos_with_menu[~pos_with_menu['is_refund']]

# Calculate item-level metrics
item_economics = pos_sales.groupby('sku').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'unit_cost_sar': 'first',
    'item_name': 'first',
    'category': 'first'
}).reset_index()

item_economics['total_cogs'] = item_economics['quantity'] * item_economics['unit_cost_sar']
item_economics['gross_profit'] = item_economics['line_total_sar'] - item_economics['total_cogs']
item_economics['gross_margin_pct'] = (item_economics['gross_profit'] / item_economics['line_total_sar'] * 100).round(2)

# Filter items with sales
item_economics_sold = item_economics[item_economics['quantity'] > 0].sort_values('gross_profit', ascending=False)

if len(item_economics_sold) > 0:
    top_profit_item = item_economics_sold.iloc[0]
    
    finding_1 = {
        "title": "Top Profit-Contributing Item",
        "claim": f"During the analysis week (2026-01-12 to 2026-01-19), {top_profit_item['item_name']} generated the highest gross profit of {top_profit_item['gross_profit']:.2f} SAR with a gross margin of {top_profit_item['gross_margin_pct']:.1f}%, based on {int(top_profit_item['quantity'])} units sold at {top_profit_item['line_total_sar']:.2f} SAR total revenue and {top_profit_item['total_cogs']:.2f} SAR COGS.",
        "finding_type": "item_economics",
        "metrics": {
            "item_name": {
                "value": top_profit_item['item_name'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "gross_profit_sar": {
                "value": round(top_profit_item['gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_profit_item['gross_profit'], 2),
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "gross_margin_pct": {
                "value": round(top_profit_item['gross_margin_pct'], 1),
                "unit": "%",
                "numerator": round(top_profit_item['gross_profit'], 2),
                "denominator": round(top_profit_item['line_total_sar'], 2),
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "units_sold": {
                "value": int(top_profit_item['quantity']),
                "unit": "units",
                "numerator": int(top_profit_item['quantity']),
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "total_revenue_sar": {
                "value": round(top_profit_item['line_total_sar'], 2),
                "unit": "SAR",
                "numerator": round(top_profit_item['line_total_sar'], 2),
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "total_cogs_sar": {
                "value": round(top_profit_item['total_cogs'], 2),
                "unit": "SAR",
                "numerator": round(top_profit_item['total_cogs'], 2),
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(item_economics_sold),
        "coverage_notes": [
            f"Analysis covers {len(pos_sales)} POS line items (excluding refunds) from {len(pos_sales['transaction_id'].unique())} transactions",
            f"Menu unit costs available for {len(item_economics_sold)} items with sales",
            "COGS calculated as quantity × unit_cost_sar from menu",
            "Gross profit = line_total_sar - total_cogs"
        ],
        "assumptions": [
            "Menu unit_cost_sar values are current and accurate for the analysis period",
            "POS line_total_sar reflects actual revenue after discounts",
            "No recipe/BOM data available; analysis is at item level only",
            "Refunds excluded from revenue calculation"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# FINDING 2: Known Waste Cost Analysis
# Filter inventory records with non-null waste cost
inventory_with_waste = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna() & (inventory_analysis['known_waste_cost_sar'] > 0)]

total_inventory_records = len(inventory_analysis)
waste_records = len(inventory_with_waste)
total_waste_cost = inventory_with_waste['known_waste_cost_sar'].sum()

if waste_records > 0:
    waste_items = inventory_with_waste[['sku', 'item', 'units_wasted', 'known_waste_cost_sar']].copy()
    waste_items = waste_items.sort_values('known_waste_cost_sar', ascending=False)
    
    finding_2 = {
        "title": "Known Waste Cost in Analysis Period",
        "claim": f"Total known waste cost for the analysis week (2026-01-12 to 2026-01-19) is {total_waste_cost:.2f} SAR across {waste_records} items with recorded waste observations. This represents waste from {int(inventory_with_waste['units_wasted'].sum())} units across {waste_records} SKUs. {total_inventory_records - waste_records} inventory records had null/blank waste values and were excluded per methodology.",
        "finding_type": "waste_cost",
        "metrics": {
            "total_known_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": round(total_waste_cost, 2),
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "items_with_waste": {
                "value": waste_records,
                "unit": "count",
                "numerator": waste_records,
                "denominator": total_inventory_records,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "total_units_wasted": {
                "value": int(inventory_with_waste['units_wasted'].sum()),
                "unit": "units",
                "numerator": int(inventory_with_waste['units_wasted'].sum()),
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "total_inventory_records": {
                "value": total_inventory_records,
                "unit": "count",
                "numerator": total_inventory_records,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "records_excluded_null_waste": {
                "value": total_inventory_records - waste_records,
                "unit": "count",
                "numerator": total_inventory_records - waste_records,
                "denominator": total_inventory_records,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            }
        },
        "source_names": ["inventory"],
        "sample_size": waste_records,
        "coverage_notes": [
            f"Only non-null waste_cost_sar values included: {waste_records} items",
            f"Total inventory records for period: {total_inventory_records}",
            f"Records excluded due to null/blank waste values: {total_inventory_records - waste_records}",
            "Known waste cost represents actual loss value from inventory.known_waste_cost_sar field"
        ],
        "assumptions": [
            "Waste cost represents actual loss value",
            "Null/blank waste values indicate no recorded waste observation (not zero waste)",
            "known_waste_cost_sar field contains only verified waste costs"
        ],
        "confidence": 0.90
    }
    findings.append(finding_2)

# FINDING 3: Supplier Price Change Detection
# Filter emails with price changes
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['category'] == 'supplier_price_change')
].copy()

if len(price_changes) > 0:
    # Calculate price change percentage
    price_changes['price_change_pct'] = ((price_changes['new_price'] - price_changes['old_price']) / price_changes['old_price'] * 100).round(2)
    
    # Sort by absolute price change
    price_changes['abs_price_change'] = abs(price_changes['new_price'] - price_changes['old_price'])
    price_changes = price_changes.sort_values('abs_price_change', ascending=False)
    
    top_change = price_changes.iloc[0]
    
    finding_3 = {
        "title": "Supplier Price Change Alert",
        "claim": f"Email analysis detected a price change notification for {top_change['entity_or_ingredient']} from {top_change['old_price']:.2f} to {top_change['new_price']:.2f} {top_change['currency']}/{top_change['unit']}, representing a {top_change['price_change_pct']:.2f}% increase, effective {top_change['effective_date'].strftime('%Y-%m-%d') if pd.notna(top_change['effective_date']) else 'date not specified'}. Business impact cannot be quantified without recipe/BOM data showing usage per drink and standing order quantities from procurement records.",
        "finding_type": "supplier_price_change",
        "metrics": {
            "entity_or_ingredient": {
                "value": top_change['entity_or_ingredient'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "old_price": {
                "value": round(top_change['old_price'], 2),
                "unit": f"{top_change['currency']}/{top_change['unit']}",
                "numerator": round(top_change['old_price'], 2),
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "new_price": {
                "value": round(top_change['new_price'], 2),
                "unit": f"{top_change['currency']}/{top_change['unit']}",
                "numerator": round(top_change['new_price'], 2),
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "price_change_pct": {
                "value": round(top_change['price_change_pct'], 2),
                "unit": "%",
                "numerator": round(top_change['new_price'] - top_change['old_price'], 2),
                "denominator": round(top_change['old_price'], 2),
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "effective_date": {
                "value": top_change['effective_date'].strftime('%Y-%m-%d') if pd.notna(top_change['effective_date']) else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            },
            "email_source": {
                "value": top_change['sender'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-12T00:00:00+03:00",
                "period_end": "2026-01-19T00:00:00+03:00"
            }
        },
        "source_names": ["emails"],
        "sample_size": len(price_changes),
        "coverage_notes": [
            f"Email analysis identified {len(price_changes)} price change notifications in the analysis period",
            "Price changes extracted from supplier emails using entity recognition",
            "Effective date may be future-dated relative to analysis period"
        ],
        "assumptions": [
            "Email extraction accurately captured old and new prices from supplier communications",
            "Price changes are from official supplier notifications (not verified against invoices or price lists)",
            "No recipe/BOM data available to calculate per-drink impact",
            "No standing order volume data available to calculate procurement cost impact",
            "Continued order volume and payment terms are unknown"
        ],
        "confidence": 0.70
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

print(f"Analysis complete. {len(findings)} findings generated.")
