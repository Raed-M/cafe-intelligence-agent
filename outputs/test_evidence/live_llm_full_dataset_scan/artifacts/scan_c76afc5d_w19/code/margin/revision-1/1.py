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
analysis_start = datetime(2026, 5, 18, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 5, 25, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 5, 11, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 5, 18, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp columns to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'], errors='coerce')
menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'], errors='coerce')
emails_df['date'] = pd.to_datetime(emails_df['date'], errors='coerce')
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'], errors='coerce')

# Filter POS data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Filter inventory for analysis period
inv_analysis = inventory_df[inventory_df['week_starting'] == pd.Timestamp('2026-05-18', tz=timezone.utc)].copy()
inv_previous = inventory_df[inventory_df['week_starting'] == pd.Timestamp('2026-05-11', tz=timezone.utc)].copy()

findings = []

# FINDING 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from POS and menu data
pos_with_menu = pos_analysis[pos_analysis['is_refund'] == False].copy()
pos_with_menu = pos_with_menu.merge(menu_df[['sku', 'unit_cost_sar']], on='sku', how='left', suffixes=('', '_menu'))

# Group by item to calculate totals
item_economics = pos_with_menu.groupby(['sku', 'item_name']).agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'unit_cost_sar': 'first',
    'transaction_id': 'nunique'
}).reset_index()

item_economics.columns = ['sku', 'item_name', 'total_quantity', 'total_revenue', 'unit_cost_sar', 'basket_count']

# Calculate COGS and gross profit
item_economics['total_cogs'] = item_economics['total_quantity'] * item_economics['unit_cost_sar']
item_economics['gross_profit'] = item_economics['total_revenue'] - item_economics['total_cogs']
item_economics['gross_margin_pct'] = (item_economics['gross_profit'] / item_economics['total_revenue'] * 100).round(2)

# Sort by gross profit
item_economics_sorted = item_economics.sort_values('gross_profit', ascending=False)

# Get top 3 items by gross profit
top_items = item_economics_sorted.head(3)

if len(top_items) > 0:
    top_item = top_items.iloc[0]
    finding_1 = {
        "title": "Top Gross Profit Item - Week of 2026-05-18",
        "claim": f"Item '{top_item['item_name']}' (SKU: {top_item['sku']}) generated the highest gross profit of {top_item['gross_profit']:.2f} SAR during the analysis week, with {top_item['total_quantity']:.0f} units sold across {top_item['basket_count']:.0f} transactions, achieving a {top_item['gross_margin_pct']:.1f}% gross margin.",
        "finding_type": "item_economics",
        "metrics": {
            "item_name": {"value": top_item['item_name'], "unit": None, "numerator": None, "denominator": None, "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"},
            "sku": {"value": top_item['sku'], "unit": None, "numerator": None, "denominator": None, "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"},
            "total_quantity": {"value": top_item['total_quantity'], "unit": "units", "numerator": top_item['total_quantity'], "denominator": None, "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"},
            "total_revenue": {"value": round(top_item['total_revenue'], 2), "unit": "SAR", "numerator": top_item['total_revenue'], "denominator": None, "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"},
            "total_cogs": {"value": round(top_item['total_cogs'], 2), "unit": "SAR", "numerator": top_item['total_cogs'], "denominator": None, "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"},
            "gross_profit": {"value": round(top_item['gross_profit'], 2), "unit": "SAR", "numerator": top_item['gross_profit'], "denominator": None, "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"},
            "gross_margin_pct": {"value": top_item['gross_margin_pct'], "unit": "%", "numerator": top_item['gross_profit'], "denominator": top_item['total_revenue'], "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"},
            "basket_count": {"value": top_item['basket_count'], "unit": "transactions", "numerator": top_item['basket_count'], "denominator": None, "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"}
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(pos_with_menu),
        "coverage_notes": [
            "Analysis period: 2026-05-18 to 2026-05-25",
            "Excludes refunds (is_refund == False)",
            "Unit costs sourced from menu.parquet",
            "Revenue calculated from line_total_sar",
            "COGS = quantity × unit_cost_sar",
            "Gross profit = revenue - COGS"
        ],
        "assumptions": [
            "Menu unit_cost_sar is current and accurate for the analysis period",
            "POS line_total_sar reflects actual transaction amounts",
            "No adjustments for waste or shrinkage beyond inventory records"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# FINDING 2: Waste Cost Analysis
# Calculate waste costs from inventory data
inv_analysis_with_waste = inv_analysis[inv_analysis['known_waste_cost_sar'].notna()].copy()

if len(inv_analysis_with_waste) > 0:
    total_waste_cost = inv_analysis_with_waste['known_waste_cost_sar'].sum()
    waste_items = inv_analysis_with_waste[['sku', 'item', 'units_wasted', 'known_waste_cost_sar']].copy()
    waste_items = waste_items.sort_values('known_waste_cost_sar', ascending=False)
    
    if len(waste_items) > 0:
        top_waste_item = waste_items.iloc[0]
        finding_2 = {
            "title": "Highest Waste Cost Item - Week of 2026-05-18",
            "claim": f"Item '{top_waste_item['item']}' (SKU: {top_waste_item['sku']}) incurred the highest waste cost of {top_waste_item['known_waste_cost_sar']:.2f} SAR with {top_waste_item['units_wasted']:.0f} units wasted during the week of 2026-05-18. Total waste cost across all items: {total_waste_cost:.2f} SAR.",
            "finding_type": "waste_cost",
            "metrics": {
                "item_name": {"value": top_waste_item['item'], "unit": None, "numerator": None, "denominator": None, "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"},
                "sku": {"value": top_waste_item['sku'], "unit": None, "numerator": None, "denominator": None, "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"},
                "units_wasted": {"value": top_waste_item['units_wasted'], "unit": "units", "numerator": top_waste_item['units_wasted'], "denominator": None, "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"},
                "waste_cost_sar": {"value": round(top_waste_item['known_waste_cost_sar'], 2), "unit": "SAR", "numerator": top_waste_item['known_waste_cost_sar'], "denominator": None, "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"},
                "total_waste_cost_all_items": {"value": round(total_waste_cost, 2), "unit": "SAR", "numerator": total_waste_cost, "denominator": None, "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"}
            },
            "source_names": ["inventory"],
            "sample_size": len(inv_analysis_with_waste),
            "coverage_notes": [
                "Analysis period: week starting 2026-05-18",
                "Only includes items with non-null known_waste_cost_sar",
                "Waste cost calculated from inventory records",
                "Items with null waste values excluded per data quality rules"
            ],
            "assumptions": [
                "known_waste_cost_sar accurately reflects waste value",
                "Waste units and costs are recorded at point of disposal"
            ],
            "confidence": 0.90
        }
        findings.append(finding_2)

# FINDING 3: Supplier Price Change Analysis
# Look for supplier price changes in emails
milk_emails = emails_df[
    (emails_df['entity_or_ingredient'].str.contains('milk', case=False, na=False)) &
    (emails_df['old_price'].notna()) &
    (emails_df['new_price'].notna())
].copy()

if len(milk_emails) > 0:
    # Get the most recent milk price change
    milk_emails = milk_emails.sort_values('date', ascending=False)
    latest_milk_email = milk_emails.iloc[0]
    
    old_price = float(latest_milk_email['old_price'])
    new_price = float(latest_milk_email['new_price'])
    price_change_pct = ((new_price - old_price) / old_price * 100) if old_price != 0 else 0
    
    finding_3 = {
        "title": "Supplier Price Notification – Full-Fat Milk Cost Increase (May 2026)",
        "claim": f"Supplier email dated {latest_milk_email['date'].strftime('%Y-%m-%d')} from {latest_milk_email['sender']} notifies of a full-fat milk price increase from {old_price:.1f} to {new_price:.1f} SAR per {latest_milk_email['unit']}, effective {latest_milk_email['effective_date'].strftime('%Y-%m-%d')}, representing an {price_change_pct:.2f}% increase. Procurement cost impact on cafe margin cannot be quantified without recipe/BOM data and standing order volumes.",
        "finding_type": "supplier_price_change",
        "metrics": {
            "entity": {"value": latest_milk_email['entity_or_ingredient'], "unit": None, "numerator": None, "denominator": None, "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"},
            "old_price": {"value": old_price, "unit": f"SAR/{latest_milk_email['unit']}", "numerator": old_price, "denominator": None, "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"},
            "new_price": {"value": new_price, "unit": f"SAR/{latest_milk_email['unit']}", "numerator": new_price, "denominator": None, "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"},
            "price_change_pct": {"value": round(price_change_pct, 2), "unit": "%", "numerator": new_price - old_price, "denominator": old_price, "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"},
            "email_date": {"value": latest_milk_email['date'].strftime('%Y-%m-%d'), "unit": None, "numerator": None, "denominator": None, "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"},
            "effective_date": {"value": latest_milk_email['effective_date'].strftime('%Y-%m-%d'), "unit": None, "numerator": None, "denominator": None, "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"},
            "sender": {"value": latest_milk_email['sender'], "unit": None, "numerator": None, "denominator": None, "period_start": "2026-05-18T00:00:00Z", "period_end": "2026-05-25T00:00:00Z"}
        },
        "source_names": ["emails"],
        "sample_size": 1,
        "coverage_notes": [
            "Evidence observation period (2026-05-18 to 2026-05-25) reflects when price change was confirmed in cafe records",
            "Email notification dated {latest_milk_email['date'].strftime('%Y-%m-%d')}",
            "Extraction confidence: {latest_milk_email['confidence']}"
        ],
        "assumptions": [
            f"Email extraction confidence: {latest_milk_email['confidence']}",
            "Actual procurement volumes and payment terms unknown",
            "Per-drink cost impact requires recipe/BOM data",
            "Standing order volumes and continued order assumptions not verified"
        ],
        "confidence": float(latest_milk_email['confidence']) if latest_milk_email['confidence'] else 0.85
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
