import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Load environment configuration
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
analysis_start = datetime.fromisoformat("2026-02-23T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-03-02T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-02-16T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-02-23T00:00:00+03:00")

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

findings = []

# Finding 1: Item-level COGS and Gross Profit Analysis
# Calculate exact item economics from menu and POS data
menu_with_margin = menu_df.copy()
menu_with_margin['gross_profit_per_unit'] = menu_with_margin['price_sar'] - menu_with_margin['unit_cost_sar']
menu_with_margin['gross_margin_pct'] = (menu_with_margin['gross_profit_per_unit'] / menu_with_margin['price_sar'] * 100).round(2)

# Aggregate POS sales by SKU for analysis period
pos_by_sku = pos_analysis[pos_analysis['is_refund'] == False].groupby('sku').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
pos_by_sku.columns = ['sku', 'total_quantity', 'total_revenue', 'basket_count']

# Merge with menu data
sku_economics = pos_by_sku.merge(menu_df[['sku', 'item_en', 'unit_cost_sar', 'price_sar']], on='sku', how='left')
sku_economics['total_cogs'] = sku_economics['total_quantity'] * sku_economics['unit_cost_sar']
sku_economics['total_gross_profit'] = sku_economics['total_revenue'] - sku_economics['total_cogs']
sku_economics['gross_margin_pct'] = (sku_economics['total_gross_profit'] / sku_economics['total_revenue'] * 100).round(2)

# Find top 3 items by revenue
top_items = sku_economics.nlargest(3, 'total_revenue')

if len(top_items) > 0:
    finding_1 = {
        "title": "Top Revenue-Generating Items: Exact Item-Level Economics",
        "claim": f"During {analysis_start.date()} to {analysis_end.date()}, the top 3 revenue items generated {top_items['total_revenue'].sum():.2f} SAR in combined revenue with {top_items['total_gross_profit'].sum():.2f} SAR gross profit. Item-level COGS calculated from menu unit costs and realized POS quantities.",
        "finding_type": "item_economics",
        "metrics": {
            "top_item_1_name": {
                "value": top_items.iloc[0]['item_en'] if len(top_items) > 0 else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_revenue": {
                "value": round(top_items.iloc[0]['total_revenue'], 2) if len(top_items) > 0 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[0]['total_revenue'], 2) if len(top_items) > 0 else None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_cogs": {
                "value": round(top_items.iloc[0]['total_cogs'], 2) if len(top_items) > 0 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[0]['total_cogs'], 2) if len(top_items) > 0 else None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_gross_profit": {
                "value": round(top_items.iloc[0]['total_gross_profit'], 2) if len(top_items) > 0 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[0]['total_gross_profit'], 2) if len(top_items) > 0 else None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_margin_pct": {
                "value": round(top_items.iloc[0]['gross_margin_pct'], 2) if len(top_items) > 0 else None,
                "unit": "%",
                "numerator": round(top_items.iloc[0]['total_gross_profit'], 2) if len(top_items) > 0 else None,
                "denominator": round(top_items.iloc[0]['total_revenue'], 2) if len(top_items) > 0 else None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_2_name": {
                "value": top_items.iloc[1]['item_en'] if len(top_items) > 1 else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_2_revenue": {
                "value": round(top_items.iloc[1]['total_revenue'], 2) if len(top_items) > 1 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[1]['total_revenue'], 2) if len(top_items) > 1 else None,
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
                "value": top_items.iloc[2]['item_en'] if len(top_items) > 2 else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_3_revenue": {
                "value": round(top_items.iloc[2]['total_revenue'], 2) if len(top_items) > 2 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[2]['total_revenue'], 2) if len(top_items) > 2 else None,
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
            "combined_top_3_revenue": {
                "value": round(top_items['total_revenue'].sum(), 2),
                "unit": "SAR",
                "numerator": round(top_items['total_revenue'].sum(), 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "combined_top_3_gross_profit": {
                "value": round(top_items['total_gross_profit'].sum(), 2),
                "unit": "SAR",
                "numerator": round(top_items['total_gross_profit'].sum(), 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(pos_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
            f"POS records analyzed: {len(pos_analysis)} line items",
            f"Unique SKUs with sales: {len(pos_by_sku)}",
            "COGS calculated from menu_items.unit_cost_sar × realized POS quantities",
            "Refunds excluded from revenue and quantity calculations"
        ],
        "assumptions": [
            "Menu unit costs are current and applicable to analysis period",
            "POS line_total_sar represents actual revenue after discounts",
            "No recipe/BOM data available; per-drink ingredient impact not calculated"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# Finding 2: Waste Cost Analysis
# Filter inventory for analysis period week
analysis_week = pd.Timestamp("2026-02-23", tz='UTC')
inventory_analysis = inventory_df[inventory_df['week_starting'] == analysis_week].copy()

# Calculate total waste cost (only non-null values)
waste_cost_total = inventory_analysis['known_waste_cost_sar'].sum()
waste_items = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna() & (inventory_analysis['known_waste_cost_sar'] > 0)]

if len(waste_items) > 0:
    finding_2 = {
        "title": "Quantified Waste Cost Impact",
        "claim": f"During week of {analysis_week.date()}, {len(waste_items)} items had documented waste with total known waste cost of {waste_cost_total:.2f} SAR. Only non-null waste observations included in calculation.",
        "finding_type": "waste_cost",
        "metrics": {
            "waste_items_count": {
                "value": len(waste_items),
                "unit": "items",
                "numerator": len(waste_items),
                "denominator": None,
                "period_start": analysis_week.isoformat(),
                "period_end": (analysis_week + pd.Timedelta(days=7)).isoformat()
            },
            "total_waste_cost_sar": {
                "value": round(waste_cost_total, 2),
                "unit": "SAR",
                "numerator": round(waste_cost_total, 2),
                "denominator": None,
                "period_start": analysis_week.isoformat(),
                "period_end": (analysis_week + pd.Timedelta(days=7)).isoformat()
            },
            "waste_items_detail": {
                "value": waste_items[['sku', 'item', 'units_wasted', 'known_waste_cost_sar']].to_dict('records'),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_week.isoformat(),
                "period_end": (analysis_week + pd.Timedelta(days=7)).isoformat()
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(inventory_analysis),
        "coverage_notes": [
            f"Inventory week: {analysis_week.date()}",
            f"Total inventory records for week: {len(inventory_analysis)}",
            f"Records with non-null waste cost: {len(waste_items)}",
            "Blank waste values treated as unknown, not zero"
        ],
        "assumptions": [
            "known_waste_cost_sar values are accurate and complete for documented waste",
            "Waste cost represents actual loss to business"
        ],
        "confidence": 0.85
    }
    findings.append(finding_2)

# Finding 3: Supplier Price Changes from Email Evidence
# Filter emails for price changes with valid old and new prices
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    # Process each price change
    for idx, change in price_changes.iterrows():
        old_price = float(change['old_price'])
        new_price = float(change['new_price'])
        price_diff = new_price - old_price
        price_change_pct = (price_diff / old_price * 100) if old_price != 0 else 0
        
        finding_3 = {
            "title": f"Supplier Price Change: {change['entity_or_ingredient']}",
            "claim": f"Email evidence documents price change for {change['entity_or_ingredient']} from {old_price:.2f} SAR/{change['unit']} to {new_price:.2f} SAR/{change['unit']} effective {change['effective_date'].date()}. Price change represents {price_change_pct:.2f}% increase. Impact on menu items requires standing order volume and payment terms confirmation.",
            "finding_type": "supplier_price_change",
            "metrics": {
                "ingredient": {
                    "value": change['entity_or_ingredient'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": change['date'].isoformat() if pd.notna(change['date']) else None,
                    "period_end": change['effective_date'].isoformat() if pd.notna(change['effective_date']) else None
                },
                "old_price": {
                    "value": round(old_price, 2),
                    "unit": f"SAR/{change['unit']}",
                    "numerator": round(old_price, 2),
                    "denominator": None,
                    "period_start": change['date'].isoformat() if pd.notna(change['date']) else None,
                    "period_end": change['effective_date'].isoformat() if pd.notna(change['effective_date']) else None
                },
                "new_price": {
                    "value": round(new_price, 2),
                    "unit": f"SAR/{change['unit']}",
                    "numerator": round(new_price, 2),
                    "denominator": None,
                    "period_start": change['date'].isoformat() if pd.notna(change['date']) else None,
                    "period_end": change['effective_date'].isoformat() if pd.notna(change['effective_date']) else None
                },
                "price_change_absolute": {
                    "value": round(price_diff, 2),
                    "unit": f"SAR/{change['unit']}",
                    "numerator": round(price_diff, 2),
                    "denominator": None,
                    "period_start": change['date'].isoformat() if pd.notna(change['date']) else None,
                    "period_end": change['effective_date'].isoformat() if pd.notna(change['effective_date']) else None
                },
                "price_change_pct": {
                    "value": round(price_change_pct, 2),
                    "unit": "%",
                    "numerator": round(price_diff, 2),
                    "denominator": round(old_price, 2),
                    "period_start": change['date'].isoformat() if pd.notna(change['date']) else None,
                    "period_end": change['effective_date'].isoformat() if pd.notna(change['effective_date']) else None
                },
                "effective_date": {
                    "value": change['effective_date'].isoformat() if pd.notna(change['effective_date']) else None,
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": change['date'].isoformat() if pd.notna(change['date']) else None,
                    "period_end": change['effective_date'].isoformat() if pd.notna(change['effective_date']) else None
                },
                "sender": {
                    "value": change['sender'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": change['date'].isoformat() if pd.notna(change['date']) else None,
                    "period_end": change['effective_date'].isoformat() if pd.notna(change['effective_date']) else None
                }
            },
            "source_names": ["emails"],
            "sample_size": 1,
            "coverage_notes": [
                f"Email date: {change['date'].date() if pd.notna(change['date']) else 'Unknown'}",
                f"Effective date: {change['effective_date'].date() if pd.notna(change['effective_date']) else 'Unknown'}",
                f"Confidence level from extraction: {change['confidence']}",
                "Price change calculation: (new_price - old_price) / old_price × 100"
            ],
            "assumptions": [
                "Email extraction accurately captured supplier price change facts",
                "No recipe/BOM data available; per-drink cost impact cannot be calculated",
                "Standing order volume and payment terms not confirmed from email evidence",
                "Price change applies only to specified ingredient/entity"
            ],
            "confidence": float(change['confidence']) if pd.notna(change['confidence']) else 0.75
        }
        findings.append(finding_3)
        
        # Limit to 3 findings total
        if len(findings) >= 3:
            break

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
