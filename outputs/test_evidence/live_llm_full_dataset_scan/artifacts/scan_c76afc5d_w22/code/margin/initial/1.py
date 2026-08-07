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
analysis_start = datetime(2026, 6, 8, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 6, 8, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)

# Filter for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Prepare findings list
findings = []

# ============================================================================
# FINDING 1: Item-level COGS and Gross Profit Analysis
# ============================================================================

# Merge POS with menu to get unit costs
pos_analysis_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'price_sar']],
    on='sku',
    how='left'
)

# Calculate item-level economics (excluding refunds)
pos_analysis_with_cost['is_refund_bool'] = pos_analysis_with_cost['is_refund'].astype(bool)
pos_sales = pos_analysis_with_cost[~pos_analysis_with_cost['is_refund_bool']].copy()

# Calculate COGS and gross profit per line
pos_sales['cogs_sar'] = pos_sales['quantity'] * pos_sales['unit_cost_sar']
pos_sales['gross_profit_sar'] = pos_sales['line_total_sar'] - pos_sales['cogs_sar']
pos_sales['gross_margin_pct'] = (pos_sales['gross_profit_sar'] / pos_sales['line_total_sar'] * 100).fillna(0)

# Aggregate by item
item_economics = pos_sales.groupby('sku').agg({
    'item_name_en': 'first',
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'cogs_sar': 'sum',
    'gross_profit_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()

item_economics.columns = ['sku', 'item_name', 'total_quantity', 'total_revenue', 'total_cogs', 'total_gross_profit', 'basket_count']
item_economics['gross_margin_pct'] = (item_economics['total_gross_profit'] / item_economics['total_revenue'] * 100).round(2)
item_economics = item_economics.sort_values('total_gross_profit', ascending=False)

# Top 3 by gross profit
top_items = item_economics.head(3)

if len(top_items) > 0:
    finding_1 = {
        "title": "Top 3 Items by Gross Profit (Analysis Period)",
        "claim": f"During {analysis_start.date()} to {analysis_end.date()}, the top 3 items by absolute gross profit contribution are {', '.join(top_items['item_name'].tolist())}. Total gross profit across all items: {item_economics['total_gross_profit'].sum():.2f} SAR.",
        "finding_type": "item_economics",
        "metrics": {
            "top_item_1_name": {
                "value": top_items.iloc[0]['item_name'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_gross_profit": {
                "value": round(top_items.iloc[0]['total_gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_items.iloc[0]['total_gross_profit'], 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_quantity": {
                "value": int(top_items.iloc[0]['total_quantity']),
                "unit": "units",
                "numerator": int(top_items.iloc[0]['total_quantity']),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_margin_pct": {
                "value": round(top_items.iloc[0]['gross_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_items.iloc[0]['gross_margin_pct'], 2),
                "denominator": 100,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_2_name": {
                "value": top_items.iloc[1]['item_name'] if len(top_items) > 1 else None,
                "unit": None,
                "numerator": None,
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
                "value": top_items.iloc[2]['item_name'] if len(top_items) > 2 else None,
                "unit": None,
                "numerator": None,
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
            "total_items_analyzed": {
                "value": len(item_economics),
                "unit": "count",
                "numerator": len(item_economics),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_gross_profit_all_items": {
                "value": round(item_economics['total_gross_profit'].sum(), 2),
                "unit": "SAR",
                "numerator": round(item_economics['total_gross_profit'].sum(), 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(pos_sales),
        "coverage_notes": [
            "Analysis period: 2026-06-08 to 2026-06-15",
            "Refunds excluded from calculations",
            "Unit costs sourced from menu.parquet",
            "Line totals used as revenue (net of discounts)"
        ],
        "assumptions": [
            "Menu unit_cost_sar is current and applicable to analysis period",
            "Line_total_sar represents actual revenue after discounts",
            "No recipe/BOM available; per-unit COGS from menu is used as-is"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Waste Cost Analysis
# ============================================================================

# Filter inventory for analysis period (week starting 2026-06-08)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'], utc=True)
inv_analysis = inventory_df[inventory_df['week_starting'] >= analysis_start].copy()

# Calculate total waste cost (only non-null values)
waste_data = inv_analysis[inv_analysis['known_waste_cost_sar'].notna()].copy()

if len(waste_data) > 0:
    total_waste_cost = waste_data['known_waste_cost_sar'].sum()
    waste_by_item = waste_data.groupby('item').agg({
        'known_waste_cost_sar': 'sum',
        'units_wasted': 'sum'
    }).reset_index().sort_values('known_waste_cost_sar', ascending=False)
    
    finding_2 = {
        "title": "Quantified Waste Cost (Analysis Period)",
        "claim": f"During the week of {analysis_start.date()}, quantified waste cost totaled {total_waste_cost:.2f} SAR across {len(waste_data)} inventory records. Top waste contributor: {waste_by_item.iloc[0]['item']} ({waste_by_item.iloc[0]['known_waste_cost_sar']:.2f} SAR).",
        "finding_type": "waste_cost",
        "metrics": {
            "total_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": round(total_waste_cost, 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "waste_records_count": {
                "value": len(waste_data),
                "unit": "count",
                "numerator": len(waste_data),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_waste_item": {
                "value": waste_by_item.iloc[0]['item'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_waste_item_cost": {
                "value": round(waste_by_item.iloc[0]['known_waste_cost_sar'], 2),
                "unit": "SAR",
                "numerator": round(waste_by_item.iloc[0]['known_waste_cost_sar'], 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_waste_item_units": {
                "value": int(waste_by_item.iloc[0]['units_wasted']),
                "unit": "units",
                "numerator": int(waste_by_item.iloc[0]['units_wasted']),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(waste_data),
        "coverage_notes": [
            "Only non-null known_waste_cost_sar values included",
            "Waste cost calculated from inventory.parquet known_waste_cost_sar column",
            "Analysis period: week starting 2026-06-08"
        ],
        "assumptions": [
            "known_waste_cost_sar represents actual quantified waste cost",
            "Null waste values are excluded (not treated as zero)"
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
    price_change_emails['effective_date'] = pd.to_datetime(price_change_emails['effective_date'], utc=True)
    
    # Calculate price change percentage
    price_change_emails['price_change_pct'] = (
        (price_change_emails['new_price'] - price_change_emails['old_price']) / 
        price_change_emails['old_price'] * 100
    ).round(2)
    
    # Sort by effective date descending
    price_change_emails = price_change_emails.sort_values('effective_date', ascending=False)
    
    # Take most recent price change
    most_recent = price_change_emails.iloc[0]
    
    finding_3 = {
        "title": "Recent Supplier Price Change",
        "claim": f"Email evidence from {most_recent['sender']} dated {most_recent['date']} documents a price change for {most_recent['entity_or_ingredient']}: {most_recent['old_price']} {most_recent['currency']}/{most_recent['unit']} → {most_recent['new_price']} {most_recent['currency']}/{most_recent['unit']} (effective {most_recent['effective_date'].date()}). This represents a {most_recent['price_change_pct']:.2f}% change.",
        "finding_type": "supplier_price_change",
        "metrics": {
            "ingredient": {
                "value": most_recent['entity_or_ingredient'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "old_price": {
                "value": round(most_recent['old_price'], 2),
                "unit": f"{most_recent['currency']}/{most_recent['unit']}",
                "numerator": round(most_recent['old_price'], 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "new_price": {
                "value": round(most_recent['new_price'], 2),
                "unit": f"{most_recent['currency']}/{most_recent['unit']}",
                "numerator": round(most_recent['new_price'], 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "price_change_pct": {
                "value": most_recent['price_change_pct'],
                "unit": "%",
                "numerator": most_recent['price_change_pct'],
                "denominator": 100,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "effective_date": {
                "value": most_recent['effective_date'].isoformat(),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "sender": {
                "value": most_recent['sender'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "email_date": {
                "value": most_recent['date'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["emails"],
        "sample_size": len(price_change_emails),
        "coverage_notes": [
            "Analysis based on emails with both old_price and new_price and effective_date",
            "Most recent price change selected",
            f"Total price change records identified: {len(price_change_emails)}"
        ],
        "assumptions": [
            "Email extraction confidence and facts field are authoritative",
            "Price change applies to the named ingredient/entity only",
            "No recipe/BOM available; impact on menu items cannot be calculated without standing order volumes"
        ],
        "confidence": 0.85
    }
    findings.append(finding_3)

# ============================================================================
# Compile output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
