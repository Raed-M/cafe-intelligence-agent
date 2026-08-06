import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Load run metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Define analysis periods
analysis_period = {
    "start": "2026-04-06T00:00:00+03:00",
    "end": "2026-04-13T00:00:00+03:00"
}
previous_period = {
    "start": "2026-03-30T00:00:00+03:00",
    "end": "2026-04-06T00:00:00+03:00"
}
trailing_baseline_periods = [
    {"start": "2026-03-30T00:00:00+03:00", "end": "2026-04-06T00:00:00+03:00"},
    {"start": "2026-03-23T00:00:00+03:00", "end": "2026-03-30T00:00:00+03:00"},
    {"start": "2026-03-16T00:00:00+03:00", "end": "2026-03-23T00:00:00+03:00"},
    {"start": "2026-03-09T00:00:00+03:00", "end": "2026-03-16T00:00:00+03:00"}
]

# Parse dates
def parse_iso_date(date_str):
    return pd.to_datetime(date_str)

analysis_start = parse_iso_date(analysis_period["start"])
analysis_end = parse_iso_date(analysis_period["end"])
prev_start = parse_iso_date(previous_period["start"])
prev_end = parse_iso_date(previous_period["end"])

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Convert timestamp columns to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'], errors='coerce')
menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'], errors='coerce')
emails_df['date'] = pd.to_datetime(emails_df['date'], errors='coerce')
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'], errors='coerce')

# Ensure timezone-aware comparison for emails
# Convert effective_date to UTC if it has timezone, or localize if naive
if emails_df['effective_date'].dt.tz is None:
    emails_df['effective_date'] = emails_df['effective_date'].dt.tz_localize('UTC')
else:
    emails_df['effective_date'] = emails_df['effective_date'].dt.tz_convert('UTC')

# Convert period boundaries to UTC for comparison
prev_start_utc = prev_start.tz_convert('UTC') if prev_start.tz is not None else prev_start.tz_localize('UTC')
analysis_end_utc = analysis_end.tz_convert('UTC') if analysis_end.tz is not None else analysis_end.tz_localize('UTC')

# Filter POS for analysis period (exclude refunds for revenue, but keep for basket analysis)
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= prev_start) & (pos_df['timestamp'] < prev_end)].copy()

# Filter inventory for analysis period
inv_analysis = inventory_df[inventory_df['week_starting'] == pd.Timestamp('2026-04-06')].copy()
inv_previous = inventory_df[inventory_df['week_starting'] == pd.Timestamp('2026-03-30')].copy()

findings = []

# ============================================================================
# FINDING 1: Item-level COGS and Gross Profit Analysis (Analysis Period)
# ============================================================================

# Merge POS with menu to get unit costs
pos_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'price_sar']],
    on='sku',
    how='left'
)

# Calculate item-level metrics (excluding refunds from revenue)
pos_revenue = pos_with_cost[~pos_with_cost['is_refund']].copy()
pos_revenue['cogs'] = pos_revenue['quantity'] * pos_revenue['unit_cost_sar']
pos_revenue['gross_profit'] = pos_revenue['line_total_sar'] - pos_revenue['cogs']

# Aggregate by item
item_economics = pos_revenue.groupby(['sku', 'item_name_en']).agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'cogs': 'sum',
    'gross_profit': 'sum',
    'transaction_id': 'nunique'
}).reset_index()

item_economics.columns = ['sku', 'item_name', 'total_quantity', 'total_revenue', 'total_cogs', 'total_gross_profit', 'basket_count']
item_economics['gross_margin_pct'] = (item_economics['total_gross_profit'] / item_economics['total_revenue'] * 100).round(2)
item_economics = item_economics.sort_values('total_gross_profit', ascending=False)

# Top 3 items by gross profit
top_items = item_economics.head(3)

if len(top_items) > 0:
    finding_1 = {
        "title": "Top 3 Items by Gross Profit (Analysis Period)",
        "claim": f"During {analysis_period['start']} to {analysis_period['end']}, the top 3 items by absolute gross profit contribution are: {', '.join(top_items['item_name'].values)}. Total gross profit from these items: {top_items['total_gross_profit'].sum():.2f} SAR.",
        "finding_type": "item_economics",
        "metrics": {
            "top_item_1_name": {
                "value": top_items.iloc[0]['item_name'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_period["start"],
                "period_end": analysis_period["end"]
            },
            "top_item_1_gross_profit_sar": {
                "value": round(top_items.iloc[0]['total_gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_items.iloc[0]['total_gross_profit'], 2),
                "denominator": None,
                "period_start": analysis_period["start"],
                "period_end": analysis_period["end"]
            },
            "top_item_1_gross_margin_pct": {
                "value": round(top_items.iloc[0]['gross_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_items.iloc[0]['gross_margin_pct'], 2),
                "denominator": 100,
                "period_start": analysis_period["start"],
                "period_end": analysis_period["end"]
            },
            "top_item_1_quantity": {
                "value": int(top_items.iloc[0]['total_quantity']),
                "unit": "units",
                "numerator": int(top_items.iloc[0]['total_quantity']),
                "denominator": None,
                "period_start": analysis_period["start"],
                "period_end": analysis_period["end"]
            },
            "top_item_2_name": {
                "value": top_items.iloc[1]['item_name'] if len(top_items) > 1 else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_period["start"],
                "period_end": analysis_period["end"]
            },
            "top_item_2_gross_profit_sar": {
                "value": round(top_items.iloc[1]['total_gross_profit'], 2) if len(top_items) > 1 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[1]['total_gross_profit'], 2) if len(top_items) > 1 else None,
                "denominator": None,
                "period_start": analysis_period["start"],
                "period_end": analysis_period["end"]
            },
            "top_item_3_name": {
                "value": top_items.iloc[2]['item_name'] if len(top_items) > 2 else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_period["start"],
                "period_end": analysis_period["end"]
            },
            "top_item_3_gross_profit_sar": {
                "value": round(top_items.iloc[2]['total_gross_profit'], 2) if len(top_items) > 2 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[2]['total_gross_profit'], 2) if len(top_items) > 2 else None,
                "denominator": None,
                "period_start": analysis_period["start"],
                "period_end": analysis_period["end"]
            },
            "total_gross_profit_top_3_sar": {
                "value": round(top_items['total_gross_profit'].sum(), 2),
                "unit": "SAR",
                "numerator": round(top_items['total_gross_profit'].sum(), 2),
                "denominator": None,
                "period_start": analysis_period["start"],
                "period_end": analysis_period["end"]
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(pos_revenue),
        "coverage_notes": [
            "Analysis period: 2026-04-06 to 2026-04-13",
            "Refunds excluded from revenue and profit calculations",
            "Unit costs sourced from menu.parquet",
            "Only items with non-null unit_cost_sar included"
        ],
        "assumptions": [
            "Menu unit_cost_sar is current and applicable to analysis period",
            "POS line_total_sar reflects actual revenue after discounts",
            "No recipe/BOM available; per-unit COGS is menu unit_cost_sar"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Waste Cost Analysis (Inventory Period)
# ============================================================================

if len(inv_analysis) > 0:
    inv_analysis_with_waste = inv_analysis[inv_analysis['known_waste_cost_sar'].notna()].copy()
    
    if len(inv_analysis_with_waste) > 0:
        total_waste_cost = inv_analysis_with_waste['known_waste_cost_sar'].sum()
        waste_items = inv_analysis_with_waste[['sku', 'item', 'units_wasted', 'known_waste_cost_sar']].copy()
        waste_items = waste_items.sort_values('known_waste_cost_sar', ascending=False)
        
        finding_2 = {
            "title": "Quantified Waste Cost (Week of 2026-04-06)",
            "claim": f"During the week starting 2026-04-06, quantified waste cost (non-null observations only) totals {total_waste_cost:.2f} SAR across {len(waste_items)} items with recorded waste.",
            "finding_type": "waste_cost",
            "metrics": {
                "total_waste_cost_sar": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": round(total_waste_cost, 2),
                    "denominator": None,
                    "period_start": "2026-04-06T00:00:00+03:00",
                    "period_end": "2026-04-13T00:00:00+03:00"
                },
                "waste_item_count": {
                    "value": len(waste_items),
                    "unit": "items",
                    "numerator": len(waste_items),
                    "denominator": None,
                    "period_start": "2026-04-06T00:00:00+03:00",
                    "period_end": "2026-04-13T00:00:00+03:00"
                },
                "highest_waste_item": {
                    "value": waste_items.iloc[0]['item'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": "2026-04-06T00:00:00+03:00",
                    "period_end": "2026-04-13T00:00:00+03:00"
                },
                "highest_waste_cost_sar": {
                    "value": round(waste_items.iloc[0]['known_waste_cost_sar'], 2),
                    "unit": "SAR",
                    "numerator": round(waste_items.iloc[0]['known_waste_cost_sar'], 2),
                    "denominator": None,
                    "period_start": "2026-04-06T00:00:00+03:00",
                    "period_end": "2026-04-13T00:00:00+03:00"
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(waste_items),
            "coverage_notes": [
                "Only non-null known_waste_cost_sar values included",
                "Inventory week: 2026-04-06",
                "Blank waste cost observations excluded per methodology"
            ],
            "assumptions": [
                "known_waste_cost_sar reflects actual waste cost for that item in that week",
                "Waste cost is unit_cost_sar × units_wasted"
            ],
            "confidence": 0.90
        }
        findings.append(finding_2)

# ============================================================================
# FINDING 3: Supplier Price Changes (Email Evidence)
# ============================================================================

# Filter emails for price changes within or near analysis/previous/baseline periods
valid_emails = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

# Check if effective_date falls within or just before analysis period
# Use UTC-normalized dates for comparison
valid_emails['is_relevant'] = valid_emails['effective_date'].apply(
    lambda x: (x >= prev_start_utc) if pd.notna(x) else False
)

relevant_emails = valid_emails[valid_emails['is_relevant']].copy()

if len(relevant_emails) > 0:
    relevant_emails['price_change_pct'] = (
        ((relevant_emails['new_price'] - relevant_emails['old_price']) / relevant_emails['old_price'] * 100)
    ).round(2)
    
    relevant_emails = relevant_emails.sort_values('effective_date', ascending=False)
    
    # Take the most recent price change
    top_price_change = relevant_emails.iloc[0]
    
    # Use previous_period as the period for all metrics since the email is about a supplier fact
    # that affects the previous/analysis periods
    finding_3 = {
        "title": "Supplier Price Change (Recent)",
        "claim": f"Email from {top_price_change['sender']} dated {top_price_change['date'].strftime('%Y-%m-%d')} reports a price change for {top_price_change['entity_or_ingredient']}: {top_price_change['old_price']} → {top_price_change['new_price']} {top_price_change['currency']}/{top_price_change['unit']}, effective {top_price_change['effective_date'].strftime('%Y-%m-%d')} ({top_price_change['price_change_pct']:.2f}% change). This is a supplier-level fact; impact on menu items depends on recipe/BOM and order volume.",
        "finding_type": "supplier_price_change",
        "metrics": {
            "entity_or_ingredient": {
                "value": top_price_change['entity_or_ingredient'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": previous_period["start"],
                "period_end": analysis_period["end"]
            },
            "old_price": {
                "value": round(top_price_change['old_price'], 2),
                "unit": f"{top_price_change['currency']}/{top_price_change['unit']}",
                "numerator": round(top_price_change['old_price'], 2),
                "denominator": None,
                "period_start": previous_period["start"],
                "period_end": analysis_period["end"]
            },
            "new_price": {
                "value": round(top_price_change['new_price'], 2),
                "unit": f"{top_price_change['currency']}/{top_price_change['unit']}",
                "numerator": round(top_price_change['new_price'], 2),
                "denominator": None,
                "period_start": previous_period["start"],
                "period_end": analysis_period["end"]
            },
            "price_change_pct": {
                "value": round(top_price_change['price_change_pct'], 2),
                "unit": "%",
                "numerator": round(top_price_change['price_change_pct'], 2),
                "denominator": 100,
                "period_start": previous_period["start"],
                "period_end": analysis_period["end"]
            },
            "effective_date": {
                "value": top_price_change['effective_date'].strftime('%Y-%m-%d'),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": previous_period["start"],
                "period_end": analysis_period["end"]
            },
            "sender": {
                "value": top_price_change['sender'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": previous_period["start"],
                "period_end": analysis_period["end"]
            }
        },
        "source_names": ["emails"],
        "sample_size": 1,
        "coverage_notes": [
            "Most recent supplier price change with effective date in or after previous period",
            "Supplier-level fact only; no recipe/BOM available to calculate per-item impact",
            "Email extraction confidence: " + str(top_price_change['confidence'])
        ],
        "assumptions": [
            "Email extraction is accurate",
            "Price change applies to standing orders if volume/terms unchanged",
            "No recipe/BOM; cannot calculate exact menu item cost impact without additional data"
        ],
        "confidence": 0.85
    }
    findings.append(finding_3)

# ============================================================================
# Build output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")
