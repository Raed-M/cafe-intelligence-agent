import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

# Load run metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Define analysis periods
analysis_period = {
    "start": "2026-04-20T00:00:00+03:00",
    "end": "2026-04-27T00:00:00+03:00"
}
previous_period = {
    "start": "2026-04-13T00:00:00+03:00",
    "end": "2026-04-20T00:00:00+03:00"
}
trailing_baseline_periods = [
    {"start": "2026-04-13T00:00:00+03:00", "end": "2026-04-20T00:00:00+03:00"},
    {"start": "2026-04-06T00:00:00+03:00", "end": "2026-04-13T00:00:00+03:00"},
    {"start": "2026-03-30T00:00:00+03:00", "end": "2026-04-06T00:00:00+03:00"},
    {"start": "2026-03-23T00:00:00+03:00", "end": "2026-03-30T00:00:00+03:00"}
]

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

# Parse analysis periods with timezone awareness
analysis_start = pd.to_datetime(analysis_period['start'])
analysis_end = pd.to_datetime(analysis_period['end'])
previous_start = pd.to_datetime(previous_period['start'])
previous_end = pd.to_datetime(previous_period['end'])

# Ensure all datetime comparisons are timezone-aware
# Convert naive datetimes to UTC-aware, then to +03:00 for consistency
if pos_df['timestamp'].dt.tz is None:
    pos_df['timestamp'] = pos_df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('+03:00')
if inventory_df['week_starting'].dt.tz is None:
    inventory_df['week_starting'] = inventory_df['week_starting'].dt.tz_localize('UTC').dt.tz_convert('+03:00')
if menu_df['launch_date'].dt.tz is None:
    menu_df['launch_date'] = menu_df['launch_date'].dt.tz_localize('UTC').dt.tz_convert('+03:00')
if menu_df['retire_date'].dt.tz is None:
    menu_df['retire_date'] = menu_df['retire_date'].dt.tz_localize('UTC').dt.tz_convert('+03:00')
if emails_df['date'].dt.tz is None:
    emails_df['date'] = emails_df['date'].dt.tz_localize('UTC').dt.tz_convert('+03:00')
if emails_df['effective_date'].dt.tz is None:
    emails_df['effective_date'] = emails_df['effective_date'].dt.tz_localize('UTC').dt.tz_convert('+03:00')

# Filter POS for analysis period (exclude refunds)
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start) & 
    (pos_df['timestamp'] < analysis_end) &
    (pos_df['is_refund'] == False)
].copy()

# Filter POS for previous period (exclude refunds)
pos_previous = pos_df[
    (pos_df['timestamp'] >= previous_start) & 
    (pos_df['timestamp'] < previous_end) &
    (pos_df['is_refund'] == False)
].copy()

findings = []
result_data = {}

# ============================================================================
# FINDING 1: Item-level COGS and Gross Profit for top revenue items
# ============================================================================

# Calculate item-level metrics for analysis period
item_metrics = pos_analysis.groupby('sku').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()
item_metrics.columns = ['sku', 'units_sold', 'revenue', 'basket_count']

# Merge with menu to get unit costs
item_metrics = item_metrics.merge(
    menu_df[['sku', 'item_en', 'unit_cost_sar', 'category']],
    on='sku',
    how='left'
)

# Calculate COGS and gross profit
item_metrics['cogs'] = item_metrics['units_sold'] * item_metrics['unit_cost_sar']
item_metrics['gross_profit'] = item_metrics['revenue'] - item_metrics['cogs']
item_metrics['margin_pct'] = (item_metrics['gross_profit'] / item_metrics['revenue'] * 100).round(2)

# Sort by revenue and get top item
item_metrics_sorted = item_metrics.dropna(subset=['unit_cost_sar']).sort_values('revenue', ascending=False)

if len(item_metrics_sorted) > 0:
    top_item = item_metrics_sorted.iloc[0]
    
    finding_1_key = f"item_{top_item['sku']}_analysis_period"
    result_data[finding_1_key] = {
        'sku': top_item['sku'],
        'item_en': top_item['item_en'],
        'units_sold': int(top_item['units_sold']),
        'revenue': round(top_item['revenue'], 2),
        'unit_cost_sar': round(top_item['unit_cost_sar'], 2),
        'cogs': round(top_item['cogs'], 2),
        'gross_profit': round(top_item['gross_profit'], 2),
        'margin_pct': top_item['margin_pct'],
        'basket_count': int(top_item['basket_count'])
    }
    
    findings.append({
        "title": f"Top Revenue Item ({top_item['item_en']}): {top_item['margin_pct']:.2f}% Gross Margin",
        "claim": f"Item {top_item['sku']} ({top_item['item_en']}) generated {top_item['revenue']:.2f} SAR revenue from {int(top_item['units_sold'])} units sold during analysis period. With menu unit cost of {top_item['unit_cost_sar']:.2f} SAR/unit, COGS totals {top_item['cogs']:.2f} SAR, yielding gross profit of {top_item['gross_profit']:.2f} SAR and margin of {top_item['margin_pct']:.2f}%.",
        "finding_type": "item_economics",
        "metrics": {
            "revenue": {
                "value": round(top_item['revenue'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "units_sold": {
                "value": int(top_item['units_sold']),
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "unit_cost_sar": {
                "value": round(top_item['unit_cost_sar'], 2),
                "unit": "SAR/unit",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "cogs": {
                "value": round(top_item['cogs'], 2),
                "unit": "SAR",
                "numerator": int(top_item['units_sold']),
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "gross_profit": {
                "value": round(top_item['gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_item['revenue'], 2),
                "denominator": round(top_item['cogs'], 2),
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "margin_pct": {
                "value": top_item['margin_pct'],
                "unit": "%",
                "numerator": round(top_item['gross_profit'], 2),
                "denominator": round(top_item['revenue'], 2),
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            },
            "basket_count": {
                "value": int(top_item['basket_count']),
                "unit": "transactions",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_period['start'],
                "period_end": analysis_period['end']
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(top_item['basket_count']),
        "coverage_notes": [
            f"Analysis period: {analysis_period['start']} to {analysis_period['end']}",
            "Refunds excluded from revenue and unit calculations",
            f"Menu unit cost of {top_item['unit_cost_sar']:.2f} SAR/unit used for COGS calculation",
            f"Top item by revenue among {len(item_metrics_sorted)} items with known unit costs"
        ],
        "assumptions": [
            "No recipe/BOM available; using menu-level unit costs",
            "Unit cost sourced from menu data only; actual ingredient costs may differ",
            "Menu unit cost is constant across analysis period",
            "Recommend validation against supplier invoices and recipe BOM when available"
        ],
        "confidence": 0.85
    })

# ============================================================================
# FINDING 2: Waste Cost Impact (if waste data available)
# ============================================================================

# Filter inventory for analysis week
inv_analysis_week = inventory_df[
    inventory_df['week_starting'] == pd.to_datetime('2026-04-20').tz_localize('UTC').tz_convert('+03:00')
].copy()

if len(inv_analysis_week) > 0:
    # Calculate total waste cost for items with known waste
    waste_items = inv_analysis_week[inv_analysis_week['known_waste_cost_sar'].notna()].copy()
    
    if len(waste_items) > 0:
        total_waste_cost = waste_items['known_waste_cost_sar'].sum()
        total_units_wasted = waste_items['units_wasted'].sum()
        
        finding_2_key = "waste_cost_analysis_period"
        result_data[finding_2_key] = {
            'total_waste_cost_sar': round(total_waste_cost, 2),
            'total_units_wasted': int(total_units_wasted),
            'items_with_waste': len(waste_items),
            'avg_waste_cost_per_item': round(total_waste_cost / len(waste_items), 2) if len(waste_items) > 0 else 0
        }
        
        findings.append({
            "title": f"Quantified Waste Cost: {total_waste_cost:.2f} SAR",
            "claim": f"During week of {inv_analysis_week.iloc[0]['week_starting'].strftime('%Y-%m-%d')}, {len(waste_items)} items with known waste observations incurred {total_waste_cost:.2f} SAR in waste cost from {int(total_units_wasted)} units wasted. Average waste cost per item: {round(total_waste_cost / len(waste_items), 2):.2f} SAR.",
            "finding_type": "waste_economics",
            "metrics": {
                "total_waste_cost_sar": {
                    "value": round(total_waste_cost, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_period['start'],
                    "period_end": analysis_period['end']
                },
                "total_units_wasted": {
                    "value": int(total_units_wasted),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_period['start'],
                    "period_end": analysis_period['end']
                },
                "items_with_waste": {
                    "value": len(waste_items),
                    "unit": "items",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_period['start'],
                    "period_end": analysis_period['end']
                },
                "avg_waste_cost_per_item": {
                    "value": round(total_waste_cost / len(waste_items), 2),
                    "unit": "SAR",
                    "numerator": round(total_waste_cost, 2),
                    "denominator": len(waste_items),
                    "period_start": analysis_period['start'],
                    "period_end": analysis_period['end']
                }
            },
            "source_names": ["inventory"],
            "sample_size": len(waste_items),
            "coverage_notes": [
                f"Analysis period: {analysis_period['start']} to {analysis_period['end']}",
                "Only items with non-null known_waste_cost_sar included",
                f"Week starting: {inv_analysis_week.iloc[0]['week_starting'].strftime('%Y-%m-%d')}",
                "Blank waste values excluded per data quality rules"
            ],
            "assumptions": [
                "Waste cost reflects actual unit cost at time of waste",
                "No recipe/BOM adjustments applied"
            ],
            "confidence": 0.90
        })

# ============================================================================
# FINDING 3: Supplier Price Changes with Procurement Impact
# ============================================================================

# Filter emails for price changes with effective dates in or near analysis period
price_change_emails = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

# Only include price changes with effective dates in analysis or previous period
valid_price_changes = price_change_emails[
    ((price_change_emails['effective_date'] >= analysis_start) & 
     (price_change_emails['effective_date'] < analysis_end)) |
    ((price_change_emails['effective_date'] >= previous_start) & 
     (price_change_emails['effective_date'] < previous_end))
].copy()

if len(valid_price_changes) > 0:
    # Take first price change as example
    price_change = valid_price_changes.iloc[0]
    
    old_price = float(price_change['old_price'])
    new_price = float(price_change['new_price'])
    price_delta = new_price - old_price
    price_delta_pct = (price_delta / old_price * 100) if old_price != 0 else 0
    
    # Determine which period this applies to
    if (price_change['effective_date'] >= analysis_start) and (price_change['effective_date'] < analysis_end):
        period_start = analysis_period['start']
        period_end = analysis_period['end']
    else:
        period_start = previous_period['start']
        period_end = previous_period['end']
    
    finding_3_key = f"price_change_{price_change['entity_or_ingredient']}"
    result_data[finding_3_key] = {
        'entity': price_change['entity_or_ingredient'],
        'old_price': round(old_price, 2),
        'new_price': round(new_price, 2),
        'currency': price_change['currency'],
        'unit': price_change['unit'],
        'price_delta': round(price_delta, 2),
        'price_delta_pct': round(price_delta_pct, 2),
        'effective_date': price_change['effective_date'].strftime('%Y-%m-%d'),
        'sender': price_change['sender'],
        'confidence': price_change['confidence']
    }
    
    findings.append({
        "title": f"Supplier Price Change: {price_change['entity_or_ingredient']} ({price_delta_pct:+.2f}%)",
        "claim": f"Email from {price_change['sender']} dated {price_change['date'].strftime('%Y-%m-%d')} reports price change for {price_change['entity_or_ingredient']}: {old_price:.2f} {price_change['currency']}/{price_change['unit']} → {new_price:.2f} {price_change['currency']}/{price_change['unit']}, effective {price_change['effective_date'].strftime('%Y-%m-%d')}. Price delta: {price_delta:+.2f} {price_change['currency']} ({price_delta_pct:+.2f}%). No standing order quantity found in email; procurement cost impact cannot be quantified without order volume data.",
        "finding_type": "supplier_price_change",
        "metrics": {
            "old_price": {
                "value": round(old_price, 2),
                "unit": f"{price_change['currency']}/{price_change['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": period_start,
                "period_end": period_end
            },
            "new_price": {
                "value": round(new_price, 2),
                "unit": f"{price_change['currency']}/{price_change['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": period_start,
                "period_end": period_end
            },
            "price_delta": {
                "value": round(price_delta, 2),
                "unit": f"{price_change['currency']}/{price_change['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": period_start,
                "period_end": period_end
            },
            "price_delta_pct": {
                "value": round(price_delta_pct, 2),
                "unit": "%",
                "numerator": round(price_delta, 2),
                "denominator": round(old_price, 2),
                "period_start": period_start,
                "period_end": period_end
            }
        },
        "source_names": ["emails"],
        "sample_size": 1,
        "coverage_notes": [
            f"Email date: {price_change['date'].strftime('%Y-%m-%d')}",
            f"Effective date: {price_change['effective_date'].strftime('%Y-%m-%d')}",
            f"Sender: {price_change['sender']}",
            "No standing order quantity found in email text",
            "Procurement cost impact cannot be calculated without order volume"
        ],
        "assumptions": [
            "Price change applies to specified entity/ingredient only",
            "No recipe/BOM available to map ingredient to menu items",
            "Standing order quantity and payment terms unknown"
        ],
        "confidence": float(price_change['confidence']) if pd.notna(price_change['confidence']) else 0.75
    })

# ============================================================================
# Build output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings[:3],  # Max 3 findings
    "result_data": result_data
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(json.dumps(output, indent=2, default=str))
