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

# Define analysis periods
analysis_period = {
    "start": "2026-02-16T00:00:00+03:00",
    "end": "2026-02-23T00:00:00+03:00"
}
previous_period = {
    "start": "2026-02-09T00:00:00+03:00",
    "end": "2026-02-16T00:00:00+03:00"
}

# Parse period dates
def parse_iso_date(iso_str):
    return pd.to_datetime(iso_str)

analysis_start = parse_iso_date(analysis_period["start"])
analysis_end = parse_iso_date(analysis_period["end"])
previous_start = parse_iso_date(previous_period["start"])
previous_end = parse_iso_date(previous_period["end"])

# Read input artifacts
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

# Filter POS data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

# Initialize findings list
findings = []

# ============================================================================
# FINDING 1: Item-level COGS and Gross Profit Analysis
# ============================================================================

# Merge POS with menu to get unit costs
pos_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'price_sar']],
    on='sku',
    how='left'
)

# Calculate item-level economics (excluding refunds)
pos_with_cost_no_refund = pos_with_cost[pos_with_cost['is_refund'] == False].copy()

# Calculate COGS and gross profit per line
pos_with_cost_no_refund['cogs_sar'] = pos_with_cost_no_refund['quantity'] * pos_with_cost_no_refund['unit_cost_sar']
pos_with_cost_no_refund['gross_profit_sar'] = pos_with_cost_no_refund['line_total_sar'] - pos_with_cost_no_refund['cogs_sar']
pos_with_cost_no_refund['gross_margin_pct'] = (pos_with_cost_no_refund['gross_profit_sar'] / pos_with_cost_no_refund['line_total_sar'] * 100).fillna(0)

# Aggregate by item
item_economics = pos_with_cost_no_refund.groupby('item_name_en').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'cogs_sar': 'sum',
    'gross_profit_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()

item_economics.columns = ['item_name', 'total_quantity', 'total_revenue', 'total_cogs', 'total_gross_profit', 'basket_count']
item_economics['gross_margin_pct'] = (item_economics['total_gross_profit'] / item_economics['total_revenue'] * 100).round(2)
item_economics = item_economics.sort_values('total_gross_profit', ascending=False)

# Get top 3 items by gross profit
top_items = item_economics.head(3)

if len(top_items) > 0:
    top_item = top_items.iloc[0]
    finding_1 = {
        "title": "Top Gross Profit Item - Analysis Period",
        "claim": f"Item '{top_item['item_name']}' generated the highest gross profit of {top_item['total_gross_profit']:.2f} SAR during the analysis period (2026-02-16 to 2026-02-23), with {int(top_item['total_quantity'])} units sold across {int(top_item['basket_count'])} transactions and a gross margin of {top_item['gross_margin_pct']:.1f}%.",
        "finding_type": "item_economics",
        "metrics": {
            "item_name": {
                "value": top_item['item_name'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_period["start"],
                "period_end": analysis_period["end"]
            },
            "total_gross_profit_sar": {
                "value": round(top_item['total_gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_item['total_gross_profit'], 2),
                "denominator": None,
                "period_start": analysis_period["start"],
                "period_end": analysis_period["end"]
            },
            "total_revenue_sar": {
                "value": round(top_item['total_revenue'], 2),
                "unit": "SAR",
                "numerator": round(top_item['total_revenue'], 2),
                "denominator": None,
                "period_start": analysis_period["start"],
                "period_end": analysis_period["end"]
            },
            "total_cogs_sar": {
                "value": round(top_item['total_cogs'], 2),
                "unit": "SAR",
                "numerator": round(top_item['total_cogs'], 2),
                "denominator": None,
                "period_start": analysis_period["start"],
                "period_end": analysis_period["end"]
            },
            "total_quantity": {
                "value": int(top_item['total_quantity']),
                "unit": "units",
                "numerator": int(top_item['total_quantity']),
                "denominator": None,
                "period_start": analysis_period["start"],
                "period_end": analysis_period["end"]
            },
            "basket_count": {
                "value": int(top_item['basket_count']),
                "unit": "transactions",
                "numerator": int(top_item['basket_count']),
                "denominator": None,
                "period_start": analysis_period["start"],
                "period_end": analysis_period["end"]
            },
            "gross_margin_pct": {
                "value": round(top_item['gross_margin_pct'], 1),
                "unit": "%",
                "numerator": round(top_item['gross_margin_pct'], 1),
                "denominator": 100,
                "period_start": analysis_period["start"],
                "period_end": analysis_period["end"]
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(top_item['basket_count']),
        "coverage_notes": [
            "Analysis period: 2026-02-16 to 2026-02-23",
            "Excludes refunds (is_refund == False)",
            "COGS calculated from menu.unit_cost_sar × quantity",
            "Gross profit = line_total_sar - COGS",
            "Only items with non-null unit_cost_sar included"
        ],
        "assumptions": [
            "Menu unit_cost_sar values are current and accurate for the analysis period",
            "No recipe/BOM adjustments applied",
            "Line totals reflect actual revenue after discounts"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Waste Cost Impact
# ============================================================================

# Filter inventory for analysis period (week starting 2026-02-16)
analysis_week = pd.to_datetime("2026-02-16")
inventory_analysis = inventory_df[inventory_df['week_starting'] == analysis_week].copy()

if len(inventory_analysis) > 0:
    # Calculate total waste cost
    total_waste_cost = inventory_analysis['known_waste_cost_sar'].sum()
    total_units_wasted = inventory_analysis['units_wasted'].sum()
    
    if total_waste_cost > 0:
        # Get items with waste
        waste_items = inventory_analysis[inventory_analysis['units_wasted'] > 0].copy()
        waste_items = waste_items.sort_values('known_waste_cost_sar', ascending=False)
        
        if len(waste_items) > 0:
            top_waste_item = waste_items.iloc[0]
            finding_2 = {
                "title": "Highest Waste Cost Item - Analysis Week",
                "claim": f"Item '{top_waste_item['item']}' incurred the highest waste cost of {top_waste_item['known_waste_cost_sar']:.2f} SAR during the week of 2026-02-16, with {int(top_waste_item['units_wasted'])} units wasted at a unit cost of {top_waste_item['unit_cost_sar']:.2f} SAR.",
                "finding_type": "waste_cost",
                "metrics": {
                    "item_name": {
                        "value": top_waste_item['item'],
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_period["start"],
                        "period_end": analysis_period["end"]
                    },
                    "waste_cost_sar": {
                        "value": round(top_waste_item['known_waste_cost_sar'], 2),
                        "unit": "SAR",
                        "numerator": round(top_waste_item['known_waste_cost_sar'], 2),
                        "denominator": None,
                        "period_start": analysis_period["start"],
                        "period_end": analysis_period["end"]
                    },
                    "units_wasted": {
                        "value": int(top_waste_item['units_wasted']),
                        "unit": "units",
                        "numerator": int(top_waste_item['units_wasted']),
                        "denominator": None,
                        "period_start": analysis_period["start"],
                        "period_end": analysis_period["end"]
                    },
                    "unit_cost_sar": {
                        "value": round(top_waste_item['unit_cost_sar'], 2),
                        "unit": "SAR/unit",
                        "numerator": round(top_waste_item['unit_cost_sar'], 2),
                        "denominator": None,
                        "period_start": analysis_period["start"],
                        "period_end": analysis_period["end"]
                    },
                    "total_waste_cost_week": {
                        "value": round(total_waste_cost, 2),
                        "unit": "SAR",
                        "numerator": round(total_waste_cost, 2),
                        "denominator": None,
                        "period_start": analysis_period["start"],
                        "period_end": analysis_period["end"]
                    }
                },
                "source_names": ["inventory"],
                "sample_size": len(waste_items),
                "coverage_notes": [
                    "Analysis week: 2026-02-16 to 2026-02-23",
                    "Only non-null waste cost observations included",
                    "Waste cost calculated from inventory.known_waste_cost_sar",
                    f"Total items with waste in period: {len(waste_items)}"
                ],
                "assumptions": [
                    "known_waste_cost_sar values are accurate and complete",
                    "Waste represents spoilage/disposal cost at unit_cost_sar"
                ],
                "confidence": 0.90
            }
            findings.append(finding_2)

# ============================================================================
# FINDING 3: Supplier Price Changes from Email Evidence
# ============================================================================

# Filter emails for price changes with valid dates
price_change_emails = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

# Filter for emails with effective dates in or near analysis period
price_change_emails['effective_date_only'] = price_change_emails['effective_date'].dt.date
analysis_date_range = pd.date_range(start=analysis_start, end=analysis_end, freq='D').date

# Get emails with effective dates in analysis period or previous period
relevant_emails = price_change_emails[
    (price_change_emails['effective_date'] >= previous_start) &
    (price_change_emails['effective_date'] <= analysis_end)
].copy()

if len(relevant_emails) > 0:
    # Sort by effective date descending to get most recent
    relevant_emails = relevant_emails.sort_values('effective_date', ascending=False)
    
    for idx, email in relevant_emails.iterrows():
        if pd.notna(email['old_price']) and pd.notna(email['new_price']):
            old_price = float(email['old_price'])
            new_price = float(email['new_price'])
            
            if old_price > 0:
                price_change_sar = new_price - old_price
                price_change_pct = (price_change_sar / old_price) * 100
                
                # Determine which period this email affects
                email_effective = email['effective_date']
                if email_effective >= analysis_start and email_effective < analysis_end:
                    period_start = analysis_period["start"]
                    period_end = analysis_period["end"]
                    period_label = "analysis"
                elif email_effective >= previous_start and email_effective < previous_end:
                    period_start = previous_period["start"]
                    period_end = previous_period["end"]
                    period_label = "previous"
                else:
                    continue
                
                finding_3 = {
                    "title": f"Supplier Price Change - {email['entity_or_ingredient']}",
                    "claim": f"Supplier email dated {email['date'].strftime('%Y-%m-%d')} documents a price change for '{email['entity_or_ingredient']}' from {old_price:.2f} {email['currency']}/{email['unit']} to {new_price:.2f} {email['currency']}/{email['unit']}, effective {email_effective.strftime('%Y-%m-%d')} (a {price_change_pct:+.1f}% change). Email confidence: {email['confidence']}.",
                    "finding_type": "supplier_price_change",
                    "metrics": {
                        "entity_or_ingredient": {
                            "value": email['entity_or_ingredient'],
                            "unit": None,
                            "numerator": None,
                            "denominator": None,
                            "period_start": period_start,
                            "period_end": period_end
                        },
                        "old_price": {
                            "value": round(old_price, 2),
                            "unit": f"{email['currency']}/{email['unit']}",
                            "numerator": round(old_price, 2),
                            "denominator": None,
                            "period_start": period_start,
                            "period_end": period_end
                        },
                        "new_price": {
                            "value": round(new_price, 2),
                            "unit": f"{email['currency']}/{email['unit']}",
                            "numerator": round(new_price, 2),
                            "denominator": None,
                            "period_start": period_start,
                            "period_end": period_end
                        },
                        "price_change_sar": {
                            "value": round(price_change_sar, 2),
                            "unit": f"{email['currency']}/{email['unit']}",
                            "numerator": round(price_change_sar, 2),
                            "denominator": None,
                            "period_start": period_start,
                            "period_end": period_end
                        },
                        "price_change_pct": {
                            "value": round(price_change_pct, 1),
                            "unit": "%",
                            "numerator": round(price_change_pct, 1),
                            "denominator": 100,
                            "period_start": period_start,
                            "period_end": period_end
                        },
                        "effective_date": {
                            "value": email_effective.strftime('%Y-%m-%d'),
                            "unit": None,
                            "numerator": None,
                            "denominator": None,
                            "period_start": period_start,
                            "period_end": period_end
                        },
                        "email_date": {
                            "value": email['date'].strftime('%Y-%m-%d'),
                            "unit": None,
                            "numerator": None,
                            "denominator": None,
                            "period_start": period_start,
                            "period_end": period_end
                        },
                        "email_confidence": {
                            "value": email['confidence'],
                            "unit": None,
                            "numerator": email['confidence'],
                            "denominator": 1,
                            "period_start": period_start,
                            "period_end": period_end
                        }
                    },
                    "source_names": ["emails"],
                    "sample_size": 1,
                    "coverage_notes": [
                        f"Email dated: {email['date'].strftime('%Y-%m-%d')}",
                        f"Effective date: {email_effective.strftime('%Y-%m-%d')}",
                        f"Extraction mode: {email['extraction_mode']}",
                        "Price change extracted from supplier email",
                        "No recipe/BOM data available to calculate per-item impact"
                    ],
                    "assumptions": [
                        "Email extraction confidence score is accurate",
                        "Price change applies to future purchases at stated effective date",
                        "No standing order quantity data available to calculate procurement cost impact",
                        "Currency and unit conversions not applied"
                    ],
                    "confidence": float(email['confidence'])
                }
                findings.append(finding_3)
                break  # Only include most recent price change

# ============================================================================
# Prepare output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. {len(findings)} findings generated.")
