import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
inventory_df = pd.read_parquet(inputs['inventory'])
emails_df = pd.read_parquet(inputs['emails'])

# Define analysis period
analysis_start = "2026-01-26"
analysis_end = "2026-02-02"
previous_start = "2026-01-19"
previous_end = "2026-01-26"

# Convert calendar_date to datetime for filtering
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])

# Filter POS for analysis period (exclude refunds)
pos_analysis = pos_df[
    (pos_df['calendar_date'] >= analysis_start) &
    (pos_df['calendar_date'] < analysis_end) &
    (pos_df['is_refund'] == False)
].copy()

pos_previous = pos_df[
    (pos_df['calendar_date'] >= previous_start) &
    (pos_df['calendar_date'] < previous_end) &
    (pos_df['is_refund'] == False)
].copy()

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

# Calculate item-level metrics
item_metrics = pos_with_cost.groupby('sku').agg({
    'item_name': 'first',
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'unit_cost_sar': 'first',
    'price_sar': 'first'
}).reset_index()

item_metrics.columns = ['sku', 'item_name', 'units_sold', 'revenue', 'unit_cost', 'menu_price']

# Calculate COGS and Gross Profit
item_metrics['cogs'] = item_metrics['units_sold'] * item_metrics['unit_cost']
item_metrics['gross_profit'] = item_metrics['revenue'] - item_metrics['cogs']
item_metrics['gross_margin_pct'] = (item_metrics['gross_profit'] / item_metrics['revenue'] * 100).round(2)

# Sort by gross profit
item_metrics_sorted = item_metrics.sort_values('gross_profit', ascending=False)

# Get top item
if len(item_metrics_sorted) > 0:
    top_item = item_metrics_sorted.iloc[0]
    
    # Verify it's the highest
    is_top = True
    top_profit = top_item['gross_profit']
    
    finding_1 = {
        "title": "Top Gross Profit Item - Analysis Week",
        "claim": f"{top_item['item_name']} generated {top_profit:.2f} SAR gross profit with {top_item['units_sold']:.0f} units sold at {top_item['gross_margin_pct']:.2f}% gross margin during the analysis week.",
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
            "units_sold": {
                "value": int(top_item['units_sold']),
                "unit": "units",
                "numerator": int(top_item['units_sold']),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "revenue": {
                "value": round(top_item['revenue'], 2),
                "unit": "SAR",
                "numerator": round(top_item['revenue'], 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "cogs": {
                "value": round(top_item['cogs'], 2),
                "unit": "SAR",
                "numerator": round(top_item['cogs'], 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "gross_profit": {
                "value": round(top_item['gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_item['gross_profit'], 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "gross_margin_percent": {
                "value": round(top_item['gross_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_item['gross_margin_pct'], 2),
                "denominator": 100,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(top_item['units_sold']),
        "coverage_notes": [
            "Analysis period: 2026-01-26 to 2026-02-02",
            "Refunds excluded from calculation",
            "Menu unit_cost_sar applied to all units sold",
            "Revenue calculated from POS line_total_sar (post-discount)"
        ],
        "assumptions": [
            "Menu unit_cost_sar is applicable to all units sold in analysis period",
            "POS line_total_sar reflects actual revenue after discounts",
            "No recipe/BOM available; unit cost is applied uniformly",
            "Comparison is among all items with sales in analysis period"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Waste Cost Impact
# ============================================================================

# Filter inventory for analysis week
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
inventory_analysis = inventory_df[
    (inventory_df['week_starting'] >= analysis_start) &
    (inventory_df['week_starting'] < analysis_end)
].copy()

# Calculate total known waste cost (only non-null values)
waste_data = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()

if len(waste_data) > 0:
    total_waste_cost = waste_data['known_waste_cost_sar'].sum()
    waste_units = waste_data['units_wasted'].sum()
    waste_items = len(waste_data)
    
    finding_2 = {
        "title": "Quantified Waste Cost - Analysis Week",
        "claim": f"Known waste cost totaled {total_waste_cost:.2f} SAR across {waste_items} items ({waste_units:.0f} units wasted) during the analysis week.",
        "finding_type": "waste_economics",
        "metrics": {
            "total_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": round(total_waste_cost, 2),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "units_wasted": {
                "value": int(waste_units),
                "unit": "units",
                "numerator": int(waste_units),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "items_with_waste": {
                "value": waste_items,
                "unit": "count",
                "numerator": waste_items,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["inventory"],
        "sample_size": waste_items,
        "coverage_notes": [
            "Analysis period: 2026-01-26 to 2026-02-02",
            "Only non-null known_waste_cost_sar values included",
            "Blank waste values treated as unknown, not zero"
        ],
        "assumptions": [
            "Inventory known_waste_cost_sar is accurate and complete for recorded waste",
            "Waste cost reflects actual unit cost at time of waste"
        ],
        "confidence": 0.85
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
    # Convert dates
    price_change_emails['effective_date'] = pd.to_datetime(price_change_emails['effective_date'])
    price_change_emails['email_date'] = pd.to_datetime(price_change_emails['date'])
    
    # Calculate price change percentage
    price_change_emails['price_change_pct'] = (
        ((price_change_emails['new_price'] - price_change_emails['old_price']) / 
         price_change_emails['old_price'] * 100)
    ).round(2)
    
    # Sort by effective date
    price_change_emails_sorted = price_change_emails.sort_values('effective_date')
    
    # Get most recent price change
    if len(price_change_emails_sorted) > 0:
        latest_change = price_change_emails_sorted.iloc[-1]
        
        finding_3 = {
            "title": "Supplier Price Change - Latest Evidence",
            "claim": f"{latest_change['entity_or_ingredient']} price changed from {latest_change['old_price']:.2f} to {latest_change['new_price']:.2f} {latest_change['currency']}/{latest_change['unit']} (effective {latest_change['effective_date'].strftime('%Y-%m-%d')}), representing a {latest_change['price_change_pct']:.2f}% change.",
            "finding_type": "supplier_pricing",
            "metrics": {
                "ingredient": {
                    "value": latest_change['entity_or_ingredient'],
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": latest_change['effective_date'].strftime('%Y-%m-%d'),
                    "period_end": latest_change['effective_date'].strftime('%Y-%m-%d')
                },
                "old_price": {
                    "value": round(latest_change['old_price'], 2),
                    "unit": f"{latest_change['currency']}/{latest_change['unit']}",
                    "numerator": round(latest_change['old_price'], 2),
                    "denominator": None,
                    "period_start": latest_change['effective_date'].strftime('%Y-%m-%d'),
                    "period_end": latest_change['effective_date'].strftime('%Y-%m-%d')
                },
                "new_price": {
                    "value": round(latest_change['new_price'], 2),
                    "unit": f"{latest_change['currency']}/{latest_change['unit']}",
                    "numerator": round(latest_change['new_price'], 2),
                    "denominator": None,
                    "period_start": latest_change['effective_date'].strftime('%Y-%m-%d'),
                    "period_end": latest_change['effective_date'].strftime('%Y-%m-%d')
                },
                "price_change_percent": {
                    "value": round(latest_change['price_change_pct'], 2),
                    "unit": "%",
                    "numerator": round(latest_change['price_change_pct'], 2),
                    "denominator": 100,
                    "period_start": latest_change['effective_date'].strftime('%Y-%m-%d'),
                    "period_end": latest_change['effective_date'].strftime('%Y-%m-%d')
                },
                "price_delta": {
                    "value": round(latest_change['new_price'] - latest_change['old_price'], 2),
                    "unit": f"{latest_change['currency']}/{latest_change['unit']}",
                    "numerator": round(latest_change['new_price'] - latest_change['old_price'], 2),
                    "denominator": None,
                    "period_start": latest_change['effective_date'].strftime('%Y-%m-%d'),
                    "period_end": latest_change['effective_date'].strftime('%Y-%m-%d')
                }
            },
            "source_names": ["emails"],
            "sample_size": 1,
            "coverage_notes": [
                "Latest price change from supplier email evidence",
                "Effective date: " + latest_change['effective_date'].strftime('%Y-%m-%d'),
                "Email date: " + latest_change['email_date'].strftime('%Y-%m-%d'),
                "No recipe/BOM available; per-drink impact cannot be calculated without consumption rates"
            ],
            "assumptions": [
                "Email extraction confidence: " + str(latest_change['confidence']),
                "Price change applies to future procurement at stated unit",
                "No assumption made about standing order quantities or payment terms without explicit email evidence"
            ],
            "confidence": float(latest_change['confidence']) if pd.notna(latest_change['confidence']) else 0.75
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
