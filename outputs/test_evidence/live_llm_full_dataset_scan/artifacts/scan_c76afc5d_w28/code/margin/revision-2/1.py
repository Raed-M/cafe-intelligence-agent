import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Load run metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Load artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
inventory_df = pd.read_parquet(inputs['inventory'])
emails_df = pd.read_parquet(inputs['emails'])

# Define analysis period
analysis_start = datetime.fromisoformat("2026-07-20T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-07-27T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-07-13T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-07-20T00:00:00+03:00")

# Convert timestamp columns to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
emails_df['date'] = pd.to_datetime(emails_df['date'])
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'])

# Filter POS for analysis period (exclude refunds for revenue, but include for transaction count)
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()

findings = []

# ============================================================================
# FINDING 1: Gross Profit & Margin Analysis (Aggregate, Analysis Period)
# ============================================================================

# Calculate revenue and COGS for analysis period (excluding refunds from revenue)
pos_analysis_non_refund = pos_analysis[pos_analysis['is_refund'] == False].copy()
pos_analysis_refund = pos_analysis[pos_analysis['is_refund'] == True].copy()

# Revenue: sum of line_total_sar for non-refunds minus refunds
revenue_non_refund = pos_analysis_non_refund['line_total_sar'].sum()
revenue_refund = pos_analysis_refund['line_total_sar'].sum()
net_revenue = revenue_non_refund + revenue_refund  # refunds are negative

# Merge POS with menu to get unit costs
pos_with_cost = pos_analysis.merge(menu_df[['sku', 'unit_cost_sar']], on='sku', how='left')

# Calculate COGS: quantity × unit_cost_sar (for all rows, including refunds which will be negative)
pos_with_cost['cogs_line'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
total_cogs = pos_with_cost['cogs_line'].sum()

# Gross profit
gross_profit = net_revenue - total_cogs

# Margin percentage
if net_revenue != 0:
    margin_pct = (gross_profit / net_revenue) * 100
else:
    margin_pct = 0

# Transaction count (unique transaction_ids, excluding refunds for basket count)
transaction_count = pos_analysis_non_refund['transaction_id'].nunique()

# Sample size: number of line items
sample_size = len(pos_analysis_non_refund)

# Coverage: SKUs with cost data
skus_with_cost = pos_with_cost[pos_with_cost['unit_cost_sar'].notna()]['sku'].nunique()
total_skus = pos_with_cost['sku'].nunique()

finding_1 = {
    "title": "Gross Profit & Margin Analysis (Analysis Period)",
    "claim": f"During the analysis period (2026-07-20 to 2026-07-27), the cafe generated net revenue of {net_revenue:.2f} SAR with total COGS of {total_cogs:.2f} SAR, resulting in gross profit of {gross_profit:.2f} SAR and a gross margin of {margin_pct:.2f}%.",
    "finding_type": "profitability_analysis",
    "metrics": {
        "net_revenue_sar": {
            "value": round(net_revenue, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-07-20T00:00:00+03:00",
            "period_end": "2026-07-27T00:00:00+03:00"
        },
        "total_cogs_sar": {
            "value": round(total_cogs, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-07-20T00:00:00+03:00",
            "period_end": "2026-07-27T00:00:00+03:00"
        },
        "gross_profit_sar": {
            "value": round(gross_profit, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-07-20T00:00:00+03:00",
            "period_end": "2026-07-27T00:00:00+03:00"
        },
        "gross_margin_pct": {
            "value": round(margin_pct, 2),
            "unit": "%",
            "numerator": round(gross_profit, 2),
            "denominator": round(net_revenue, 2),
            "period_start": "2026-07-20T00:00:00+03:00",
            "period_end": "2026-07-27T00:00:00+03:00"
        },
        "transaction_count": {
            "value": transaction_count,
            "unit": "baskets",
            "numerator": None,
            "denominator": None,
            "period_start": "2026-07-20T00:00:00+03:00",
            "period_end": "2026-07-27T00:00:00+03:00"
        }
    },
    "source_names": ["pos", "menu"],
    "sample_size": sample_size,
    "coverage_notes": [
        f"SKUs with unit cost data: {skus_with_cost} of {total_skus} menu items",
        "Revenue includes net of refunds (refund line_total_sar values are negative)",
        "COGS calculated as quantity × unit_cost_sar from menu.parquet",
        "Transaction count based on unique transaction_ids excluding refunds"
    ],
    "assumptions": [
        "Menu unit_cost_sar values are current and applicable to all sales in analysis period",
        "Unit costs in menu.parquet reflect actual procurement costs at time of sale",
        "No recipe/BOM data available; unit costs are treated as stated in menu",
        "Refunds are included in net revenue calculation (negative line_total_sar)"
    ],
    "confidence": 0.85
}

findings.append(finding_1)

# ============================================================================
# FINDING 2: Waste Cost Impact (Inventory Data)
# ============================================================================

# Filter inventory for analysis period week
inventory_analysis = inventory_df[inventory_df['week_starting'] == pd.Timestamp('2026-07-20')]

# Calculate total waste cost (only non-null values)
waste_cost_total = inventory_analysis['known_waste_cost_sar'].sum()
waste_rows = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()]
waste_count = len(waste_rows)

if waste_count > 0:
    waste_items = waste_rows[['sku', 'item', 'units_wasted', 'known_waste_cost_sar']].to_dict('records')
    
    finding_2 = {
        "title": "Quantified Waste Cost (Analysis Period)",
        "claim": f"During the week of 2026-07-20, {waste_count} items with known waste observations incurred a total waste cost of {waste_cost_total:.2f} SAR.",
        "finding_type": "waste_cost_analysis",
        "metrics": {
            "total_waste_cost_sar": {
                "value": round(waste_cost_total, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-07-20T00:00:00+03:00",
                "period_end": "2026-07-27T00:00:00+03:00"
            },
            "waste_items_count": {
                "value": waste_count,
                "unit": "items",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-07-20T00:00:00+03:00",
                "period_end": "2026-07-27T00:00:00+03:00"
            }
        },
        "source_names": ["inventory"],
        "sample_size": waste_count,
        "coverage_notes": [
            "Only non-null known_waste_cost_sar values included",
            "Waste cost sourced from inventory.parquet known_waste_cost_sar column",
            "Week starting 2026-07-20 corresponds to analysis period"
        ],
        "assumptions": [
            "known_waste_cost_sar values are accurate and complete for observed waste",
            "Blank waste values are treated as unknown, not zero"
        ],
        "confidence": 0.90
    }
    
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Supplier Price Changes from Email Evidence
# ============================================================================

# Filter emails for price changes with effective dates in or near analysis period
price_change_emails = emails_df[
    (emails_df['category'] == 'price_change') & 
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna())
].copy()

if len(price_change_emails) > 0:
    # Sort by effective_date descending to get most recent
    price_change_emails = price_change_emails.sort_values('effective_date', ascending=False)
    
    # Calculate price change percentage for each
    price_change_emails['price_delta'] = price_change_emails['new_price'] - price_change_emails['old_price']
    price_change_emails['price_change_pct'] = (price_change_emails['price_delta'] / price_change_emails['old_price']) * 100
    
    # Get most recent price change
    most_recent = price_change_emails.iloc[0]
    
    finding_3 = {
        "title": "Supplier Price Changes Detected",
        "claim": f"Email evidence identifies {len(price_change_emails)} supplier price changes. Most recent: {most_recent['entity_or_ingredient']} price changed from {most_recent['old_price']:.2f} to {most_recent['new_price']:.2f} SAR per {most_recent['unit']} (effective {most_recent['effective_date'].strftime('%Y-%m-%d')}), representing a {most_recent['price_change_pct']:.2f}% increase. No recipe/BOM available to calculate exact menu item cost impact.",
        "finding_type": "supplier_cost_analysis",
        "metrics": {
            "total_price_changes_identified": {
                "value": len(price_change_emails),
                "unit": "changes",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-07-20T00:00:00+03:00",
                "period_end": "2026-07-27T00:00:00+03:00"
            },
            "ingredient_name": {
                "value": most_recent['entity_or_ingredient'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": most_recent['effective_date'].isoformat() if pd.notna(most_recent['effective_date']) else None,
                "period_end": most_recent['effective_date'].isoformat() if pd.notna(most_recent['effective_date']) else None
            },
            "old_price": {
                "value": round(most_recent['old_price'], 2),
                "unit": f"SAR per {most_recent['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": most_recent['effective_date'].isoformat() if pd.notna(most_recent['effective_date']) else None,
                "period_end": most_recent['effective_date'].isoformat() if pd.notna(most_recent['effective_date']) else None
            },
            "new_price": {
                "value": round(most_recent['new_price'], 2),
                "unit": f"SAR per {most_recent['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": most_recent['effective_date'].isoformat() if pd.notna(most_recent['effective_date']) else None,
                "period_end": most_recent['effective_date'].isoformat() if pd.notna(most_recent['effective_date']) else None
            },
            "price_change_pct": {
                "value": round(most_recent['price_change_pct'], 2),
                "unit": "%",
                "numerator": round(most_recent['price_delta'], 2),
                "denominator": round(most_recent['old_price'], 2),
                "period_start": most_recent['effective_date'].isoformat() if pd.notna(most_recent['effective_date']) else None,
                "period_end": most_recent['effective_date'].isoformat() if pd.notna(most_recent['effective_date']) else None
            }
        },
        "source_names": ["emails"],
        "sample_size": len(price_change_emails),
        "coverage_notes": [
            f"Email evidence contains {len(price_change_emails)} price change records",
            "Most recent price change shown; others available in source data",
            "Effective dates extracted from email evidence",
            "No recipe/BOM data available to link ingredient costs to menu items"
        ],
        "assumptions": [
            "Email extraction accuracy and completeness",
            "Effective dates as stated in supplier communications",
            "Price units (SAR per unit) as specified in email evidence",
            "No retroactive price adjustments or contract modifications"
        ],
        "confidence": 0.80
    }
    
    findings.append(finding_3)

# ============================================================================
# Build output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write to output path
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")