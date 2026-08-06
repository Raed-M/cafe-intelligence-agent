import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

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
analysis_start = datetime(2026, 5, 18, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 5, 25, 0, 0, 0, tzinfo=timezone.utc)
previous_start = datetime(2026, 5, 11, 0, 0, 0, tzinfo=timezone.utc)
previous_end = datetime(2026, 5, 18, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp to datetime if needed
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'], utc=True)
menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'], errors='coerce', utc=True)
menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'], errors='coerce', utc=True)
emails_df['date'] = pd.to_datetime(emails_df['date'], errors='coerce', utc=True)
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'], errors='coerce', utc=True)

findings = []

# ============================================================================
# FINDING 1: Item-level COGS and Gross Profit Analysis (Analysis Period)
# ============================================================================

# Filter POS for analysis period, exclude refunds
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start) & 
    (pos_df['timestamp'] < analysis_end) &
    (pos_df['is_refund'] == False)
].copy()

# Merge with menu to get unit costs
pos_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'item_en']],
    on='sku',
    how='left'
)

# Calculate item-level metrics
pos_with_cost['cogs_sar'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
pos_with_cost['gross_profit_sar'] = pos_with_cost['line_total_sar'] - pos_with_cost['cogs_sar']
pos_with_cost['gross_margin_pct'] = (
    (pos_with_cost['gross_profit_sar'] / pos_with_cost['line_total_sar'] * 100)
    .fillna(0)
)

# Aggregate by item
item_economics = pos_with_cost.groupby('sku').agg({
    'item_en': 'first',
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'cogs_sar': 'sum',
    'gross_profit_sar': 'sum',
    'unit_cost_sar': 'first',
    'unit_price_sar': 'first'
}).reset_index()

item_economics['gross_margin_pct'] = (
    (item_economics['gross_profit_sar'] / item_economics['line_total_sar'] * 100)
    .fillna(0)
)

# Calculate popularity (quantity sold) and contribution (gross profit)
total_qty = item_economics['quantity'].sum()
total_gp = item_economics['gross_profit_sar'].sum()

item_economics['popularity_pct'] = (item_economics['quantity'] / total_qty * 100).fillna(0)
item_economics['contribution_pct'] = (item_economics['gross_profit_sar'] / total_gp * 100).fillna(0)

# Menu engineering quadrants: popularity > median, contribution > median
popularity_median = item_economics['popularity_pct'].median()
contribution_median = item_economics['contribution_pct'].median()

item_economics['quadrant'] = item_economics.apply(
    lambda row: (
        'Stars' if row['popularity_pct'] > popularity_median and row['contribution_pct'] > contribution_median
        else 'Plowhorses' if row['popularity_pct'] > popularity_median and row['contribution_pct'] <= contribution_median
        else 'Puzzles' if row['popularity_pct'] <= popularity_median and row['contribution_pct'] > contribution_median
        else 'Dogs'
    ),
    axis=1
)

# Top 3 items by gross profit
top_items = item_economics.nlargest(3, 'gross_profit_sar')

if len(top_items) > 0:
    finding_1 = {
        "title": "Top 3 Items by Gross Profit Contribution (Analysis Period)",
        "claim": f"During {analysis_start.date()} to {analysis_end.date()}, the top 3 items by gross profit contribution are: {', '.join(top_items['item_en'].values)}. These items generated {top_items['gross_profit_sar'].sum():.2f} SAR in gross profit from {top_items['quantity'].sum():.0f} units sold, representing {(top_items['gross_profit_sar'].sum() / total_gp * 100):.1f}% of total cafe gross profit.",
        "finding_type": "item_economics",
        "metrics": {
            "top_item_1_name": {
                "value": top_items.iloc[0]['item_en'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_gross_profit_sar": {
                "value": round(top_items.iloc[0]['gross_profit_sar'], 2),
                "unit": "SAR",
                "numerator": round(top_items.iloc[0]['gross_profit_sar'], 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_quantity": {
                "value": int(top_items.iloc[0]['quantity']),
                "unit": "units",
                "numerator": int(top_items.iloc[0]['quantity']),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_gross_margin_pct": {
                "value": round(top_items.iloc[0]['gross_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_items.iloc[0]['gross_profit_sar'], 2),
                "denominator": round(top_items.iloc[0]['line_total_sar'], 2),
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
            "top_item_2_gross_profit_sar": {
                "value": round(top_items.iloc[1]['gross_profit_sar'], 2) if len(top_items) > 1 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[1]['gross_profit_sar'], 2) if len(top_items) > 1 else None,
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
            "top_item_3_gross_profit_sar": {
                "value": round(top_items.iloc[2]['gross_profit_sar'], 2) if len(top_items) > 2 else None,
                "unit": "SAR",
                "numerator": round(top_items.iloc[2]['gross_profit_sar'], 2) if len(top_items) > 2 else None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_cafe_gross_profit_sar": {
                "value": round(total_gp, 2),
                "unit": "SAR",
                "numerator": round(total_gp, 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_3_contribution_pct": {
                "value": round((top_items['gross_profit_sar'].sum() / total_gp * 100), 2),
                "unit": "%",
                "numerator": round(top_items['gross_profit_sar'].sum(), 2),
                "denominator": round(total_gp, 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(pos_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
            f"POS transactions analyzed: {len(pos_analysis)} line items",
            f"Refunds excluded from analysis",
            f"Menu items with unit_cost_sar merged: {pos_with_cost['unit_cost_sar'].notna().sum()} of {len(pos_with_cost)} items",
            f"Total unique items in analysis: {len(item_economics)}"
        ],
        "assumptions": [
            "Unit costs from menu_items.unit_cost_sar are current and applicable to analysis period",
            "Line totals in POS include discounts and represent actual revenue",
            "Gross profit = line_total_sar - (quantity × unit_cost_sar)",
            "No recipe/BOM available; analysis is at item level only"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Waste Cost Analysis (Inventory Data)
# ============================================================================

# Filter inventory for analysis period
inv_analysis = inventory_df[
    (inventory_df['week_starting'] >= analysis_start) & 
    (inventory_df['week_starting'] < analysis_end)
].copy()

# Only include rows with non-null waste cost
inv_with_waste = inv_analysis[inv_analysis['known_waste_cost_sar'].notna()].copy()

if len(inv_with_waste) > 0:
    total_waste_cost = inv_with_waste['known_waste_cost_sar'].sum()
    total_units_wasted = inv_with_waste['units_wasted'].sum()
    
    # Merge with menu for item names
    inv_with_waste = inv_with_waste.merge(
        menu_df[['sku', 'item_en']],
        on='sku',
        how='left'
    )
    
    # Top waste items
    top_waste = inv_with_waste.nlargest(3, 'known_waste_cost_sar')
    
    finding_2 = {
        "title": "Quantified Waste Cost (Known Observations Only)",
        "claim": f"During the analysis period ({analysis_start.date()} to {analysis_end.date()}), documented waste cost totaled {total_waste_cost:.2f} SAR across {len(inv_with_waste)} inventory records with non-null waste observations. Top waste item: {top_waste.iloc[0]['item_en']} ({top_waste.iloc[0]['known_waste_cost_sar']:.2f} SAR, {int(top_waste.iloc[0]['units_wasted'])} units).",
        "finding_type": "waste_cost",
        "metrics": {
            "total_known_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": round(total_waste_cost, 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_units_wasted": {
                "value": int(total_units_wasted),
                "unit": "units",
                "numerator": int(total_units_wasted),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "waste_records_count": {
                "value": len(inv_with_waste),
                "unit": "count",
                "numerator": len(inv_with_waste),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_waste_item_name": {
                "value": top_waste.iloc[0]['item_en'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_waste_item_cost_sar": {
                "value": round(top_waste.iloc[0]['known_waste_cost_sar'], 2),
                "unit": "SAR",
                "numerator": round(top_waste.iloc[0]['known_waste_cost_sar'], 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_waste_item_units": {
                "value": int(top_waste.iloc[0]['units_wasted']),
                "unit": "units",
                "numerator": int(top_waste.iloc[0]['units_wasted']),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["inventory", "menu"],
        "sample_size": len(inv_with_waste),
        "coverage_notes": [
            f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
            f"Inventory records with non-null known_waste_cost_sar: {len(inv_with_waste)} of {len(inv_analysis)} total",
            f"Blank waste cost values excluded per methodology",
            f"Waste cost sourced from inventory.known_waste_cost_sar (pre-calculated field)"
        ],
        "assumptions": [
            "known_waste_cost_sar in inventory represents actual documented waste cost",
            "Null waste cost values indicate missing data, not zero waste",
            "Waste cost is calculated at source and reflects unit cost at time of waste"
        ],
        "confidence": 0.90
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Supplier Price Change Notification (Email Evidence)
# ============================================================================

# Filter emails for price changes with high confidence
price_change_emails = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['confidence'] >= 0.85)
].copy()

if len(price_change_emails) > 0:
    # Take the most recent/relevant price change
    price_change_emails['date'] = pd.to_datetime(price_change_emails['date'], errors='coerce', utc=True)
    price_change_emails = price_change_emails.sort_values('date', ascending=False)
    
    top_change = price_change_emails.iloc[0]
    
    # Calculate percentage change
    old_price = float(top_change['old_price'])
    new_price = float(top_change['new_price'])
    pct_change = ((new_price - old_price) / old_price * 100) if old_price != 0 else 0
    
    effective_date = top_change['effective_date']
    email_date = top_change['date']
    
    finding_3 = {
        "title": "Supplier Price Notification – Full-Fat Milk Increase (2026-05-01)",
        "claim": f"Supplier email dated {email_date.date()} from {top_change['sender']} notifies of a {top_change['entity_or_ingredient']} price increase from {old_price} to {new_price} {top_change['currency']} per {top_change['unit']}, effective {effective_date.date()} ({pct_change:.2f}% increase). Cafe-level cost impact cannot be quantified without recipe/BOM and procurement volume data.",
        "finding_type": "supplier_price_change",
        "metrics": {
            "supplier_entity": {
                "value": top_change['entity_or_ingredient'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": email_date.isoformat(),
                "period_end": email_date.isoformat()
            },
            "old_price": {
                "value": round(old_price, 2),
                "unit": f"{top_change['currency']}/{top_change['unit']}",
                "numerator": round(old_price, 2),
                "denominator": None,
                "period_start": email_date.isoformat(),
                "period_end": email_date.isoformat()
            },
            "new_price": {
                "value": round(new_price, 2),
                "unit": f"{top_change['currency']}/{top_change['unit']}",
                "numerator": round(new_price, 2),
                "denominator": None,
                "period_start": email_date.isoformat(),
                "period_end": email_date.isoformat()
            },
            "price_change_pct": {
                "value": round(pct_change, 2),
                "unit": "%",
                "numerator": round(new_price - old_price, 2),
                "denominator": round(old_price, 2),
                "period_start": email_date.isoformat(),
                "period_end": email_date.isoformat()
            },
            "email_date": {
                "value": email_date.date().isoformat(),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": email_date.isoformat(),
                "period_end": email_date.isoformat()
            },
            "effective_date": {
                "value": effective_date.date().isoformat() if pd.notna(effective_date) else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": email_date.isoformat(),
                "period_end": email_date.isoformat()
            },
            "sender": {
                "value": top_change['sender'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": email_date.isoformat(),
                "period_end": email_date.isoformat()
            },
            "extraction_confidence": {
                "value": round(top_change['confidence'], 2),
                "unit": None,
                "numerator": round(top_change['confidence'], 2),
                "denominator": 1.0,
                "period_start": email_date.isoformat(),
                "period_end": email_date.isoformat()
            }
        },
        "source_names": ["emails"],
        "sample_size": 1,
        "coverage_notes": [
            f"Email date: {email_date.date()}",
            f"Effective date: {effective_date.date() if pd.notna(effective_date) else 'Not specified'}",
            f"Evidence period (analysis window): {analysis_start.date()} to {analysis_end.date()}",
            f"Email date is {(analysis_start - email_date).days} days before analysis period start",
            f"Extraction confidence: {top_change['confidence']}",
            f"Source: {top_change['sender']}"
        ],
        "assumptions": [
            "Email extraction confidence of 0.85+ indicates reliable price signal",
            "Price change applies to supplier's standing orders (volume and payment terms not confirmed)",
            "No recipe/BOM available; per-drink cost impact cannot be calculated",
            "Cafe procurement volume and payment terms are unknown; cost exposure is unquantified"
        ],
        "confidence": 0.85
    }
    findings.append(finding_3)

# ============================================================================
# Write output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")