import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
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

# Parse analysis period
analysis_start = datetime.fromisoformat("2026-04-06T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-04-13T00:00:00+03:00")
previous_start = datetime.fromisoformat("2026-03-30T00:00:00+03:00")
previous_end = datetime.fromisoformat("2026-04-06T00:00:00+03:00")

# Convert timestamp columns to datetime
pos_df['timestamp_local'] = pd.to_datetime(pos_df['timestamp_local'], utc=True)
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])

# Filter POS for analysis period
pos_analysis = pos_df[
    (pos_df['timestamp_local'] >= analysis_start) & 
    (pos_df['timestamp_local'] < analysis_end)
].copy()

pos_previous = pos_df[
    (pos_df['timestamp_local'] >= previous_start) & 
    (pos_df['timestamp_local'] < previous_end)
].copy()

# Prepare findings list
findings = []

# ============================================================================
# FINDING 1: Item-Level Gross Profit Analysis (Analysis Period)
# ============================================================================

# Merge POS with menu to get unit costs
pos_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'price_sar']],
    on='sku',
    how='left'
)

# Calculate line-level metrics
pos_with_cost['cogs_sar'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
pos_with_cost['gross_profit_sar'] = pos_with_cost['line_total_sar'] - pos_with_cost['cogs_sar']
pos_with_cost['gross_margin_pct'] = np.where(
    pos_with_cost['line_total_sar'] != 0,
    (pos_with_cost['gross_profit_sar'] / pos_with_cost['line_total_sar']) * 100,
    0
)

# Aggregate by item
item_economics = pos_with_cost.groupby('item_name_en').agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'cogs_sar': 'sum',
    'gross_profit_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()

item_economics.columns = ['item_name', 'total_quantity', 'total_revenue', 'total_cogs', 'total_gross_profit', 'basket_count']
item_economics['gross_margin_pct'] = np.where(
    item_economics['total_revenue'] != 0,
    (item_economics['total_gross_profit'] / item_economics['total_revenue']) * 100,
    0
)

# Sort by gross profit contribution
item_economics = item_economics.sort_values('total_gross_profit', ascending=False)

# Top 3 items by gross profit
top_3_items = item_economics.head(3)

if len(top_3_items) > 0:
    finding_1 = {
        "title": "Top 3 Items by Gross Profit Contribution (Analysis Period)",
        "claim": f"During {analysis_start.date()} to {analysis_end.date()}, the top 3 items by gross profit contribution are: {', '.join(top_3_items['item_name'].tolist())}. Combined, they generated {top_3_items['total_gross_profit'].sum():.2f} SAR in gross profit from {top_3_items['total_quantity'].sum():.0f} units sold across {top_3_items['basket_count'].sum():.0f} baskets.",
        "finding_type": "item_economics",
        "metrics": {
            "top_item_1_name": {
                "value": top_3_items.iloc[0]['item_name'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_gross_profit_sar": {
                "value": round(top_3_items.iloc[0]['total_gross_profit'], 2),
                "unit": "SAR",
                "numerator": round(top_3_items.iloc[0]['total_gross_profit'], 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_quantity": {
                "value": int(top_3_items.iloc[0]['total_quantity']),
                "unit": "units",
                "numerator": int(top_3_items.iloc[0]['total_quantity']),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_margin_pct": {
                "value": round(top_3_items.iloc[0]['gross_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_3_items.iloc[0]['gross_margin_pct'], 2),
                "denominator": 100,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_2_name": {
                "value": top_3_items.iloc[1]['item_name'] if len(top_3_items) > 1 else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_2_gross_profit_sar": {
                "value": round(top_3_items.iloc[1]['total_gross_profit'], 2) if len(top_3_items) > 1 else None,
                "unit": "SAR",
                "numerator": round(top_3_items.iloc[1]['total_gross_profit'], 2) if len(top_3_items) > 1 else None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_3_name": {
                "value": top_3_items.iloc[2]['item_name'] if len(top_3_items) > 2 else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_3_gross_profit_sar": {
                "value": round(top_3_items.iloc[2]['total_gross_profit'], 2) if len(top_3_items) > 2 else None,
                "unit": "SAR",
                "numerator": round(top_3_items.iloc[2]['total_gross_profit'], 2) if len(top_3_items) > 2 else None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "combined_gross_profit_sar": {
                "value": round(top_3_items['total_gross_profit'].sum(), 2),
                "unit": "SAR",
                "numerator": round(top_3_items['total_gross_profit'].sum(), 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "combined_quantity": {
                "value": int(top_3_items['total_quantity'].sum()),
                "unit": "units",
                "numerator": int(top_3_items['total_quantity'].sum()),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "combined_basket_count": {
                "value": int(top_3_items['basket_count'].sum()),
                "unit": "baskets",
                "numerator": int(top_3_items['basket_count'].sum()),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(pos_analysis.shape[0]),
        "coverage_notes": [
            f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
            f"POS records analyzed: {pos_analysis.shape[0]} line items",
            f"Unique items in analysis: {len(item_economics)}",
            "Gross profit calculated as: line_total_sar - (quantity × unit_cost_sar)",
            "Refunds included in net calculations per metric definition"
        ],
        "assumptions": [
            "Menu unit_cost_sar values are current and accurate for the analysis period",
            "POS line_total_sar reflects actual revenue after discounts",
            "No recipe/BOM adjustments applied; unit costs are as-stated in menu"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Waste Cost Impact Analysis
# ============================================================================

# Filter inventory for analysis period (week starting 2026-04-06)
inventory_analysis = inventory_df[
    inventory_df['week_starting'] == '2026-04-06'
].copy()

# Calculate total waste cost
total_waste_cost = inventory_analysis['known_waste_cost_sar'].sum()
waste_items = inventory_analysis[inventory_analysis['known_waste_cost_sar'] > 0].copy()

if len(waste_items) > 0 and total_waste_cost > 0:
    waste_items = waste_items.sort_values('known_waste_cost_sar', ascending=False)
    
    finding_2 = {
        "title": "Quantified Waste Cost Impact (Week of 2026-04-06)",
        "claim": f"During the week of 2026-04-06, {len(waste_items)} items had documented waste with a total known waste cost of {total_waste_cost:.2f} SAR. The highest waste cost item was {waste_items.iloc[0]['item']} with {waste_items.iloc[0]['known_waste_cost_sar']:.2f} SAR.",
        "finding_type": "waste_cost_analysis",
        "metrics": {
            "total_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": round(total_waste_cost, 2),
                "denominator": None,
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            },
            "waste_items_count": {
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
            },
            "highest_waste_units": {
                "value": int(waste_items.iloc[0]['units_wasted']),
                "unit": "units",
                "numerator": int(waste_items.iloc[0]['units_wasted']),
                "denominator": None,
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(inventory_analysis),
        "coverage_notes": [
            "Analysis period: Week of 2026-04-06",
            f"Inventory records with waste data: {len(waste_items)} out of {len(inventory_analysis)}",
            "Only non-null waste cost values included per metric definition",
            "Waste cost calculated from known_waste_cost_sar field"
        ],
        "assumptions": [
            "known_waste_cost_sar values are accurate and complete for documented waste",
            "Blank waste values are treated as unknown, not zero",
            "Unit cost × units_wasted = known_waste_cost_sar"
        ],
        "confidence": 0.90
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Supplier Price Changes and Procurement Impact
# ============================================================================

# Filter emails for price changes with effective dates
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    price_changes['effective_date'] = pd.to_datetime(price_changes['effective_date'])
    price_changes['date'] = pd.to_datetime(price_changes['date'])
    
    # Calculate price change percentage
    price_changes['price_change_pct'] = (
        (price_changes['new_price'] - price_changes['old_price']) / 
        price_changes['old_price'] * 100
    )
    
    # Sort by effective date
    price_changes = price_changes.sort_values('effective_date', ascending=False)
    
    # Get most recent price change
    most_recent = price_changes.iloc[0]
    
    finding_3 = {
        "title": "Detected Supplier Price Change",
        "claim": f"Email from {most_recent['sender']} dated {most_recent['date'].date()} documents a price change for {most_recent['entity_or_ingredient']} effective {most_recent['effective_date'].date()}. Price changed from {most_recent['old_price']} to {most_recent['new_price']} {most_recent['currency']} per {most_recent['unit']}, representing a {most_recent['price_change_pct']:.2f}% change.",
        "finding_type": "supplier_price_change",
        "metrics": {
            "entity_or_ingredient": {
                "value": most_recent['entity_or_ingredient'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": most_recent['date'].isoformat(),
                "period_end": most_recent['date'].isoformat()
            },
            "old_price": {
                "value": round(most_recent['old_price'], 4),
                "unit": most_recent['currency'],
                "numerator": round(most_recent['old_price'], 4),
                "denominator": None,
                "period_start": most_recent['date'].isoformat(),
                "period_end": most_recent['date'].isoformat()
            },
            "new_price": {
                "value": round(most_recent['new_price'], 4),
                "unit": most_recent['currency'],
                "numerator": round(most_recent['new_price'], 4),
                "denominator": None,
                "period_start": most_recent['date'].isoformat(),
                "period_end": most_recent['date'].isoformat()
            },
            "price_unit": {
                "value": most_recent['unit'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": most_recent['date'].isoformat(),
                "period_end": most_recent['date'].isoformat()
            },
            "price_change_pct": {
                "value": round(most_recent['price_change_pct'], 2),
                "unit": "%",
                "numerator": round(most_recent['price_change_pct'], 2),
                "denominator": 100,
                "period_start": most_recent['date'].isoformat(),
                "period_end": most_recent['date'].isoformat()
            },
            "effective_date": {
                "value": most_recent['effective_date'].isoformat(),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": most_recent['date'].isoformat(),
                "period_end": most_recent['date'].isoformat()
            },
            "sender": {
                "value": most_recent['sender'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": most_recent['date'].isoformat(),
                "period_end": most_recent['date'].isoformat()
            }
        },
        "source_names": ["emails"],
        "sample_size": len(price_changes),
        "coverage_notes": [
            f"Total price change emails identified: {len(price_changes)}",
            "Analysis shows most recent price change",
            "Price change extracted from supplier email evidence",
            "No standing order quantities or payment terms data available for scenario modeling"
        ],
        "assumptions": [
            "Email extraction confidence and facts field are accurate",
            "Price change applies to the specified entity/ingredient only",
            "No recipe/BOM exists to calculate per-drink impact",
            "Standing order volume and payment terms are unknown"
        ],
        "confidence": 0.85
    }
    findings.append(finding_3)

# ============================================================================
# Prepare final output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings generated.")
print(f"Output written to {output_path}")
