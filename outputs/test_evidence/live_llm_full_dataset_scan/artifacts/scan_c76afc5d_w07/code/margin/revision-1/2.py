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

# Parse analysis periods
analysis_start = "2026-02-23T00:00:00+03:00"
analysis_end = "2026-03-02T00:00:00+03:00"
previous_start = "2026-02-16T00:00:00+03:00"
previous_end = "2026-02-23T00:00:00+03:00"

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Convert timestamps to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'], errors='coerce')
menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'], errors='coerce')
emails_df['date'] = pd.to_datetime(emails_df['date'], errors='coerce')
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'], errors='coerce')

analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)
previous_start_dt = pd.to_datetime(previous_start)
previous_end_dt = pd.to_datetime(previous_end)

findings = []

# ============================================================================
# FINDING 1: Item-level COGS and Gross Profit Analysis (Analysis Period)
# ============================================================================

# Filter POS for analysis period, exclude refunds
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start_dt) &
    (pos_df['timestamp'] < analysis_end_dt) &
    (pos_df['is_refund'] == False)
].copy()

# Merge with menu to get unit costs
pos_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'item_en', 'category']],
    on='sku',
    how='left'
)

# Calculate item-level economics
pos_with_cost['cogs_sar'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
pos_with_cost['gross_profit_sar'] = pos_with_cost['line_total_sar'] - pos_with_cost['cogs_sar']
pos_with_cost['gross_margin_pct'] = (
    (pos_with_cost['gross_profit_sar'] / pos_with_cost['line_total_sar'] * 100)
    .fillna(0)
)

# Aggregate by SKU - only include columns that exist in pos_with_cost
sku_economics = pos_with_cost.groupby('sku').agg({
    'item_en': 'first',
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'cogs_sar': 'sum',
    'gross_profit_sar': 'sum',
    'unit_cost_sar': 'first'
}).reset_index()

# Add category from menu separately to avoid aggregation issues
sku_to_category = menu_df[['sku', 'category']].drop_duplicates()
sku_economics = sku_economics.merge(sku_to_category, on='sku', how='left')

# Add unit_price_sar from menu
sku_to_price = menu_df[['sku', 'price_sar']].drop_duplicates()
sku_economics = sku_economics.merge(sku_to_price, on='sku', how='left')
sku_economics.rename(columns={'price_sar': 'unit_price_sar'}, inplace=True)

sku_economics['gross_margin_pct'] = (
    (sku_economics['gross_profit_sar'] / sku_economics['line_total_sar'] * 100)
    .fillna(0)
)

# Sort by gross profit contribution
sku_economics_sorted = sku_economics.sort_values('gross_profit_sar', ascending=False)

# Top 5 contributors
top_5 = sku_economics_sorted.head(5)

if len(top_5) > 0:
    total_revenue = sku_economics['line_total_sar'].sum()
    total_cogs = sku_economics['cogs_sar'].sum()
    total_gp = sku_economics['gross_profit_sar'].sum()
    overall_margin = (total_gp / total_revenue * 100) if total_revenue > 0 else 0
    
    top_5_revenue = top_5['line_total_sar'].sum()
    top_5_gp = top_5['gross_profit_sar'].sum()
    top_5_pct_of_total = (top_5_revenue / total_revenue * 100) if total_revenue > 0 else 0
    
    findings.append({
        "title": "Top 5 Gross Profit Contributors (Analysis Period)",
        "claim": f"The top 5 SKUs by gross profit contribution generated {top_5_gp:.2f} SAR in gross profit, representing {top_5_pct_of_total:.1f}% of total revenue ({top_5_revenue:.2f} SAR). Overall cafe gross margin is {overall_margin:.1f}%.",
        "finding_type": "item_economics",
        "metrics": {
            "total_revenue_sar": {
                "value": round(total_revenue, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_cogs_sar": {
                "value": round(total_cogs, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_gross_profit_sar": {
                "value": round(total_gp, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "overall_gross_margin_pct": {
                "value": round(overall_margin, 2),
                "unit": "%",
                "numerator": round(total_gp, 2),
                "denominator": round(total_revenue, 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "top_5_gross_profit_sar": {
                "value": round(top_5_gp, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "top_5_pct_of_total_revenue": {
                "value": round(top_5_pct_of_total, 2),
                "unit": "%",
                "numerator": round(top_5_revenue, 2),
                "denominator": round(total_revenue, 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": len(pos_analysis),
        "coverage_notes": [
            "Analysis period: 2026-02-23 to 2026-03-02",
            "Refunds excluded from calculation",
            "Menu unit costs merged on SKU; missing costs treated as null",
            f"Total transactions in period: {pos_analysis['transaction_id'].nunique()}"
        ],
        "assumptions": [
            "Menu unit_cost_sar is current and applicable to analysis period",
            "Line totals are net of discounts",
            "No recipe/BOM data; economics are at menu-item level only"
        ],
        "confidence": 0.95
    })

# ============================================================================
# FINDING 2: Waste Cost Impact (Known Waste Only)
# ============================================================================

# Filter inventory for analysis period
inv_analysis = inventory_df[
    (inventory_df['week_starting'] >= analysis_start_dt) &
    (inventory_df['week_starting'] < analysis_end_dt)
].copy()

# Only include rows with non-null waste cost
inv_with_waste = inv_analysis[inv_analysis['known_waste_cost_sar'].notna()].copy()

if len(inv_with_waste) > 0:
    total_waste_cost = inv_with_waste['known_waste_cost_sar'].sum()
    waste_items = len(inv_with_waste)
    
    # Get total revenue for context
    total_revenue_waste_context = pos_analysis['line_total_sar'].sum()
    waste_pct_of_revenue = (total_waste_cost / total_revenue_waste_context * 100) if total_revenue_waste_context > 0 else 0
    
    findings.append({
        "title": "Quantified Waste Cost (Known Observations Only)",
        "claim": f"Documented waste cost in analysis period totals {total_waste_cost:.2f} SAR across {waste_items} inventory records with non-null waste observations, representing {waste_pct_of_revenue:.2f}% of period revenue.",
        "finding_type": "waste_cost",
        "metrics": {
            "total_known_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "waste_cost_pct_of_revenue": {
                "value": round(waste_pct_of_revenue, 2),
                "unit": "%",
                "numerator": round(total_waste_cost, 2),
                "denominator": round(total_revenue_waste_context, 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "inventory_records_with_waste": {
                "value": waste_items,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["inventory"],
        "sample_size": waste_items,
        "coverage_notes": [
            "Only non-null known_waste_cost_sar values included",
            "Blank waste observations excluded per methodology",
            f"Total inventory records in period: {len(inv_analysis)}",
            f"Records with quantified waste: {waste_items}"
        ],
        "assumptions": [
            "known_waste_cost_sar reflects actual waste cost incurred",
            "Waste cost is already calculated and provided in inventory data"
        ],
        "confidence": 0.90
    })

# ============================================================================
# FINDING 3: Supplier Price Changes from Email Evidence
# ============================================================================

# Filter emails for price changes with both old and new prices
price_changes = emails_df[
    (emails_df['old_price'].notna()) &
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(price_changes) > 0:
    # Take the most recent/relevant price change
    price_changes_sorted = price_changes.sort_values('effective_date', ascending=False)
    
    for idx, row in price_changes_sorted.iterrows():
        old_price = float(row['old_price'])
        new_price = float(row['new_price'])
        currency = row['currency'] if pd.notna(row['currency']) else 'SAR'
        unit = row['unit'] if pd.notna(row['unit']) else 'unit'
        entity = row['entity_or_ingredient'] if pd.notna(row['entity_or_ingredient']) else 'Unknown'
        effective_date = row['effective_date']
        evidence_text = row['evidence_text'] if pd.notna(row['evidence_text']) else ''
        
        price_diff = new_price - old_price
        price_change_pct = (price_diff / old_price * 100) if old_price != 0 else 0
        
        findings.append({
            "title": f"Supplier Price Change: {entity}",
            "claim": f"Supplier email evidence documents a price change for {entity} from {old_price:.2f} {currency}/{unit} to {new_price:.2f} {currency}/{unit}, effective {effective_date.strftime('%Y-%m-%d')}. Price change: {price_change_pct:+.2f}%. Impact on menu items requires standing order volume and payment terms confirmation.",
            "finding_type": "supplier_cost_change",
            "metrics": {
                "old_price": {
                    "value": round(old_price, 2),
                    "unit": f"{currency}/{unit}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": None,
                    "period_end": None
                },
                "new_price": {
                    "value": round(new_price, 2),
                    "unit": f"{currency}/{unit}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": None,
                    "period_end": None
                },
                "price_change_pct": {
                    "value": round(price_change_pct, 2),
                    "unit": "%",
                    "numerator": round(price_diff, 2),
                    "denominator": round(old_price, 2),
                    "period_start": None,
                    "period_end": None
                },
                "effective_date": {
                    "value": effective_date.strftime('%Y-%m-%d') if pd.notna(effective_date) else None,
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": None,
                    "period_end": None
                }
            },
            "source_names": ["emails"],
            "sample_size": 1,
            "coverage_notes": [
                "Evidence sourced from supplier email extraction",
                "Price change applies to ingredient/supplier entity, not menu items",
                "No recipe/BOM data available to calculate per-drink impact"
            ],
            "assumptions": [
                "No standing order volumes provided; procurement cost impact is unquantified",
                "No payment terms data; timing of cost realization is unknown",
                "Price applies only to named entity; applicability to menu items requires menu engineering review",
                "Price change percentage calculated as (new_price - old_price) / old_price × 100"
            ],
            "confidence": 0.85
        })
        
        # Only report first (most recent) price change to stay within 3-finding limit
        break

# ============================================================================
# Output
# ============================================================================

result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings[:3]  # Limit to 3 findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")
