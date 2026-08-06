import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
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

# Parse analysis period
analysis_start = datetime.fromisoformat("2026-04-13T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-04-20T00:00:00+03:00")

# Convert timestamp to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter POS to analysis period
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start) & 
    (pos_df['timestamp'] < analysis_end)
].copy()

# Exclude refunds for revenue/profit calculations
pos_sales = pos_analysis[pos_analysis['is_refund'] == False].copy()

# ============================================================================
# FINDING 1: Item-level gross profit contribution analysis
# ============================================================================

# Merge POS with menu to get unit costs
pos_with_cost = pos_sales.merge(
    menu_df[['sku', 'unit_cost_sar', 'item_en']],
    on='sku',
    how='left'
)

# Calculate line-level COGS and gross profit
pos_with_cost['line_cogs_sar'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
pos_with_cost['line_gross_profit_sar'] = pos_with_cost['line_total_sar'] - pos_with_cost['line_cogs_sar']

# Aggregate by item
item_metrics = pos_with_cost.groupby('sku').agg({
    'item_name_en': 'first',
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'line_cogs_sar': 'sum',
    'line_gross_profit_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()

item_metrics.columns = ['sku', 'item_name', 'total_quantity', 'total_revenue_sar', 'total_cogs_sar', 'total_gross_profit_sar', 'basket_count']

# Calculate margin percentage
item_metrics['gross_margin_pct'] = (item_metrics['total_gross_profit_sar'] / item_metrics['total_revenue_sar'] * 100).round(1)

# Sort by gross profit contribution
item_metrics_sorted = item_metrics.sort_values('total_gross_profit_sar', ascending=False)

# Get top 3 items
top_3_items = item_metrics_sorted.head(3)

# Overall metrics
total_revenue = pos_sales['line_total_sar'].sum()
total_cogs = pos_with_cost['line_cogs_sar'].sum()
total_gross_profit = total_revenue - total_cogs
overall_margin_pct = (total_gross_profit / total_revenue * 100) if total_revenue > 0 else 0

# Count items with cost data
items_with_cost = pos_with_cost[pos_with_cost['unit_cost_sar'].notna()]['sku'].nunique()
items_sold = pos_sales['sku'].nunique()

# ============================================================================
# FINDING 2: Waste cost impact analysis
# ============================================================================

# Filter inventory to analysis period
inventory_analysis = inventory_df[
    (pd.to_datetime(inventory_df['week_starting']) >= analysis_start) &
    (pd.to_datetime(inventory_df['week_starting']) < analysis_end)
].copy()

# Calculate waste metrics
waste_metrics = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()
total_waste_cost = waste_metrics['known_waste_cost_sar'].sum()
waste_item_count = len(waste_metrics)
total_units_wasted = waste_metrics['units_wasted'].sum()

# ============================================================================
# FINDING 3: Supplier price changes and procurement impact
# ============================================================================

# Filter emails to analysis period and price changes
emails_analysis = emails_df[
    (pd.to_datetime(emails_df['date']) >= analysis_start) &
    (pd.to_datetime(emails_df['date']) < analysis_end) &
    (emails_df['old_price'].notna()) &
    (emails_df['new_price'].notna())
].copy()

price_changes = []
for idx, row in emails_analysis.iterrows():
    if pd.notna(row['old_price']) and pd.notna(row['new_price']):
        old_price = float(row['old_price'])
        new_price = float(row['new_price'])
        pct_change = ((new_price - old_price) / old_price * 100) if old_price != 0 else 0
        
        price_changes.append({
            'entity': row['entity_or_ingredient'],
            'old_price': old_price,
            'new_price': new_price,
            'currency': row['currency'],
            'unit': row['unit'],
            'pct_change': round(pct_change, 1),
            'effective_date': row['effective_date'],
            'date': row['date']
        })

# ============================================================================
# Build findings
# ============================================================================

findings = []

# Finding 1: Top items by gross profit
if len(top_3_items) > 0:
    top_1 = top_3_items.iloc[0]
    
    claim_items = [top_1['item_name']]
    if len(top_3_items) > 1:
        claim_items.append(top_3_items.iloc[1]['item_name'])
    if len(top_3_items) > 2:
        claim_items.append(top_3_items.iloc[2]['item_name'])
    
    claim_text = f"Top item by gross profit contribution: {top_1['item_name']} (SAR {top_1['total_gross_profit_sar']:.2f}, {top_1['gross_margin_pct']:.1f}% margin). Overall cafe gross margin: {overall_margin_pct:.1f}% (SAR {total_gross_profit:.2f} on SAR {total_revenue:.2f} revenue). Item-level costs based on menu unit costs rather than recipe-level ingredient analysis."
    
    finding_1 = {
        "title": "Item-level Gross Profit Contribution & Cafe Margin",
        "claim": claim_text,
        "finding_type": "margin_analysis",
        "metrics": {
            "top_item_1_name": {
                "value": top_1['item_name'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_gross_profit_sar": {
                "value": round(top_1['total_gross_profit_sar'], 2),
                "unit": "SAR",
                "numerator": round(top_1['total_gross_profit_sar'], 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_margin_pct": {
                "value": top_1['gross_margin_pct'],
                "unit": "%",
                "numerator": top_1['gross_margin_pct'],
                "denominator": 100,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "overall_gross_profit_sar": {
                "value": round(total_gross_profit, 2),
                "unit": "SAR",
                "numerator": round(total_gross_profit, 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "overall_gross_margin_pct": {
                "value": round(overall_margin_pct, 1),
                "unit": "%",
                "numerator": round(overall_margin_pct, 1),
                "denominator": 100,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_revenue_sar": {
                "value": round(total_revenue, 2),
                "unit": "SAR",
                "numerator": round(total_revenue, 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_cogs_sar": {
                "value": round(total_cogs, 2),
                "unit": "SAR",
                "numerator": round(total_cogs, 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(pos_sales.shape[0]),
        "coverage_notes": [
            f"{items_with_cost} of {items_sold} items sold have menu unit cost data",
            f"Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
            "Refunds excluded from revenue and profit calculations"
        ],
        "assumptions": [
            "Item-level unit costs sourced from menu.unit_cost_sar applied uniformly across all sales",
            "No recipe/BOM data available; per-drink ingredient costs not calculated",
            "Line totals used as-is from POS; discount_sar already reflected in line_total_sar"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# Finding 2: Waste cost impact
if total_waste_cost > 0 and waste_item_count > 0:
    waste_pct_of_cogs = (total_waste_cost / total_cogs * 100) if total_cogs > 0 else 0
    
    claim_text = f"Quantified waste cost: SAR {total_waste_cost:.2f} across {waste_item_count} items ({total_units_wasted:.0f} units). Waste represents {waste_pct_of_cogs:.2f}% of total COGS. Only non-null waste observations included."
    
    finding_2 = {
        "title": "Waste Cost Impact on COGS",
        "claim": claim_text,
        "finding_type": "waste_analysis",
        "metrics": {
            "total_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": round(total_waste_cost, 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "waste_items_count": {
                "value": waste_item_count,
                "unit": None,
                "numerator": waste_item_count,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_units_wasted": {
                "value": round(total_units_wasted, 0),
                "unit": "units",
                "numerator": round(total_units_wasted, 0),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "waste_pct_of_cogs": {
                "value": round(waste_pct_of_cogs, 2),
                "unit": "%",
                "numerator": round(waste_pct_of_cogs, 2),
                "denominator": 100,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["inventory"],
        "sample_size": waste_item_count,
        "coverage_notes": [
            "Only inventory records with non-null known_waste_cost_sar included",
            f"Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
            "Blank waste values treated as missing, not zero"
        ],
        "assumptions": [
            "known_waste_cost_sar values are accurate and complete for reported waste",
            "Waste cost is incremental to COGS and represents lost margin"
        ],
        "confidence": 0.85
    }
    findings.append(finding_2)

# Finding 3: Supplier price changes
if len(price_changes) > 0:
    price_change = price_changes[0]
    
    claim_text = f"Supplier price change detected: {price_change['entity']} price changed from {price_change['currency']} {price_change['old_price']:.2f} to {price_change['currency']} {price_change['new_price']:.2f} per {price_change['unit']} ({price_change['pct_change']:+.1f}%), effective {price_change['effective_date']}. Impact on procurement costs depends on standing order volumes and payment terms (not confirmed in available data)."
    
    finding_3 = {
        "title": "Supplier Price Change Alert",
        "claim": claim_text,
        "finding_type": "supplier_cost_change",
        "metrics": {
            "entity": {
                "value": price_change['entity'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "old_price": {
                "value": round(price_change['old_price'], 2),
                "unit": price_change['currency'],
                "numerator": round(price_change['old_price'], 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "new_price": {
                "value": round(price_change['new_price'], 2),
                "unit": price_change['currency'],
                "numerator": round(price_change['new_price'], 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "price_change_pct": {
                "value": price_change['pct_change'],
                "unit": "%",
                "numerator": price_change['pct_change'],
                "denominator": 100,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "unit": {
                "value": price_change['unit'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "effective_date": {
                "value": price_change['effective_date'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["emails"],
        "sample_size": 1,
        "coverage_notes": [
            f"Email date: {price_change['date']}",
            "Price change extracted from supplier email",
            "No standing order quantities or payment terms confirmed in available data"
        ],
        "assumptions": [
            "Price change applies to future procurement only",
            "Standing order volumes and payment terms are unknown",
            "Impact on menu item costs requires recipe/BOM mapping (not available)"
        ],
        "confidence": 0.75
    }
    findings.append(finding_3)

# ============================================================================
# Output result
# ============================================================================

result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)