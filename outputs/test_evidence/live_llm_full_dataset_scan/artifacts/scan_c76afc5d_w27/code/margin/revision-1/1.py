import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any

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
analysis_start = datetime(2026, 7, 13, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 7, 20, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamps to UTC for comparison
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)

# Filter POS data to analysis period
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start) & 
    (pos_df['timestamp'] < analysis_end)
].copy()

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

# Calculate line-level metrics
pos_with_cost['cogs_sar'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
pos_with_cost['gross_profit_sar'] = pos_with_cost['line_total_sar'] - pos_with_cost['cogs_sar']
pos_with_cost['gross_margin_pct'] = (
    pos_with_cost['gross_profit_sar'] / pos_with_cost['line_total_sar'] * 100
).replace([np.inf, -np.inf], np.nan)

# Aggregate by item (excluding refunds for net calculation)
non_refund_pos = pos_with_cost[pos_with_cost['is_refund'] == False].copy()

item_economics = non_refund_pos.groupby('sku').agg({
    'item_name_en': 'first',
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'cogs_sar': 'sum',
    'gross_profit_sar': 'sum',
    'unit_cost_sar': 'first',
    'price_sar': 'first',
    'transaction_id': 'nunique'
}).reset_index()

item_economics.columns = ['sku', 'item_name', 'total_qty', 'total_revenue', 'total_cogs', 'total_gp', 'unit_cost', 'menu_price', 'basket_count']
item_economics['gp_margin_pct'] = (item_economics['total_gp'] / item_economics['total_revenue'] * 100).round(2)

# Sort by gross profit contribution
item_economics_sorted = item_economics.sort_values('total_gp', ascending=False)

# Top 3 items by gross profit
top_3_gp = item_economics_sorted.head(3)

if len(top_3_gp) > 0:
    finding_1 = {
        "title": "Top 3 Items by Gross Profit Contribution (Week 27)",
        "claim": f"The top 3 items by gross profit contribution in the analysis period are {', '.join(top_3_gp['item_name'].values)}, collectively generating {top_3_gp['total_gp'].sum():.2f} SAR in gross profit.",
        "finding_type": "item_economics",
        "metrics": {
            "top_item_1_name": {
                "value": top_3_gp.iloc[0]['item_name'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_gp_sar": {
                "value": round(top_3_gp.iloc[0]['total_gp'], 2),
                "unit": "SAR",
                "numerator": round(top_3_gp.iloc[0]['total_gp'], 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_margin_pct": {
                "value": round(top_3_gp.iloc[0]['gp_margin_pct'], 2),
                "unit": "%",
                "numerator": round(top_3_gp.iloc[0]['gp_margin_pct'], 2),
                "denominator": 100,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_item_1_qty": {
                "value": int(top_3_gp.iloc[0]['total_qty']),
                "unit": "units",
                "numerator": int(top_3_gp.iloc[0]['total_qty']),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_3_combined_gp_sar": {
                "value": round(top_3_gp['total_gp'].sum(), 2),
                "unit": "SAR",
                "numerator": round(top_3_gp['total_gp'].sum(), 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(non_refund_pos.shape[0]),
        "coverage_notes": [
            f"Analysis period: 2026-07-13 to 2026-07-20 (Week 27)",
            f"POS records analyzed: {int(non_refund_pos.shape[0])} line items",
            f"Unique baskets (transactions): {int(non_refund_pos['transaction_id'].nunique())}",
            f"Items with menu cost data: {int(item_economics[item_economics['unit_cost'].notna()].shape[0])} of {int(item_economics.shape[0])}",
            "Refunds excluded from net revenue and profit calculations"
        ],
        "assumptions": [
            "Menu unit_cost_sar represents actual COGS per unit",
            "Line totals are accurate and consistent with quantity × unit_price - discount",
            "No recipe/BOM data available; COGS is item-level only"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Waste Cost Impact Analysis
# ============================================================================

# Filter inventory to analysis period (week starting 2026-07-13)
inventory_analysis = inventory_df[
    inventory_df['week_starting'] == '2026-07-13'
].copy()

# Calculate total waste cost (only non-null values)
waste_with_cost = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()

if len(waste_with_cost) > 0:
    total_waste_cost = waste_with_cost['known_waste_cost_sar'].sum()
    waste_items = waste_with_cost[['sku', 'item', 'units_wasted', 'known_waste_cost_sar']].copy()
    waste_items = waste_items.sort_values('known_waste_cost_sar', ascending=False)
    
    # Calculate as percentage of total revenue
    total_revenue_analysis = non_refund_pos['line_total_sar'].sum()
    waste_pct_revenue = (total_waste_cost / total_revenue_analysis * 100) if total_revenue_analysis > 0 else 0
    
    finding_2 = {
        "title": "Quantified Waste Cost Impact (Week 27)",
        "claim": f"Documented waste cost in Week 27 totals {total_waste_cost:.2f} SAR, representing {waste_pct_revenue:.2f}% of gross revenue. {len(waste_items)} items had recorded waste.",
        "finding_type": "waste_economics",
        "metrics": {
            "total_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": round(total_waste_cost, 2),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "waste_pct_of_revenue": {
                "value": round(waste_pct_revenue, 2),
                "unit": "%",
                "numerator": round(waste_pct_revenue, 2),
                "denominator": 100,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "items_with_waste": {
                "value": len(waste_items),
                "unit": "count",
                "numerator": len(waste_items),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_waste_item": {
                "value": waste_items.iloc[0]['item'] if len(waste_items) > 0 else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "top_waste_item_cost_sar": {
                "value": round(waste_items.iloc[0]['known_waste_cost_sar'], 2) if len(waste_items) > 0 else None,
                "unit": "SAR",
                "numerator": round(waste_items.iloc[0]['known_waste_cost_sar'], 2) if len(waste_items) > 0 else None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["inventory"],
        "sample_size": len(waste_items),
        "coverage_notes": [
            f"Analysis period: Week starting 2026-07-13",
            f"Waste records with non-null cost: {len(waste_items)} items",
            f"Total inventory records for period: {len(inventory_analysis)}",
            "Only items with known_waste_cost_sar populated are included"
        ],
        "assumptions": [
            "known_waste_cost_sar represents actual cost of wasted units",
            "Blank waste cost values are treated as unknown, not zero",
            "Waste cost is calculated from unit_cost_sar × units_wasted"
        ],
        "confidence": 0.90
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Supplier Price Change Analysis (Temporal Alignment)
# ============================================================================

# Filter emails for price changes with valid dates
emails_price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(emails_price_changes) > 0:
    # Convert dates to datetime
    emails_price_changes['effective_date'] = pd.to_datetime(emails_price_changes['effective_date'], utc=True)
    emails_price_changes['email_date'] = pd.to_datetime(emails_price_changes['date'], utc=True)
    
    # Calculate price change percentage
    emails_price_changes['price_change_pct'] = (
        (emails_price_changes['new_price'] - emails_price_changes['old_price']) / 
        emails_price_changes['old_price'] * 100
    ).round(2)
    
    # Sort by effective date descending to get most recent
    emails_price_changes_sorted = emails_price_changes.sort_values('effective_date', ascending=False)
    
    # Get the most recent price change
    most_recent = emails_price_changes_sorted.iloc[0]
    
    # Check if effective date is within or before analysis period
    effective_date_dt = pd.to_datetime(most_recent['effective_date'], utc=True)
    
    # For temporal clarity: if effective date is before analysis period, note the lag
    days_lag = (analysis_start - effective_date_dt).days
    
    finding_3 = {
        "title": "Supplier Price Change: Full-Fat Milk (Temporal Context)",
        "claim": f"Email dated {most_recent['email_date'].strftime('%Y-%m-%d')} announced a price change for {most_recent['entity_or_ingredient']} effective {most_recent['effective_date']}: {most_recent['old_price']} → {most_recent['new_price']} {most_recent['currency']}/{most_recent['unit']}, a {most_recent['price_change_pct']:.2f}% increase. This change was effective {days_lag} days before the analysis period began.",
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
                "value": float(most_recent['old_price']),
                "unit": f"{most_recent['currency']}/{most_recent['unit']}",
                "numerator": float(most_recent['old_price']),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "new_price": {
                "value": float(most_recent['new_price']),
                "unit": f"{most_recent['currency']}/{most_recent['unit']}",
                "numerator": float(most_recent['new_price']),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "price_change_pct": {
                "value": float(most_recent['price_change_pct']),
                "unit": "%",
                "numerator": float(most_recent['price_change_pct']),
                "denominator": 100,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "effective_date": {
                "value": most_recent['effective_date'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "email_date": {
                "value": most_recent['email_date'].strftime('%Y-%m-%d'),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "days_before_analysis_period": {
                "value": int(days_lag),
                "unit": "days",
                "numerator": int(days_lag),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["emails"],
        "sample_size": 1,
        "coverage_notes": [
            f"Email extraction confidence: {most_recent['confidence']}",
            f"Effective date: {most_recent['effective_date']} (before analysis period start by {days_lag} days)",
            "No recipe/BOM data available; cannot calculate per-drink impact without ingredient quantities",
            "No purchase order or invoice data available to confirm actual implementation"
        ],
        "assumptions": [
            "Email announcement reflects actual supplier price change",
            "Price change applies to cafe's procurement contract (not verified)",
            "Standing order quantity and payment terms are not confirmed in email data",
            "No recipe data available; cannot estimate impact on menu item costs without ingredient specifications"
        ],
        "confidence": 0.70
    }
    findings.append(finding_3)

# ============================================================================
# Compile final output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write to output path
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")
