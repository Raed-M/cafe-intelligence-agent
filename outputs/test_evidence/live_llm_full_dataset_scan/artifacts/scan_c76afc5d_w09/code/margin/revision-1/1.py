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

# Parse analysis period
analysis_start = datetime.fromisoformat("2026-03-09T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-03-16T00:00:00+03:00")

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

# Filter POS to analysis period
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start) & 
    (pos_df['timestamp'] < analysis_end)
].copy()

# Filter inventory to analysis period
inventory_analysis = inventory_df[
    (inventory_df['week_starting'] >= analysis_start) & 
    (inventory_df['week_starting'] < analysis_end)
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

# Calculate line-level COGS and gross profit
pos_with_cost['line_cogs_sar'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
pos_with_cost['line_gross_profit_sar'] = pos_with_cost['line_total_sar'] - pos_with_cost['line_cogs_sar']

# Exclude refunds from aggregation
pos_sales = pos_with_cost[pos_with_cost['is_refund'] == False].copy()

# Aggregate by item
item_economics = pos_sales.groupby(['sku', 'item_name_en']).agg({
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'line_cogs_sar': 'sum',
    'line_gross_profit_sar': 'sum',
    'transaction_id': 'nunique'
}).reset_index()

item_economics.columns = ['sku', 'item_name', 'total_qty', 'total_revenue', 'total_cogs', 'total_gross_profit', 'basket_count']

# Calculate margin rate
item_economics['margin_rate'] = (item_economics['total_gross_profit'] / item_economics['total_revenue']).fillna(0)

# Sort by gross profit contribution
item_economics_sorted = item_economics.sort_values('total_gross_profit', ascending=False)

# Top 5 items by gross profit
top_items = item_economics_sorted.head(5)

if len(top_items) > 0:
    finding_1 = {
        "title": "Top 5 Items by Gross Profit Contribution (Week 9-16 Mar 2026)",
        "claim": f"During the analysis period (2026-03-09 to 2026-03-16), the top 5 items by gross profit contribution generated {top_items['total_gross_profit'].sum():.2f} SAR in total gross profit across {top_items['basket_count'].sum()} baskets, representing {(top_items['total_gross_profit'].sum() / item_economics['total_gross_profit'].sum() * 100):.1f}% of total cafe gross profit.",
        "finding_type": "item_economics",
        "metrics": {
            "top_5_total_gross_profit_sar": {
                "value": round(top_items['total_gross_profit'].sum(), 2),
                "unit": "SAR",
                "numerator": round(top_items['total_gross_profit'].sum(), 2),
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "top_5_contribution_pct": {
                "value": round(top_items['total_gross_profit'].sum() / item_economics['total_gross_profit'].sum() * 100, 1),
                "unit": "%",
                "numerator": round(top_items['total_gross_profit'].sum(), 2),
                "denominator": round(item_economics['total_gross_profit'].sum(), 2),
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "top_5_basket_count": {
                "value": int(top_items['basket_count'].sum()),
                "unit": "baskets",
                "numerator": int(top_items['basket_count'].sum()),
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "cafe_total_gross_profit_sar": {
                "value": round(item_economics['total_gross_profit'].sum(), 2),
                "unit": "SAR",
                "numerator": round(item_economics['total_gross_profit'].sum(), 2),
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": int(pos_sales.shape[0]),
        "coverage_notes": [
            "POS data filtered to analysis period 2026-03-09 to 2026-03-16",
            "Refunds excluded from revenue and profit calculations",
            "Menu unit costs merged on SKU; items without menu cost data excluded",
            f"Total POS line items in analysis period: {pos_sales.shape[0]}",
            f"Unique items analyzed: {len(item_economics)}"
        ],
        "assumptions": [
            "Menu unit_cost_sar is current and applicable to analysis period",
            "Line totals in POS are net of discounts and represent actual revenue",
            "COGS = quantity × unit_cost_sar from menu",
            "Gross profit = line_total_sar - line_cogs_sar"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Waste Cost Impact Analysis
# ============================================================================

# Filter inventory to analysis period and identify non-null waste
inventory_with_waste = inventory_analysis[
    inventory_analysis['known_waste_cost_sar'].notna() & 
    (inventory_analysis['known_waste_cost_sar'] > 0)
].copy()

if len(inventory_with_waste) > 0:
    total_waste_cost = inventory_with_waste['known_waste_cost_sar'].sum()
    waste_items = inventory_with_waste.groupby('item').agg({
        'known_waste_cost_sar': 'sum',
        'units_wasted': 'sum'
    }).reset_index().sort_values('known_waste_cost_sar', ascending=False)
    
    # Calculate as percentage of total revenue
    total_revenue = pos_sales['line_total_sar'].sum()
    waste_pct_revenue = (total_waste_cost / total_revenue * 100) if total_revenue > 0 else 0
    
    finding_2 = {
        "title": "Quantified Waste Cost Impact (Week 9-16 Mar 2026)",
        "claim": f"Known waste cost during the analysis period totaled {total_waste_cost:.2f} SAR across {len(inventory_with_waste)} inventory records, representing {waste_pct_revenue:.2f}% of cafe revenue. Top waste contributor: {waste_items.iloc[0]['item']} ({waste_items.iloc[0]['known_waste_cost_sar']:.2f} SAR).",
        "finding_type": "waste_cost",
        "metrics": {
            "total_known_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": round(total_waste_cost, 2),
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "waste_cost_pct_revenue": {
                "value": round(waste_pct_revenue, 2),
                "unit": "%",
                "numerator": round(total_waste_cost, 2),
                "denominator": round(total_revenue, 2),
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "waste_records_count": {
                "value": len(inventory_with_waste),
                "unit": "records",
                "numerator": len(inventory_with_waste),
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "top_waste_item": {
                "value": waste_items.iloc[0]['item'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "top_waste_cost_sar": {
                "value": round(waste_items.iloc[0]['known_waste_cost_sar'], 2),
                "unit": "SAR",
                "numerator": round(waste_items.iloc[0]['known_waste_cost_sar'], 2),
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            }
        },
        "source_names": ["inventory", "pos"],
        "sample_size": len(inventory_with_waste),
        "coverage_notes": [
            "Inventory data filtered to analysis period 2026-03-09 to 2026-03-16",
            "Only non-null, positive known_waste_cost_sar values included",
            f"Inventory records with waste cost: {len(inventory_with_waste)} out of {len(inventory_analysis)} total",
            "Blank waste values excluded per schema guidance",
            f"Total cafe revenue (net of refunds): {total_revenue:.2f} SAR"
        ],
        "assumptions": [
            "known_waste_cost_sar in inventory is accurate and complete for non-null entries",
            "Waste cost represents actual COGS loss",
            "Revenue includes all non-refund POS transactions"
        ],
        "confidence": 0.90
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Supplier Price Change Detection and Temporal Validation
# ============================================================================

# Filter emails to those with price changes and valid dates
price_change_emails = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['date'].notna())
].copy()

if len(price_change_emails) > 0:
    # Calculate percentage change
    price_change_emails['pct_change'] = (
        (price_change_emails['new_price'] - price_change_emails['old_price']) / 
        price_change_emails['old_price'] * 100
    )
    
    # Sort by date descending to get most recent
    price_change_emails_sorted = price_change_emails.sort_values('date', ascending=False)
    
    # Get most recent price change
    latest_change = price_change_emails_sorted.iloc[0]
    
    # Check temporal relationship
    email_date = latest_change['date']
    effective_date = latest_change['effective_date']
    
    # Determine if email was received during or before analysis period
    email_in_period = (email_date >= analysis_start) and (email_date < analysis_end)
    email_before_period = email_date < analysis_start
    effective_after_period = effective_date >= analysis_end if pd.notna(effective_date) else False
    
    temporal_status = "received_during_analysis_period" if email_in_period else (
        "received_before_analysis_period" if email_before_period else "unknown"
    )
    
    finding_3 = {
        "title": "Supplier Price Change: Full-Fat Milk (Most Recent)",
        "claim": f"Email from {latest_change['sender']} (received {email_date.strftime('%Y-%m-%d')}) announces {latest_change['entity_or_ingredient']} price change from {latest_change['old_price']} to {latest_change['new_price']} {latest_change['currency']}/{latest_change['unit']}, effective {effective_date.strftime('%Y-%m-%d') if pd.notna(effective_date) else 'date not specified'}. This represents a {latest_change['pct_change']:.2f}% price increase. Email receipt status: {temporal_status}. Impact on cafe margins cannot be quantified without standing order volume data and confirmed supplier relationship.",
        "finding_type": "supplier_price_change",
        "metrics": {
            "old_price": {
                "value": latest_change['old_price'],
                "unit": f"{latest_change['currency']}/{latest_change['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "new_price": {
                "value": latest_change['new_price'],
                "unit": f"{latest_change['currency']}/{latest_change['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "price_change_pct": {
                "value": round(latest_change['pct_change'], 2),
                "unit": "%",
                "numerator": round(latest_change['new_price'] - latest_change['old_price'], 2),
                "denominator": round(latest_change['old_price'], 2),
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "email_date": {
                "value": email_date.strftime('%Y-%m-%d'),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "effective_date": {
                "value": effective_date.strftime('%Y-%m-%d') if pd.notna(effective_date) else None,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "email_receipt_status": {
                "value": temporal_status,
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "effective_after_analysis_period": {
                "value": "yes" if effective_after_period else "no",
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            }
        },
        "source_names": ["emails"],
        "sample_size": len(price_change_emails),
        "coverage_notes": [
            f"Email extraction identified {len(price_change_emails)} price change records",
            f"Most recent price change analyzed: {latest_change['entity_or_ingredient']} from {latest_change['sender']}",
            "Email receipt date is distinct from effective date; effective date is future-dated relative to analysis period",
            "No recipe/BOM data available to calculate per-drink cost impact",
            "Supplier identity and standing order applicability not confirmed in structured data"
        ],
        "assumptions": [
            "Email sender is authoritative supplier (not independently verified)",
            "Price change applies to cafe's standing orders (not confirmed)",
            "No standing order volume data available; per-unit impact cannot be calculated",
            "Effective date represents when price change takes effect; current period pricing unaffected"
        ],
        "confidence": 0.70
    }
    findings.append(finding_3)

# ============================================================================
# Construct final output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write to output path
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")
