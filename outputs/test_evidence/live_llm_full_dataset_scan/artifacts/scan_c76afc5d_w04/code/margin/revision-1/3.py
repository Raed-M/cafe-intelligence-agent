import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read input artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Define analysis period
analysis_start = datetime(2026, 2, 2, 0, 0, 0, tzinfo=timezone.utc)
analysis_end = datetime(2026, 2, 9, 0, 0, 0, tzinfo=timezone.utc)

# Convert timestamp columns to datetime
pos_df['timestamp_local'] = pd.to_datetime(pos_df['timestamp_local'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
emails_df['date'] = pd.to_datetime(emails_df['date'])
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'])

# Ensure all datetime columns are timezone-naive for consistent comparison
if pos_df['timestamp_local'].dt.tz is not None:
    pos_df['timestamp_local'] = pos_df['timestamp_local'].dt.tz_localize(None)
if inventory_df['week_starting'].dt.tz is not None:
    inventory_df['week_starting'] = inventory_df['week_starting'].dt.tz_localize(None)
if emails_df['date'].dt.tz is not None:
    emails_df['date'] = emails_df['date'].dt.tz_localize(None)
if emails_df['effective_date'].dt.tz is not None:
    emails_df['effective_date'] = emails_df['effective_date'].dt.tz_localize(None)

# Convert analysis period to timezone-naive for comparison
analysis_start_naive = analysis_start.replace(tzinfo=None)
analysis_end_naive = analysis_end.replace(tzinfo=None)

# Filter POS data to analysis period
pos_analysis = pos_df[
    (pos_df['timestamp_local'] >= analysis_start_naive) & 
    (pos_df['timestamp_local'] < analysis_end_naive)
].copy()

# Filter inventory to analysis period
inventory_analysis = inventory_df[
    (inventory_df['week_starting'] >= analysis_start_naive) & 
    (inventory_df['week_starting'] < analysis_end_naive)
].copy()

findings = []

# ============================================================================
# FINDING 1: Item-level gross profit and margin analysis
# ============================================================================

# Exclude refunds from revenue/profit calculations
pos_sales = pos_analysis[pos_analysis['is_refund'] == False].copy()

# Calculate item-level metrics
item_metrics = {}
for sku in pos_sales['sku'].unique():
    sku_data = pos_sales[pos_sales['sku'] == sku]
    
    # Get menu cost
    menu_row = menu_df[menu_df['sku'] == sku]
    if menu_row.empty:
        continue
    
    unit_cost = menu_row['unit_cost_sar'].values[0]
    item_name = menu_row['item_en'].values[0]
    
    # Calculate totals
    total_quantity = sku_data['quantity'].sum()
    total_revenue = sku_data['line_total_sar'].sum()
    total_cogs = total_quantity * unit_cost
    gross_profit = total_revenue - total_cogs
    
    if total_revenue > 0:
        margin_pct = (gross_profit / total_revenue) * 100
    else:
        margin_pct = 0
    
    basket_count = sku_data['transaction_id'].nunique()
    
    item_metrics[sku] = {
        'item_name': item_name,
        'total_quantity': total_quantity,
        'total_revenue': round(total_revenue, 2),
        'unit_cost': unit_cost,
        'total_cogs': round(total_cogs, 2),
        'gross_profit': round(gross_profit, 2),
        'margin_pct': round(margin_pct, 2),
        'basket_count': basket_count
    }

# Find highest gross profit item
if item_metrics:
    top_item_sku = max(item_metrics.keys(), key=lambda x: item_metrics[x]['gross_profit'])
    top_item = item_metrics[top_item_sku]
    
    finding_1 = {
        "title": "Highest Gross Profit Item",
        "claim": f"{top_item['item_name']} generated the highest gross profit of {top_item['gross_profit']} SAR with {top_item['margin_pct']}% margin across {top_item['basket_count']} transactions ({top_item['total_quantity']} units sold).",
        "finding_type": "item_economics",
        "metrics": {
            "gross_profit_sar": {
                "value": top_item['gross_profit'],
                "unit": "SAR",
                "numerator": top_item['total_revenue'],
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "gross_margin_pct": {
                "value": top_item['margin_pct'],
                "unit": "%",
                "numerator": top_item['gross_profit'],
                "denominator": top_item['total_revenue'],
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_quantity": {
                "value": top_item['total_quantity'],
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "basket_count": {
                "value": top_item['basket_count'],
                "unit": "transactions",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_revenue": {
                "value": top_item['total_revenue'],
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_cogs": {
                "value": top_item['total_cogs'],
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["pos", "menu"],
        "sample_size": top_item['basket_count'],
        "coverage_notes": [
            f"Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
            f"Refunds excluded from revenue and profit calculations",
            f"Menu unit costs applied from menu.parquet as of analysis period",
            f"Total POS transactions in period: {pos_analysis['transaction_id'].nunique()}",
            f"Total items analyzed: {len(item_metrics)}"
        ],
        "assumptions": [
            "Menu unit_cost_sar values are current and accurate for the analysis period",
            "POS line_total_sar reflects post-discount revenue",
            "No recipe/BOM adjustments applied; unit cost is as declared in menu",
            "Waste costs not included in item-level COGS (tracked separately in inventory)"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Waste cost impact analysis
# ============================================================================

# Calculate waste costs from inventory data
waste_analysis = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()

if not waste_analysis.empty:
    total_waste_cost = waste_analysis['known_waste_cost_sar'].sum()
    total_waste_units = waste_analysis['units_wasted'].sum()
    waste_items = len(waste_analysis)
    
    # Calculate as percentage of total revenue
    total_revenue_period = pos_sales['line_total_sar'].sum()
    waste_pct_of_revenue = (total_waste_cost / total_revenue_period * 100) if total_revenue_period > 0 else 0
    
    finding_2 = {
        "title": "Quantified Waste Cost Impact",
        "claim": f"Known waste cost totaled {round(total_waste_cost, 2)} SAR across {waste_items} items ({round(total_waste_units, 0)} units wasted) in the analysis period, representing {round(waste_pct_of_revenue, 2)}% of total revenue.",
        "finding_type": "cost_analysis",
        "metrics": {
            "total_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_waste_units": {
                "value": round(total_waste_units, 0),
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "waste_pct_of_revenue": {
                "value": round(waste_pct_of_revenue, 2),
                "unit": "%",
                "numerator": round(total_waste_cost, 2),
                "denominator": round(total_revenue_period, 2),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "items_with_waste": {
                "value": waste_items,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["inventory", "pos"],
        "sample_size": waste_items,
        "coverage_notes": [
            f"Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
            f"Only non-null known_waste_cost_sar values included",
            f"Inventory records with null waste cost excluded (treated per schema rules)",
            f"Total inventory records in period: {len(inventory_analysis)}",
            f"Records with quantified waste: {waste_items}"
        ],
        "assumptions": [
            "known_waste_cost_sar values are accurate and complete for reported waste",
            "Null waste values are not treated as zero (per schema rules)",
            "Waste cost is calculated at unit_cost_sar from inventory records",
            "Revenue baseline includes all non-refund POS transactions"
        ],
        "confidence": 0.90
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Supplier price change analysis
# ============================================================================

# Filter emails for price changes with effective dates
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if not price_changes.empty:
    # Sort by effective date and take most recent
    price_changes = price_changes.sort_values('effective_date', ascending=False)
    latest_change = price_changes.iloc[0]
    
    old_price = latest_change['old_price']
    new_price = latest_change['new_price']
    price_delta = new_price - old_price
    pct_change = (price_delta / old_price * 100) if old_price != 0 else 0
    
    entity = latest_change['entity_or_ingredient']
    unit = latest_change['unit'] if pd.notna(latest_change['unit']) else "unit"
    effective_date = latest_change['effective_date']
    email_date = latest_change['date'] if pd.notna(latest_change['date']) else None
    
    # Check if effective date is in the future relative to analysis period
    is_future = effective_date > analysis_end_naive
    
    claim_text = f"Latest supplier price update: {entity} price changing from {old_price} to {new_price} SAR per {unit} (effective {effective_date.date()}), a {round(pct_change, 2)}% increase."
    if is_future:
        days_ahead = (effective_date - analysis_end_naive).days
        claim_text += f" Note: effective date is {days_ahead} days after analysis period end (prospective change)."
    
    # Adjust confidence based on temporal status
    confidence_score = 0.85 if is_future else 0.90
    
    finding_3 = {
        "title": "Supplier Price Change Alert",
        "claim": claim_text,
        "finding_type": "supplier_cost_change",
        "metrics": {
            "old_price_sar": {
                "value": old_price,
                "unit": f"SAR per {unit}",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "new_price_sar": {
                "value": new_price,
                "unit": f"SAR per {unit}",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "price_delta_sar": {
                "value": round(price_delta, 2),
                "unit": f"SAR per {unit}",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "pct_change": {
                "value": round(pct_change, 2),
                "unit": "%",
                "numerator": round(price_delta, 2),
                "denominator": old_price,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "effective_date": {
                "value": effective_date.isoformat(),
                "unit": "ISO date",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": ["emails"],
        "sample_size": 1,
        "coverage_notes": [
            f"Analysis period: {analysis_start.isoformat()} to {analysis_end.isoformat()}",
            f"Email extraction date: {email_date.isoformat() if email_date is not None else 'unknown'}",
            f"Effective date: {effective_date.isoformat()}",
            f"Total price change records in emails: {len(price_changes)}",
            f"Temporal status: {'FUTURE (not yet effective)' if is_future else 'CURRENT or PAST'}"
        ],
        "assumptions": [
            "Email extraction confidence: supplier price changes are as stated in source emails",
            "Business impact unverified—recipe/BOM evidence required to confirm ingredient is used in menu items",
            "Standing order quantities and payment terms not verified; actual procurement cost impact depends on order volume and timing",
            "No corroborating supplier invoice or contract amendment reviewed"
        ],
        "confidence": confidence_score
    }
    findings.append(finding_3)

# ============================================================================
# Build output
# ============================================================================

output = {
    "status": "success" if findings else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")
