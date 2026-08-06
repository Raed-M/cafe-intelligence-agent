import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)
inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Define analysis period
analysis_start = "2026-03-23T00:00:00+03:00"
analysis_end = "2026-03-30T00:00:00+03:00"
previous_start = "2026-03-16T00:00:00+03:00"
previous_end = "2026-03-23T00:00:00+03:00"

# Parse timestamps with timezone awareness
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)
previous_start_dt = pd.to_datetime(previous_start)
previous_end_dt = pd.to_datetime(previous_end)

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Convert timestamp columns to datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
emails_df['date'] = pd.to_datetime(emails_df['date'])
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'])

# Ensure timezone-aware comparison: convert naive datetimes to UTC-aware if needed
if emails_df['effective_date'].dt.tz is None:
    emails_df['effective_date'] = emails_df['effective_date'].dt.tz_localize('UTC')

findings = []

# ============================================================================
# FINDING 1: Item-level COGS and Gross Profit for Analysis Period
# ============================================================================

# Filter POS for analysis period, exclude refunds
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start_dt) &
    (pos_df['timestamp'] < analysis_end_dt) &
    (pos_df['is_refund'] == False)
].copy()

# Merge with menu to get unit costs
pos_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'item_en']],
    on='sku',
    how='left'
)

# Calculate COGS and gross profit per line
pos_with_cost['cogs_sar'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
pos_with_cost['gross_profit_sar'] = pos_with_cost['line_total_sar'] - pos_with_cost['cogs_sar']
pos_with_cost['gross_margin_pct'] = (
    pos_with_cost['gross_profit_sar'] / pos_with_cost['line_total_sar'] * 100
).replace([np.inf, -np.inf], np.nan)

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
    item_economics['gross_profit_sar'] / item_economics['line_total_sar'] * 100
)

# Sort by gross profit descending
item_economics_sorted = item_economics.sort_values('gross_profit_sar', ascending=False)

# Total metrics
total_revenue = pos_with_cost['line_total_sar'].sum()
total_cogs = pos_with_cost['cogs_sar'].sum()
total_gross_profit = pos_with_cost['gross_profit_sar'].sum()
total_gross_margin = (total_gross_profit / total_revenue * 100) if total_revenue > 0 else 0

# Count transactions
transaction_count = pos_analysis['transaction_id'].nunique()
line_item_count = len(pos_analysis)

finding_1 = {
    "title": "Item-Level COGS and Gross Profit Analysis (Mar 23-30, 2026)",
    "claim": f"During the analysis period (Mar 23-30, 2026), total gross profit across all items was SAR {total_gross_profit:.2f} on revenue of SAR {total_revenue:.2f}, representing a {total_gross_margin:.2f}% gross margin. Top 3 profit contributors: {item_economics_sorted.iloc[0]['item_en']} (SAR {item_economics_sorted.iloc[0]['gross_profit_sar']:.2f}), {item_economics_sorted.iloc[1]['item_en']} (SAR {item_economics_sorted.iloc[1]['gross_profit_sar']:.2f}), {item_economics_sorted.iloc[2]['item_en']} (SAR {item_economics_sorted.iloc[2]['gross_profit_sar']:.2f}).",
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
            "value": round(total_gross_profit, 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "gross_margin_pct": {
            "value": round(total_gross_margin, 2),
            "unit": "%",
            "numerator": round(total_gross_profit, 2),
            "denominator": round(total_revenue, 2),
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "top_profit_item_1_name": {
            "value": item_economics_sorted.iloc[0]['item_en'],
            "unit": None,
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "top_profit_item_1_sar": {
            "value": round(item_economics_sorted.iloc[0]['gross_profit_sar'], 2),
            "unit": "SAR",
            "numerator": None,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    },
    "source_names": ["pos", "menu"],
    "sample_size": transaction_count,
    "coverage_notes": [
        f"Analysis period: {analysis_start} to {analysis_end}",
        f"Transactions: {transaction_count} unique baskets, {line_item_count} line items",
        "Refunds explicitly excluded from revenue and COGS calculations",
        "Unit costs sourced from menu.unit_cost_sar",
        "All items with known SKU and menu cost included"
    ],
    "assumptions": [
        "Menu unit costs are current and applicable to analysis period",
        "Line-item unit_price_sar and quantity are accurate",
        "No adjustments for waste, shrinkage, or promotional cost-absorption beyond recorded discounts"
    ],
    "confidence": 0.95
}

findings.append(finding_1)

# ============================================================================
# FINDING 2: Waste Cost Analysis (Non-Null Waste Records Only)
# ============================================================================

# Filter inventory for analysis period week
# Week starting 2026-03-23 corresponds to the analysis period
inventory_analysis = inventory_df[
    inventory_df['week_starting'] == pd.to_datetime('2026-03-23')
].copy()

# Filter for non-null waste units and known waste cost
waste_records = inventory_analysis[
    (inventory_analysis['units_wasted'].notna()) &
    (inventory_analysis['units_wasted'] > 0) &
    (inventory_analysis['known_waste_cost_sar'].notna())
].copy()

if len(waste_records) > 0:
    total_waste_cost = waste_records['known_waste_cost_sar'].sum()
    total_waste_units = waste_records['units_wasted'].sum()
    waste_items_count = len(waste_records)
    
    # Calculate waste cost as percentage of revenue
    waste_cost_pct = (total_waste_cost / total_revenue * 100) if total_revenue > 0 else 0
    
    finding_2 = {
        "title": "Quantified Waste Cost Impact (Mar 23-30, 2026)",
        "claim": f"During the analysis period, {waste_items_count} items with non-null waste records incurred a total known waste cost of SAR {total_waste_cost:.2f} ({waste_cost_pct:.2f}% of period revenue). This represents {total_waste_units:.0f} units wasted. Waste cost is calculated from inventory.known_waste_cost_sar for records with units_wasted > 0 only.",
        "finding_type": "waste_cost",
        "metrics": {
            "total_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_waste_units": {
                "value": round(total_waste_units, 2),
                "unit": "units",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "waste_cost_pct_of_revenue": {
                "value": round(waste_cost_pct, 2),
                "unit": "%",
                "numerator": round(total_waste_cost, 2),
                "denominator": round(total_revenue, 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "waste_items_count": {
                "value": waste_items_count,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["inventory", "pos"],
        "sample_size": waste_items_count,
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            "Only non-null waste records (units_wasted > 0) included",
            "Only records with known_waste_cost_sar populated included",
            f"Waste items: {waste_items_count} SKUs",
            "Revenue denominator: POS sales (refunds excluded) for same period",
            "Waste records sourced from inventory.known_waste_cost_sar; no refund or data quality filtering applied at inventory level"
        ],
        "assumptions": [
            "Waste units include all non-null inventory loss records, which may include spoilage, disposal, shrinkage, and potential data entry errors",
            "known_waste_cost_sar reflects recorded economic cost of waste",
            "No recovery value assumed for wasted units",
            "Inventory waste records are period-aligned with POS sales",
            "Unit costs in inventory records are current as of analysis period start (2026-03-23)"
        ],
        "confidence": 0.85
    }
    
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Supplier Price Changes and Procurement Cost Scenario
# ============================================================================

# Filter emails for price changes with effective dates
# Ensure timezone-aware comparison by converting analysis_end_dt to UTC if needed
analysis_end_dt_utc = analysis_end_dt.tz_convert('UTC') if analysis_end_dt.tz else analysis_end_dt.tz_localize('UTC')

price_changes = emails_df[
    (emails_df['old_price'].notna()) &
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna()) &
    (emails_df['effective_date'] <= analysis_end_dt_utc)
].copy()

if len(price_changes) > 0:
    # Calculate percentage change
    price_changes['pct_change'] = (
        (price_changes['new_price'] - price_changes['old_price']) / price_changes['old_price'] * 100
    )
    
    # Sort by effective date descending to get most recent
    price_changes_sorted = price_changes.sort_values('effective_date', ascending=False)
    
    # Take the most recent price change
    latest_change = price_changes_sorted.iloc[0]
    
    entity = latest_change['entity_or_ingredient']
    old_price = latest_change['old_price']
    new_price = latest_change['new_price']
    currency = latest_change['currency']
    unit = latest_change['unit']
    effective_date = latest_change['effective_date']
    pct_change = latest_change['pct_change']
    
    # Look for standing order quantity in facts
    facts_text = str(latest_change['facts']).lower() if pd.notna(latest_change['facts']) else ""
    
    # Extract standing order quantity if mentioned
    standing_qty = None
    if 'standing order' in facts_text or 'monthly order' in facts_text:
        # Try to extract a number from facts
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', facts_text)
        if numbers:
            standing_qty = float(numbers[0])
    
    # Calculate procurement cost scenario if standing quantity available
    if standing_qty:
        price_delta = new_price - old_price
        procurement_cost_impact = standing_qty * price_delta
        
        finding_3 = {
            "title": f"Supplier Price Change: {entity} (Effective {effective_date.strftime('%Y-%m-%d')})",
            "claim": f"Supplier {latest_change['sender']} notified a price change for {entity} effective {effective_date.strftime('%Y-%m-%d')}: {old_price} {currency}/{unit} → {new_price} {currency}/{unit} ({pct_change:+.2f}%). Based on extracted standing order quantity of {standing_qty:.0f} {unit}, the monthly procurement cost impact is estimated at {procurement_cost_impact:+.2f} {currency}. This scenario assumes continued order volume and current payment terms.",
            "finding_type": "supplier_price_change",
            "metrics": {
                "entity": {
                    "value": entity,
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "old_price": {
                    "value": round(old_price, 4),
                    "unit": f"{currency}/{unit}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "new_price": {
                    "value": round(new_price, 4),
                    "unit": f"{currency}/{unit}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "price_change_pct": {
                    "value": round(pct_change, 2),
                    "unit": "%",
                    "numerator": round(new_price - old_price, 4),
                    "denominator": round(old_price, 4),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "effective_date": {
                    "value": effective_date.strftime('%Y-%m-%d'),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "standing_order_qty": {
                    "value": round(standing_qty, 2) if standing_qty else None,
                    "unit": unit,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "monthly_procurement_impact": {
                    "value": round(procurement_cost_impact, 2) if standing_qty else None,
                    "unit": currency,
                    "numerator": round(standing_qty * (new_price - old_price), 2) if standing_qty else None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": ["emails"],
            "sample_size": 1,
            "coverage_notes": [
                f"Email date: {latest_change['date'].strftime('%Y-%m-%d')}",
                f"Effective date: {effective_date.strftime('%Y-%m-%d')}",
                f"Sender: {latest_change['sender']}",
                "Standing order quantity extracted from email facts text",
                "Procurement cost scenario is illustrative; assumes continued order volume and payment terms"
            ],
            "assumptions": [
                "Standing order quantity of {:.0f} {} continues unchanged".format(standing_qty, unit),
                "Price delta applies uniformly to all units ordered",
                "No volume discounts or contract renegotiations",
                "Payment terms and delivery schedules remain constant",
                "Extracted standing quantity is accurate and current"
            ],
            "confidence": 0.70
        }
        
        findings.append(finding_3)
    else:
        # Price change without standing order quantity
        finding_3 = {
            "title": f"Supplier Price Change: {entity} (Effective {effective_date.strftime('%Y-%m-%d')})",
            "claim": f"Supplier {latest_change['sender']} notified a price change for {entity} effective {effective_date.strftime('%Y-%m-%d')}: {old_price} {currency}/{unit} → {new_price} {currency}/{unit} ({pct_change:+.2f}%). No standing order quantity was extracted from the email; procurement cost impact cannot be quantified without order volume data.",
            "finding_type": "supplier_price_change",
            "metrics": {
                "entity": {
                    "value": entity,
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "old_price": {
                    "value": round(old_price, 4),
                    "unit": f"{currency}/{unit}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "new_price": {
                    "value": round(new_price, 4),
                    "unit": f"{currency}/{unit}",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "price_change_pct": {
                    "value": round(pct_change, 2),
                    "unit": "%",
                    "numerator": round(new_price - old_price, 4),
                    "denominator": round(old_price, 4),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "effective_date": {
                    "value": effective_date.strftime('%Y-%m-%d'),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": ["emails"],
            "sample_size": 1,
            "coverage_notes": [
                f"Email date: {latest_change['date'].strftime('%Y-%m-%d')}",
                f"Effective date: {effective_date.strftime('%Y-%m-%d')}",
                f"Sender: {latest_change['sender']}",
                "Standing order quantity not found in email facts"
            ],
            "assumptions": [
                "Price change is confirmed and applicable to cafe orders",
                "No volume discounts or contract renegotiations"
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
