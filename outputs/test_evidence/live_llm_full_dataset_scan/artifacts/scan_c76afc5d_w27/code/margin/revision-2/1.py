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
menu_df = pd.read_parquet(inputs['menu'])
inventory_df = pd.read_parquet(inputs['inventory'])
emails_df = pd.read_parquet(inputs['emails'])

# Define analysis period
analysis_start = "2026-07-13"
analysis_end = "2026-07-20"

# Convert timestamp columns to datetime
pos_df['timestamp_local'] = pd.to_datetime(pos_df['timestamp_local'])
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
emails_df['date'] = pd.to_datetime(emails_df['date'])
emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'])

# Filter POS data for analysis period (Week 27: 2026-07-13 to 2026-07-20)
pos_analysis = pos_df[
    (pos_df['calendar_date'] >= analysis_start) & 
    (pos_df['calendar_date'] < analysis_end)
].copy()

# Filter inventory for analysis period
inventory_analysis = inventory_df[
    (inventory_df['week_starting'] >= analysis_start) & 
    (inventory_df['week_starting'] < analysis_end)
].copy()

findings = []

# ============================================================================
# FINDING 1: Item-Level Gross Profit Analysis (Exact Economics)
# ============================================================================

# Exclude refunds from POS analysis
pos_sales = pos_analysis[pos_analysis['is_refund'] == False].copy()

# Merge POS with menu to get unit costs
pos_with_cost = pos_sales.merge(
    menu_df[['sku', 'unit_cost_sar', 'item_en']],
    on='sku',
    how='left'
)

# Calculate item-level metrics
pos_with_cost['cogs'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
pos_with_cost['gross_profit'] = pos_with_cost['line_total_sar'] - pos_with_cost['cogs']

# Group by item to find top contributors
item_metrics = pos_with_cost.groupby('sku').agg({
    'item_en': 'first',
    'quantity': 'sum',
    'line_total_sar': 'sum',
    'cogs': 'sum',
    'gross_profit': 'sum',
    'transaction_id': 'nunique'
}).reset_index()

item_metrics.columns = ['sku', 'item_name', 'units_sold', 'revenue', 'cogs', 'gross_profit', 'transactions']
item_metrics['gross_margin_pct'] = (item_metrics['gross_profit'] / item_metrics['revenue'] * 100).round(2)
item_metrics = item_metrics.sort_values('gross_profit', ascending=False)

# Get top item by gross profit
if len(item_metrics) > 0:
    top_item = item_metrics.iloc[0]
    
    # Verify no nulls in cost data
    top_item_detail = pos_with_cost[pos_with_cost['sku'] == top_item['sku']]
    null_cost_count = top_item_detail['unit_cost_sar'].isna().sum()
    
    if null_cost_count == 0:
        finding_1 = {
            "title": "Top Gross Profit Item: Exact Item Economics",
            "claim": f"{top_item['item_name']} generated {top_item['gross_profit']:.2f} SAR gross profit ({top_item['gross_margin_pct']:.2f}% margin) from {int(top_item['units_sold'])} units across {int(top_item['transactions'])} transactions during Week 27.",
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
                "gross_profit_sar": {
                    "value": round(top_item['gross_profit'], 2),
                    "unit": "SAR",
                    "numerator": round(top_item['revenue'], 2),
                    "denominator": round(top_item['cogs'], 2),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "gross_margin_pct": {
                    "value": top_item['gross_margin_pct'],
                    "unit": "%",
                    "numerator": round(top_item['gross_profit'], 2),
                    "denominator": round(top_item['revenue'], 2),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "units_sold": {
                    "value": int(top_item['units_sold']),
                    "unit": "units",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "revenue_sar": {
                    "value": round(top_item['revenue'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "cogs_sar": {
                    "value": round(top_item['cogs'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "transactions": {
                    "value": int(top_item['transactions']),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": int(top_item['transactions']),
            "coverage_notes": [
                f"Analysis period: Week 27 (2026-07-13 to 2026-07-20)",
                f"Total POS transactions in period: {pos_sales['transaction_id'].nunique()}",
                f"Total items in menu: {len(menu_df)}",
                f"Items with sales in period: {len(item_metrics)}",
                f"Refunds excluded from revenue and COGS calculations",
                f"Unit cost sourced from menu.unit_cost_sar; no recipe/BOM adjustments applied"
            ],
            "assumptions": [
                "Menu unit_cost_sar is current and applies to all units sold in period",
                "No recipe/BOM data available; per-ingredient cost variance not quantified",
                "Discount amounts (discount_sar) are deducted from line_total_sar before COGS calculation",
                "Refunds (is_refund=True) excluded from all calculations"
            ],
            "confidence": 0.95
        }
        findings.append(finding_1)

# ============================================================================
# FINDING 2: Waste Cost Analysis (Exact Waste Records Only)
# ============================================================================

# Filter inventory records with non-null waste cost
waste_records = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()

if len(waste_records) > 0:
    total_waste_cost = waste_records['known_waste_cost_sar'].sum()
    total_revenue_period = pos_sales['line_total_sar'].sum()
    waste_pct_revenue = (total_waste_cost / total_revenue_period * 100) if total_revenue_period > 0 else 0
    
    # Find top waste item
    waste_by_item = waste_records.sort_values('known_waste_cost_sar', ascending=False)
    top_waste_item = waste_by_item.iloc[0] if len(waste_by_item) > 0 else None
    
    finding_2 = {
        "title": "Quantified Waste Cost Impact",
        "claim": f"Week 27 recorded {len(waste_records)} inventory items with documented waste totaling {total_waste_cost:.2f} SAR, representing {waste_pct_revenue:.2f}% of gross revenue. Top waste item: {top_waste_item['item']} ({top_waste_item['known_waste_cost_sar']:.2f} SAR).",
        "finding_type": "waste_economics",
        "metrics": {
            "total_waste_cost_sar": {
                "value": round(total_waste_cost, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "waste_pct_of_revenue": {
                "value": round(waste_pct_revenue, 2),
                "unit": "%",
                "numerator": round(total_waste_cost, 2),
                "denominator": round(total_revenue_period, 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "waste_items_count": {
                "value": len(waste_records),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "top_waste_item": {
                "value": top_waste_item['item'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "top_waste_cost_sar": {
                "value": round(top_waste_item['known_waste_cost_sar'], 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "gross_revenue_sar": {
                "value": round(total_revenue_period, 2),
                "unit": "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["inventory", "pos"],
        "sample_size": len(waste_records),
        "coverage_notes": [
            f"Analysis period: Week 27 (2026-07-13 to 2026-07-20)",
            f"Total inventory records for period: {len(inventory_analysis)}",
            f"Waste records with non-null known_waste_cost_sar: {len(waste_records)}",
            f"Waste cost calculated from inventory.known_waste_cost_sar (pre-computed field)",
            f"Blank/null waste costs excluded per data quality rules",
            f"Revenue denominator: POS gross revenue (refunds excluded)"
        ],
        "assumptions": [
            "known_waste_cost_sar field represents actual waste cost incurred (not estimated)",
            "Waste cost is calculated as units_wasted × unit_cost_sar in source data",
            "All inventory items with waste activity in period are captured",
            "Revenue used for percentage calculation is net of refunds"
        ],
        "confidence": 0.90
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Supplier Price Change Signal (Email Evidence Only)
# ============================================================================

# Filter emails for price changes with clear old/new prices
price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['category'] == 'supplier_price_change')
].copy()

if len(price_changes) > 0:
    # Take first price change (most recent or relevant)
    price_change = price_changes.iloc[0]
    
    old_price = float(price_change['old_price'])
    new_price = float(price_change['new_price'])
    price_delta = new_price - old_price
    pct_change = (price_delta / old_price * 100) if old_price > 0 else 0
    
    finding_3 = {
        "title": "Supplier Price Change Signal (Email Announcement)",
        "claim": f"Email dated {price_change['date'].strftime('%Y-%m-%d')} announces {price_change['entity_or_ingredient']} price increase from {old_price} to {new_price} {price_change['currency']}/{price_change['unit']} (effective {price_change['effective_date'].strftime('%Y-%m-%d')}), representing a {pct_change:.2f}% increase. No recipe/BOM data available to quantify per-item margin impact.",
        "finding_type": "supplier_price_signal",
        "metrics": {
            "ingredient": {
                "value": price_change['entity_or_ingredient'],
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "old_price": {
                "value": old_price,
                "unit": f"{price_change['currency']}/{price_change['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "new_price": {
                "value": new_price,
                "unit": f"{price_change['currency']}/{price_change['unit']}",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "price_change_pct": {
                "value": round(pct_change, 2),
                "unit": "%",
                "numerator": round(price_delta, 2),
                "denominator": old_price,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "effective_date": {
                "value": price_change['effective_date'].strftime('%Y-%m-%d'),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "email_date": {
                "value": price_change['date'].strftime('%Y-%m-%d'),
                "unit": None,
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "extraction_confidence": {
                "value": price_change['confidence'],
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
            f"Analysis period: Week 27 (2026-07-13 to 2026-07-20)",
            f"Email announcement date: {price_change['date'].strftime('%Y-%m-%d')}",
            f"Effective date: {price_change['effective_date'].strftime('%Y-%m-%d')} (temporal gap: {(pd.to_datetime(analysis_start) - price_change['effective_date']).days} days before analysis period)",
            f"No recipe/BOM data available; cannot calculate per-drink impact without ingredient quantities",
            f"No purchase order or invoice data available to confirm actual implementation",
            f"Email extraction confidence: {price_change['confidence']}"
        ],
        "assumptions": [
            "Price change applies to cafe's procurement contract (not verified in transactional data)",
            "Standing order quantity and payment terms not confirmed in email data",
            "Email announcement reflects actual supplier price change (not verified against invoices)",
            "Temporal gap between effective date (May 2026) and analysis period (July 2026) noted; actual cafe procurement cost impact during analysis period not confirmed"
        ],
        "confidence": 0.65
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
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")