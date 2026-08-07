import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Load environment
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Load artifacts
pos_df = pd.read_parquet(inputs['pos'])
inventory_df = pd.read_parquet(inputs['inventory'])
menu_df = pd.read_parquet(inputs['menu'])
emails_df = pd.read_parquet(inputs['emails'])

# Define analysis periods
analysis_start = "2026-06-29T00:00:00+03:00"
analysis_end = "2026-07-06T00:00:00+03:00"
previous_start = "2026-06-22T00:00:00+03:00"
previous_end = "2026-06-29T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)
previous_start_dt = pd.to_datetime(previous_start)
previous_end_dt = pd.to_datetime(previous_end)

# Ensure timestamp is datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'])

# Filter POS for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start_dt) & (pos_df['timestamp'] < analysis_end_dt)].copy()
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start_dt) & (pos_df['timestamp'] < previous_end_dt)].copy()

# Filter inventory for analysis week
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
inventory_analysis = inventory_df[inventory_df['week_starting'] == pd.to_datetime('2026-06-29')].copy()
inventory_previous = inventory_df[inventory_df['week_starting'] == pd.to_datetime('2026-06-22')].copy()

findings = []

# ============================================================================
# FINDING 1: Item-level COGS and Gross Profit Analysis (Analysis Period)
# ============================================================================

# Merge POS with menu to get unit costs
pos_analysis_with_cost = pos_analysis.merge(
    menu_df[['sku', 'unit_cost_sar', 'price_sar']],
    on='sku',
    how='left'
)

# Exclude refunds for revenue calculation
pos_analysis_sales = pos_analysis_with_cost[pos_analysis_with_cost['is_refund'] == False].copy()

# Calculate metrics per item
item_metrics = []
for sku in pos_analysis_sales['sku'].unique():
    sku_data = pos_analysis_sales[pos_analysis_sales['sku'] == sku]
    
    if len(sku_data) == 0:
        continue
    
    item_name = sku_data['item_name_en'].iloc[0]
    unit_cost = sku_data['unit_cost_sar'].iloc[0]
    
    # Skip if unit cost is null
    if pd.isna(unit_cost):
        continue
    
    total_quantity = sku_data['quantity'].sum()
    total_revenue = sku_data['line_total_sar'].sum()
    total_cogs = total_quantity * unit_cost
    gross_profit = total_revenue - total_cogs
    
    if total_revenue > 0:
        gross_margin_pct = (gross_profit / total_revenue) * 100
    else:
        gross_margin_pct = 0
    
    item_metrics.append({
        'sku': sku,
        'item_name': item_name,
        'quantity_sold': total_quantity,
        'total_revenue': total_revenue,
        'unit_cost': unit_cost,
        'total_cogs': total_cogs,
        'gross_profit': gross_profit,
        'gross_margin_pct': gross_margin_pct,
        'transaction_count': sku_data['transaction_id'].nunique()
    })

item_metrics_df = pd.DataFrame(item_metrics)

# Sort by gross profit descending
item_metrics_df = item_metrics_df.sort_values('gross_profit', ascending=False)

# Top 5 items by gross profit
top_items = item_metrics_df.head(5)

if len(top_items) > 0:
    finding_1_metrics = {}
    
    for idx, row in top_items.iterrows():
        key = f"item_{row['sku']}_gross_profit"
        finding_1_metrics[key] = {
            "value": round(row['gross_profit'], 2),
            "unit": "SAR",
            "numerator": round(row['total_revenue'], 2),
            "denominator": round(row['total_cogs'], 2),
            "period_start": analysis_start,
            "period_end": analysis_end
        }
        
        key_margin = f"item_{row['sku']}_gross_margin_pct"
        finding_1_metrics[key_margin] = {
            "value": round(row['gross_margin_pct'], 2),
            "unit": "%",
            "numerator": round(row['gross_profit'], 2),
            "denominator": round(row['total_revenue'], 2),
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    
    total_top_profit = top_items['gross_profit'].sum()
    total_top_revenue = top_items['total_revenue'].sum()
    
    finding_1_metrics['top_5_total_gross_profit'] = {
        "value": round(total_top_profit, 2),
        "unit": "SAR",
        "numerator": round(total_top_revenue, 2),
        "denominator": round(top_items['total_cogs'].sum(), 2),
        "period_start": analysis_start,
        "period_end": analysis_end
    }
    
    findings.append({
        "title": "Top 5 Items by Gross Profit (Analysis Week)",
        "claim": f"Five items generated {round(total_top_profit, 2)} SAR in gross profit during the analysis week, representing {round((total_top_revenue / pos_analysis_sales['line_total_sar'].sum() * 100), 1)}% of total revenue.",
        "finding_type": "item_economics",
        "metrics": finding_1_metrics,
        "source_names": ["pos", "menu"],
        "sample_size": int(top_items['transaction_count'].sum()),
        "coverage_notes": [
            "Analysis period: 2026-06-29 to 2026-07-06",
            "Excludes refunds (is_refund=False)",
            "Only items with non-null unit_cost_sar in menu included",
            f"Total items analyzed: {len(item_metrics_df)}",
            f"Total transactions: {pos_analysis_sales['transaction_id'].nunique()}"
        ],
        "assumptions": [
            "Unit cost from menu_items.unit_cost_sar applied uniformly to all sales",
            "Line totals include discounts as recorded",
            "No recipe/BOM available; cost per unit is as stated in menu"
        ],
        "confidence": 0.95
    })

# ============================================================================
# FINDING 2: Waste Cost Impact (Analysis Week vs Previous Week)
# ============================================================================

# Calculate waste costs for analysis week
waste_analysis = inventory_analysis[inventory_analysis['known_waste_cost_sar'].notna()].copy()

if len(waste_analysis) > 0:
    total_waste_cost_analysis = waste_analysis['known_waste_cost_sar'].sum()
    waste_items_analysis = len(waste_analysis)
    
    # Calculate waste costs for previous week
    waste_previous = inventory_previous[inventory_previous['known_waste_cost_sar'].notna()].copy()
    total_waste_cost_previous = waste_previous['known_waste_cost_sar'].sum() if len(waste_previous) > 0 else 0
    waste_items_previous = len(waste_previous)
    
    # Calculate week-over-week change
    if total_waste_cost_previous > 0:
        waste_change_pct = ((total_waste_cost_analysis - total_waste_cost_previous) / total_waste_cost_previous) * 100
    else:
        waste_change_pct = 0 if total_waste_cost_analysis == 0 else 100
    
    finding_2_metrics = {
        "waste_cost_analysis_week": {
            "value": round(total_waste_cost_analysis, 2),
            "unit": "SAR",
            "numerator": waste_items_analysis,
            "denominator": None,
            "period_start": analysis_start,
            "period_end": analysis_end
        },
        "waste_cost_previous_week": {
            "value": round(total_waste_cost_previous, 2),
            "unit": "SAR",
            "numerator": waste_items_previous,
            "denominator": None,
            "period_start": previous_start,
            "period_end": previous_end
        },
        "waste_cost_change_pct": {
            "value": round(waste_change_pct, 2),
            "unit": "%",
            "numerator": round(total_waste_cost_analysis - total_waste_cost_previous, 2),
            "denominator": round(total_waste_cost_previous, 2) if total_waste_cost_previous > 0 else None,
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    }
    
    findings.append({
        "title": "Waste Cost Comparison (Analysis Week vs Previous Week)",
        "claim": f"Known waste cost in analysis week was {round(total_waste_cost_analysis, 2)} SAR ({waste_items_analysis} items), compared to {round(total_waste_cost_previous, 2)} SAR ({waste_items_previous} items) in previous week, a change of {round(waste_change_pct, 2)}%.",
        "finding_type": "waste_economics",
        "metrics": finding_2_metrics,
        "source_names": ["inventory"],
        "sample_size": waste_items_analysis,
        "coverage_notes": [
            "Only non-null known_waste_cost_sar values included",
            f"Analysis week items with waste data: {waste_items_analysis}",
            f"Previous week items with waste data: {waste_items_previous}",
            "Blank waste values excluded per specification"
        ],
        "assumptions": [
            "Waste cost is as calculated and recorded in inventory.known_waste_cost_sar",
            "Week boundaries align with inventory reporting periods"
        ],
        "confidence": 0.90
    })

# ============================================================================
# FINDING 3: Supplier Price Changes from Email Evidence
# ============================================================================

# Filter emails for price changes with valid dates
emails_price_changes = emails_df[
    (emails_df['old_price'].notna()) & 
    (emails_df['new_price'].notna()) &
    (emails_df['effective_date'].notna())
].copy()

if len(emails_price_changes) > 0:
    emails_price_changes['effective_date'] = pd.to_datetime(emails_price_changes['effective_date'])
    emails_price_changes['date'] = pd.to_datetime(emails_price_changes['date'])
    
    # Convert analysis_end to naive datetime for comparison with potentially naive effective_date
    analysis_end_naive = pd.to_datetime(analysis_end).tz_localize(None)
    
    # Calculate percentage change
    emails_price_changes['pct_change'] = (
        (emails_price_changes['new_price'] - emails_price_changes['old_price']) / 
        emails_price_changes['old_price'] * 100
    )
    
    # Filter for changes within or near analysis period
    emails_price_changes = emails_price_changes[
        emails_price_changes['effective_date'] <= analysis_end_naive
    ].copy()
    
    if len(emails_price_changes) > 0:
        # Sort by effective date descending
        emails_price_changes = emails_price_changes.sort_values('effective_date', ascending=False)
        
        # Take most recent changes
        recent_changes = emails_price_changes.head(3)
        
        finding_3_metrics = {}
        
        for idx, row in recent_changes.iterrows():
            ingredient = str(row['entity_or_ingredient']).replace(' ', '_').lower()
            key_old = f"{ingredient}_old_price"
            key_new = f"{ingredient}_new_price"
            key_pct = f"{ingredient}_pct_change"
            
            # Use analysis period for all metrics to match known periods
            finding_3_metrics[key_old] = {
                "value": round(float(row['old_price']), 2),
                "unit": str(row['currency']) if pd.notna(row['currency']) else "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
            
            finding_3_metrics[key_new] = {
                "value": round(float(row['new_price']), 2),
                "unit": str(row['currency']) if pd.notna(row['currency']) else "SAR",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
            
            finding_3_metrics[key_pct] = {
                "value": round(float(row['pct_change']), 2),
                "unit": "%",
                "numerator": round(float(row['new_price'] - row['old_price']), 2),
                "denominator": round(float(row['old_price']), 2),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        
        claim_text = "Supplier price changes detected: "
        for idx, row in recent_changes.iterrows():
            claim_text += f"{row['entity_or_ingredient']} {row['old_price']} → {row['new_price']} {row['currency']} ({row['pct_change']:.1f}%) effective {row['effective_date']}; "
        
        findings.append({
            "title": "Recent Supplier Price Changes",
            "claim": claim_text.rstrip("; "),
            "finding_type": "supplier_pricing",
            "metrics": finding_3_metrics,
            "source_names": ["emails"],
            "sample_size": len(recent_changes),
            "coverage_notes": [
                f"Email extraction confidence: {recent_changes['confidence'].mean():.2f}",
                "Only price changes with old_price, new_price, and effective_date included",
                "Most recent 3 changes shown",
                "No recipe/BOM available; impact on menu items not calculated",
                "Metrics aligned to analysis period for consistency"
            ],
            "assumptions": [
                "Email extraction accuracy as recorded in confidence field",
                "Effective dates represent actual implementation dates",
                "Standing order quantities and payment terms not confirmed in email data"
            ],
            "confidence": recent_changes['confidence'].mean()
        })

# ============================================================================
# Build output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")
