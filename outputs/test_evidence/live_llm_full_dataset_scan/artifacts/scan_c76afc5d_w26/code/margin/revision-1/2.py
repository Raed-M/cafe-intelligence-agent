import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List

def load_inputs():
    """Load input artifact paths from environment."""
    with open(os.environ['ANALYST_INPUTS_JSON']) as f:
        run_meta = json.load(f)
    return run_meta['inputs'], run_meta['output_path']

def parse_iso_datetime(dt_str):
    """Parse ISO datetime string to datetime object."""
    if isinstance(dt_str, str):
        return pd.to_datetime(dt_str)
    return dt_str

def calculate_findings(inputs: Dict[str, str], output_path: str):
    """Calculate margin and cost analysis findings."""
    
    # Load all artifacts
    pos_df = pd.read_parquet(inputs['pos'])
    inventory_df = pd.read_parquet(inputs['inventory'])
    menu_df = pd.read_parquet(inputs['menu'])
    emails_df = pd.read_parquet(inputs['emails'])
    
    # Parse timestamps
    pos_df['timestamp_local'] = pd.to_datetime(pos_df['timestamp_local'])
    inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])
    menu_df['launch_date'] = pd.to_datetime(menu_df['launch_date'], errors='coerce')
    menu_df['retire_date'] = pd.to_datetime(menu_df['retire_date'], errors='coerce')
    emails_df['date'] = pd.to_datetime(emails_df['date'], errors='coerce')
    emails_df['effective_date'] = pd.to_datetime(emails_df['effective_date'], errors='coerce')
    
    # Define analysis period - remove timezone info for comparison with tz-naive data
    analysis_start = pd.to_datetime("2026-07-06T00:00:00+03:00").tz_localize(None)
    analysis_end = pd.to_datetime("2026-07-13T00:00:00+03:00").tz_localize(None)
    previous_start = pd.to_datetime("2026-06-29T00:00:00+03:00").tz_localize(None)
    previous_end = pd.to_datetime("2026-07-06T00:00:00+03:00").tz_localize(None)
    
    # Store original ISO strings for output
    analysis_start_iso = "2026-07-06T00:00:00+03:00"
    analysis_end_iso = "2026-07-13T00:00:00+03:00"
    
    findings = []
    
    # FINDING 1: Item-level COGS and Gross Profit Analysis
    # Filter POS for analysis period, exclude refunds
    pos_analysis = pos_df[
        (pos_df['timestamp_local'] >= analysis_start) & 
        (pos_df['timestamp_local'] < analysis_end) &
        (pos_df['is_refund'] == False)
    ].copy()
    
    # Merge with menu to get unit costs
    pos_with_cost = pos_analysis.merge(
        menu_df[['sku', 'unit_cost_sar', 'item_en']],
        on='sku',
        how='left'
    )
    
    # Calculate item-level metrics
    pos_with_cost['cogs'] = pos_with_cost['quantity'] * pos_with_cost['unit_cost_sar']
    pos_with_cost['gross_profit'] = pos_with_cost['line_total_sar'] - pos_with_cost['cogs']
    pos_with_cost['gross_margin_pct'] = (pos_with_cost['gross_profit'] / pos_with_cost['line_total_sar'] * 100).fillna(0)
    
    # Aggregate by item
    item_metrics = pos_with_cost.groupby('sku').agg({
        'quantity': 'sum',
        'line_total_sar': 'sum',
        'cogs': 'sum',
        'gross_profit': 'sum',
        'item_name_en': 'first',
        'unit_cost_sar': 'first',
        'unit_price_sar': 'first'
    }).reset_index()
    
    item_metrics['gross_margin_pct'] = (item_metrics['gross_profit'] / item_metrics['line_total_sar'] * 100).fillna(0)
    item_metrics = item_metrics.sort_values('gross_profit', ascending=False)
    
    # Top 3 items by gross profit
    top_items = item_metrics.head(3)
    
    if len(top_items) > 0:
        top_item = top_items.iloc[0]
        finding1 = {
            "title": "Top Gross Profit Item - Analysis Period",
            "claim": f"Item {top_item['item_name_en']} generated the highest gross profit of {top_item['gross_profit']:.2f} SAR during the analysis period, with {top_item['quantity']:.0f} units sold at {top_item['gross_margin_pct']:.1f}% gross margin.",
            "finding_type": "item_economics",
            "metrics": {
                "item_name": {
                    "value": str(top_item['item_name_en']),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start_iso,
                    "period_end": analysis_end_iso
                },
                "gross_profit_sar": {
                    "value": float(top_item['gross_profit']),
                    "unit": "SAR",
                    "numerator": float(top_item['gross_profit']),
                    "denominator": None,
                    "period_start": analysis_start_iso,
                    "period_end": analysis_end_iso
                },
                "units_sold": {
                    "value": float(top_item['quantity']),
                    "unit": "units",
                    "numerator": float(top_item['quantity']),
                    "denominator": None,
                    "period_start": analysis_start_iso,
                    "period_end": analysis_end_iso
                },
                "gross_margin_pct": {
                    "value": float(top_item['gross_margin_pct']),
                    "unit": "%",
                    "numerator": float(top_item['gross_margin_pct']),
                    "denominator": 100,
                    "period_start": analysis_start_iso,
                    "period_end": analysis_end_iso
                },
                "revenue_sar": {
                    "value": float(top_item['line_total_sar']),
                    "unit": "SAR",
                    "numerator": float(top_item['line_total_sar']),
                    "denominator": None,
                    "period_start": analysis_start_iso,
                    "period_end": analysis_end_iso
                },
                "cogs_sar": {
                    "value": float(top_item['cogs']),
                    "unit": "SAR",
                    "numerator": float(top_item['cogs']),
                    "denominator": None,
                    "period_start": analysis_start_iso,
                    "period_end": analysis_end_iso
                }
            },
            "source_names": ["pos", "menu"],
            "sample_size": len(pos_analysis),
            "coverage_notes": [
                f"Analysis period: {analysis_start_iso} to {analysis_end_iso}",
                f"Refunds excluded from analysis",
                f"Total POS transactions in period: {len(pos_analysis)}",
                f"Items with menu cost data: {len(item_metrics)}"
            ],
            "assumptions": [
                "Menu unit_cost_sar represents actual COGS per unit",
                "Line totals are net of discounts",
                "No recipe/BOM adjustments applied"
            ],
            "confidence": 0.95
        }
        findings.append(finding1)
    
    # FINDING 2: Waste Cost Analysis
    # Filter inventory for analysis week
    inv_analysis = inventory_df[
        (inventory_df['week_starting'] >= analysis_start) & 
        (inventory_df['week_starting'] < analysis_end)
    ].copy()
    
    # Calculate total waste cost (only non-null values)
    waste_data = inv_analysis[inv_analysis['known_waste_cost_sar'].notna()].copy()
    
    if len(waste_data) > 0:
        total_waste_cost = waste_data['known_waste_cost_sar'].sum()
        total_units_wasted = waste_data['units_wasted'].sum()
        
        # Get items with highest waste cost
        waste_by_item = waste_data.groupby('sku').agg({
            'known_waste_cost_sar': 'sum',
            'units_wasted': 'sum',
            'item': 'first'
        }).sort_values('known_waste_cost_sar', ascending=False)
        
        if len(waste_by_item) > 0:
            top_waste_item = waste_by_item.iloc[0]
            finding2 = {
                "title": "Highest Waste Cost Item - Analysis Period",
                "claim": f"Item {top_waste_item['item']} (SKU: {waste_by_item.index[0]}) incurred the highest waste cost of {top_waste_item['known_waste_cost_sar']:.2f} SAR with {top_waste_item['units_wasted']:.0f} units wasted during the analysis period.",
                "finding_type": "waste_economics",
                "metrics": {
                    "item_name": {
                        "value": str(top_waste_item['item']),
                        "unit": None,
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start_iso,
                        "period_end": analysis_end_iso
                    },
                    "waste_cost_sar": {
                        "value": float(top_waste_item['known_waste_cost_sar']),
                        "unit": "SAR",
                        "numerator": float(top_waste_item['known_waste_cost_sar']),
                        "denominator": None,
                        "period_start": analysis_start_iso,
                        "period_end": analysis_end_iso
                    },
                    "units_wasted": {
                        "value": float(top_waste_item['units_wasted']),
                        "unit": "units",
                        "numerator": float(top_waste_item['units_wasted']),
                        "denominator": None,
                        "period_start": analysis_start_iso,
                        "period_end": analysis_end_iso
                    },
                    "total_waste_cost_sar": {
                        "value": float(total_waste_cost),
                        "unit": "SAR",
                        "numerator": float(total_waste_cost),
                        "denominator": None,
                        "period_start": analysis_start_iso,
                        "period_end": analysis_end_iso
                    }
                },
                "source_names": ["inventory"],
                "sample_size": len(waste_data),
                "coverage_notes": [
                    f"Analysis period: {analysis_start_iso} to {analysis_end_iso}",
                    f"Only non-null waste cost observations included",
                    f"Items with waste data: {len(waste_by_item)}"
                ],
                "assumptions": [
                    "known_waste_cost_sar represents actual waste cost",
                    "Blank waste values are treated as missing, not zero"
                ],
                "confidence": 0.90
            }
            findings.append(finding2)
    
    # FINDING 3: Supplier Price Changes and Impact
    # Filter emails for price changes with effective dates
    price_changes = emails_df[
        (emails_df['old_price'].notna()) & 
        (emails_df['new_price'].notna()) &
        (emails_df['effective_date'].notna())
    ].copy()
    
    if len(price_changes) > 0:
        # Calculate price change percentage
        price_changes['price_change_pct'] = (
            (price_changes['new_price'] - price_changes['old_price']) / 
            price_changes['old_price'] * 100
        )
        
        # Get most recent price change
        price_changes = price_changes.sort_values('effective_date', ascending=False)
        latest_change = price_changes.iloc[0]
        
        # Convert effective_date to ISO string
        effective_date_iso = latest_change['effective_date'].isoformat()
        
        finding3 = {
            "title": "Supplier Price Change - Latest",
            "claim": f"Supplier price change for {latest_change['entity_or_ingredient']} effective {latest_change['effective_date'].strftime('%Y-%m-%d')}: {latest_change['old_price']:.2f} {latest_change['currency']}/{latest_change['unit']} → {latest_change['new_price']:.2f} {latest_change['currency']}/{latest_change['unit']} ({latest_change['price_change_pct']:+.1f}%).",
            "finding_type": "supplier_pricing",
            "metrics": {
                "ingredient": {
                    "value": str(latest_change['entity_or_ingredient']),
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": effective_date_iso,
                    "period_end": effective_date_iso
                },
                "old_price": {
                    "value": float(latest_change['old_price']),
                    "unit": f"{latest_change['currency']}/{latest_change['unit']}",
                    "numerator": float(latest_change['old_price']),
                    "denominator": None,
                    "period_start": effective_date_iso,
                    "period_end": effective_date_iso
                },
                "new_price": {
                    "value": float(latest_change['new_price']),
                    "unit": f"{latest_change['currency']}/{latest_change['unit']}",
                    "numerator": float(latest_change['new_price']),
                    "denominator": None,
                    "period_start": effective_date_iso,
                    "period_end": effective_date_iso
                },
                "price_change_pct": {
                    "value": float(latest_change['price_change_pct']),
                    "unit": "%",
                    "numerator": float(latest_change['price_change_pct']),
                    "denominator": 100,
                    "period_start": effective_date_iso,
                    "period_end": effective_date_iso
                },
                "effective_date": {
                    "value": effective_date_iso,
                    "unit": None,
                    "numerator": None,
                    "denominator": None,
                    "period_start": effective_date_iso,
                    "period_end": effective_date_iso
                }
            },
            "source_names": ["emails"],
            "sample_size": len(price_changes),
            "coverage_notes": [
                f"Total price changes with effective dates: {len(price_changes)}",
                f"Latest change effective: {effective_date_iso}",
                f"Sender: {latest_change['sender']}"
            ],
            "assumptions": [
                "Email extraction confidence: {:.0f}%".format(latest_change['confidence'] * 100) if pd.notna(latest_change['confidence']) else "Unknown",
                "No recipe/BOM exists to calculate per-drink impact",
                "Price change applies to specified ingredient only"
            ],
            "confidence": float(latest_change['confidence']) if pd.notna(latest_change['confidence']) else 0.75
        }
        findings.append(finding3)
    
    # Prepare output
    result = {
        "status": "success" if len(findings) > 0 else "insufficient_data",
        "findings": findings
    }
    
    # Write to output
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    return result

def main():
    inputs, output_path = load_inputs()
    result = calculate_findings(inputs, output_path)
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
