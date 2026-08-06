import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
traffic_df = pd.read_parquet(inputs['traffic'])
staff_df = pd.read_parquet(inputs['staff'])
inventory_df = pd.read_parquet(inputs['inventory'])

# Define periods (UTC+3)
tz = timezone('Asia/Riyadh')
analysis_start = tz.localize(datetime(2026, 4, 6, 0, 0, 0))
analysis_end = tz.localize(datetime(2026, 4, 13, 0, 0, 0))
previous_start = tz.localize(datetime(2026, 3, 30, 0, 0, 0))
previous_end = tz.localize(datetime(2026, 4, 6, 0, 0, 0))

# Convert timestamps to timezone-aware
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True).dt.tz_convert(tz)
traffic_df['date'] = pd.to_datetime(traffic_df['date']).dt.tz_localize(tz)
staff_df['date'] = pd.to_datetime(staff_df['date']).dt.tz_localize(tz)
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True).dt.tz_convert(tz)
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True).dt.tz_convert(tz)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting']).dt.tz_localize(tz)

findings = []

# ============================================================================
# FINDING 1: Conversion Rate Analysis (POS + Traffic)
# ============================================================================

def compute_conversion(pos_data, traffic_data, period_start, period_end, period_name):
    """Compute conversion rate for a period, excluding dead sensor days."""
    
    # Filter POS: valid sales transactions (not refunds)
    pos_period = pos_data[
        (pos_data['timestamp'] >= period_start) &
        (pos_data['timestamp'] < period_end) &
        (pos_data['is_refund'] == False)
    ].copy()
    
    # Count unique transactions
    unique_transactions = pos_period['transaction_id'].nunique()
    
    # Filter traffic: exclude dead sensor days
    traffic_period = traffic_data[
        (traffic_data['date'] >= period_start) &
        (traffic_data['date'] < period_end) &
        (traffic_data['is_dead_sensor_day'] == False)
    ].copy()
    
    # Sum door counts
    total_footfall = traffic_period['door_count'].sum()
    
    # Compute conversion
    if total_footfall > 0:
        conversion = unique_transactions / total_footfall
    else:
        conversion = None
    
    return {
        'unique_transactions': unique_transactions,
        'total_footfall': total_footfall,
        'conversion': conversion,
        'traffic_rows': len(traffic_period),
        'pos_rows': len(pos_period)
    }

analysis_conv = compute_conversion(pos_df, traffic_df, analysis_start, analysis_end, "analysis")
previous_conv = compute_conversion(pos_df, traffic_df, previous_start, previous_end, "previous")

if analysis_conv['conversion'] is not None and previous_conv['conversion'] is not None:
    conv_change = analysis_conv['conversion'] - previous_conv['conversion']
    conv_pct_change = (conv_change / previous_conv['conversion']) * 100 if previous_conv['conversion'] > 0 else None
    
    if conv_pct_change is not None and abs(conv_pct_change) > 5:  # Threshold for significance
        finding_1 = {
            "title": "Conversion Rate Change Week-over-Week",
            "claim": f"Conversion rate changed from {previous_conv['conversion']:.4f} ({previous_conv['unique_transactions']} transactions / {previous_conv['total_footfall']} footfall) in the previous period to {analysis_conv['conversion']:.4f} ({analysis_conv['unique_transactions']} transactions / {analysis_conv['total_footfall']} footfall) in the analysis period, representing a {conv_pct_change:.1f}% change.",
            "finding_type": "operational_metric",
            "metrics": {
                "conversion_rate_analysis": {
                    "value": round(analysis_conv['conversion'], 4),
                    "unit": "transactions_per_visitor",
                    "numerator": analysis_conv['unique_transactions'],
                    "denominator": analysis_conv['total_footfall'],
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "conversion_rate_previous": {
                    "value": round(previous_conv['conversion'], 4),
                    "unit": "transactions_per_visitor",
                    "numerator": previous_conv['unique_transactions'],
                    "denominator": previous_conv['total_footfall'],
                    "period_start": previous_start.isoformat(),
                    "period_end": previous_end.isoformat()
                },
                "conversion_change_pct": {
                    "value": round(conv_pct_change, 1),
                    "unit": "percent",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["pos", "traffic"],
            "sample_size": analysis_conv['pos_rows'],
            "coverage_notes": [
                f"Analysis period: {analysis_conv['traffic_rows']} traffic records (dead sensor days excluded)",
                f"Previous period: {previous_conv['traffic_rows']} traffic records (dead sensor days excluded)",
                f"POS transactions counted by unique transaction_id, refunds excluded"
            ],
            "assumptions": [
                "Traffic data is timezone-aligned to UTC+3",
                "Dead sensor days are correctly flagged in traffic data",
                "Refunds are correctly marked in POS data",
                "Transaction_id uniquely identifies a basket"
            ],
            "confidence": 0.75
        }
        findings.append(finding_1)

# ============================================================================
# FINDING 2: Labour Cost as % of Revenue
# ============================================================================

def compute_labour_revenue_ratio(pos_data, staff_data, period_start, period_end):
    """Compute labour cost as % of revenue."""
    
    # Filter POS: all transactions (including refunds, which reduce revenue)
    pos_period = pos_data[
        (pos_data['timestamp'] >= period_start) &
        (pos_data['timestamp'] < period_end)
    ].copy()
    
    # Compute net revenue (line_total_sar includes refunds as negative)
    net_revenue = pos_period['line_total_sar'].sum()
    
    # Filter staff: shifts that overlap with period
    staff_period = staff_data[
        (staff_data['shift_start'] < period_end) &
        (staff_data['shift_end'] > period_start)
    ].copy()
    
    # Sum labour cost
    total_labour_cost = staff_period['labour_cost_sar'].sum()
    
    # Compute ratio
    if net_revenue > 0:
        ratio = total_labour_cost / net_revenue
    else:
        ratio = None
    
    return {
        'net_revenue': net_revenue,
        'total_labour_cost': total_labour_cost,
        'ratio': ratio,
        'staff_rows': len(staff_period),
        'pos_rows': len(pos_period)
    }

analysis_labour = compute_labour_revenue_ratio(pos_df, staff_df, analysis_start, analysis_end)
previous_labour = compute_labour_revenue_ratio(pos_df, staff_df, previous_start, previous_end)

if analysis_labour['ratio'] is not None and previous_labour['ratio'] is not None:
    labour_ratio_change = analysis_labour['ratio'] - previous_labour['ratio']
    labour_ratio_pct_change = (labour_ratio_change / previous_labour['ratio']) * 100 if previous_labour['ratio'] > 0 else None
    
    if labour_ratio_pct_change is not None:
        # Determine direction: if ratio decreased, efficiency improved (lower cost per revenue)
        direction = "decreased" if labour_ratio_change < 0 else "increased"
        efficiency_implication = "improved" if labour_ratio_change < 0 else "deteriorated"
        
        finding_2 = {
            "title": f"Labour Cost Ratio {direction.capitalize()} Week-over-Week",
            "claim": f"Labour cost as % of revenue {direction} from {previous_labour['ratio']:.4f} ({previous_labour['total_labour_cost']:.2f} SAR / {previous_labour['net_revenue']:.2f} SAR revenue) in the previous period to {analysis_labour['ratio']:.4f} ({analysis_labour['total_labour_cost']:.2f} SAR / {analysis_labour['net_revenue']:.2f} SAR revenue) in the analysis period, a {abs(labour_ratio_pct_change):.1f}% {direction} change. This indicates {efficiency_implication} labour cost efficiency relative to revenue generation.",
            "finding_type": "operational_metric",
            "metrics": {
                "labour_cost_ratio_analysis": {
                    "value": round(analysis_labour['ratio'], 4),
                    "unit": "ratio",
                    "numerator": round(analysis_labour['total_labour_cost'], 2),
                    "denominator": round(analysis_labour['net_revenue'], 2),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "labour_cost_ratio_previous": {
                    "value": round(previous_labour['ratio'], 4),
                    "unit": "ratio",
                    "numerator": round(previous_labour['total_labour_cost'], 2),
                    "denominator": round(previous_labour['net_revenue'], 2),
                    "period_start": previous_start.isoformat(),
                    "period_end": previous_end.isoformat()
                },
                "labour_cost_ratio_change": {
                    "value": round(labour_ratio_change, 4),
                    "unit": "ratio_points",
                    "numerator": None,
                    "denominator": None,
                    "period_start": previous_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["pos", "staff"],
            "sample_size": analysis_labour['staff_rows'],
            "coverage_notes": [
                f"Analysis period: {analysis_labour['staff_rows']} staff records with overlapping shifts",
                f"Previous period: {previous_labour['staff_rows']} staff records with overlapping shifts",
                f"Revenue includes all POS line items (refunds reduce net revenue)",
                f"Labour cost from staff.labour_cost_sar field"
            ],
            "assumptions": [
                "Staff shift times are timezone-aligned to UTC+3",
                "Labour cost includes all shifts with any overlap to period",
                "No imputation for missing staff records",
                "Revenue and labour cost are in same currency (SAR)"
            ],
            "confidence": 0.70
        }
        findings.append(finding_2)

# ============================================================================
# FINDING 3: Known Waste Rate Analysis
# ============================================================================

def compute_waste_rate(inventory_data, period_start, period_end):
    """Compute known waste rate for a period."""
    
    # Filter inventory by week_starting within period
    inv_period = inventory_data[
        (inventory_data['week_starting'] >= period_start) &
        (inventory_data['week_starting'] < period_end)
    ].copy()
    
    # Sum units sold and units wasted
    total_sold = inv_period['units_sold'].sum()
    total_wasted = inv_period['units_wasted'].sum()
    total_waste_cost = inv_period['known_waste_cost_sar'].sum()
    
    # Compute waste rate
    if (total_sold + total_wasted) > 0:
        waste_rate = total_wasted / (total_sold + total_wasted)
    else:
        waste_rate = None
    
    return {
        'total_sold': total_sold,
        'total_wasted': total_wasted,
        'waste_rate': waste_rate,
        'waste_cost': total_waste_cost,
        'inv_rows': len(inv_period)
    }

analysis_waste = compute_waste_rate(inventory_df, analysis_start, analysis_end)
previous_waste = compute_waste_rate(inventory_df, previous_start, previous_end)

if analysis_waste['waste_rate'] is not None and previous_waste['waste_rate'] is not None:
    waste_rate_change = analysis_waste['waste_rate'] - previous_waste['waste_rate']
    waste_rate_pct_change = (waste_rate_change / previous_waste['waste_rate']) * 100 if previous_waste['waste_rate'] > 0 else None
    
    if waste_rate_pct_change is not None:
        direction = "decreased" if waste_rate_change < 0 else "increased"
        
        finding_3 = {
            "title": f"Known Waste Rate {direction.capitalize()} Week-over-Week",
            "claim": f"Known waste rate {direction} from {previous_waste['waste_rate']:.4f} ({previous_waste['total_wasted']} units wasted / {previous_waste['total_sold'] + previous_waste['total_wasted']} total units) in the previous period to {analysis_waste['waste_rate']:.4f} ({analysis_waste['total_wasted']} units wasted / {analysis_waste['total_sold'] + analysis_waste['total_wasted']} total units) in the analysis period, a {abs(waste_rate_pct_change):.1f}% {direction} change. Known waste cost in analysis period: {analysis_waste['waste_cost']:.2f} SAR.",
            "finding_type": "operational_metric",
            "metrics": {
                "waste_rate_analysis": {
                    "value": round(analysis_waste['waste_rate'], 4),
                    "unit": "ratio",
                    "numerator": analysis_waste['total_wasted'],
                    "denominator": analysis_waste['total_sold'] + analysis_waste['total_wasted'],
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "waste_rate_previous": {
                    "value": round(previous_waste['waste_rate'], 4),
                    "unit": "ratio",
                    "numerator": previous_waste['total_wasted'],
                    "denominator": previous_waste['total_sold'] + previous_waste['total_wasted'],
                    "period_start": previous_start.isoformat(),
                    "period_end": previous_end.isoformat()
                },
                "waste_cost_analysis": {
                    "value": round(analysis_waste['waste_cost'], 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["inventory"],
            "sample_size": analysis_waste['inv_rows'],
            "coverage_notes": [
                f"Analysis period: {analysis_waste['inv_rows']} inventory records",
                f"Previous period: {previous_waste['inv_rows']} inventory records",
                f"Waste rate calculated from units_wasted / (units_sold + units_wasted)",
                f"Waste cost from known_waste_cost_sar field"
            ],
            "assumptions": [
                "Inventory week_starting dates are timezone-aligned to UTC+3",
                "Known waste values are accurate; unknown waste is excluded",
                "Units sold and wasted are mutually exclusive categories"
            ],
            "confidence": 0.75
        }
        findings.append(finding_3)

# ============================================================================
# Output
# ============================================================================

result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)
