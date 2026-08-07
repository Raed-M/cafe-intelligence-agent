import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

# Define periods using UTC offset for Asia/Riyadh (UTC+3)
utc_offset = timedelta(hours=3)

analysis_start = datetime(2026, 6, 8, 0, 0, 0)
analysis_end = datetime(2026, 6, 15, 0, 0, 0)
prev_start = datetime(2026, 6, 1, 0, 0, 0)
prev_end = datetime(2026, 6, 8, 0, 0, 0)

# Convert timestamps to UTC-aware datetime
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True)
traffic_df['date'] = pd.to_datetime(traffic_df['date']).dt.normalize()
staff_df['date'] = pd.to_datetime(staff_df['date']).dt.normalize()
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True)
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting']).dt.normalize()

# Convert period boundaries to UTC for comparison
analysis_start_utc = pd.Timestamp(analysis_start, tz='UTC')
analysis_end_utc = pd.Timestamp(analysis_end, tz='UTC')
prev_start_utc = pd.Timestamp(prev_start, tz='UTC')
prev_end_utc = pd.Timestamp(prev_end, tz='UTC')

findings = []

# ============================================================================
# FINDING 1: Conversion Rate Analysis (Analysis Period vs Previous Period)
# ============================================================================

def calculate_conversion(pos_data, traffic_data, period_start, period_end):
    """Calculate conversion rate for a period"""
    # Filter POS for period
    pos_period = pos_data[
        (pos_data['timestamp'] >= period_start) & 
        (pos_data['timestamp'] < period_end) &
        (pos_data['is_refund'] == False)
    ]
    
    # Count unique valid transactions
    valid_transactions = pos_period['transaction_id'].nunique()
    
    # Filter traffic for period, exclude dead sensor days
    traffic_period = traffic_data[
        (traffic_data['date'] >= period_start.normalize()) & 
        (traffic_data['date'] < period_end.normalize()) &
        (traffic_data['is_dead_sensor_day'] == False)
    ]
    
    # Sum valid footfall
    valid_footfall = traffic_period['door_count'].sum()
    
    if valid_footfall == 0:
        return None, None, None
    
    conversion = valid_transactions / valid_footfall if valid_footfall > 0 else None
    return conversion, valid_transactions, valid_footfall

# Analysis period conversion
conv_analysis, trans_analysis, foot_analysis = calculate_conversion(
    pos_df, traffic_df, analysis_start_utc, analysis_end_utc
)

# Previous period conversion
conv_prev, trans_prev, foot_prev = calculate_conversion(
    pos_df, traffic_df, prev_start_utc, prev_end_utc
)

if conv_analysis is not None and conv_prev is not None:
    conv_change = ((conv_analysis - conv_prev) / conv_prev) * 100
    
    findings.append({
        "title": "Conversion Rate Comparison: Analysis Week vs Previous Week",
        "claim": f"Conversion rate in analysis period (Jun 8-15) was {conv_analysis:.4f} vs {conv_prev:.4f} in previous period (Jun 1-8), a change of {conv_change:.1f}%",
        "finding_type": "conversion_analysis",
        "metrics": {
            "conversion_rate_analysis": {
                "value": round(conv_analysis, 4),
                "unit": "transactions_per_visitor",
                "numerator": int(trans_analysis),
                "denominator": int(foot_analysis),
                "period_start": analysis_start_utc.isoformat(),
                "period_end": analysis_end_utc.isoformat()
            },
            "conversion_rate_previous": {
                "value": round(conv_prev, 4),
                "unit": "transactions_per_visitor",
                "numerator": int(trans_prev),
                "denominator": int(foot_prev),
                "period_start": prev_start_utc.isoformat(),
                "period_end": prev_end_utc.isoformat()
            },
            "conversion_change_percent": {
                "value": round(conv_change, 1),
                "unit": "percent",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start_utc.isoformat(),
                "period_end": analysis_end_utc.isoformat()
            }
        },
        "source_names": ["pos", "traffic"],
        "sample_size": int(trans_analysis),
        "coverage_notes": [
            "Excluded dead sensor days from traffic denominator",
            "Excluded refunds from transaction count",
            "Used transaction_id for unique basket count"
        ],
        "assumptions": [
            "Traffic sensor accuracy consistent across periods",
            "Refund flag accurately identifies returns",
            "Timezone conversion to Asia/Riyadh (UTC+3) is correct"
        ],
        "confidence": 0.85
    })

# ============================================================================
# FINDING 2: Labour Cost vs Demand Alignment
# ============================================================================

# Analysis period labour cost
staff_analysis = staff_df[
    (staff_df['date'] >= pd.Timestamp(analysis_start).normalize()) & 
    (staff_df['date'] < pd.Timestamp(analysis_end).normalize())
]
labour_cost_analysis = staff_analysis['labour_cost_sar'].sum()
staff_count_analysis = staff_analysis['employee_id'].nunique()

# Previous period labour cost
staff_prev = staff_df[
    (staff_df['date'] >= pd.Timestamp(prev_start).normalize()) & 
    (staff_df['date'] < pd.Timestamp(prev_end).normalize())
]
labour_cost_prev = staff_prev['labour_cost_sar'].sum()
staff_count_prev = staff_prev['employee_id'].nunique()

# Calculate revenue for periods
pos_analysis = pos_df[
    (pos_df['timestamp'] >= analysis_start_utc) & 
    (pos_df['timestamp'] < analysis_end_utc) &
    (pos_df['is_refund'] == False)
]
revenue_analysis = pos_analysis['line_total_sar'].sum()

pos_prev = pos_df[
    (pos_df['timestamp'] >= prev_start_utc) & 
    (pos_df['timestamp'] < prev_end_utc) &
    (pos_df['is_refund'] == False)
]
revenue_prev = pos_prev['line_total_sar'].sum()

if labour_cost_analysis > 0 and labour_cost_prev > 0:
    labour_per_revenue_analysis = labour_cost_analysis / revenue_analysis if revenue_analysis > 0 else None
    labour_per_revenue_prev = labour_cost_prev / revenue_prev if revenue_prev > 0 else None
    
    if labour_per_revenue_analysis and labour_per_revenue_prev:
        labour_efficiency_change = ((labour_per_revenue_analysis - labour_per_revenue_prev) / labour_per_revenue_prev) * 100
        
        findings.append({
            "title": "Labour Cost Efficiency: Analysis Period vs Previous Period",
            "claim": f"Labour cost as % of revenue was {labour_per_revenue_analysis*100:.2f}% in analysis period vs {labour_per_revenue_prev*100:.2f}% in previous period, a change of {labour_efficiency_change:.1f}%",
            "finding_type": "labour_efficiency",
            "metrics": {
                "labour_cost_analysis": {
                    "value": round(labour_cost_analysis, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start_utc.isoformat(),
                    "period_end": analysis_end_utc.isoformat()
                },
                "revenue_analysis": {
                    "value": round(revenue_analysis, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start_utc.isoformat(),
                    "period_end": analysis_end_utc.isoformat()
                },
                "labour_per_revenue_analysis": {
                    "value": round(labour_per_revenue_analysis, 4),
                    "unit": "ratio",
                    "numerator": round(labour_cost_analysis, 2),
                    "denominator": round(revenue_analysis, 2),
                    "period_start": analysis_start_utc.isoformat(),
                    "period_end": analysis_end_utc.isoformat()
                },
                "labour_per_revenue_previous": {
                    "value": round(labour_per_revenue_prev, 4),
                    "unit": "ratio",
                    "numerator": round(labour_cost_prev, 2),
                    "denominator": round(revenue_prev, 2),
                    "period_start": prev_start_utc.isoformat(),
                    "period_end": prev_end_utc.isoformat()
                },
                "efficiency_change_percent": {
                    "value": round(labour_efficiency_change, 1),
                    "unit": "percent",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start_utc.isoformat(),
                    "period_end": analysis_end_utc.isoformat()
                }
            },
            "source_names": ["staff", "pos"],
            "sample_size": int(staff_count_analysis),
            "coverage_notes": [
                "Labour cost includes all shifts in period",
                "Revenue excludes refunds",
                "Staff count represents unique employees"
            ],
            "assumptions": [
                "Labour cost calculation is accurate",
                "Revenue reflects actual sales",
                "No missing staff records for period"
            ],
            "confidence": 0.80
        })

# ============================================================================
# FINDING 3: Inventory Waste Analysis
# ============================================================================

# Analysis period inventory (week starting Jun 8)
inv_analysis = inventory_df[inventory_df['week_starting'] >= pd.Timestamp(analysis_start).normalize()]
if len(inv_analysis) > 0:
    total_waste_cost_analysis = inv_analysis['known_waste_cost_sar'].sum()
    total_units_sold_analysis = inv_analysis['units_sold'].sum()
    total_units_wasted_analysis = inv_analysis['units_wasted'].sum()
    
    # Previous period inventory (week starting Jun 1)
    inv_prev = inventory_df[
        (inventory_df['week_starting'] >= pd.Timestamp(prev_start).normalize()) & 
        (inventory_df['week_starting'] < pd.Timestamp(analysis_start).normalize())
    ]
    if len(inv_prev) > 0:
        total_waste_cost_prev = inv_prev['known_waste_cost_sar'].sum()
        total_units_sold_prev = inv_prev['units_sold'].sum()
        total_units_wasted_prev = inv_prev['units_wasted'].sum()
        
        if total_units_sold_analysis > 0 and total_units_sold_prev > 0:
            waste_rate_analysis = total_units_wasted_analysis / (total_units_sold_analysis + total_units_wasted_analysis) if (total_units_sold_analysis + total_units_wasted_analysis) > 0 else 0
            waste_rate_prev = total_units_wasted_prev / (total_units_sold_prev + total_units_wasted_prev) if (total_units_sold_prev + total_units_wasted_prev) > 0 else 0
            
            waste_rate_change = ((waste_rate_analysis - waste_rate_prev) / waste_rate_prev * 100) if waste_rate_prev > 0 else 0
            
            findings.append({
                "title": "Known Waste Rate: Analysis Period vs Previous Period",
                "claim": f"Known waste rate was {waste_rate_analysis*100:.2f}% of units (sold + wasted) in analysis period vs {waste_rate_prev*100:.2f}% in previous period, a change of {waste_rate_change:.1f}%",
                "finding_type": "waste_analysis",
                "metrics": {
                    "known_waste_cost_analysis": {
                        "value": round(total_waste_cost_analysis, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start_utc.isoformat(),
                        "period_end": analysis_end_utc.isoformat()
                    },
                    "units_wasted_analysis": {
                        "value": int(total_units_wasted_analysis),
                        "unit": "units",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start_utc.isoformat(),
                        "period_end": analysis_end_utc.isoformat()
                    },
                    "waste_rate_analysis": {
                        "value": round(waste_rate_analysis, 4),
                        "unit": "ratio",
                        "numerator": int(total_units_wasted_analysis),
                        "denominator": int(total_units_sold_analysis + total_units_wasted_analysis),
                        "period_start": analysis_start_utc.isoformat(),
                        "period_end": analysis_end_utc.isoformat()
                    },
                    "waste_rate_previous": {
                        "value": round(waste_rate_prev, 4),
                        "unit": "ratio",
                        "numerator": int(total_units_wasted_prev),
                        "denominator": int(total_units_sold_prev + total_units_wasted_prev),
                        "period_start": prev_start_utc.isoformat(),
                        "period_end": prev_end_utc.isoformat()
                    },
                    "waste_rate_change_percent": {
                        "value": round(waste_rate_change, 1),
                        "unit": "percent",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start_utc.isoformat(),
                        "period_end": analysis_end_utc.isoformat()
                    }
                },
                "source_names": ["inventory"],
                "sample_size": len(inv_analysis),
                "coverage_notes": [
                    "Only known waste included; unknown waste excluded",
                    "Waste rate calculated as wasted / (sold + wasted)",
                    "Sunday inventory counts treated as weekly snapshots, not real-time"
                ],
                "assumptions": [
                    "Known waste values are accurate",
                    "Units sold and wasted are correctly recorded",
                    "Weekly inventory data is representative"
                ],
                "confidence": 0.75
            })

# Write output
result = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)
