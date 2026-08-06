import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

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

# Define periods
tz = pytz.timezone('Asia/Riyadh')
analysis_start = tz.localize(datetime(2026, 2, 9, 0, 0, 0))
analysis_end = tz.localize(datetime(2026, 2, 16, 0, 0, 0))
prev_start = tz.localize(datetime(2026, 2, 2, 0, 0, 0))
prev_end = tz.localize(datetime(2026, 2, 9, 0, 0, 0))

findings = []
execution_notes = []

try:
    # ===== FINDING 1: Conversion Rate Analysis =====
    # Convert timestamp to datetime
    pos_df['timestamp_dt'] = pd.to_datetime(pos_df['timestamp'])
    traffic_df['date_dt'] = pd.to_datetime(traffic_df['date'])
    
    # Filter analysis period
    pos_analysis = pos_df[
        (pos_df['timestamp_dt'] >= analysis_start) & 
        (pos_df['timestamp_dt'] < analysis_end)
    ].copy()
    
    traffic_analysis = traffic_df[
        (traffic_df['date_dt'] >= analysis_start.date()) & 
        (traffic_df['date_dt'] < analysis_end.date())
    ].copy()
    
    # Count valid transactions (exclude refunds, use transaction_id)
    valid_transactions = pos_analysis[pos_analysis['is_refund'] == False]['transaction_id'].nunique()
    
    # Count valid footfall (exclude dead sensor days)
    valid_traffic = traffic_analysis[traffic_analysis['is_dead_sensor_day'] == False]['door_count'].sum()
    
    if valid_traffic > 0:
        conversion_rate = valid_transactions / valid_traffic
        
        findings.append({
            "title": "Conversion Rate - Analysis Period",
            "claim": f"During the analysis period (Feb 9-16, 2026), the cafe achieved a conversion rate of {conversion_rate:.4f}, with {valid_transactions} valid transactions from {valid_traffic} valid footfall events.",
            "finding_type": "operational_metric",
            "metrics": {
                "conversion_rate": {
                    "value": round(conversion_rate, 4),
                    "unit": "ratio",
                    "numerator": valid_transactions,
                    "denominator": valid_traffic,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "valid_transactions": {
                    "value": valid_transactions,
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "valid_footfall": {
                    "value": valid_traffic,
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["pos", "traffic"],
            "sample_size": valid_transactions,
            "coverage_notes": [
                "Excluded refund transactions from numerator",
                "Excluded dead sensor days from denominator",
                "Analysis period: 2026-02-09 to 2026-02-16"
            ],
            "assumptions": [
                "transaction_id uniqueness identifies distinct sales baskets",
                "is_dead_sensor_day flag accurately marks non-operational traffic sensors",
                "is_refund flag correctly identifies refund transactions"
            ],
            "confidence": 0.85
        })
    else:
        execution_notes.append("Conversion rate calculation skipped: no valid footfall in analysis period")
    
    # ===== FINDING 2: Labour Cost vs Demand Alignment =====
    # Filter staff data for analysis period
    staff_df['date_dt'] = pd.to_datetime(staff_df['date'])
    staff_analysis = staff_df[
        (staff_df['date_dt'] >= analysis_start.date()) & 
        (staff_df['date_dt'] < analysis_end.date())
    ].copy()
    
    # Calculate total labour cost and hours
    total_labour_cost = staff_analysis['labour_cost_sar'].sum()
    total_hours = staff_analysis['computed_duration_hours'].sum()
    
    # Calculate daily metrics
    daily_labour = staff_analysis.groupby('date_dt').agg({
        'labour_cost_sar': 'sum',
        'computed_duration_hours': 'sum'
    }).reset_index()
    
    # Get daily transaction counts
    pos_analysis['date_dt_only'] = pos_analysis['timestamp_dt'].dt.date
    daily_transactions = pos_analysis[pos_analysis['is_refund'] == False].groupby('date_dt_only')['transaction_id'].nunique().reset_index()
    daily_transactions.columns = ['date_dt', 'transactions']
    daily_transactions['date_dt'] = pd.to_datetime(daily_transactions['date_dt'])
    
    # Merge
    daily_merged = daily_labour.merge(daily_transactions, on='date_dt', how='inner')
    
    if len(daily_merged) > 0 and total_hours > 0:
        avg_labour_per_transaction = total_labour_cost / valid_transactions if valid_transactions > 0 else 0
        avg_labour_per_hour = total_labour_cost / total_hours
        
        findings.append({
            "title": "Labour Cost Efficiency",
            "claim": f"During the analysis period, total labour cost was {total_labour_cost:.2f} SAR across {total_hours:.1f} hours, yielding {avg_labour_per_transaction:.2f} SAR labour cost per transaction.",
            "finding_type": "cost_metric",
            "metrics": {
                "total_labour_cost_sar": {
                    "value": round(total_labour_cost, 2),
                    "unit": "SAR",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "total_labour_hours": {
                    "value": round(total_hours, 1),
                    "unit": "hours",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "labour_cost_per_transaction": {
                    "value": round(avg_labour_per_transaction, 2),
                    "unit": "SAR/transaction",
                    "numerator": total_labour_cost,
                    "denominator": valid_transactions,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "labour_cost_per_hour": {
                    "value": round(avg_labour_per_hour, 2),
                    "unit": "SAR/hour",
                    "numerator": total_labour_cost,
                    "denominator": total_hours,
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": ["staff", "pos"],
            "sample_size": len(daily_merged),
            "coverage_notes": [
                f"Staff data covers {len(staff_analysis)} shifts across {len(daily_merged)} days",
                "Labour cost includes computed_duration_hours-based calculations",
                "Transactions exclude refunds"
            ],
            "assumptions": [
                "computed_duration_hours accurately reflects actual staff time on floor",
                "labour_cost_sar is correctly calculated from hourly_rate_sar and duration",
                "All staff shifts in period are captured in staff artifact"
            ],
            "confidence": 0.80
        })
    else:
        execution_notes.append("Labour cost analysis skipped: insufficient daily overlap data")
    
    # ===== FINDING 3: Waste and Inventory Management =====
    # Filter inventory for analysis period
    inventory_df['week_starting_dt'] = pd.to_datetime(inventory_df['week_starting'])
    inventory_analysis = inventory_df[
        (inventory_df['week_starting_dt'] >= analysis_start.date()) & 
        (inventory_df['week_starting_dt'] < analysis_end.date())
    ].copy()
    
    if len(inventory_analysis) > 0:
        total_units_sold = inventory_analysis['units_sold'].sum()
        total_units_wasted = inventory_analysis['units_wasted'].sum()
        total_known_waste_cost = inventory_analysis['known_waste_cost_sar'].sum()
        total_units_ordered = inventory_analysis['units_ordered'].sum()
        
        if total_units_sold > 0:
            waste_to_sales_ratio = total_units_wasted / total_units_sold
            
            findings.append({
                "title": "Waste Management - Known Waste Analysis",
                "claim": f"During the analysis period, {total_units_wasted} units were wasted (known waste cost: {total_known_waste_cost:.2f} SAR) against {total_units_sold} units sold, yielding a waste-to-sales ratio of {waste_to_sales_ratio:.4f}.",
                "finding_type": "inventory_metric",
                "metrics": {
                    "total_units_wasted": {
                        "value": total_units_wasted,
                        "unit": "units",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    },
                    "total_units_sold": {
                        "value": total_units_sold,
                        "unit": "units",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    },
                    "known_waste_cost_sar": {
                        "value": round(total_known_waste_cost, 2),
                        "unit": "SAR",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    },
                    "waste_to_sales_ratio": {
                        "value": round(waste_to_sales_ratio, 4),
                        "unit": "ratio",
                        "numerator": total_units_wasted,
                        "denominator": total_units_sold,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    },
                    "total_units_ordered": {
                        "value": total_units_ordered,
                        "unit": "units",
                        "numerator": None,
                        "denominator": None,
                        "period_start": analysis_start.isoformat(),
                        "period_end": analysis_end.isoformat()
                    }
                },
                "source_names": ["inventory"],
                "sample_size": len(inventory_analysis),
                "coverage_notes": [
                    f"Inventory data covers {len(inventory_analysis)} SKU-week records",
                    "Known waste cost is explicitly tracked; unknown waste is excluded from cost calculations",
                    "units_wasted includes only known waste values"
                ],
                "assumptions": [
                    "units_wasted represents only known/documented waste",
                    "known_waste_cost_sar is accurately calculated from unit_cost_sar",
                    "units_sold and units_ordered are accurately recorded"
                ],
                "confidence": 0.75
            })
    else:
        execution_notes.append("Waste analysis skipped: no inventory data in analysis period")

except Exception as e:
    execution_notes.append(f"Error during analysis: {str(e)}")

# Construct output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings[:3],  # Max 3 findings
    "execution_notes": execution_notes
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings generated.")
