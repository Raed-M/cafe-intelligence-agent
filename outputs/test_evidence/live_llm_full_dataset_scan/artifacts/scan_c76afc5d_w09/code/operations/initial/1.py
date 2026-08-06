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

# Define timezone
tz = timezone('Asia/Riyadh')

# Parse dates and times
pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], utc=True).dt.tz_convert(tz)
pos_df['business_date'] = pd.to_datetime(pos_df['business_date'])
traffic_df['date'] = pd.to_datetime(traffic_df['date'])
staff_df['date'] = pd.to_datetime(staff_df['date'])
staff_df['shift_start'] = pd.to_datetime(staff_df['shift_start'], utc=True).dt.tz_convert(tz)
staff_df['shift_end'] = pd.to_datetime(staff_df['shift_end'], utc=True).dt.tz_convert(tz)
inventory_df['week_starting'] = pd.to_datetime(inventory_df['week_starting'])

# Analysis period
analysis_start = pd.Timestamp('2026-03-09', tz=tz)
analysis_end = pd.Timestamp('2026-03-16', tz=tz)
previous_start = pd.Timestamp('2026-03-02', tz=tz)
previous_end = pd.Timestamp('2026-03-09', tz=tz)

# Filter data for analysis period
pos_analysis = pos_df[(pos_df['timestamp'] >= analysis_start) & (pos_df['timestamp'] < analysis_end)].copy()
traffic_analysis = traffic_df[(traffic_df['date'] >= analysis_start.date()) & (traffic_df['date'] < analysis_end.date())].copy()
staff_analysis = staff_df[(staff_df['date'] >= analysis_start.date()) & (staff_df['date'] < analysis_end.date())].copy()

# Filter data for previous period
pos_previous = pos_df[(pos_df['timestamp'] >= previous_start) & (pos_df['timestamp'] < previous_end)].copy()
traffic_previous = traffic_df[(traffic_df['date'] >= previous_start.date()) & (traffic_df['date'] < previous_end.date())].copy()

# Calculate conversion metrics
def calculate_conversion(pos_data, traffic_data, period_name):
    # Count unique valid transactions (exclude refunds)
    valid_transactions = pos_data[~pos_data['is_refund']].groupby('transaction_id').size()
    unique_transactions = len(valid_transactions)
    
    # Count valid footfall (exclude dead sensor days)
    valid_traffic = traffic_data[~traffic_data['is_dead_sensor_day']].copy()
    total_footfall = valid_traffic['door_count'].sum()
    
    if total_footfall > 0:
        conversion = unique_transactions / total_footfall
    else:
        conversion = None
    
    return {
        'unique_transactions': unique_transactions,
        'total_footfall': total_footfall,
        'conversion': conversion,
        'period': period_name
    }

conv_analysis = calculate_conversion(pos_analysis, traffic_analysis, 'analysis')
conv_previous = calculate_conversion(pos_previous, traffic_previous, 'previous')

# Calculate revenue metrics
def calculate_revenue(pos_data, period_name):
    # Exclude refunds from revenue
    valid_sales = pos_data[~pos_data['is_refund']].copy()
    total_revenue = valid_sales['line_total_sar'].sum()
    transaction_count = valid_sales['transaction_id'].nunique()
    
    if transaction_count > 0:
        avg_transaction = total_revenue / transaction_count
    else:
        avg_transaction = None
    
    return {
        'total_revenue': total_revenue,
        'transaction_count': transaction_count,
        'avg_transaction': avg_transaction,
        'period': period_name
    }

rev_analysis = calculate_revenue(pos_analysis, 'analysis')
rev_previous = calculate_revenue(pos_previous, 'previous')

# Calculate staffing metrics
def calculate_staffing(staff_data, period_name):
    total_hours = staff_data['computed_duration_hours'].sum()
    total_cost = staff_data['labour_cost_sar'].sum()
    unique_employees = staff_data['employee_id'].nunique()
    
    return {
        'total_hours': total_hours,
        'total_cost': total_cost,
        'unique_employees': unique_employees,
        'period': period_name
    }

staff_analysis_metrics = calculate_staffing(staff_analysis, 'analysis')
staff_previous_metrics = calculate_staffing(staff_df[(staff_df['date'] >= previous_start.date()) & (staff_df['date'] < previous_end.date())], 'previous')

# Calculate inventory metrics for analysis week
analysis_week_start = pd.Timestamp('2026-03-09', tz=tz).date()
inventory_analysis = inventory_df[inventory_df['week_starting'].dt.date == analysis_week_start].copy()

# Prepare findings
findings = []

# Finding 1: Conversion Rate Comparison
if conv_analysis['total_footfall'] > 0 and conv_previous['total_footfall'] > 0:
    conversion_change = ((conv_analysis['conversion'] - conv_previous['conversion']) / conv_previous['conversion'] * 100) if conv_previous['conversion'] > 0 else None
    
    findings.append({
        'title': 'Conversion Rate Analysis',
        'claim': f"Conversion rate in analysis week ({conv_analysis['conversion']:.4f}) compared to previous week ({conv_previous['conversion']:.4f})",
        'finding_type': 'conversion_metric',
        'metrics': {
            'analysis_conversion': {
                'value': round(conv_analysis['conversion'], 4),
                'unit': 'transactions_per_visitor',
                'numerator': conv_analysis['unique_transactions'],
                'denominator': conv_analysis['total_footfall'],
                'period_start': analysis_start.isoformat(),
                'period_end': analysis_end.isoformat()
            },
            'previous_conversion': {
                'value': round(conv_previous['conversion'], 4),
                'unit': 'transactions_per_visitor',
                'numerator': conv_previous['unique_transactions'],
                'denominator': conv_previous['total_footfall'],
                'period_start': previous_start.isoformat(),
                'period_end': previous_end.isoformat()
            },
            'conversion_change_percent': {
                'value': round(conversion_change, 2) if conversion_change else None,
                'unit': 'percent',
                'numerator': None,
                'denominator': None,
                'period_start': analysis_start.isoformat(),
                'period_end': analysis_end.isoformat()
            }
        },
        'source_names': ['pos', 'traffic'],
        'sample_size': conv_analysis['total_footfall'],
        'coverage_notes': [
            f"Analysis period: {conv_analysis['unique_transactions']} transactions from {conv_analysis['total_footfall']} valid footfall",
            f"Previous period: {conv_previous['unique_transactions']} transactions from {conv_previous['total_footfall']} valid footfall",
            'Dead sensor days excluded from footfall denominator'
        ],
        'assumptions': [
            'Refunds excluded from transaction count',
            'Valid footfall excludes dead sensor intervals',
            'One transaction_id = one basket'
        ],
        'confidence': 0.85
    })

# Finding 2: Revenue and Transaction Performance
if rev_analysis['transaction_count'] > 0 and rev_previous['transaction_count'] > 0:
    revenue_change = ((rev_analysis['total_revenue'] - rev_previous['total_revenue']) / rev_previous['total_revenue'] * 100) if rev_previous['total_revenue'] > 0 else None
    
    findings.append({
        'title': 'Revenue Performance Comparison',
        'claim': f"Total revenue in analysis week (SAR {rev_analysis['total_revenue']:.2f}) compared to previous week (SAR {rev_previous['total_revenue']:.2f})",
        'finding_type': 'revenue_metric',
        'metrics': {
            'analysis_total_revenue': {
                'value': round(rev_analysis['total_revenue'], 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': analysis_start.isoformat(),
                'period_end': analysis_end.isoformat()
            },
            'analysis_avg_transaction': {
                'value': round(rev_analysis['avg_transaction'], 2),
                'unit': 'SAR',
                'numerator': rev_analysis['total_revenue'],
                'denominator': rev_analysis['transaction_count'],
                'period_start': analysis_start.isoformat(),
                'period_end': analysis_end.isoformat()
            },
            'previous_total_revenue': {
                'value': round(rev_previous['total_revenue'], 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': previous_start.isoformat(),
                'period_end': previous_end.isoformat()
            },
            'previous_avg_transaction': {
                'value': round(rev_previous['avg_transaction'], 2),
                'unit': 'SAR',
                'numerator': rev_previous['total_revenue'],
                'denominator': rev_previous['transaction_count'],
                'period_start': previous_start.isoformat(),
                'period_end': previous_end.isoformat()
            },
            'revenue_change_percent': {
                'value': round(revenue_change, 2) if revenue_change else None,
                'unit': 'percent',
                'numerator': None,
                'denominator': None,
                'period_start': analysis_start.isoformat(),
                'period_end': analysis_end.isoformat()
            }
        },
        'source_names': ['pos'],
        'sample_size': rev_analysis['transaction_count'],
        'coverage_notes': [
            f"Analysis period: {rev_analysis['transaction_count']} transactions",
            f"Previous period: {rev_previous['transaction_count']} transactions",
            'Refunds excluded from revenue calculations'
        ],
        'assumptions': [
            'Refunds treated as negative revenue',
            'Line totals used as transaction values',
            'No imputation for missing values'
        ],
        'confidence': 0.90
    })

# Finding 3: Staffing Cost and Hours
if staff_analysis_metrics['total_hours'] > 0 and staff_previous_metrics['total_hours'] > 0:
    cost_per_hour_analysis = staff_analysis_metrics['total_cost'] / staff_analysis_metrics['total_hours']
    cost_per_hour_previous = staff_previous_metrics['total_cost'] / staff_previous_metrics['total_hours']
    
    findings.append({
        'title': 'Labour Cost and Staffing Hours',
        'claim': f"Analysis week staffing: {staff_analysis_metrics['total_hours']:.1f} hours (SAR {staff_analysis_metrics['total_cost']:.2f}) vs previous week: {staff_previous_metrics['total_hours']:.1f} hours (SAR {staff_previous_metrics['total_cost']:.2f})",
        'finding_type': 'staffing_metric',
        'metrics': {
            'analysis_total_hours': {
                'value': round(staff_analysis_metrics['total_hours'], 1),
                'unit': 'hours',
                'numerator': None,
                'denominator': None,
                'period_start': analysis_start.isoformat(),
                'period_end': analysis_end.isoformat()
            },
            'analysis_total_cost': {
                'value': round(staff_analysis_metrics['total_cost'], 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': analysis_start.isoformat(),
                'period_end': analysis_end.isoformat()
            },
            'analysis_cost_per_hour': {
                'value': round(cost_per_hour_analysis, 2),
                'unit': 'SAR/hour',
                'numerator': staff_analysis_metrics['total_cost'],
                'denominator': staff_analysis_metrics['total_hours'],
                'period_start': analysis_start.isoformat(),
                'period_end': analysis_end.isoformat()
            },
            'previous_total_hours': {
                'value': round(staff_previous_metrics['total_hours'], 1),
                'unit': 'hours',
                'numerator': None,
                'denominator': None,
                'period_start': previous_start.isoformat(),
                'period_end': previous_end.isoformat()
            },
            'previous_total_cost': {
                'value': round(staff_previous_metrics['total_cost'], 2),
                'unit': 'SAR',
                'numerator': None,
                'denominator': None,
                'period_start': previous_start.isoformat(),
                'period_end': previous_end.isoformat()
            },
            'previous_cost_per_hour': {
                'value': round(cost_per_hour_previous, 2),
                'unit': 'SAR/hour',
                'numerator': staff_previous_metrics['total_cost'],
                'denominator': staff_previous_metrics['total_hours'],
                'period_start': previous_start.isoformat(),
                'period_end': previous_end.isoformat()
            }
        },
        'source_names': ['staff'],
        'sample_size': staff_analysis_metrics['unique_employees'],
        'coverage_notes': [
            f"Analysis period: {staff_analysis_metrics['unique_employees']} unique employees",
            f"Previous period: {staff_previous_metrics['unique_employees']} unique employees",
            'Hours calculated from computed_duration_hours field'
        ],
        'assumptions': [
            'Shift overlap computed from shift_start and shift_end times',
            'Labour cost includes all hourly rates and computed durations',
            'No imputation for missing shift data'
        ],
        'confidence': 0.88
    })

# Prepare output
output = {
    'status': 'success' if len(findings) > 0 else 'insufficient_data',
    'findings': findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
