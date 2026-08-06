import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter
import numpy as np

# Load input/output paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)
inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Define analysis period
analysis_start = "2026-02-09T00:00:00+03:00"
analysis_end = "2026-02-16T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Filter reviews to analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Handle timezone-aware vs timezone-naive comparison
if reviews_df['date'].dt.tz is None:
    # reviews_df dates are naive, make comparison dates naive
    analysis_start_dt = analysis_start_dt.tz_localize(None)
    analysis_end_dt = analysis_end_dt.tz_localize(None)
else:
    # reviews_df dates are aware, ensure comparison dates are aware
    if analysis_start_dt.tz is None:
        analysis_start_dt = analysis_start_dt.tz_localize('UTC').tz_convert(reviews_df['date'].dt.tz.iloc[0])
        analysis_end_dt = analysis_end_dt.tz_localize('UTC').tz_convert(reviews_df['date'].dt.tz.iloc[0])

reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

# Initialize findings list
findings = []

# ============================================================================
# FINDING 1: Rating Distribution and Average
# ============================================================================
if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Get source names from analysis period reviews
    source_names = sorted(reviews_analysis['source'].unique().tolist())
    
    # Get language distribution
    language_dist = reviews_analysis['language'].value_counts()
    
    finding_1 = {
        "title": "Review Rating Distribution and Average (Analysis Period)",
        "claim": f"During {analysis_start[:10]} to {analysis_end[:10]}, the average rating across {len(reviews_analysis)} reviews was {avg_rating:.2f} out of 5, with {rating_counts.get(5, 0)} five-star, {rating_counts.get(4, 0)} four-star, {rating_counts.get(3, 0)} three-star, {rating_counts.get(2, 0)} two-star, and {rating_counts.get(1, 0)} one-star ratings.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": len(reviews_analysis),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "five_star_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "four_star_count": {
                "value": int(rating_counts.get(4, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(4, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "three_star_count": {
                "value": int(rating_counts.get(3, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(3, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "two_star_count": {
                "value": int(rating_counts.get(2, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(2, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "one_star_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start[:10]} to {analysis_end[:10]}",
            f"Total reviews in analysis period: {len(reviews_analysis)}",
            f"Language distribution: {dict(language_dist)}",
            f"Sources represented: {', '.join(source_names)}"
        ],
        "assumptions": [
            "Rating values are numeric and valid (1-5 scale)",
            "All reviews in the analysis period are included",
            "No filtering applied beyond date range"
        ],
        "confidence": 1.0
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Language Distribution
# ============================================================================
if len(reviews_analysis) > 0:
    language_counts = reviews_analysis['language'].value_counts()
    
    finding_2 = {
        "title": "Review Language Distribution",
        "claim": f"Of {len(reviews_analysis)} reviews in the analysis period, {language_counts.get('en', 0)} were in English and {language_counts.get('ar', 0)} were in Arabic.",
        "finding_type": "language_distribution",
        "metrics": {
            "english_reviews": {
                "value": int(language_counts.get('en', 0)),
                "unit": "reviews",
                "numerator": int(language_counts.get('en', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": int(language_counts.get('ar', 0)),
                "unit": "reviews",
                "numerator": int(language_counts.get('ar', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_percentage": {
                "value": round(100 * language_counts.get('en', 0) / len(reviews_analysis), 1) if len(reviews_analysis) > 0 else 0,
                "unit": "percent",
                "numerator": int(language_counts.get('en', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_percentage": {
                "value": round(100 * language_counts.get('ar', 0) / len(reviews_analysis), 1) if len(reviews_analysis) > 0 else 0,
                "unit": "percent",
                "numerator": int(language_counts.get('ar', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": sorted(reviews_analysis['source'].unique().tolist()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start[:10]} to {analysis_end[:10]}",
            f"Total reviews analyzed: {len(reviews_analysis)}"
        ],
        "assumptions": [
            "Language field is accurately populated",
            "Language values are either 'en' or 'ar'"
        ],
        "confidence": 1.0
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Source Coverage
# ============================================================================
if len(reviews_analysis) > 0:
    source_counts = reviews_analysis['source'].value_counts()
    source_names = sorted(reviews_analysis['source'].unique().tolist())
    
    finding_3 = {
        "title": "Review Source Distribution",
        "claim": f"During the analysis period, reviews came from {len(source_counts)} source(s): {', '.join([f'{src} ({count} reviews)' for src, count in source_counts.items()])}.",
        "finding_type": "source_distribution",
        "metrics": {}
    }
    
    # Add metrics for each source
    for src, count in source_counts.items():
        safe_src_key = src.lower().replace(' ', '_').replace('-', '_')
        finding_3["metrics"][f"{safe_src_key}_count"] = {
            "value": int(count),
            "unit": "reviews",
            "numerator": int(count),
            "denominator": len(reviews_analysis),
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    
    finding_3["source_names"] = source_names
    finding_3["sample_size"] = len(reviews_analysis)
    finding_3["coverage_notes"] = [
        f"Analysis period: {analysis_start[:10]} to {analysis_end[:10]}",
        f"Total reviews: {len(reviews_analysis)}",
        f"Number of sources: {len(source_counts)}",
        f"Sources: {', '.join(source_names)}"
    ]
    finding_3["assumptions"] = [
        "Source field is accurately populated",
        "All reviews in the analysis period are included"
    ]
    finding_3["confidence"] = 1.0
    
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
    json.dump(output, f, indent=2)
