import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
reviews_df = pd.read_parquet(inputs['reviews'])
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Analysis period
analysis_start = "2026-01-19T00:00:00+03:00"
analysis_end = "2026-01-26T00:00:00+03:00"

# Convert to datetime for filtering - handle timezone awareness
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Convert reviews date to datetime and remove timezone info for comparison
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
# Remove timezone info from both dataframe and comparison datetimes
reviews_df['date'] = reviews_df['date'].dt.tz_localize(None)
analysis_start_dt = analysis_start_dt.tz_localize(None)
analysis_end_dt = analysis_end_dt.tz_localize(None)

# Filter reviews for analysis period
analysis_reviews = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

# Get all unique sources in the data
all_sources = sorted(reviews_df['source'].unique().tolist())

findings = []

# Finding 1: Rating Distribution and Average
if len(analysis_reviews) > 0:
    rating_counts = analysis_reviews['rating'].value_counts().sort_index()
    avg_rating = analysis_reviews['rating'].mean()
    
    finding1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During the analysis period ({analysis_start} to {analysis_end}), the average rating across all reviews is {avg_rating:.2f} out of 5, with {len(analysis_reviews)} reviews collected.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(avg_rating * len(analysis_reviews), 2),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_reviews": {
                "value": len(analysis_reviews),
                "unit": "count",
                "numerator": len(analysis_reviews),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_1_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_2_count": {
                "value": int(rating_counts.get(2, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(2, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_3_count": {
                "value": int(rating_counts.get(3, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(3, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_4_count": {
                "value": int(rating_counts.get(4, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(4, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_5_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": all_sources,
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Total reviews in dataset: {len(reviews_df)}",
            f"Reviews in analysis period: {len(analysis_reviews)}",
            f"Sources included: {', '.join(all_sources)}"
        ],
        "assumptions": [
            "Rating values are numeric and valid (1-5 scale)",
            "Date field is properly parsed",
            "All reviews in the dataset are included regardless of language"
        ],
        "confidence": 1.0 if len(analysis_reviews) > 0 else 0.0
    }
    findings.append(finding1)

# Finding 2: Language Distribution
if len(analysis_reviews) > 0:
    language_counts = analysis_reviews['language'].value_counts()
    
    finding2 = {
        "title": "Review Language Distribution",
        "claim": f"During the analysis period, {language_counts.get('en', 0)} reviews were in English and {language_counts.get('ar', 0)} reviews were in Arabic out of {len(analysis_reviews)} total reviews.",
        "finding_type": "language_distribution",
        "metrics": {
            "english_reviews": {
                "value": int(language_counts.get('en', 0)),
                "unit": "count",
                "numerator": int(language_counts.get('en', 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": int(language_counts.get('ar', 0)),
                "unit": "count",
                "numerator": int(language_counts.get('ar', 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_percentage": {
                "value": round(100 * language_counts.get('en', 0) / len(analysis_reviews), 1) if len(analysis_reviews) > 0 else 0,
                "unit": "percent",
                "numerator": int(language_counts.get('en', 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_percentage": {
                "value": round(100 * language_counts.get('ar', 0) / len(analysis_reviews), 1) if len(analysis_reviews) > 0 else 0,
                "unit": "percent",
                "numerator": int(language_counts.get('ar', 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": all_sources,
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Language values found: {list(language_counts.index)}"
        ],
        "assumptions": [
            "Language field contains valid ISO 639-1 codes (en/ar)",
            "Language classification is accurate as provided in the dataset"
        ],
        "confidence": 1.0 if len(analysis_reviews) > 0 else 0.0
    }
    findings.append(finding2)

# Finding 3: Source Distribution
if len(analysis_reviews) > 0:
    source_counts = analysis_reviews['source'].value_counts()
    
    finding3 = {
        "title": "Review Source Distribution",
        "claim": f"During the analysis period, reviews were collected from {len(source_counts)} sources: {', '.join([f'{source} ({count})' for source, count in source_counts.items()])}.",
        "finding_type": "source_distribution",
        "metrics": {}
    }
    
    # Add metrics for each source
    for source, count in source_counts.items():
        safe_source_name = source.replace(' ', '_').replace('-', '_').lower()
        finding3["metrics"][f"{safe_source_name}_count"] = {
            "value": int(count),
            "unit": "count",
            "numerator": int(count),
            "denominator": len(analysis_reviews),
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    
    finding3["source_names"] = all_sources
    finding3["sample_size"] = len(analysis_reviews)
    finding3["coverage_notes"] = [
        f"Analysis period: {analysis_start} to {analysis_end}",
        f"Sources in analysis period: {list(source_counts.index)}"
    ]
    finding3["assumptions"] = [
        "Source field accurately identifies the platform/channel from which reviews were collected",
        "All sources in the dataset are represented"
    ]
    finding3["confidence"] = 1.0 if len(analysis_reviews) > 0 else 0.0
    
    findings.append(finding3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Analysis complete. Output written to {output_path}")
print(f"Findings generated: {len(findings)}")
