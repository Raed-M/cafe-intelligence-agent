import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter, defaultdict

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Define analysis period
analysis_start = "2026-06-15T00:00:00+03:00"
analysis_end = "2026-06-22T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Filter reviews for analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Handle timezone-aware vs timezone-naive comparison
# If the dataframe dates are timezone-naive, make the comparison datetimes timezone-naive
if reviews_df['date'].dt.tz is None:
    analysis_start_dt = analysis_start_dt.tz_localize(None)
    analysis_end_dt = analysis_end_dt.tz_localize(None)
else:
    # If dataframe dates are timezone-aware, ensure comparison datetimes are too
    if analysis_start_dt.tz is None:
        analysis_start_dt = analysis_start_dt.tz_localize('UTC').tz_convert(reviews_df['date'].dt.tz.iloc[0])
        analysis_end_dt = analysis_end_dt.tz_localize('UTC').tz_convert(reviews_df['date'].dt.tz.iloc[0])

analysis_reviews = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

findings = []

# Finding 1: Rating Distribution and Average
if len(analysis_reviews) > 0:
    rating_counts = analysis_reviews['rating'].value_counts().sort_index()
    avg_rating = analysis_reviews['rating'].mean()
    
    finding1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During the analysis period (2026-06-15 to 2026-06-22), the average rating across {len(analysis_reviews)} reviews was {avg_rating:.2f} out of 5.0. Rating distribution: 1-star: {rating_counts.get(1, 0)} reviews, 2-star: {rating_counts.get(2, 0)} reviews, 3-star: {rating_counts.get(3, 0)} reviews, 4-star: {rating_counts.get(4, 0)} reviews, 5-star: {rating_counts.get(5, 0)} reviews.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": len(analysis_reviews),
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
            "rating_1_star_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_2_star_count": {
                "value": int(rating_counts.get(2, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(2, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_3_star_count": {
                "value": int(rating_counts.get(3, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(3, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_4_star_count": {
                "value": int(rating_counts.get(4, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(4, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_5_star_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": sorted(list(analysis_reviews['source'].unique())),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period: 2026-06-15 to 2026-06-22",
            f"Total reviews in analysis period: {len(analysis_reviews)}",
            f"Review sources: {', '.join(sorted(analysis_reviews['source'].unique()))}",
            f"Language distribution: {dict(analysis_reviews['language'].value_counts())}"
        ],
        "assumptions": [
            "Rating values are numeric and valid (1-5 scale)",
            "All reviews in the analysis period are included",
            "Missing or null ratings are excluded from average calculation"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Language Distribution
if len(analysis_reviews) > 0:
    language_counts = analysis_reviews['language'].value_counts()
    
    finding2 = {
        "title": "Review Language Distribution",
        "claim": f"During the analysis period, reviews were submitted in {len(language_counts)} languages. Language breakdown: {', '.join([f'{lang}: {count} reviews' for lang, count in language_counts.items()])}.",
        "finding_type": "language_distribution",
        "metrics": {
            "total_reviews_by_language": {
                "value": dict(language_counts),
                "unit": "reviews",
                "numerator": len(analysis_reviews),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": sorted(list(analysis_reviews['source'].unique())),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period: 2026-06-15 to 2026-06-22",
            f"Total reviews analyzed: {len(analysis_reviews)}",
            f"Languages detected: {sorted(list(language_counts.index))}"
        ],
        "assumptions": [
            "Language field is accurately populated in source data",
            "All reviews have a language designation"
        ],
        "confidence": 0.95
    }
    findings.append(finding2)

# Finding 3: Review Source Distribution
if len(analysis_reviews) > 0:
    source_counts = analysis_reviews['source'].value_counts()
    
    finding3 = {
        "title": "Review Source Distribution",
        "claim": f"During the analysis period, reviews came from {len(source_counts)} sources. Source breakdown: {', '.join([f'{source}: {count} reviews' for source, count in source_counts.items()])}.",
        "finding_type": "source_distribution",
        "metrics": {
            "reviews_by_source": {
                "value": dict(source_counts),
                "unit": "reviews",
                "numerator": len(analysis_reviews),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": sorted(list(source_counts.index)),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period: 2026-06-15 to 2026-06-22",
            f"Total reviews analyzed: {len(analysis_reviews)}",
            f"Sources identified: {sorted(list(source_counts.index))}"
        ],
        "assumptions": [
            "Source field is accurately populated in review data",
            "All reviews have a source designation"
        ],
        "confidence": 0.95
    }
    findings.append(finding3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
