import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter
import re

# Load input paths from environment
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Define analysis periods
analysis_period_start = "2026-04-20T00:00:00+03:00"
analysis_period_end = "2026-04-27T00:00:00+03:00"
previous_period_start = "2026-04-13T00:00:00+03:00"
previous_period_end = "2026-04-20T00:00:00+03:00"

# Convert to datetime for filtering - handle timezone awareness
analysis_start = pd.to_datetime(analysis_period_start)
analysis_end = pd.to_datetime(analysis_period_end)
previous_start = pd.to_datetime(previous_period_start)
previous_end = pd.to_datetime(previous_period_end)

# Convert review dates to datetime and ensure timezone-naive for comparison
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
# Remove timezone info to make tz-naive for comparison
if reviews_df['date'].dt.tz is not None:
    reviews_df['date'] = reviews_df['date'].dt.tz_localize(None)

# Also convert comparison dates to tz-naive
analysis_start = analysis_start.tz_localize(None) if analysis_start.tz is not None else analysis_start
analysis_end = analysis_end.tz_localize(None) if analysis_end.tz is not None else analysis_end
previous_start = previous_start.tz_localize(None) if previous_start.tz is not None else previous_start
previous_end = previous_end.tz_localize(None) if previous_end.tz is not None else previous_end

# Filter reviews for analysis period
analysis_reviews = reviews_df[(reviews_df['date'] >= analysis_start) & (reviews_df['date'] < analysis_end)].copy()
previous_reviews = reviews_df[(reviews_df['date'] >= previous_start) & (reviews_df['date'] < previous_end)].copy()

# Get unique sources in the data
unique_sources = sorted(reviews_df['source'].unique().tolist())

findings = []

# Finding 1: Rating distribution and average for analysis period
if len(analysis_reviews) > 0:
    rating_counts = analysis_reviews['rating'].value_counts().sort_index().to_dict()
    avg_rating = analysis_reviews['rating'].mean()
    
    finding1 = {
        "title": "Analysis Period Rating Distribution",
        "claim": f"During the analysis period (2026-04-20 to 2026-04-27), the average rating across all review sources was {avg_rating:.2f} out of 5, based on {len(analysis_reviews)} reviews.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating_analysis_period": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": len(analysis_reviews),
                "denominator": len(analysis_reviews),
                "period_start": analysis_period_start,
                "period_end": analysis_period_end
            },
            "total_reviews_analysis_period": {
                "value": len(analysis_reviews),
                "unit": "count",
                "numerator": len(analysis_reviews),
                "denominator": None,
                "period_start": analysis_period_start,
                "period_end": analysis_period_end
            }
        },
        "source_names": unique_sources,
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Reviews from {len(unique_sources)} source(s): {', '.join(unique_sources)}",
            f"Language distribution: {dict(analysis_reviews['language'].value_counts())}"
        ],
        "assumptions": [
            "All reviews in the artifact are valid and represent genuine customer feedback",
            "Rating scale is 1-5 stars",
            "No filtering applied for review quality or authenticity"
        ],
        "confidence": 0.95 if len(analysis_reviews) >= 10 else 0.70
    }
    findings.append(finding1)

# Finding 2: Language distribution in analysis period
if len(analysis_reviews) > 0:
    language_dist = analysis_reviews['language'].value_counts().to_dict()
    
    finding2 = {
        "title": "Language Distribution in Reviews",
        "claim": f"During the analysis period, {len(analysis_reviews)} reviews were received across {len(language_dist)} language(s). {', '.join([f'{lang}: {count} reviews' for lang, count in language_dist.items()])}.",
        "finding_type": "language_coverage",
        "metrics": {
            "total_reviews_analysis_period": {
                "value": len(analysis_reviews),
                "unit": "count",
                "numerator": len(analysis_reviews),
                "denominator": None,
                "period_start": analysis_period_start,
                "period_end": analysis_period_end
            }
        },
        "source_names": unique_sources,
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Language breakdown: {language_dist}",
            f"All {len(unique_sources)} sources represented: {', '.join(unique_sources)}"
        ],
        "assumptions": [
            "Language field accurately reflects the language of each review",
            "Reviews are not translated; original language is preserved"
        ],
        "confidence": 0.95
    }
    findings.append(finding2)

# Finding 3: Rating comparison between analysis and previous period
if len(analysis_reviews) > 0 and len(previous_reviews) > 0:
    avg_rating_analysis = analysis_reviews['rating'].mean()
    avg_rating_previous = previous_reviews['rating'].mean()
    rating_change = avg_rating_analysis - avg_rating_previous
    
    finding3 = {
        "title": "Rating Trend: Analysis vs Previous Period",
        "claim": f"Average rating in the analysis period (2026-04-20 to 2026-04-27) was {avg_rating_analysis:.2f} stars ({len(analysis_reviews)} reviews), compared to {avg_rating_previous:.2f} stars ({len(previous_reviews)} reviews) in the previous period (2026-04-13 to 2026-04-20), representing a change of {rating_change:+.2f} stars.",
        "finding_type": "rating_trend",
        "metrics": {
            "average_rating_analysis_period": {
                "value": round(avg_rating_analysis, 2),
                "unit": "stars",
                "numerator": len(analysis_reviews),
                "denominator": len(analysis_reviews),
                "period_start": analysis_period_start,
                "period_end": analysis_period_end
            },
            "average_rating_previous_period": {
                "value": round(avg_rating_previous, 2),
                "unit": "stars",
                "numerator": len(previous_reviews),
                "denominator": len(previous_reviews),
                "period_start": previous_period_start,
                "period_end": previous_period_end
            },
            "rating_change": {
                "value": round(rating_change, 2),
                "unit": "stars",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_period_start,
                "period_end": analysis_period_end
            }
        },
        "source_names": unique_sources,
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period: {len(analysis_reviews)} reviews from {len(unique_sources)} source(s)",
            f"Previous period: {len(previous_reviews)} reviews from {len(unique_sources)} source(s)",
            f"Sources: {', '.join(unique_sources)}"
        ],
        "assumptions": [
            "Both periods have comparable review volume and source distribution",
            "Rating scale is consistent across both periods",
            "No systematic bias in review collection between periods"
        ],
        "confidence": 0.85 if len(analysis_reviews) >= 10 and len(previous_reviews) >= 10 else 0.65
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

print(f"Analysis complete. {len(findings)} finding(s) generated.")
print(f"Output written to {output_path}")
