import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter
import re

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
reviews_df = pd.read_parquet(inputs['reviews'])
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define analysis period
analysis_start = "2026-01-12T00:00:00+03:00"
analysis_end = "2026-01-19T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Convert review dates to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Filter reviews for analysis period
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

# Get all reviews for baseline comparison
reviews_all = reviews_df.copy()

findings = []

# Finding 1: Rating Distribution and Average
if len(reviews_analysis) > 0:
    rating_dist = reviews_analysis['rating'].value_counts().sort_index().to_dict()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Get source names from analysis period
    source_names_analysis = reviews_analysis['source'].unique().tolist()
    
    finding1 = {
        "title": "Rating Distribution and Average (Analysis Period)",
        "claim": f"During the analysis period (2026-01-12 to 2026-01-19), the average rating across {len(reviews_analysis)} reviews is {avg_rating:.2f} out of 5, with distribution: {rating_dist}",
        "finding_type": "rating_analysis",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": len(reviews_analysis),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names_analysis,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: 2026-01-12 to 2026-01-19",
            f"Total reviews in analysis period: {len(reviews_analysis)}",
            f"Sources represented: {', '.join(source_names_analysis)}"
        ],
        "assumptions": [
            "Rating values are numeric and valid",
            "Review dates are accurate and in UTC+3 timezone",
            "All reviews in the period are included"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Language Distribution
if len(reviews_analysis) > 0:
    lang_dist = reviews_analysis['language'].value_counts().to_dict()
    
    finding2 = {
        "title": "Language Distribution in Reviews",
        "claim": f"During the analysis period, reviews are distributed across languages: {lang_dist}. English reviews comprise {lang_dist.get('en', 0)} ({100*lang_dist.get('en', 0)/len(reviews_analysis):.1f}%), Arabic reviews comprise {lang_dist.get('ar', 0)} ({100*lang_dist.get('ar', 0)/len(reviews_analysis):.1f}%)",
        "finding_type": "language_coverage",
        "metrics": {
            "english_reviews": {
                "value": lang_dist.get('en', 0),
                "unit": "count",
                "numerator": lang_dist.get('en', 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": lang_dist.get('ar', 0),
                "unit": "count",
                "numerator": lang_dist.get('ar', 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_percentage": {
                "value": round(100*lang_dist.get('en', 0)/len(reviews_analysis), 1),
                "unit": "percent",
                "numerator": lang_dist.get('en', 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names_analysis,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Language distribution based on {len(reviews_analysis)} reviews",
            f"Languages detected: {', '.join(lang_dist.keys())}"
        ],
        "assumptions": [
            "Language field accurately reflects review language",
            "All reviews have a language classification"
        ],
        "confidence": 0.95
    }
    findings.append(finding2)

# Finding 3: Sentiment Analysis (based on rating thresholds)
if len(reviews_analysis) > 0:
    # Classify sentiment based on rating
    positive_reviews = len(reviews_analysis[reviews_analysis['rating'] >= 4])
    neutral_reviews = len(reviews_analysis[(reviews_analysis['rating'] >= 2) & (reviews_analysis['rating'] < 4)])
    negative_reviews = len(reviews_analysis[reviews_analysis['rating'] < 2])
    
    finding3 = {
        "title": "Sentiment Distribution by Rating",
        "claim": f"During the analysis period, sentiment distribution shows {positive_reviews} positive reviews (rating ≥4), {neutral_reviews} neutral reviews (rating 2-3), and {negative_reviews} negative reviews (rating <2) out of {len(reviews_analysis)} total reviews",
        "finding_type": "sentiment_analysis",
        "metrics": {
            "positive_reviews": {
                "value": positive_reviews,
                "unit": "count",
                "numerator": positive_reviews,
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "neutral_reviews": {
                "value": neutral_reviews,
                "unit": "count",
                "numerator": neutral_reviews,
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_reviews": {
                "value": negative_reviews,
                "unit": "count",
                "numerator": negative_reviews,
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "positive_percentage": {
                "value": round(100*positive_reviews/len(reviews_analysis), 1),
                "unit": "percent",
                "numerator": positive_reviews,
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names_analysis,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Sentiment classification based on rating thresholds",
            f"Positive: rating ≥4, Neutral: 2-3, Negative: <2",
            f"Total reviews analyzed: {len(reviews_analysis)}"
        ],
        "assumptions": [
            "Rating is a reliable proxy for sentiment",
            "Rating scale is 1-5",
            "Threshold boundaries are appropriate for this business context"
        ],
        "confidence": 0.85
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

print(f"Analysis complete. Output written to {output_path}")
print(f"Total findings: {len(findings)}")
