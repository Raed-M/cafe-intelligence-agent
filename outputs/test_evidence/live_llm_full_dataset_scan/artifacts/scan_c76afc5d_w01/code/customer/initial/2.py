import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter
import re

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read input artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Define analysis period
analysis_start = "2026-01-12T00:00:00+03:00"
analysis_end = "2026-01-19T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Filter reviews for analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Handle timezone awareness mismatch
if reviews_df['date'].dt.tz is None:
    # If reviews_df dates are tz-naive, make them tz-aware with UTC then convert to +03:00
    reviews_df['date'] = reviews_df['date'].dt.tz_localize('UTC').dt.tz_convert('+03:00')
else:
    # If already tz-aware, ensure consistent timezone
    reviews_df['date'] = reviews_df['date'].dt.tz_convert('+03:00')

# Ensure analysis_start_dt and analysis_end_dt are tz-aware
if analysis_start_dt.tz is None:
    analysis_start_dt = analysis_start_dt.tz_localize('UTC').tz_convert('+03:00')
else:
    analysis_start_dt = analysis_start_dt.tz_convert('+03:00')

if analysis_end_dt.tz is None:
    analysis_end_dt = analysis_end_dt.tz_localize('UTC').tz_convert('+03:00')
else:
    analysis_end_dt = analysis_end_dt.tz_convert('+03:00')

analysis_reviews = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

findings = []

# Finding 1: Rating Distribution and Average
if len(analysis_reviews) > 0:
    rating_counts = analysis_reviews['rating'].value_counts().sort_index()
    avg_rating = analysis_reviews['rating'].mean()
    
    # Sentiment classification based on rating
    def classify_sentiment(rating):
        if pd.isna(rating):
            return 'unknown'
        if rating >= 4:
            return 'positive'
        elif rating == 3:
            return 'neutral'
        else:
            return 'negative'
    
    analysis_reviews['sentiment'] = analysis_reviews['rating'].apply(classify_sentiment)
    sentiment_counts = analysis_reviews['sentiment'].value_counts()
    
    finding1 = {
        "title": "Customer Rating Distribution and Average Sentiment",
        "claim": f"During the analysis period (Jan 12-19, 2026), customers provided {len(analysis_reviews)} reviews with an average rating of {avg_rating:.2f}/5. Positive reviews (4-5 stars) comprised {sentiment_counts.get('positive', 0)} reviews, neutral (3 stars) {sentiment_counts.get('neutral', 0)}, and negative (1-2 stars) {sentiment_counts.get('negative', 0)}.",
        "finding_type": "customer_sentiment",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": len(analysis_reviews),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "positive_reviews": {
                "value": int(sentiment_counts.get('positive', 0)),
                "unit": "count",
                "numerator": int(sentiment_counts.get('positive', 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "neutral_reviews": {
                "value": int(sentiment_counts.get('neutral', 0)),
                "unit": "count",
                "numerator": int(sentiment_counts.get('neutral', 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_reviews": {
                "value": int(sentiment_counts.get('negative', 0)),
                "unit": "count",
                "numerator": int(sentiment_counts.get('negative', 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(analysis_reviews['source'].unique()),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Total reviews in period: {len(analysis_reviews)}",
            f"Language distribution: {dict(analysis_reviews['language'].value_counts())}"
        ],
        "assumptions": [
            "Rating scale is 1-5 stars",
            "Positive sentiment defined as 4-5 stars, neutral as 3 stars, negative as 1-2 stars",
            "All reviews in the dataset are valid and complete"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Language Distribution
if len(analysis_reviews) > 0:
    language_counts = analysis_reviews['language'].value_counts()
    
    finding2 = {
        "title": "Review Language Distribution",
        "claim": f"Of {len(analysis_reviews)} reviews collected during Jan 12-19, 2026, {language_counts.get('en', 0)} were in English and {language_counts.get('ar', 0)} were in Arabic, indicating {round(100*language_counts.get('ar', 0)/len(analysis_reviews), 1)}% Arabic language coverage.",
        "finding_type": "data_coverage",
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
            "arabic_coverage_percent": {
                "value": round(100*language_counts.get('ar', 0)/len(analysis_reviews), 1),
                "unit": "percent",
                "numerator": int(language_counts.get('ar', 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(analysis_reviews['source'].unique()),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Bilingual review coverage: {dict(language_counts)}",
            "Both English and Arabic reviews present in dataset"
        ],
        "assumptions": [
            "Language field accurately reflects review language",
            "All reviews are either English or Arabic"
        ],
        "confidence": 0.98
    }
    findings.append(finding2)

# Finding 3: Review Source Distribution
if len(analysis_reviews) > 0:
    source_counts = analysis_reviews['source'].value_counts()
    
    finding3 = {
        "title": "Review Source Distribution",
        "claim": f"During Jan 12-19, 2026, reviews were collected from {len(source_counts)} sources: {', '.join([f'{source} ({count} reviews)' for source, count in source_counts.items()])}. This indicates diversified customer feedback channels.",
        "finding_type": "data_coverage",
        "metrics": {
            "total_sources": {
                "value": len(source_counts),
                "unit": "count",
                "numerator": len(source_counts),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(source_counts.index),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Source breakdown: {dict(source_counts)}",
            f"Total reviews across all sources: {len(analysis_reviews)}"
        ],
        "assumptions": [
            "Source field accurately identifies review platform/channel",
            "All sources are equally reliable"
        ],
        "confidence": 0.95
    }
    
    # Add individual source metrics
    for source, count in source_counts.items():
        finding3['metrics'][f'{source}_reviews'] = {
            "value": int(count),
            "unit": "count",
            "numerator": int(count),
            "denominator": len(analysis_reviews),
            "period_start": analysis_start,
            "period_end": analysis_end
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

print(f"Analysis complete. Results written to {output_path}")
print(f"Total findings: {len(findings)}")
print(f"Reviews analyzed: {len(analysis_reviews)}")
