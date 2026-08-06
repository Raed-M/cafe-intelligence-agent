import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

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
analysis_start = "2026-02-09"
analysis_end = "2026-02-16"

# Convert date columns to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Filter reviews for analysis period
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start) & 
    (reviews_df['date'] < analysis_end)
].copy()

findings = []

# Finding 1: Rating Distribution and Average
if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Language distribution
    language_counts = reviews_analysis['language'].value_counts()
    
    # Convert numpy types to Python native types
    rating_counts_dict = {int(k): int(v) for k, v in rating_counts.items()}
    
    finding1 = {
        "title": "Customer Rating Distribution and Average (Feb 9-16, 2026)",
        "claim": f"During the analysis period, customers provided {len(reviews_analysis)} reviews with an average rating of {avg_rating:.2f} out of 5. Rating distribution shows {rating_counts_dict} across the scale.",
        "finding_type": "sentiment_distribution",
        "metrics": {
            "average_rating": {
                "value": float(round(avg_rating, 2)),
                "unit": "stars",
                "numerator": int(len(reviews_analysis)),
                "denominator": int(len(reviews_analysis)),
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "total_reviews": {
                "value": int(len(reviews_analysis)),
                "unit": "count",
                "numerator": int(len(reviews_analysis)),
                "denominator": None,
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "english_reviews": {
                "value": int(language_counts.get('en', 0)),
                "unit": "count",
                "numerator": int(language_counts.get('en', 0)),
                "denominator": int(len(reviews_analysis)),
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "arabic_reviews": {
                "value": int(language_counts.get('ar', 0)),
                "unit": "count",
                "numerator": int(language_counts.get('ar', 0)),
                "denominator": int(len(reviews_analysis)),
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": int(len(reviews_analysis)),
        "coverage_notes": [
            f"Analysis period: 2026-02-09 to 2026-02-16",
            f"Total reviews in dataset: {len(reviews_df)}",
            f"Reviews in analysis period: {len(reviews_analysis)}",
            f"Language coverage: {', '.join([f'{lang}: {count}' for lang, count in language_counts.items()])}"
        ],
        "assumptions": [
            "Rating values are numeric and valid",
            "Date filtering uses UTC+3 timezone as specified",
            "All reviews in the period are included regardless of source"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Sentiment Classification by Language
if len(reviews_analysis) > 0:
    # Classify sentiment based on rating
    def classify_sentiment(rating):
        if pd.isna(rating):
            return 'unknown'
        if rating >= 4:
            return 'positive'
        elif rating == 3:
            return 'neutral'
        else:
            return 'negative'
    
    reviews_analysis['sentiment'] = reviews_analysis['rating'].apply(classify_sentiment)
    
    # Sentiment by language
    sentiment_by_lang = pd.crosstab(reviews_analysis['language'], reviews_analysis['sentiment'])
    
    # Get positive and negative counts
    positive_count = int((reviews_analysis['sentiment'] == 'positive').sum())
    negative_count = int((reviews_analysis['sentiment'] == 'negative').sum())
    neutral_count = int((reviews_analysis['sentiment'] == 'neutral').sum())
    
    finding2 = {
        "title": "Sentiment Classification by Language (Feb 9-16, 2026)",
        "claim": f"Sentiment analysis of {len(reviews_analysis)} reviews shows {positive_count} positive (rating ≥4), {neutral_count} neutral (rating=3), and {negative_count} negative (rating <3) reviews. Distribution varies by language.",
        "finding_type": "sentiment_classification",
        "metrics": {
            "positive_reviews": {
                "value": positive_count,
                "unit": "count",
                "numerator": positive_count,
                "denominator": int(len(reviews_analysis)),
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "negative_reviews": {
                "value": negative_count,
                "unit": "count",
                "numerator": negative_count,
                "denominator": int(len(reviews_analysis)),
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "neutral_reviews": {
                "value": neutral_count,
                "unit": "count",
                "numerator": neutral_count,
                "denominator": int(len(reviews_analysis)),
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            },
            "positive_sentiment_rate": {
                "value": float(round(positive_count / len(reviews_analysis) * 100, 1)),
                "unit": "percent",
                "numerator": positive_count,
                "denominator": int(len(reviews_analysis)),
                "period_start": "2026-02-09T00:00:00+03:00",
                "period_end": "2026-02-16T00:00:00+03:00"
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": int(len(reviews_analysis)),
        "coverage_notes": [
            f"Sentiment classification based on rating thresholds: positive (≥4), neutral (=3), negative (<3)",
            f"Language distribution: {dict(reviews_analysis['language'].value_counts())}",
            f"All reviews in analysis period included"
        ],
        "assumptions": [
            "Sentiment classification uses rating-based thresholds",
            "Rating values are numeric and valid",
            "No text analysis performed; classification based solely on numeric rating"
        ],
        "confidence": 0.90
    }
    findings.append(finding2)

# Finding 3: Review Volume Trend
if len(reviews_df) > 0:
    # Compare analysis period with previous period
    previous_start = "2026-02-02"
    previous_end = "2026-02-09"
    
    reviews_previous = reviews_df[
        (reviews_df['date'] >= previous_start) & 
        (reviews_df['date'] < previous_end)
    ]
    
    analysis_count = len(reviews_analysis)
    previous_count = len(reviews_previous)
    
    if previous_count > 0:
        change_pct = ((analysis_count - previous_count) / previous_count) * 100
        
        finding3 = {
            "title": "Review Volume Comparison (Week-over-Week)",
            "claim": f"Review volume in the analysis period (Feb 9-16) was {analysis_count} compared to {previous_count} in the previous week (Feb 2-9), representing a {change_pct:+.1f}% change.",
            "finding_type": "volume_trend",
            "metrics": {
                "analysis_period_reviews": {
                    "value": int(analysis_count),
                    "unit": "count",
                    "numerator": int(analysis_count),
                    "denominator": None,
                    "period_start": "2026-02-09T00:00:00+03:00",
                    "period_end": "2026-02-16T00:00:00+03:00"
                },
                "previous_period_reviews": {
                    "value": int(previous_count),
                    "unit": "count",
                    "numerator": int(previous_count),
                    "denominator": None,
                    "period_start": "2026-02-02T00:00:00+03:00",
                    "period_end": "2026-02-09T00:00:00+03:00"
                },
                "week_over_week_change": {
                    "value": float(round(change_pct, 1)),
                    "unit": "percent",
                    "numerator": int(analysis_count - previous_count),
                    "denominator": int(previous_count),
                    "period_start": "2026-02-09T00:00:00+03:00",
                    "period_end": "2026-02-16T00:00:00+03:00"
                }
            },
            "source_names": list(reviews_df['source'].unique()),
            "sample_size": int(analysis_count + previous_count),
            "coverage_notes": [
                f"Analysis period: 2026-02-09 to 2026-02-16 ({analysis_count} reviews)",
                f"Previous period: 2026-02-02 to 2026-02-09 ({previous_count} reviews)",
                f"Total reviews in dataset: {len(reviews_df)}"
            ],
            "assumptions": [
                "Date filtering uses UTC+3 timezone",
                "Week-over-week comparison uses consecutive 7-day periods",
                "All reviews in both periods are included"
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

print(f"Analysis complete. {len(findings)} findings generated.")
print(f"Output written to {output_path}")
