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

# Analysis period
analysis_start = "2026-05-04T00:00:00+03:00"
analysis_end = "2026-05-11T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Filter reviews for analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

findings = []

# Finding 1: Rating Distribution and Average
if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Language breakdown
    language_counts = reviews_analysis['language'].value_counts()
    
    finding1 = {
        "title": "Review Rating Distribution and Average (Analysis Period)",
        "claim": f"During the analysis period (May 4-11, 2026), the average rating across {len(reviews_analysis)} reviews was {avg_rating:.2f} out of 5, with {int(rating_counts.get(5, 0))} five-star ratings and {int(rating_counts.get(1, 0))} one-star ratings.",
        "finding_type": "customer_sentiment_distribution",
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
            "one_star_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Language distribution: {dict(language_counts)}",
            f"Rating distribution: {dict(rating_counts)}"
        ],
        "assumptions": [
            "Rating values are as provided in the reviews artifact",
            "Analysis period is May 4-11, 2026 (UTC+3)"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Sentiment Classification by Language
if len(reviews_analysis) > 0:
    # Separate by language
    english_reviews = reviews_analysis[reviews_analysis['language'] == 'en'].copy()
    arabic_reviews = reviews_analysis[reviews_analysis['language'] == 'ar'].copy()
    
    # Simple sentiment classification based on rating
    def classify_sentiment(rating):
        if rating >= 4:
            return 'positive'
        elif rating == 3:
            return 'neutral'
        else:
            return 'negative'
    
    english_reviews['sentiment'] = english_reviews['rating'].apply(classify_sentiment)
    arabic_reviews['sentiment'] = arabic_reviews['rating'].apply(classify_sentiment)
    
    en_sentiment_counts = english_reviews['sentiment'].value_counts()
    ar_sentiment_counts = arabic_reviews['sentiment'].value_counts()
    
    if len(english_reviews) > 0:
        finding2 = {
            "title": "Sentiment Distribution by Language",
            "claim": f"English reviews (n={len(english_reviews)}) showed {int(en_sentiment_counts.get('positive', 0))} positive, {int(en_sentiment_counts.get('neutral', 0))} neutral, and {int(en_sentiment_counts.get('negative', 0))} negative sentiments. Arabic reviews (n={len(arabic_reviews)}) showed {int(ar_sentiment_counts.get('positive', 0))} positive, {int(ar_sentiment_counts.get('neutral', 0))} neutral, and {int(ar_sentiment_counts.get('negative', 0))} negative sentiments.",
            "finding_type": "sentiment_by_language",
            "metrics": {
                "english_positive_count": {
                    "value": int(en_sentiment_counts.get('positive', 0)),
                    "unit": "reviews",
                    "numerator": int(en_sentiment_counts.get('positive', 0)),
                    "denominator": len(english_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "english_negative_count": {
                    "value": int(en_sentiment_counts.get('negative', 0)),
                    "unit": "reviews",
                    "numerator": int(en_sentiment_counts.get('negative', 0)),
                    "denominator": len(english_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "arabic_positive_count": {
                    "value": int(ar_sentiment_counts.get('positive', 0)),
                    "unit": "reviews",
                    "numerator": int(ar_sentiment_counts.get('positive', 0)),
                    "denominator": len(arabic_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "arabic_negative_count": {
                    "value": int(ar_sentiment_counts.get('negative', 0)),
                    "unit": "reviews",
                    "numerator": int(ar_sentiment_counts.get('negative', 0)),
                    "denominator": len(arabic_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": list(reviews_analysis['source'].unique()),
            "sample_size": len(reviews_analysis),
            "coverage_notes": [
                f"English reviews: {len(english_reviews)}",
                f"Arabic reviews: {len(arabic_reviews)}",
                "Sentiment classified as: positive (rating >= 4), neutral (rating = 3), negative (rating < 3)"
            ],
            "assumptions": [
                "Sentiment classification based on rating thresholds",
                "Language field accurately reflects review language"
            ],
            "confidence": 0.90
        }
        findings.append(finding2)

# Finding 3: Review Source Distribution
if len(reviews_analysis) > 0:
    source_counts = reviews_analysis['source'].value_counts()
    
    finding3 = {
        "title": "Review Source Distribution",
        "claim": f"During the analysis period, reviews came from {len(source_counts)} sources: {', '.join([f'{source} ({count})' for source, count in source_counts.items()])}.",
        "finding_type": "source_coverage",
        "metrics": {
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "reviews",
                "numerator": len(reviews_analysis),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(source_counts.index),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Source distribution: {dict(source_counts)}"
        ],
        "assumptions": [
            "Source field accurately identifies review platform"
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

print(f"Analysis complete. {len(findings)} findings generated.")
