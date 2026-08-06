import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

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
analysis_start = "2026-03-30T00:00:00+03:00"
analysis_end = "2026-04-06T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Filter reviews for analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
analysis_reviews = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

findings = []

# Finding 1: Rating Distribution and Average
if len(analysis_reviews) > 0:
    rating_counts = analysis_reviews['rating'].value_counts().sort_index()
    avg_rating = analysis_reviews['rating'].mean()
    
    # Language distribution
    language_counts = analysis_reviews['language'].value_counts()
    
    finding1 = {
        "title": "Review Rating Distribution and Language Coverage",
        "claim": f"During the analysis period (Mar 30 - Apr 6, 2026), the average review rating was {avg_rating:.2f} out of 5, based on {len(analysis_reviews)} reviews. English reviews comprised {language_counts.get('en', 0)} reviews and Arabic reviews comprised {language_counts.get('ar', 0)} reviews.",
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
            "english_reviews": {
                "value": language_counts.get('en', 0),
                "unit": "count",
                "numerator": language_counts.get('en', 0),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": language_counts.get('ar', 0),
                "unit": "count",
                "numerator": language_counts.get('ar', 0),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(analysis_reviews['source'].unique()),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Total reviews in dataset: {len(reviews_df)}",
            f"Reviews in analysis period: {len(analysis_reviews)}",
            f"Language distribution: {dict(language_counts)}"
        ],
        "assumptions": [
            "Rating values are numeric and valid",
            "Date field is accurate and timezone-aware",
            "Language classification is accurate"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Sentiment Classification by Language
# Simple sentiment classification based on rating thresholds
if len(analysis_reviews) > 0:
    analysis_reviews['sentiment'] = analysis_reviews['rating'].apply(
        lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral')
    )
    
    sentiment_counts = analysis_reviews['sentiment'].value_counts()
    
    # Separate by language
    en_reviews = analysis_reviews[analysis_reviews['language'] == 'en']
    ar_reviews = analysis_reviews[analysis_reviews['language'] == 'ar']
    
    en_sentiment = en_reviews['sentiment'].value_counts() if len(en_reviews) > 0 else {}
    ar_sentiment = ar_reviews['sentiment'].value_counts() if len(ar_reviews) > 0 else {}
    
    finding2 = {
        "title": "Sentiment Distribution by Language",
        "claim": f"Among {len(analysis_reviews)} reviews in the analysis period, {sentiment_counts.get('positive', 0)} were positive (rating ≥4), {sentiment_counts.get('negative', 0)} were negative (rating ≤2), and {sentiment_counts.get('neutral', 0)} were neutral (rating 3). English reviews showed {en_sentiment.get('positive', 0)} positive, {en_sentiment.get('negative', 0)} negative, and {en_sentiment.get('neutral', 0)} neutral sentiments. Arabic reviews showed {ar_sentiment.get('positive', 0)} positive, {ar_sentiment.get('negative', 0)} negative, and {ar_sentiment.get('neutral', 0)} neutral sentiments.",
        "finding_type": "sentiment_distribution",
        "metrics": {
            "positive_reviews": {
                "value": sentiment_counts.get('positive', 0),
                "unit": "count",
                "numerator": sentiment_counts.get('positive', 0),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_reviews": {
                "value": sentiment_counts.get('negative', 0),
                "unit": "count",
                "numerator": sentiment_counts.get('negative', 0),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "neutral_reviews": {
                "value": sentiment_counts.get('neutral', 0),
                "unit": "count",
                "numerator": sentiment_counts.get('neutral', 0),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "positive_percentage": {
                "value": round((sentiment_counts.get('positive', 0) / len(analysis_reviews) * 100), 1) if len(analysis_reviews) > 0 else 0,
                "unit": "percent",
                "numerator": sentiment_counts.get('positive', 0),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(analysis_reviews['source'].unique()),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Sentiment classification: positive (rating ≥4), neutral (rating=3), negative (rating ≤2)",
            f"English reviews: {len(en_reviews)}, Arabic reviews: {len(ar_reviews)}",
            f"Total reviews analyzed: {len(analysis_reviews)}"
        ],
        "assumptions": [
            "Rating thresholds for sentiment classification are appropriate",
            "All reviews have valid rating values",
            "Language classification is accurate"
        ],
        "confidence": 0.90
    }
    findings.append(finding2)

# Finding 3: Review Source Distribution
if len(analysis_reviews) > 0:
    source_counts = analysis_reviews['source'].value_counts()
    
    finding3 = {
        "title": "Review Source Distribution",
        "claim": f"During the analysis period, reviews came from {len(source_counts)} different sources. The distribution was: {', '.join([f'{source}: {count} reviews' for source, count in source_counts.items()])}.",
        "finding_type": "source_distribution",
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
            f"Source distribution: {dict(source_counts)}",
            f"Total reviews: {len(analysis_reviews)}",
            f"Analysis period: {analysis_start} to {analysis_end}"
        ],
        "assumptions": [
            "Source field is accurately populated",
            "All reviews have a valid source"
        ],
        "confidence": 0.95
    }
    
    # Add source-specific metrics
    for source, count in source_counts.items():
        finding3['metrics'][f'{source}_count'] = {
            "value": count,
            "unit": "count",
            "numerator": count,
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
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. Output written to {output_path}")
print(f"Total findings: {len(findings)}")
