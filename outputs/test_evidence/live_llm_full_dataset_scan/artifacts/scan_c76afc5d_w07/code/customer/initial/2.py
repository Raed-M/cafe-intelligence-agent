import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter
import re

# Load environment
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Analysis period
analysis_start = "2026-02-23T00:00:00+03:00"
analysis_end = "2026-03-02T00:00:00+03:00"
previous_start = "2026-02-16T00:00:00+03:00"
previous_end = "2026-02-23T00:00:00+03:00"

# Parse dates for filtering - use utc=False to get naive datetime, then handle timezone
analysis_start_dt = pd.to_datetime(analysis_start, utc=True).tz_localize(None)
analysis_end_dt = pd.to_datetime(analysis_end, utc=True).tz_localize(None)
previous_start_dt = pd.to_datetime(previous_start, utc=True).tz_localize(None)
previous_end_dt = pd.to_datetime(previous_end, utc=True).tz_localize(None)

# Convert review dates to datetime (naive)
reviews_df['date'] = pd.to_datetime(reviews_df['date'], errors='coerce', utc=True).dt.tz_localize(None)

# Filter reviews for analysis period
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

reviews_previous = reviews_df[
    (reviews_df['date'] >= previous_start_dt) & 
    (reviews_df['date'] < previous_end_dt)
].copy()

findings = []

# FINDING 1: Rating Distribution and Average
if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    finding1 = {
        "title": "Customer Rating Distribution (Analysis Period)",
        "claim": f"Average rating is {avg_rating:.2f} across {len(reviews_analysis)} reviews in the analysis period (2026-02-23 to 2026-03-02). Distribution: {dict(rating_counts)}",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": float(reviews_analysis['rating'].sum()),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Reviews from {len(reviews_analysis['source'].unique())} source(s)",
            f"Language distribution: {dict(reviews_analysis['language'].value_counts())}"
        ],
        "assumptions": [
            "Rating values are valid numeric ratings",
            "Review dates are accurate and in UTC+3 timezone"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# FINDING 2: Sentiment Classification by Language
if len(reviews_analysis) > 0:
    # Classify sentiment based on rating thresholds
    def classify_sentiment(rating):
        if pd.isna(rating):
            return "unknown"
        if rating >= 4:
            return "positive"
        elif rating == 3:
            return "neutral"
        else:
            return "negative"
    
    reviews_analysis['sentiment'] = reviews_analysis['rating'].apply(classify_sentiment)
    
    # Count by language and sentiment
    lang_sentiment = reviews_analysis.groupby(['language', 'sentiment']).size().reset_index(name='count')
    
    # Get language breakdown
    lang_counts = reviews_analysis['language'].value_counts()
    
    # Identify dominant language
    dominant_lang = lang_counts.idxmax() if len(lang_counts) > 0 else "unknown"
    dominant_count = lang_counts.max() if len(lang_counts) > 0 else 0
    
    sentiment_by_lang = {}
    for lang in reviews_analysis['language'].unique():
        lang_data = reviews_analysis[reviews_analysis['language'] == lang]
        sentiment_dist = lang_data['sentiment'].value_counts().to_dict()
        sentiment_by_lang[lang] = sentiment_dist
    
    finding2 = {
        "title": "Sentiment Distribution by Language",
        "claim": f"Of {len(reviews_analysis)} reviews, {dominant_count} ({100*dominant_count/len(reviews_analysis):.1f}%) are in {dominant_lang}. Sentiment distribution: {dict(reviews_analysis['sentiment'].value_counts())}",
        "finding_type": "sentiment_by_language",
        "metrics": {
            "total_reviews_analyzed": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "positive_sentiment_count": {
                "value": len(reviews_analysis[reviews_analysis['sentiment'] == 'positive']),
                "unit": "count",
                "numerator": len(reviews_analysis[reviews_analysis['sentiment'] == 'positive']),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_sentiment_count": {
                "value": len(reviews_analysis[reviews_analysis['sentiment'] == 'negative']),
                "unit": "count",
                "numerator": len(reviews_analysis[reviews_analysis['sentiment'] == 'negative']),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Language coverage: {dict(lang_counts)}",
            f"Sentiment by language: {json.dumps(sentiment_by_lang)}"
        ],
        "assumptions": [
            "Sentiment classified by rating threshold: 4+ = positive, 3 = neutral, <3 = negative",
            "Language field accurately reflects review language"
        ],
        "confidence": 0.90
    }
    findings.append(finding2)

# FINDING 3: Period-over-Period Rating Change
if len(reviews_previous) > 0 and len(reviews_analysis) > 0:
    avg_rating_previous = reviews_previous['rating'].mean()
    avg_rating_current = reviews_analysis['rating'].mean()
    rating_change = avg_rating_current - avg_rating_previous
    pct_change = (rating_change / avg_rating_previous * 100) if avg_rating_previous != 0 else 0
    
    finding3 = {
        "title": "Rating Trend: Week-over-Week Change",
        "claim": f"Average rating changed from {avg_rating_previous:.2f} (previous week) to {avg_rating_current:.2f} (analysis week), a change of {rating_change:+.2f} stars ({pct_change:+.1f}%)",
        "finding_type": "rating_trend",
        "metrics": {
            "average_rating_current_period": {
                "value": round(avg_rating_current, 2),
                "unit": "stars",
                "numerator": float(reviews_analysis['rating'].sum()),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "average_rating_previous_period": {
                "value": round(avg_rating_previous, 2),
                "unit": "stars",
                "numerator": float(reviews_previous['rating'].sum()),
                "denominator": len(reviews_previous),
                "period_start": previous_start,
                "period_end": previous_end
            },
            "rating_change": {
                "value": round(rating_change, 2),
                "unit": "stars",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(set(list(reviews_analysis['source'].unique()) + list(reviews_previous['source'].unique()))),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Current period reviews: {len(reviews_analysis)}",
            f"Previous period reviews: {len(reviews_previous)}",
            f"Current period language distribution: {dict(reviews_analysis['language'].value_counts())}",
            f"Previous period language distribution: {dict(reviews_previous['language'].value_counts())}"
        ],
        "assumptions": [
            "Review dates are accurate",
            "Both periods have sufficient sample size for comparison",
            "No systematic bias in review collection between periods"
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
