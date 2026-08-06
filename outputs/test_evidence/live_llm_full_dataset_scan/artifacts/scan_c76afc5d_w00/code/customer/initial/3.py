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
analysis_start = "2026-01-05"
analysis_end = "2026-01-12"

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
    
    # Calculate rating distribution
    rating_dist = {}
    for rating in sorted(reviews_analysis['rating'].unique()):
        count = len(reviews_analysis[reviews_analysis['rating'] == rating])
        rating_dist[f"rating_{int(rating)}_count"] = count
    
    finding1 = {
        "title": "Customer Rating Distribution (Jan 5-12, 2026)",
        "claim": f"Average customer rating is {avg_rating:.2f} out of 5, based on {len(reviews_analysis)} reviews during the analysis period.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": float(round(avg_rating, 2)),
                "unit": "stars",
                "numerator": float(round(avg_rating * len(reviews_analysis), 2)),
                "denominator": int(len(reviews_analysis)),
                "period_start": "2026-01-05T00:00:00+03:00",
                "period_end": "2026-01-12T00:00:00+03:00"
            },
            "total_reviews": {
                "value": int(len(reviews_analysis)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-05T00:00:00+03:00",
                "period_end": "2026-01-12T00:00:00+03:00"
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": int(len(reviews_analysis)),
        "coverage_notes": [
            f"Reviews from {len(reviews_analysis['source'].unique())} source(s)",
            f"Language distribution: {dict(reviews_analysis['language'].value_counts())}"
        ],
        "assumptions": [
            "Rating values are numeric and valid",
            "Review dates are accurate and in UTC+3",
            "All reviews in the period are included"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Sentiment/Topic Analysis by Language
if len(reviews_analysis) > 0:
    # Separate by language
    english_reviews = reviews_analysis[reviews_analysis['language'] == 'en'].copy()
    arabic_reviews = reviews_analysis[reviews_analysis['language'] == 'ar'].copy()
    
    # Analyze sentiment based on rating
    sentiment_mapping = {
        5: "positive",
        4: "positive", 
        3: "neutral",
        2: "negative",
        1: "negative"
    }
    
    english_reviews['sentiment'] = english_reviews['rating'].map(sentiment_mapping)
    arabic_reviews['sentiment'] = arabic_reviews['rating'].map(sentiment_mapping)
    
    # Count sentiments
    en_sentiment_counts = english_reviews['sentiment'].value_counts().to_dict() if len(english_reviews) > 0 else {}
    ar_sentiment_counts = arabic_reviews['sentiment'].value_counts().to_dict() if len(arabic_reviews) > 0 else {}
    
    # Extract key topics from text (simple keyword analysis)
    topics_en = []
    topics_ar = []
    
    if len(english_reviews) > 0:
        en_texts = english_reviews[english_reviews['text'].notna()]['text'].str.lower()
        en_keywords = ['coffee', 'taste', 'quality', 'service', 'price', 'fast', 'slow', 'friendly', 'rude', 'clean', 'dirty', 'hot', 'cold', 'fresh', 'stale']
        for keyword in en_keywords:
            count = sum(en_texts.str.contains(keyword, na=False))
            if count > 0:
                topics_en.append((keyword, count))
    
    if len(arabic_reviews) > 0:
        ar_texts = arabic_reviews[arabic_reviews['text'].notna()]['text'].str.lower()
        ar_keywords = ['قهوة', 'طعم', 'جودة', 'خدمة', 'سعر', 'سريع', 'بطيء', 'ودود', 'وقح', 'نظيف', 'وسخ', 'ساخن', 'بارد', 'طازج', 'قديم']
        for keyword in ar_keywords:
            count = sum(ar_texts.str.contains(keyword, na=False))
            if count > 0:
                topics_ar.append((keyword, count))
    
    # Create finding for sentiment analysis
    total_en = len(english_reviews)
    total_ar = len(arabic_reviews)
    
    if total_en > 0 or total_ar > 0:
        finding2 = {
            "title": "Customer Sentiment by Language (Jan 5-12, 2026)",
            "claim": f"English reviews ({total_en} reviews) show {en_sentiment_counts.get('positive', 0)} positive, {en_sentiment_counts.get('neutral', 0)} neutral, {en_sentiment_counts.get('negative', 0)} negative sentiments. Arabic reviews ({total_ar} reviews) show {ar_sentiment_counts.get('positive', 0)} positive, {ar_sentiment_counts.get('neutral', 0)} neutral, {ar_sentiment_counts.get('negative', 0)} negative sentiments.",
            "finding_type": "sentiment_analysis",
            "metrics": {
                "english_positive_count": {
                    "value": int(en_sentiment_counts.get('positive', 0)),
                    "unit": "reviews",
                    "numerator": int(en_sentiment_counts.get('positive', 0)),
                    "denominator": int(total_en) if total_en > 0 else None,
                    "period_start": "2026-01-05T00:00:00+03:00",
                    "period_end": "2026-01-12T00:00:00+03:00"
                },
                "english_negative_count": {
                    "value": int(en_sentiment_counts.get('negative', 0)),
                    "unit": "reviews",
                    "numerator": int(en_sentiment_counts.get('negative', 0)),
                    "denominator": int(total_en) if total_en > 0 else None,
                    "period_start": "2026-01-05T00:00:00+03:00",
                    "period_end": "2026-01-12T00:00:00+03:00"
                },
                "arabic_positive_count": {
                    "value": int(ar_sentiment_counts.get('positive', 0)),
                    "unit": "reviews",
                    "numerator": int(ar_sentiment_counts.get('positive', 0)),
                    "denominator": int(total_ar) if total_ar > 0 else None,
                    "period_start": "2026-01-05T00:00:00+03:00",
                    "period_end": "2026-01-12T00:00:00+03:00"
                },
                "arabic_negative_count": {
                    "value": int(ar_sentiment_counts.get('negative', 0)),
                    "unit": "reviews",
                    "numerator": int(ar_sentiment_counts.get('negative', 0)),
                    "denominator": int(total_ar) if total_ar > 0 else None,
                    "period_start": "2026-01-05T00:00:00+03:00",
                    "period_end": "2026-01-12T00:00:00+03:00"
                }
            },
            "source_names": list(reviews_analysis['source'].unique()),
            "sample_size": int(len(reviews_analysis)),
            "coverage_notes": [
                f"English reviews: {total_en}",
                f"Arabic reviews: {total_ar}",
                f"Total reviews analyzed: {len(reviews_analysis)}"
            ],
            "assumptions": [
                "Sentiment derived from rating: 4-5 stars = positive, 3 = neutral, 1-2 = negative",
                "Language classification is accurate",
                "Review dates are in UTC+3 timezone"
            ],
            "confidence": 0.90
        }
        findings.append(finding2)

# Finding 3: High-Volume Review Sources
if len(reviews_analysis) > 0:
    source_counts = reviews_analysis['source'].value_counts()
    
    if len(source_counts) > 0:
        top_source = source_counts.index[0]
        top_source_count = int(source_counts.iloc[0])
        top_source_avg_rating = reviews_analysis[reviews_analysis['source'] == top_source]['rating'].mean()
        
        finding3 = {
            "title": "Primary Review Source (Jan 5-12, 2026)",
            "claim": f"The primary review source is '{top_source}' with {top_source_count} reviews ({top_source_count/len(reviews_analysis)*100:.1f}% of total), averaging {top_source_avg_rating:.2f} stars.",
            "finding_type": "source_analysis",
            "metrics": {
                "top_source_review_count": {
                    "value": int(top_source_count),
                    "unit": "reviews",
                    "numerator": int(top_source_count),
                    "denominator": int(len(reviews_analysis)),
                    "period_start": "2026-01-05T00:00:00+03:00",
                    "period_end": "2026-01-12T00:00:00+03:00"
                },
                "top_source_average_rating": {
                    "value": float(round(top_source_avg_rating, 2)),
                    "unit": "stars",
                    "numerator": float(round(top_source_avg_rating * top_source_count, 2)),
                    "denominator": int(top_source_count),
                    "period_start": "2026-01-05T00:00:00+03:00",
                    "period_end": "2026-01-12T00:00:00+03:00"
                },
                "source_percentage": {
                    "value": float(round(top_source_count/len(reviews_analysis)*100, 1)),
                    "unit": "percent",
                    "numerator": int(top_source_count),
                    "denominator": int(len(reviews_analysis)),
                    "period_start": "2026-01-05T00:00:00+03:00",
                    "period_end": "2026-01-12T00:00:00+03:00"
                }
            },
            "source_names": [top_source],
            "sample_size": int(top_source_count),
            "coverage_notes": [
                f"Total sources: {len(source_counts)}",
                f"Source distribution: {dict(source_counts)}"
            ],
            "assumptions": [
                "Source names are accurate and consistent",
                "Review dates are in UTC+3 timezone",
                "All reviews in the period are included"
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
