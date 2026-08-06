import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

# Load environment
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Define analysis period
analysis_start = "2026-03-02"
analysis_end = "2026-03-09"

# Convert date columns to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])

# Filter reviews for analysis period
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start) & 
    (reviews_df['date'] < analysis_end)
].copy()

# Initialize findings list
findings = []

# FINDING 1: Rating Distribution and Average
if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    finding1 = {
        "title": "Customer Rating Distribution (Analysis Period)",
        "claim": f"Average rating is {avg_rating:.2f} out of 5 across {len(reviews_analysis)} reviews in the analysis period (2026-03-02 to 2026-03-09).",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": float(reviews_analysis['rating'].sum()),
                "denominator": len(reviews_analysis),
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            },
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Reviews from {len(reviews_analysis['source'].unique())} source(s)",
            f"Language distribution: {dict(reviews_analysis['language'].value_counts())}"
        ],
        "assumptions": [
            "Rating values are valid integers between 1-5",
            "Review dates are accurate and in UTC+3 timezone"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# FINDING 2: Sentiment Classification by Language
if len(reviews_analysis) > 0:
    # Separate by language
    english_reviews = reviews_analysis[reviews_analysis['language'] == 'en'].copy()
    arabic_reviews = reviews_analysis[reviews_analysis['language'] == 'ar'].copy()
    
    # Simple sentiment classification based on rating
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
    
    sentiment_counts = reviews_analysis['sentiment'].value_counts()
    
    # Get sample reviews for each sentiment
    positive_reviews = reviews_analysis[reviews_analysis['sentiment'] == 'positive']
    negative_reviews = reviews_analysis[reviews_analysis['sentiment'] == 'negative']
    
    finding2 = {
        "title": "Sentiment Distribution by Language",
        "claim": f"Sentiment analysis of {len(reviews_analysis)} reviews shows {sentiment_counts.get('positive', 0)} positive, {sentiment_counts.get('neutral', 0)} neutral, and {sentiment_counts.get('negative', 0)} negative reviews. English reviews: {len(english_reviews)}, Arabic reviews: {len(arabic_reviews)}.",
        "finding_type": "sentiment_classification",
        "metrics": {
            "positive_reviews": {
                "value": int(sentiment_counts.get('positive', 0)),
                "unit": "count",
                "numerator": int(sentiment_counts.get('positive', 0)),
                "denominator": len(reviews_analysis),
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            },
            "negative_reviews": {
                "value": int(sentiment_counts.get('negative', 0)),
                "unit": "count",
                "numerator": int(sentiment_counts.get('negative', 0)),
                "denominator": len(reviews_analysis),
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            },
            "english_review_count": {
                "value": len(english_reviews),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            },
            "arabic_review_count": {
                "value": len(arabic_reviews),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-03-02T00:00:00+03:00",
                "period_end": "2026-03-09T00:00:00+03:00"
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Bilingual coverage: {len(english_reviews)} English, {len(arabic_reviews)} Arabic",
            f"Sentiment based on rating thresholds: 4-5 stars = positive, 3 = neutral, 1-2 = negative"
        ],
        "assumptions": [
            "Sentiment classification derived from numerical ratings only",
            "Rating scale is consistent across all sources",
            "Language field accurately reflects review language"
        ],
        "confidence": 0.85
    }
    findings.append(finding2)

# FINDING 3: Review Text Analysis - Topics/Keywords
if len(reviews_analysis) > 0 and reviews_analysis['text'].notna().sum() > 0:
    # Get non-empty reviews
    non_empty_reviews = reviews_analysis[reviews_analysis['text'].notna() & (reviews_analysis['text'].str.len() > 0)].copy()
    
    if len(non_empty_reviews) > 0:
        # Simple keyword extraction for common cafe topics
        keywords_en = ['coffee', 'taste', 'quality', 'service', 'price', 'fast', 'slow', 'friendly', 'rude', 'clean', 'dirty', 'hot', 'cold', 'fresh', 'stale']
        keywords_ar = ['قهوة', 'طعم', 'جودة', 'خدمة', 'سعر', 'سريع', 'بطيء', 'ودود', 'وقح', 'نظيف', 'متسخ', 'ساخن', 'بارد', 'طازج', 'قديم']
        
        # Count keyword mentions
        keyword_mentions = {}
        for review_text in non_empty_reviews['text']:
            if isinstance(review_text, str):
                text_lower = review_text.lower()
                for keyword in keywords_en:
                    if keyword in text_lower:
                        keyword_mentions[keyword] = keyword_mentions.get(keyword, 0) + 1
                for keyword in keywords_ar:
                    if keyword in review_text:
                        keyword_mentions[keyword] = keyword_mentions.get(keyword, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(keyword_mentions.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if top_keywords:
            finding3 = {
                "title": "Common Topics in Customer Reviews",
                "claim": f"Analysis of {len(non_empty_reviews)} non-empty reviews identifies recurring topics. Most mentioned: {top_keywords[0][0]} ({top_keywords[0][1]} mentions).",
                "finding_type": "topic_analysis",
                "metrics": {
                    "reviews_with_text": {
                        "value": len(non_empty_reviews),
                        "unit": "count",
                        "numerator": None,
                        "denominator": None,
                        "period_start": "2026-03-02T00:00:00+03:00",
                        "period_end": "2026-03-09T00:00:00+03:00"
                    },
                    "top_keyword": {
                        "value": top_keywords[0][0],
                        "unit": "keyword",
                        "numerator": top_keywords[0][1],
                        "denominator": len(non_empty_reviews),
                        "period_start": "2026-03-02T00:00:00+03:00",
                        "period_end": "2026-03-09T00:00:00+03:00"
                    }
                },
                "source_names": list(non_empty_reviews['source'].unique()),
                "sample_size": len(non_empty_reviews),
                "coverage_notes": [
                    f"Text analysis based on {len(non_empty_reviews)} reviews with non-empty text",
                    f"Keyword matching performed in both English and Arabic",
                    f"Top 5 keywords identified: {', '.join([f'{k[0]} ({k[1]})' for k in top_keywords])}"
                ],
                "assumptions": [
                    "Keyword matching is case-insensitive for English",
                    "Keywords are representative of common cafe topics",
                    "Text field contains original review text without translation"
                ],
                "confidence": 0.75
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

print(f"Analysis complete. {len(findings)} findings generated.")
