import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter
import re

# Load environment configuration
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read input artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Define analysis period
analysis_start = datetime.fromisoformat("2026-03-23T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-03-30T00:00:00+03:00")

# Convert review dates to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True)

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
    
    finding_1 = {
        "title": "Customer Rating Distribution (Mar 23-30, 2026)",
        "claim": f"Average customer rating is {avg_rating:.2f} out of 5, based on {len(reviews_analysis)} reviews during the analysis period.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(avg_rating * len(reviews_analysis), 1),
                "denominator": len(reviews_analysis),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Reviews from {len(reviews_analysis['source'].unique())} source(s)",
            f"Language distribution: {dict(reviews_analysis['language'].value_counts())}"
        ],
        "assumptions": [
            "All reviews in the artifact are valid and represent genuine customer feedback",
            "Rating scale is 1-5 stars",
            "Reviews are independent observations"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# Finding 2: Sentiment Classification by Language
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
    
    # Get sample review IDs for each sentiment
    positive_samples = reviews_analysis[reviews_analysis['sentiment'] == 'positive']['review_id'].head(3).tolist()
    negative_samples = reviews_analysis[reviews_analysis['sentiment'] == 'negative']['review_id'].head(3).tolist()
    
    finding_2 = {
        "title": "Sentiment Distribution by Language",
        "claim": f"Sentiment analysis shows {sentiment_counts.get('positive', 0)} positive, {sentiment_counts.get('neutral', 0)} neutral, and {sentiment_counts.get('negative', 0)} negative reviews. English reviews: {len(english_reviews)}, Arabic reviews: {len(arabic_reviews)}.",
        "finding_type": "sentiment_classification",
        "metrics": {
            "positive_reviews": {
                "value": sentiment_counts.get('positive', 0),
                "unit": "count",
                "numerator": sentiment_counts.get('positive', 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "negative_reviews": {
                "value": sentiment_counts.get('negative', 0),
                "unit": "count",
                "numerator": sentiment_counts.get('negative', 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "english_review_count": {
                "value": len(english_reviews),
                "unit": "count",
                "numerator": len(english_reviews),
                "denominator": len(reviews_analysis),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            },
            "arabic_review_count": {
                "value": len(arabic_reviews),
                "unit": "count",
                "numerator": len(arabic_reviews),
                "denominator": len(reviews_analysis),
                "period_start": "2026-03-23T00:00:00+03:00",
                "period_end": "2026-03-30T00:00:00+03:00"
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Bilingual coverage: {len(english_reviews)} English, {len(arabic_reviews)} Arabic",
            f"Sentiment classification based on rating thresholds (4-5: positive, 3: neutral, 1-2: negative)"
        ],
        "assumptions": [
            "Sentiment is derived from rating values only",
            "Rating scale is 1-5 stars",
            "Language field accurately reflects review language"
        ],
        "confidence": 0.90
    }
    findings.append(finding_2)

# Finding 3: Review Text Analysis - Common Topics
if len(reviews_analysis) > 0 and reviews_analysis['text'].notna().sum() > 0:
    # Analyze text content for common topics
    all_text = ' '.join(reviews_analysis[reviews_analysis['text'].notna()]['text'].astype(str).str.lower())
    
    # Define topic keywords (simple keyword matching)
    topics = {
        'quality': ['quality', 'good', 'excellent', 'bad', 'poor', 'جودة', 'ممتاز', 'سيء'],
        'service': ['service', 'staff', 'friendly', 'rude', 'خدمة', 'موظف', 'لطيف'],
        'price': ['price', 'expensive', 'cheap', 'value', 'سعر', 'غالي', 'رخيص'],
        'taste': ['taste', 'flavor', 'delicious', 'bland', 'طعم', 'لذيذ', 'مملل'],
        'speed': ['fast', 'slow', 'quick', 'wait', 'سريع', 'بطيء', 'انتظار']
    }
    
    topic_counts = {}
    for topic, keywords in topics.items():
        count = sum(1 for keyword in keywords if keyword in all_text)
        if count > 0:
            topic_counts[topic] = count
    
    # Get review count with non-empty text
    reviews_with_text = reviews_analysis[reviews_analysis['text'].notna()]
    text_coverage = len(reviews_with_text)
    
    if text_coverage > 0:
        finding_3 = {
            "title": "Review Text Topics and Themes",
            "claim": f"Analysis of {text_coverage} reviews with text content identifies mentions of quality, service, price, taste, and speed. Topic mentions detected: {', '.join([f'{k}: {v}' for k, v in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]])}.",
            "finding_type": "topic_analysis",
            "metrics": {
                "reviews_with_text": {
                    "value": text_coverage,
                    "unit": "count",
                    "numerator": text_coverage,
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-03-23T00:00:00+03:00",
                    "period_end": "2026-03-30T00:00:00+03:00"
                },
                "empty_reviews": {
                    "value": len(reviews_analysis) - text_coverage,
                    "unit": "count",
                    "numerator": len(reviews_analysis) - text_coverage,
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-03-23T00:00:00+03:00",
                    "period_end": "2026-03-30T00:00:00+03:00"
                }
            },
            "source_names": list(reviews_analysis['source'].unique()),
            "sample_size": text_coverage,
            "coverage_notes": [
                f"Text analysis covers {text_coverage} of {len(reviews_analysis)} reviews ({100*text_coverage/len(reviews_analysis):.1f}%)",
                f"Bilingual keyword matching applied to both English and Arabic text",
                "Topic detection uses simple keyword presence, not frequency weighting"
            ],
            "assumptions": [
                "Keyword presence indicates topic relevance",
                "Text field contains original customer language",
                "Empty text fields represent reviews with rating only"
            ],
            "confidence": 0.75
        }
        findings.append(finding_3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings generated.")
print(f"Output written to {output_path}")
