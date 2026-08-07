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
analysis_start = "2026-06-22"
analysis_end = "2026-06-29"

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
    
    finding1 = {
        "title": "Review Rating Distribution (Jun 22-29, 2026)",
        "claim": f"Average rating is {avg_rating:.2f} out of 5 across {len(reviews_analysis)} reviews during the analysis period.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": len(reviews_analysis),
                "denominator": len(reviews_analysis),
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            },
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
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

# Finding 2: Sentiment/Topic Classification by Language
if len(reviews_analysis) > 0:
    # Separate by language
    english_reviews = reviews_analysis[reviews_analysis['language'] == 'en'].copy()
    arabic_reviews = reviews_analysis[reviews_analysis['language'] == 'ar'].copy()
    
    # Simple sentiment classification based on rating
    def classify_sentiment(rating):
        if rating >= 4:
            return "positive"
        elif rating == 3:
            return "neutral"
        else:
            return "negative"
    
    reviews_analysis['sentiment'] = reviews_analysis['rating'].apply(classify_sentiment)
    
    sentiment_counts = reviews_analysis['sentiment'].value_counts()
    
    # Extract topics from text (simple keyword-based approach)
    topics = []
    topic_keywords = {
        'taste': ['taste', 'flavor', 'delicious', 'good', 'bad', 'طعم', 'لذيذ', 'سيء'],
        'service': ['service', 'staff', 'friendly', 'rude', 'خدمة', 'موظف', 'لطيف'],
        'price': ['price', 'expensive', 'cheap', 'cost', 'سعر', 'غالي', 'رخيص'],
        'quality': ['quality', 'fresh', 'clean', 'dirty', 'جودة', 'نظيف', 'وسخ'],
        'speed': ['fast', 'slow', 'quick', 'wait', 'سريع', 'بطيء', 'انتظار']
    }
    
    for idx, row in reviews_analysis.iterrows():
        text = str(row['text']).lower() if pd.notna(row['text']) else ""
        for topic, keywords in topic_keywords.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)
    
    topic_counts = Counter(topics)
    
    finding2 = {
        "title": "Sentiment Distribution by Language (Jun 22-29, 2026)",
        "claim": f"Sentiment distribution shows {sentiment_counts.get('positive', 0)} positive, {sentiment_counts.get('neutral', 0)} neutral, and {sentiment_counts.get('negative', 0)} negative reviews. English reviews: {len(english_reviews)}, Arabic reviews: {len(arabic_reviews)}.",
        "finding_type": "sentiment_analysis",
        "metrics": {
            "positive_reviews": {
                "value": sentiment_counts.get('positive', 0),
                "unit": "count",
                "numerator": sentiment_counts.get('positive', 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            },
            "neutral_reviews": {
                "value": sentiment_counts.get('neutral', 0),
                "unit": "count",
                "numerator": sentiment_counts.get('neutral', 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            },
            "negative_reviews": {
                "value": sentiment_counts.get('negative', 0),
                "unit": "count",
                "numerator": sentiment_counts.get('negative', 0),
                "denominator": len(reviews_analysis),
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            },
            "english_reviews": {
                "value": len(english_reviews),
                "unit": "count",
                "numerator": len(english_reviews),
                "denominator": len(reviews_analysis),
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            },
            "arabic_reviews": {
                "value": len(arabic_reviews),
                "unit": "count",
                "numerator": len(arabic_reviews),
                "denominator": len(reviews_analysis),
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Sentiment classified based on rating thresholds (4-5: positive, 3: neutral, 1-2: negative)",
            f"Language coverage: {len(english_reviews)} English, {len(arabic_reviews)} Arabic"
        ],
        "assumptions": [
            "Sentiment classification based on rating values only",
            "Topic extraction uses simple keyword matching",
            "Review text may be empty or missing for some reviews"
        ],
        "confidence": 0.85
    }
    findings.append(finding2)

# Finding 3: Top Topics Mentioned
if len(reviews_analysis) > 0 and len(topic_counts) > 0:
    top_topics = dict(topic_counts.most_common(3))
    
    finding3 = {
        "title": "Most Mentioned Topics in Reviews (Jun 22-29, 2026)",
        "claim": f"The most frequently mentioned topics are: {', '.join([f'{topic} ({count} mentions)' for topic, count in top_topics.items()])}.",
        "finding_type": "topic_analysis",
        "metrics": {
            "top_topic_1": {
                "value": list(top_topics.keys())[0] if len(top_topics) > 0 else None,
                "unit": "topic",
                "numerator": list(top_topics.values())[0] if len(top_topics) > 0 else None,
                "denominator": len(reviews_analysis),
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            },
            "top_topic_2": {
                "value": list(top_topics.keys())[1] if len(top_topics) > 1 else None,
                "unit": "topic",
                "numerator": list(top_topics.values())[1] if len(top_topics) > 1 else None,
                "denominator": len(reviews_analysis),
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            },
            "top_topic_3": {
                "value": list(top_topics.keys())[2] if len(top_topics) > 2 else None,
                "unit": "topic",
                "numerator": list(top_topics.values())[2] if len(top_topics) > 2 else None,
                "denominator": len(reviews_analysis),
                "period_start": "2026-06-22T00:00:00+03:00",
                "period_end": "2026-06-29T00:00:00+03:00"
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Topics identified through keyword matching in review text",
            f"Total topic mentions: {sum(topic_counts.values())}",
            f"Unique topics identified: {len(topic_counts)}"
        ],
        "assumptions": [
            "Topic keywords are predefined and may not capture all variations",
            "A single review may mention multiple topics",
            "Keyword matching is case-insensitive"
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
