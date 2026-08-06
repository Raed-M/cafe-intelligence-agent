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
analysis_start = "2026-05-18"
analysis_end = "2026-05-25"

# Convert date columns to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
pos_df['calendar_date'] = pd.to_datetime(pos_df['calendar_date'])

# Filter reviews for analysis period
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start) & 
    (reviews_df['date'] < analysis_end)
].copy()

# Calculate rating distribution and average
rating_counts = reviews_analysis['rating'].value_counts().sort_index()
avg_rating = reviews_analysis['rating'].mean()
total_reviews = len(reviews_analysis)

# Language distribution
language_counts = reviews_analysis['language'].value_counts()

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

reviews_analysis['sentiment'] = reviews_analysis['rating'].apply(classify_sentiment)
sentiment_counts = reviews_analysis['sentiment'].value_counts()

# Topic extraction from reviews (simple keyword-based)
topics = {
    'quality': 0,
    'service': 0,
    'price': 0,
    'taste': 0,
    'atmosphere': 0,
    'speed': 0,
    'cleanliness': 0
}

quality_keywords = ['quality', 'جودة', 'excellent', 'poor', 'bad', 'good']
service_keywords = ['service', 'خدمة', 'staff', 'friendly', 'rude', 'helpful']
price_keywords = ['price', 'سعر', 'expensive', 'cheap', 'cost', 'value']
taste_keywords = ['taste', 'طعم', 'flavor', 'delicious', 'bland', 'fresh']
atmosphere_keywords = ['atmosphere', 'جو', 'ambiance', 'cozy', 'crowded', 'clean']
speed_keywords = ['speed', 'سرعة', 'fast', 'slow', 'wait', 'quick']
cleanliness_keywords = ['clean', 'نظيف', 'dirty', 'hygiene', 'sanitary']

for idx, row in reviews_analysis.iterrows():
    text = str(row['text']).lower() if pd.notna(row['text']) else ''
    
    if any(kw in text for kw in quality_keywords):
        topics['quality'] += 1
    if any(kw in text for kw in service_keywords):
        topics['service'] += 1
    if any(kw in text for kw in price_keywords):
        topics['price'] += 1
    if any(kw in text for kw in taste_keywords):
        topics['taste'] += 1
    if any(kw in text for kw in atmosphere_keywords):
        topics['atmosphere'] += 1
    if any(kw in text for kw in speed_keywords):
        topics['speed'] += 1
    if any(kw in text for kw in cleanliness_keywords):
        topics['cleanliness'] += 1

# Find top topics
top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]

# Analyze sentiment by language
sentiment_by_language = {}
for lang in reviews_analysis['language'].unique():
    lang_reviews = reviews_analysis[reviews_analysis['language'] == lang]
    sentiment_by_language[lang] = {
        'positive': len(lang_reviews[lang_reviews['sentiment'] == 'positive']),
        'neutral': len(lang_reviews[lang_reviews['sentiment'] == 'neutral']),
        'negative': len(lang_reviews[lang_reviews['sentiment'] == 'negative']),
        'avg_rating': lang_reviews['rating'].mean(),
        'count': len(lang_reviews)
    }

# Get sample reviews for evidence
positive_reviews = reviews_analysis[reviews_analysis['sentiment'] == 'positive'].head(3)
negative_reviews = reviews_analysis[reviews_analysis['sentiment'] == 'negative'].head(3)

# Build findings
findings = []

# Finding 1: Rating Distribution and Average
if total_reviews > 0:
    finding1 = {
        "title": "Overall Customer Rating Distribution",
        "claim": f"Analysis period {analysis_start} to {analysis_end}: Average rating is {avg_rating:.2f}/5 across {total_reviews} reviews. Rating distribution shows {sentiment_counts.get('positive', 0)} positive (4-5 stars), {sentiment_counts.get('neutral', 0)} neutral (3 stars), and {sentiment_counts.get('negative', 0)} negative (1-2 stars) reviews.",
        "finding_type": "customer_sentiment",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": float(reviews_analysis['rating'].sum()),
                "denominator": total_reviews,
                "period_start": f"{analysis_start}T00:00:00+03:00",
                "period_end": f"{analysis_end}T00:00:00+03:00"
            },
            "total_reviews": {
                "value": total_reviews,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": f"{analysis_start}T00:00:00+03:00",
                "period_end": f"{analysis_end}T00:00:00+03:00"
            },
            "positive_reviews": {
                "value": sentiment_counts.get('positive', 0),
                "unit": "count",
                "numerator": sentiment_counts.get('positive', 0),
                "denominator": total_reviews,
                "period_start": f"{analysis_start}T00:00:00+03:00",
                "period_end": f"{analysis_end}T00:00:00+03:00"
            },
            "negative_reviews": {
                "value": sentiment_counts.get('negative', 0),
                "unit": "count",
                "numerator": sentiment_counts.get('negative', 0),
                "denominator": total_reviews,
                "period_start": f"{analysis_start}T00:00:00+03:00",
                "period_end": f"{analysis_end}T00:00:00+03:00"
            }
        },
        "source_names": ["reviews"],
        "sample_size": total_reviews,
        "coverage_notes": [
            f"Reviews collected from {len(reviews_analysis['source'].unique())} source(s)",
            f"Language coverage: {', '.join([f'{lang} ({sentiment_by_language[lang][\"count\"]} reviews)' for lang in sentiment_by_language.keys()])}",
            f"Rating distribution: {dict(rating_counts)}"
        ],
        "assumptions": [
            "Rating scale is 1-5 stars",
            "Positive sentiment = 4-5 stars, Neutral = 3 stars, Negative = 1-2 stars",
            "All reviews in the period are valid and complete"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Language-specific sentiment patterns
if len(sentiment_by_language) > 1:
    lang_comparison = []
    for lang, stats in sentiment_by_language.items():
        lang_comparison.append(f"{lang}: avg {stats['avg_rating']:.2f}/5 ({stats['count']} reviews)")
    
    finding2 = {
        "title": "Sentiment Patterns by Language",
        "claim": f"Customer sentiment varies by language. {', '.join(lang_comparison)}. This suggests potential differences in customer experience perception across language groups.",
        "finding_type": "customer_sentiment",
        "metrics": {
            "language_coverage": {
                "value": len(sentiment_by_language),
                "unit": "languages",
                "numerator": None,
                "denominator": None,
                "period_start": f"{analysis_start}T00:00:00+03:00",
                "period_end": f"{analysis_end}T00:00:00+03:00"
            }
        },
        "source_names": ["reviews"],
        "sample_size": total_reviews,
        "coverage_notes": [
            f"Language breakdown: {json.dumps({k: v['count'] for k, v in sentiment_by_language.items()})}"
        ],
        "assumptions": [
            "Language classification is accurate",
            "Sample sizes per language are sufficient for comparison"
        ],
        "confidence": 0.85
    }
    findings.append(finding2)

# Finding 3: Top mentioned topics
if top_topics and top_topics[0][1] > 0:
    topic_summary = ", ".join([f"{topic} ({count} mentions)" for topic, count in top_topics])
    
    finding3 = {
        "title": "Most Frequently Mentioned Topics in Reviews",
        "claim": f"Customer reviews most frequently mention: {topic_summary}. These topics appear in {top_topics[0][1]} to {top_topics[-1][1]} reviews respectively, indicating key areas of customer focus.",
        "finding_type": "customer_feedback",
        "metrics": {
            "top_topic_1": {
                "value": top_topics[0][0],
                "unit": "topic",
                "numerator": top_topics[0][1],
                "denominator": total_reviews,
                "period_start": f"{analysis_start}T00:00:00+03:00",
                "period_end": f"{analysis_end}T00:00:00+03:00"
            },
            "top_topic_2": {
                "value": top_topics[1][0] if len(top_topics) > 1 else None,
                "unit": "topic",
                "numerator": top_topics[1][1] if len(top_topics) > 1 else None,
                "denominator": total_reviews if len(top_topics) > 1 else None,
                "period_start": f"{analysis_start}T00:00:00+03:00",
                "period_end": f"{analysis_end}T00:00:00+03:00"
            },
            "top_topic_3": {
                "value": top_topics[2][0] if len(top_topics) > 2 else None,
                "unit": "topic",
                "numerator": top_topics[2][1] if len(top_topics) > 2 else None,
                "denominator": total_reviews if len(top_topics) > 2 else None,
                "period_start": f"{analysis_start}T00:00:00+03:00",
                "period_end": f"{analysis_end}T00:00:00+03:00"
            }
        },
        "source_names": ["reviews"],
        "sample_size": total_reviews,
        "coverage_notes": [
            "Topic extraction based on keyword matching in review text",
            f"Topics identified: {json.dumps(topics)}"
        ],
        "assumptions": [
            "Keyword-based topic extraction captures main themes",
            "Multiple topics can be mentioned in a single review",
            "Topic keywords are relevant and comprehensive"
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
    json.dump(output, f, indent=2)

print(f"Analysis complete. {len(findings)} findings generated.")
print(f"Total reviews analyzed: {total_reviews}")
print(f"Average rating: {avg_rating:.2f}/5")
