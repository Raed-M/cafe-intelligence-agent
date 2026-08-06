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
analysis_start = "2026-04-06"
analysis_end = "2026-04-13"

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

# Topic extraction from review text
def extract_topics(text, language):
    if pd.isna(text) or text == '':
        return []
    
    text_lower = str(text).lower()
    topics = []
    
    # Common topic keywords in English and Arabic
    topic_keywords = {
        'quality': ['quality', 'good', 'excellent', 'bad', 'poor', 'جودة', 'ممتاز', 'سيء'],
        'taste': ['taste', 'flavor', 'delicious', 'sweet', 'bitter', 'طعم', 'لذيذ', 'حلو'],
        'service': ['service', 'staff', 'friendly', 'rude', 'slow', 'خدمة', 'موظف', 'سريع'],
        'price': ['price', 'expensive', 'cheap', 'value', 'سعر', 'غالي', 'رخيص'],
        'temperature': ['hot', 'cold', 'warm', 'ice', 'iced', 'درجة', 'بارد', 'ساخن'],
        'cleanliness': ['clean', 'dirty', 'hygiene', 'fresh', 'نظيف', 'وسخ', 'نظافة'],
        'atmosphere': ['atmosphere', 'ambiance', 'cozy', 'crowded', 'جو', 'مريح', 'مزدحم']
    }
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            topics.append(topic)
    
    return topics

reviews_analysis['topics'] = reviews_analysis.apply(
    lambda row: extract_topics(row['text'], row['language']), 
    axis=1
)

# Flatten topics for counting
all_topics = []
for topics_list in reviews_analysis['topics']:
    all_topics.extend(topics_list)

topic_counts = Counter(all_topics)

# Source coverage
source_counts = reviews_analysis['source'].value_counts()

# Prepare findings
findings = []

# Finding 1: Rating Distribution and Average
if total_reviews > 0:
    finding1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"Average rating is {avg_rating:.2f} out of 5 across {total_reviews} reviews in the analysis period (2026-04-06 to 2026-04-13). Rating distribution shows {int(rating_counts.get(5, 0))} five-star, {int(rating_counts.get(4, 0))} four-star, {int(rating_counts.get(3, 0))} three-star, {int(rating_counts.get(2, 0))} two-star, and {int(rating_counts.get(1, 0))} one-star reviews.",
        "finding_type": "customer_sentiment",
        "metrics": {
            "average_rating": {
                "value": round(float(avg_rating), 2),
                "unit": "stars",
                "numerator": round(float(reviews_analysis['rating'].sum()), 2),
                "denominator": int(total_reviews),
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            },
            "total_reviews": {
                "value": int(total_reviews),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            },
            "five_star_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            },
            "one_star_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            }
        },
        "source_names": list(source_counts.index),
        "sample_size": int(total_reviews),
        "coverage_notes": [
            f"Reviews from {len(source_counts)} sources",
            f"Language distribution: {dict((k, int(v)) for k, v in language_counts.items())}",
            f"Analysis period: 2026-04-06 to 2026-04-13"
        ],
        "assumptions": [
            "Rating scale is 1-5 stars",
            "All reviews in the dataset are valid and complete",
            "Review dates are accurate and in the specified timezone"
        ],
        "confidence": 0.95 if total_reviews >= 30 else 0.70
    }
    findings.append(finding1)

# Finding 2: Sentiment Distribution
if total_reviews > 0:
    positive_count = int(sentiment_counts.get('positive', 0))
    negative_count = int(sentiment_counts.get('negative', 0))
    neutral_count = int(sentiment_counts.get('neutral', 0))
    
    finding2 = {
        "title": "Sentiment Classification (Analysis Period)",
        "claim": f"Sentiment analysis of {total_reviews} reviews shows {positive_count} positive ({100*positive_count/total_reviews:.1f}%), {neutral_count} neutral ({100*neutral_count/total_reviews:.1f}%), and {negative_count} negative ({100*negative_count/total_reviews:.1f}%) reviews based on rating thresholds (4-5 stars = positive, 3 = neutral, 1-2 = negative).",
        "finding_type": "customer_sentiment",
        "metrics": {
            "positive_reviews": {
                "value": positive_count,
                "unit": "count",
                "numerator": positive_count,
                "denominator": int(total_reviews),
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            },
            "negative_reviews": {
                "value": negative_count,
                "unit": "count",
                "numerator": negative_count,
                "denominator": int(total_reviews),
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            },
            "neutral_reviews": {
                "value": neutral_count,
                "unit": "count",
                "numerator": neutral_count,
                "denominator": int(total_reviews),
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            },
            "positive_percentage": {
                "value": round(100*positive_count/total_reviews, 1),
                "unit": "percent",
                "numerator": positive_count,
                "denominator": int(total_reviews),
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            }
        },
        "source_names": list(source_counts.index),
        "sample_size": int(total_reviews),
        "coverage_notes": [
            f"Sentiment classification based on rating thresholds",
            f"Language distribution: {dict((k, int(v)) for k, v in language_counts.items())}",
            f"All {total_reviews} reviews classified"
        ],
        "assumptions": [
            "Rating 4-5 = positive sentiment",
            "Rating 3 = neutral sentiment",
            "Rating 1-2 = negative sentiment",
            "No text analysis required for sentiment classification"
        ],
        "confidence": 0.95 if total_reviews >= 30 else 0.70
    }
    findings.append(finding2)

# Finding 3: Topic Frequency (if topics found)
if topic_counts and total_reviews > 0:
    top_topics = topic_counts.most_common(3)
    
    topic_summary = ", ".join([f"{topic} ({int(count)} mentions)" for topic, count in top_topics])
    
    finding3 = {
        "title": "Most Mentioned Topics in Reviews (Analysis Period)",
        "claim": f"Topic analysis of {total_reviews} reviews identifies {len(topic_counts)} distinct topics. Most frequently mentioned: {topic_summary}. Topics extracted from review text using keyword matching in both English and Arabic.",
        "finding_type": "customer_feedback_themes",
        "metrics": {
            "total_topics_identified": {
                "value": int(len(topic_counts)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            },
            "top_topic_mentions": {
                "value": int(top_topics[0][1]) if top_topics else 0,
                "unit": "count",
                "numerator": int(top_topics[0][1]) if top_topics else 0,
                "denominator": int(total_reviews),
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            }
        },
        "source_names": list(source_counts.index),
        "sample_size": int(total_reviews),
        "coverage_notes": [
            f"Topics extracted from {len(reviews_analysis[reviews_analysis['text'].notna()])} reviews with text",
            f"Language distribution: {dict((k, int(v)) for k, v in language_counts.items())}",
            f"Topic keywords matched in both English and Arabic"
        ],
        "assumptions": [
            "Topic extraction uses predefined keyword lists",
            "Multiple topics can be identified in a single review",
            "Keyword matching is case-insensitive",
            "Topics are language-agnostic (same keywords searched in both languages)"
        ],
        "confidence": 0.80 if total_reviews >= 30 else 0.60
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

print(f"Analysis complete. Output written to {output_path}")
print(f"Total reviews analyzed: {total_reviews}")
print(f"Findings generated: {len(findings)}")
