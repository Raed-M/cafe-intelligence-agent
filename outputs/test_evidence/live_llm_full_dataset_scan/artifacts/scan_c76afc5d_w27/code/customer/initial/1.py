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
analysis_start = "2026-07-13"
analysis_end = "2026-07-20"

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

# Analyze by source platform
source_distribution = reviews_analysis['source'].value_counts()

# Analyze by language
language_distribution = reviews_analysis['language'].value_counts()

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
sentiment_distribution = reviews_analysis['sentiment'].value_counts()

# Extract topics from review text
def extract_topics(text):
    if pd.isna(text) or text == '':
        return []
    
    text_lower = text.lower()
    topics = []
    
    # Common cafe-related topics
    topic_keywords = {
        'coffee_quality': ['coffee', 'espresso', 'latte', 'cappuccino', 'taste', 'flavor', 'bitter', 'strong', 'weak'],
        'service': ['service', 'staff', 'waiter', 'barista', 'friendly', 'rude', 'slow', 'fast', 'attentive'],
        'ambiance': ['ambiance', 'atmosphere', 'clean', 'dirty', 'cozy', 'comfortable', 'music', 'noise', 'decor'],
        'price': ['price', 'expensive', 'cheap', 'cost', 'value', 'overpriced', 'affordable'],
        'wait_time': ['wait', 'queue', 'long', 'quick', 'fast', 'slow'],
        'food_quality': ['food', 'pastry', 'cake', 'sandwich', 'fresh', 'stale', 'delicious', 'bad'],
        'temperature': ['hot', 'cold', 'warm', 'temperature', 'iced'],
        'location': ['location', 'convenient', 'accessible', 'parking', 'near', 'far']
    }
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            topics.append(topic)
    
    return topics

reviews_analysis['topics'] = reviews_analysis['text'].apply(extract_topics)

# Flatten topics for analysis
all_topics = []
for topics_list in reviews_analysis['topics']:
    all_topics.extend(topics_list)

topic_counts = Counter(all_topics)

# Prepare findings
findings = []

# Finding 1: Overall rating distribution and average
if total_reviews > 0:
    finding1 = {
        "title": "Review Rating Distribution (Jul 13-20, 2026)",
        "claim": f"Average rating is {avg_rating:.2f}/5 across {total_reviews} reviews during the analysis period, with {sentiment_distribution.get('positive', 0)} positive, {sentiment_distribution.get('neutral', 0)} neutral, and {sentiment_distribution.get('negative', 0)} negative reviews.",
        "finding_type": "voice_of_customer",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": total_reviews,
                "denominator": total_reviews,
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-20T00:00:00+03:00"
            },
            "total_reviews": {
                "value": total_reviews,
                "unit": "count",
                "numerator": total_reviews,
                "denominator": None,
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-20T00:00:00+03:00"
            },
            "positive_reviews": {
                "value": sentiment_distribution.get('positive', 0),
                "unit": "count",
                "numerator": sentiment_distribution.get('positive', 0),
                "denominator": total_reviews,
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-20T00:00:00+03:00"
            },
            "negative_reviews": {
                "value": sentiment_distribution.get('negative', 0),
                "unit": "count",
                "numerator": sentiment_distribution.get('negative', 0),
                "denominator": total_reviews,
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-20T00:00:00+03:00"
            }
        },
        "source_names": ["reviews"],
        "sample_size": total_reviews,
        "coverage_notes": [
            f"Analysis period: 2026-07-13 to 2026-07-20",
            f"Total reviews in period: {total_reviews}",
            f"Language distribution: {dict(language_distribution)}",
            f"Platform distribution: {dict(source_distribution)}"
        ],
        "assumptions": [
            "Rating >= 4 classified as positive, rating = 3 as neutral, rating < 3 as negative",
            "All reviews with valid ratings included in sentiment classification",
            "Review date field used to filter analysis period"
        ],
        "confidence": 0.95 if total_reviews >= 10 else 0.7
    }
    findings.append(finding1)

# Finding 2: Top topics mentioned in reviews
if len(topic_counts) > 0:
    top_topics = topic_counts.most_common(3)
    topic_mentions = {topic: count for topic, count in top_topics}
    
    finding2 = {
        "title": "Most Frequently Mentioned Topics in Reviews",
        "claim": f"The most discussed topics in customer reviews are {', '.join([f'{topic} ({count} mentions)' for topic, count in top_topics])}.",
        "finding_type": "voice_of_customer",
        "metrics": {
            "top_topic_1": {
                "value": top_topics[0][0] if len(top_topics) > 0 else None,
                "unit": "topic",
                "numerator": top_topics[0][1] if len(top_topics) > 0 else None,
                "denominator": total_reviews,
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-20T00:00:00+03:00"
            },
            "top_topic_2": {
                "value": top_topics[1][0] if len(top_topics) > 1 else None,
                "unit": "topic",
                "numerator": top_topics[1][1] if len(top_topics) > 1 else None,
                "denominator": total_reviews,
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-20T00:00:00+03:00"
            },
            "top_topic_3": {
                "value": top_topics[2][0] if len(top_topics) > 2 else None,
                "unit": "topic",
                "numerator": top_topics[2][1] if len(top_topics) > 2 else None,
                "denominator": total_reviews,
                "period_start": "2026-07-13T00:00:00+03:00",
                "period_end": "2026-07-20T00:00:00+03:00"
            }
        },
        "source_names": ["reviews"],
        "sample_size": total_reviews,
        "coverage_notes": [
            f"Topic extraction based on keyword matching in review text",
            f"Total unique topics identified: {len(topic_counts)}",
            f"Reviews with extractable text: {len(reviews_analysis[reviews_analysis['text'].notna()])}"
        ],
        "assumptions": [
            "Topics identified through keyword matching in review text",
            "Multiple topics can be mentioned in a single review",
            "Topic keywords are case-insensitive"
        ],
        "confidence": 0.8 if total_reviews >= 10 else 0.6
    }
    findings.append(finding2)

# Finding 3: Sentiment by platform
if len(source_distribution) > 0:
    platform_sentiments = reviews_analysis.groupby('source')['sentiment'].value_counts().unstack(fill_value=0)
    
    # Find platform with highest positive ratio
    if 'positive' in platform_sentiments.columns:
        platform_sentiments['positive_ratio'] = platform_sentiments.get('positive', 0) / platform_sentiments.sum(axis=1)
        best_platform = platform_sentiments['positive_ratio'].idxmax()
        best_platform_positive = platform_sentiments.loc[best_platform, 'positive']
        best_platform_total = platform_sentiments.loc[best_platform].sum()
        
        finding3 = {
            "title": "Sentiment Distribution by Review Platform",
            "claim": f"Platform '{best_platform}' has the highest proportion of positive reviews with {int(best_platform_positive)} positive out of {int(best_platform_total)} total reviews.",
            "finding_type": "voice_of_customer",
            "metrics": {
                "best_platform": {
                    "value": best_platform,
                    "unit": "platform",
                    "numerator": int(best_platform_positive),
                    "denominator": int(best_platform_total),
                    "period_start": "2026-07-13T00:00:00+03:00",
                    "period_end": "2026-07-20T00:00:00+03:00"
                },
                "positive_ratio": {
                    "value": round(best_platform_positive / best_platform_total, 2),
                    "unit": "ratio",
                    "numerator": int(best_platform_positive),
                    "denominator": int(best_platform_total),
                    "period_start": "2026-07-13T00:00:00+03:00",
                    "period_end": "2026-07-20T00:00:00+03:00"
                }
            },
            "source_names": ["reviews"],
            "sample_size": total_reviews,
            "coverage_notes": [
                f"Platforms analyzed: {list(source_distribution.index)}",
                f"Platform distribution: {dict(source_distribution)}"
            ],
            "assumptions": [
                "Sentiment classification based on rating thresholds",
                "Platform refers to review source platform (google, instagram, talabat, etc.)"
            ],
            "confidence": 0.85 if best_platform_total >= 5 else 0.65
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

print(f"Analysis complete. Results written to {output_path}")
print(f"Total findings: {len(findings)}")
print(f"Total reviews analyzed: {total_reviews}")
