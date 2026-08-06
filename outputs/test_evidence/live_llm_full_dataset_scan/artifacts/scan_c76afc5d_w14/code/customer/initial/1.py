import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter

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
analysis_start = "2026-04-13T00:00:00+03:00"
analysis_end = "2026-04-20T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Filter reviews for analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
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

# Topic extraction from text (simple keyword-based)
topics = {
    'quality': ['quality', 'جودة', 'excellent', 'great', 'good', 'bad', 'poor'],
    'taste': ['taste', 'flavor', 'طعم', 'delicious', 'sweet', 'bitter'],
    'service': ['service', 'staff', 'خدمة', 'friendly', 'rude', 'slow'],
    'price': ['price', 'expensive', 'cheap', 'سعر', 'value'],
    'temperature': ['hot', 'cold', 'warm', 'iced', 'درجة الحرارة'],
    'cleanliness': ['clean', 'dirty', 'hygiene', 'نظافة']
}

def extract_topics(text):
    if pd.isna(text):
        return []
    text_lower = str(text).lower()
    found_topics = []
    for topic, keywords in topics.items():
        for keyword in keywords:
            if keyword in text_lower:
                found_topics.append(topic)
                break
    return found_topics

reviews_analysis['topics'] = reviews_analysis['text'].apply(extract_topics)

# Flatten topics for counting
all_topics = []
for topic_list in reviews_analysis['topics']:
    all_topics.extend(topic_list)
topic_counts = Counter(all_topics)

# Identify reviews with specific sentiments and topics
positive_reviews = reviews_analysis[reviews_analysis['sentiment'] == 'positive']
negative_reviews = reviews_analysis[reviews_analysis['sentiment'] == 'negative']

# Find most common topics in positive and negative reviews
positive_topics = []
for topic_list in positive_reviews['topics']:
    positive_topics.extend(topic_list)
positive_topic_counts = Counter(positive_topics)

negative_topics = []
for topic_list in negative_reviews['topics']:
    negative_topics.extend(topic_list)
negative_topic_counts = Counter(negative_topics)

# Prepare findings
findings = []

# Finding 1: Rating Distribution and Average
if total_reviews > 0:
    finding1 = {
        "title": "Customer Rating Distribution and Average",
        "claim": f"During the analysis period (2026-04-13 to 2026-04-20), customers provided {total_reviews} reviews with an average rating of {avg_rating:.2f} out of 5. The distribution shows {sentiment_counts.get('positive', 0)} positive (4-5 stars), {sentiment_counts.get('neutral', 0)} neutral (3 stars), and {sentiment_counts.get('negative', 0)} negative (1-2 stars) reviews.",
        "finding_type": "customer_satisfaction",
        "metrics": {
            "total_reviews": {
                "value": total_reviews,
                "unit": "count",
                "numerator": total_reviews,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(avg_rating * total_reviews, 2),
                "denominator": total_reviews,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "positive_reviews": {
                "value": sentiment_counts.get('positive', 0),
                "unit": "count",
                "numerator": sentiment_counts.get('positive', 0),
                "denominator": total_reviews,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_reviews": {
                "value": sentiment_counts.get('negative', 0),
                "unit": "count",
                "numerator": sentiment_counts.get('negative', 0),
                "denominator": total_reviews,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": total_reviews,
        "coverage_notes": [
            f"Reviews collected from {len(reviews_analysis['source'].unique())} source(s)",
            f"Language distribution: {dict(language_counts)}",
            f"Rating distribution: {dict(rating_counts)}"
        ],
        "assumptions": [
            "Ratings are on a 1-5 scale",
            "Positive sentiment defined as 4-5 stars, neutral as 3 stars, negative as 1-2 stars",
            "All reviews in the dataset are valid and complete"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Top Topics in Reviews
if len(topic_counts) > 0:
    top_topics = topic_counts.most_common(3)
    top_topics_str = ", ".join([f"{topic} ({count})" for topic, count in top_topics])
    
    finding2 = {
        "title": "Most Discussed Topics in Customer Reviews",
        "claim": f"The most frequently mentioned topics in customer reviews during the analysis period are: {top_topics_str}. These topics appear across {sum(dict(topic_counts).values())} total topic mentions in {total_reviews} reviews.",
        "finding_type": "voice_of_customer",
        "metrics": {
            "total_topic_mentions": {
                "value": sum(dict(topic_counts).values()),
                "unit": "count",
                "numerator": sum(dict(topic_counts).values()),
                "denominator": total_reviews,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": total_reviews,
        "coverage_notes": [
            f"Topic extraction based on keyword matching in review text",
            f"Top 3 topics: {dict(top_topics)}",
            f"Total unique topics identified: {len(topic_counts)}"
        ],
        "assumptions": [
            "Topics identified through keyword matching in both English and Arabic",
            "A review may mention multiple topics",
            "Keyword list is representative but not exhaustive"
        ],
        "confidence": 0.75
    }
    findings.append(finding2)

# Finding 3: Sentiment-Topic Correlation
if len(positive_topic_counts) > 0 and len(negative_topic_counts) > 0:
    top_positive_topic = positive_topic_counts.most_common(1)[0] if positive_topic_counts else None
    top_negative_topic = negative_topic_counts.most_common(1)[0] if negative_topic_counts else None
    
    if top_positive_topic and top_negative_topic:
        finding3 = {
            "title": "Topic Sentiment Association",
            "claim": f"In positive reviews (4-5 stars, n={len(positive_reviews)}), the most discussed topic is '{top_positive_topic[0]}' ({top_positive_topic[1]} mentions). In negative reviews (1-2 stars, n={len(negative_reviews)}), the most discussed topic is '{top_negative_topic[0]}' ({top_negative_topic[1]} mentions). This suggests different aspects drive satisfaction vs. dissatisfaction.",
            "finding_type": "sentiment_analysis",
            "metrics": {
                "positive_reviews_count": {
                    "value": len(positive_reviews),
                    "unit": "count",
                    "numerator": len(positive_reviews),
                    "denominator": total_reviews,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "negative_reviews_count": {
                    "value": len(negative_reviews),
                    "unit": "count",
                    "numerator": len(negative_reviews),
                    "denominator": total_reviews,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "top_positive_topic": {
                    "value": top_positive_topic[0],
                    "unit": "topic",
                    "numerator": top_positive_topic[1],
                    "denominator": len(positive_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "top_negative_topic": {
                    "value": top_negative_topic[0],
                    "unit": "topic",
                    "numerator": top_negative_topic[1],
                    "denominator": len(negative_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": list(reviews_analysis['source'].unique()),
            "sample_size": total_reviews,
            "coverage_notes": [
                f"Positive reviews: {len(positive_reviews)} ({len(positive_reviews)/total_reviews*100:.1f}%)",
                f"Negative reviews: {len(negative_reviews)} ({len(negative_reviews)/total_reviews*100:.1f}%)",
                f"Topic distribution differs between sentiment groups"
            ],
            "assumptions": [
                "Sentiment classification based on rating thresholds",
                "Topic extraction through keyword matching",
                "Correlation observed but causation not established"
            ],
            "confidence": 0.70
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
print(f"Average rating: {avg_rating:.2f}")
