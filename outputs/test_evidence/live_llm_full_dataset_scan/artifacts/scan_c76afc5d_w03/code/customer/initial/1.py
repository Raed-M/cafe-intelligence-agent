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

# Load artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Define analysis period
analysis_start = "2026-01-26"
analysis_end = "2026-02-02"

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
        'taste': ['taste', 'flavor', 'flavour', 'delicious', 'good', 'bad', 'sweet', 'bitter', 'طعم', 'لذيذ', 'سيء'],
        'quality': ['quality', 'fresh', 'stale', 'old', 'new', 'جودة', 'طازة', 'قديم'],
        'service': ['service', 'staff', 'friendly', 'rude', 'slow', 'fast', 'خدمة', 'موظف', 'سريع', 'بطيء'],
        'price': ['price', 'expensive', 'cheap', 'cost', 'value', 'سعر', 'غالي', 'رخيص'],
        'temperature': ['hot', 'cold', 'warm', 'ice', 'iced', 'حار', 'بارد', 'دافئ'],
        'portion': ['portion', 'size', 'small', 'large', 'big', 'حجم', 'صغير', 'كبير'],
        'cleanliness': ['clean', 'dirty', 'hygiene', 'sanitary', 'نظيف', 'وسخ', 'نظافة'],
        'ambiance': ['ambiance', 'atmosphere', 'cozy', 'comfortable', 'noise', 'جو', 'مريح', 'ضوضاء']
    }
    
    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                topics.append(topic)
                break
    
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

# Prepare findings
findings = []

# Finding 1: Rating Distribution and Average
if total_reviews > 0:
    finding1 = {
        "title": "Review Rating Distribution (Jan 26 - Feb 2, 2026)",
        "claim": f"Average rating is {avg_rating:.2f} out of 5 across {total_reviews} reviews. Rating distribution shows {rating_counts.get(5, 0)} 5-star, {rating_counts.get(4, 0)} 4-star, {rating_counts.get(3, 0)} 3-star, {rating_counts.get(2, 0)} 2-star, and {rating_counts.get(1, 0)} 1-star reviews.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(reviews_analysis['rating'].sum(), 2),
                "denominator": total_reviews,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "total_reviews": {
                "value": total_reviews,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "five_star_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "four_star_count": {
                "value": int(rating_counts.get(4, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "three_star_count": {
                "value": int(rating_counts.get(3, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "two_star_count": {
                "value": int(rating_counts.get(2, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "one_star_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            }
        },
        "source_names": ["reviews"],
        "sample_size": total_reviews,
        "coverage_notes": [
            f"Analysis period: 2026-01-26 to 2026-02-02",
            f"Language distribution: {dict(language_counts)}",
            "All reviews with valid ratings included"
        ],
        "assumptions": [
            "Rating scale is 1-5 stars",
            "All reviews in the period are included",
            "No filtering by source or language"
        ],
        "confidence": 0.95 if total_reviews >= 10 else 0.7
    }
    findings.append(finding1)

# Finding 2: Sentiment Distribution
if total_reviews > 0:
    positive_count = sentiment_counts.get('positive', 0)
    neutral_count = sentiment_counts.get('neutral', 0)
    negative_count = sentiment_counts.get('negative', 0)
    
    finding2 = {
        "title": "Sentiment Distribution by Rating (Jan 26 - Feb 2, 2026)",
        "claim": f"Sentiment analysis shows {positive_count} positive reviews (rating ≥4), {neutral_count} neutral reviews (rating=3), and {negative_count} negative reviews (rating <3) out of {total_reviews} total reviews.",
        "finding_type": "sentiment_distribution",
        "metrics": {
            "positive_reviews": {
                "value": positive_count,
                "unit": "count",
                "numerator": positive_count,
                "denominator": total_reviews,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "neutral_reviews": {
                "value": neutral_count,
                "unit": "count",
                "numerator": neutral_count,
                "denominator": total_reviews,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "negative_reviews": {
                "value": negative_count,
                "unit": "count",
                "numerator": negative_count,
                "denominator": total_reviews,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            },
            "positive_percentage": {
                "value": round((positive_count / total_reviews * 100) if total_reviews > 0 else 0, 1),
                "unit": "percent",
                "numerator": positive_count,
                "denominator": total_reviews,
                "period_start": "2026-01-26T00:00:00+03:00",
                "period_end": "2026-02-02T00:00:00+03:00"
            }
        },
        "source_names": ["reviews"],
        "sample_size": total_reviews,
        "coverage_notes": [
            f"Analysis period: 2026-01-26 to 2026-02-02",
            f"Positive defined as rating ≥4, Neutral as rating=3, Negative as rating <3",
            "All reviews with valid ratings included"
        ],
        "assumptions": [
            "Rating-based sentiment classification",
            "No text analysis for sentiment override",
            "All reviews equally weighted"
        ],
        "confidence": 0.95 if total_reviews >= 10 else 0.7
    }
    findings.append(finding2)

# Finding 3: Top Topics Mentioned
if len(topic_counts) > 0:
    top_topics = topic_counts.most_common(3)
    
    topic_metrics = {}
    for topic, count in top_topics:
        topic_metrics[f"{topic}_mentions"] = {
            "value": count,
            "unit": "count",
            "numerator": count,
            "denominator": len(reviews_analysis),
            "period_start": "2026-01-26T00:00:00+03:00",
            "period_end": "2026-02-02T00:00:00+03:00"
        }
    
    top_topics_str = ", ".join([f"{topic} ({count} mentions)" for topic, count in top_topics])
    
    finding3 = {
        "title": "Most Mentioned Topics in Reviews (Jan 26 - Feb 2, 2026)",
        "claim": f"Top topics mentioned in reviews are: {top_topics_str}. These topics were identified through keyword analysis of review text in both English and Arabic.",
        "finding_type": "topic_frequency",
        "metrics": topic_metrics,
        "source_names": ["reviews"],
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: 2026-01-26 to 2026-02-02",
            f"Total reviews analyzed: {len(reviews_analysis)}",
            f"Reviews with extractable topics: {len([r for r in reviews_analysis['topics'] if len(r) > 0])}",
            "Topic extraction based on keyword matching in English and Arabic"
        ],
        "assumptions": [
            "Topic keywords are representative of actual topics",
            "Keyword matching is sufficient for topic identification",
            "Multiple topics can be mentioned in a single review",
            "Empty or missing review text is excluded from topic analysis"
        ],
        "confidence": 0.75 if len(reviews_analysis) >= 10 else 0.5
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
print(f"Total findings: {len(findings)}")
print(f"Total reviews analyzed: {total_reviews}")
