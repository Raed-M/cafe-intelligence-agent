import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import re

# Load input paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Analysis period
analysis_start = "2026-01-26T00:00:00+03:00"
analysis_end = "2026-02-02T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Filter reviews to analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Handle timezone-aware comparison: convert both to UTC or strip timezone
if reviews_df['date'].dt.tz is not None:
    # Reviews have timezone info
    analysis_start_dt = analysis_start_dt.tz_convert('UTC')
    analysis_end_dt = analysis_end_dt.tz_convert('UTC')
    reviews_df['date'] = reviews_df['date'].dt.tz_convert('UTC')
else:
    # Reviews are naive, strip timezone from analysis dates
    analysis_start_dt = analysis_start_dt.tz_localize(None)
    analysis_end_dt = analysis_end_dt.tz_localize(None)

reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

# Compute rating distribution and average
rating_counts = reviews_analysis['rating'].value_counts().sort_index()
avg_rating = reviews_analysis['rating'].mean()
total_reviews_period = len(reviews_analysis)

# Define keyword sets for topic classification (English and Arabic)
service_keywords_en = ['service', 'staff', 'waiter', 'barista', 'friendly', 'rude', 'slow', 'fast', 'attentive', 'helpful']
service_keywords_ar = ['خدمة', 'موظف', 'باريستا', 'ودود', 'وقح', 'بطيء', 'سريع', 'منتبه', 'مفيد']

taste_keywords_en = ['taste', 'flavor', 'delicious', 'bland', 'sweet', 'bitter', 'sour', 'fresh', 'stale', 'good taste']
taste_keywords_ar = ['طعم', 'نكهة', 'لذيذ', 'مملح', 'حلو', 'مرير', 'حامض', 'طازج', 'قديم']

quality_keywords_en = ['quality', 'fresh', 'clean', 'dirty', 'hygiene', 'sanitary', 'premium', 'cheap', 'worth']
quality_keywords_ar = ['جودة', 'طازج', 'نظيف', 'وسخ', 'نظافة', 'صحي', 'فاخر', 'رخيص', 'يستحق']

price_keywords_en = ['price', 'expensive', 'cheap', 'cost', 'value', 'overpriced', 'affordable', 'worth']
price_keywords_ar = ['سعر', 'غالي', 'رخيص', 'تكلفة', 'قيمة', 'مبالغ', 'معقول', 'يستحق']

def extract_topics(text, language):
    """Extract topics from review text based on keywords"""
    if pd.isna(text) or text == '':
        return []
    
    text_lower = str(text).lower()
    topics = []
    
    if language == 'en' or language == 'EN':
        if any(kw in text_lower for kw in service_keywords_en):
            topics.append('service')
        if any(kw in text_lower for kw in taste_keywords_en):
            topics.append('taste')
        if any(kw in text_lower for kw in quality_keywords_en):
            topics.append('quality')
        if any(kw in text_lower for kw in price_keywords_en):
            topics.append('price')
    elif language == 'ar' or language == 'AR':
        if any(kw in text_lower for kw in service_keywords_ar):
            topics.append('service')
        if any(kw in text_lower for kw in taste_keywords_ar):
            topics.append('taste')
        if any(kw in text_lower for kw in quality_keywords_ar):
            topics.append('quality')
        if any(kw in text_lower for kw in price_keywords_ar):
            topics.append('price')
    
    return list(set(topics))

# Extract topics for all reviews in analysis period
reviews_analysis['topics'] = reviews_analysis.apply(
    lambda row: extract_topics(row['text'], row['language']), 
    axis=1
)

# Count reviews with extractable topics
reviews_with_topics = reviews_analysis[reviews_analysis['topics'].apply(len) > 0]
total_with_topics = len(reviews_with_topics)

# Count topic mentions
topic_mentions = Counter()
for topics_list in reviews_with_topics['topics']:
    for topic in topics_list:
        topic_mentions[topic] += 1

# Get top 3 topics
top_topics = topic_mentions.most_common(3)

# Prepare findings
findings = []

# Finding 1: Rating Distribution and Average
if total_reviews_period > 0:
    finding1 = {
        "title": "Review Rating Distribution (Jan 26 - Feb 2, 2026)",
        "claim": f"During the analysis period (Jan 26 - Feb 2, 2026), {total_reviews_period} reviews were collected with an average rating of {avg_rating:.2f} out of 5. Rating distribution: {dict(rating_counts)}",
        "finding_type": "rating_analysis",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(reviews_analysis['rating'].sum(), 2),
                "denominator": total_reviews_period,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_reviews": {
                "value": total_reviews_period,
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": total_reviews_period,
        "coverage_notes": [
            f"Total reviews in analysis period: {total_reviews_period}",
            f"Sources covered: {', '.join(reviews_analysis['source'].unique())}",
            f"Language distribution: {dict(reviews_analysis['language'].value_counts())}"
        ],
        "assumptions": [
            "Rating values are valid integers between 1 and 5",
            "All reviews in the specified date range are included"
        ],
        "confidence": 1.0
    }
    findings.append(finding1)

# Finding 2: Topic Analysis
if total_reviews_period > 0 and len(top_topics) > 0:
    claim_text = f"Among {total_reviews_period} reviews analyzed (Jan 26 - Feb 2, 2026), keyword analysis identified topics in {total_with_topics} reviews. "
    claim_text += f"The most frequently mentioned topics were: "
    topic_parts = []
    for topic, count in top_topics:
        topic_parts.append(f"{topic} ({count} mentions)")
    claim_text += ", ".join(topic_parts) + ". "
    claim_text += f"Note: {total_reviews_period - total_with_topics} reviews contained no extractable topics using the keyword matching approach. Analysis covered both English and Arabic text."
    
    metrics_dict = {}
    for i, (topic, count) in enumerate(top_topics):
        metrics_dict[f"{topic}_mentions"] = {
            "value": count,
            "unit": "mentions",
            "numerator": count,
            "denominator": total_with_topics,
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    
    metrics_dict["reviews_with_topics"] = {
        "value": total_with_topics,
        "unit": "count",
        "numerator": total_with_topics,
        "denominator": total_reviews_period,
        "period_start": analysis_start,
        "period_end": analysis_end
    }
    
    finding2 = {
        "title": "Review Topic Analysis (Jan 26 - Feb 2, 2026)",
        "claim": claim_text,
        "finding_type": "topic_analysis",
        "metrics": metrics_dict,
        "source_names": list(reviews_with_topics['source'].unique()) if total_with_topics > 0 else list(reviews_analysis['source'].unique()),
        "sample_size": total_with_topics,
        "coverage_notes": [
            f"Total reviews analyzed: {total_reviews_period}",
            f"Reviews with extractable topics: {total_with_topics}",
            f"Reviews with no extractable topics: {total_reviews_period - total_with_topics}",
            f"Topic extraction method: keyword matching in English and Arabic",
            f"Language coverage: {dict(reviews_with_topics['language'].value_counts()) if total_with_topics > 0 else 'N/A'}"
        ],
        "assumptions": [
            "Empty or missing review text is excluded from topic analysis",
            "Keyword matching approach may miss nuanced or implicit topic references; manual validation of a sample is recommended to assess precision",
            "Each review can contain multiple topics",
            "Keywords are case-insensitive"
        ],
        "confidence": 0.7
    }
    findings.append(finding2)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
