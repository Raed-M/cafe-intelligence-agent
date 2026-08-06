import os
import json
import pandas as pd
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

# Define analysis period
analysis_start = "2026-01-26T00:00:00+03:00"
analysis_end = "2026-02-02T00:00:00+03:00"

# Convert to datetime for filtering, handling timezone awareness
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Convert reviews date to datetime and handle timezone
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# If the dataframe dates are timezone-naive, localize them to UTC then convert to +03:00
# If they are timezone-aware, convert to +03:00
if reviews_df['date'].dt.tz is None:
    # Assume UTC if naive
    reviews_df['date'] = reviews_df['date'].dt.tz_localize('UTC').dt.tz_convert('+03:00')
else:
    # Convert to +03:00 if already aware
    reviews_df['date'] = reviews_df['date'].dt.tz_convert('+03:00')

# Also convert analysis boundaries to +03:00 for comparison
if analysis_start_dt.tz is None:
    analysis_start_dt = analysis_start_dt.tz_localize('UTC').tz_convert('+03:00')
else:
    analysis_start_dt = analysis_start_dt.tz_convert('+03:00')

if analysis_end_dt.tz is None:
    analysis_end_dt = analysis_end_dt.tz_localize('UTC').tz_convert('+03:00')
else:
    analysis_end_dt = analysis_end_dt.tz_convert('+03:00')

# Filter reviews for analysis period
analysis_reviews = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

findings = []

# Finding 1: Rating Distribution and Average
if len(analysis_reviews) > 0:
    rating_counts = analysis_reviews['rating'].value_counts().sort_index()
    avg_rating = analysis_reviews['rating'].mean()
    
    finding1 = {
        "title": "Review Rating Distribution (Jan 26 - Feb 2, 2026)",
        "claim": f"During the analysis period (Jan 26 - Feb 2, 2026), {int(len(analysis_reviews))} reviews were collected with an average rating of {float(avg_rating):.2f} out of 5. Rating distribution: {dict((int(k), int(v)) for k, v in rating_counts.items())}",
        "finding_type": "customer_sentiment",
        "metrics": {
            "total_reviews": {
                "value": int(len(analysis_reviews)),
                "unit": "count",
                "numerator": int(len(analysis_reviews)),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "average_rating": {
                "value": float(round(avg_rating, 2)),
                "unit": "stars",
                "numerator": float(round(analysis_reviews['rating'].sum(), 2)),
                "denominator": int(len(analysis_reviews)),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_5_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": int(len(analysis_reviews)),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_4_count": {
                "value": int(rating_counts.get(4, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(4, 0)),
                "denominator": int(len(analysis_reviews)),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_3_count": {
                "value": int(rating_counts.get(3, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(3, 0)),
                "denominator": int(len(analysis_reviews)),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_2_count": {
                "value": int(rating_counts.get(2, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(2, 0)),
                "denominator": int(len(analysis_reviews)),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_1_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": int(len(analysis_reviews)),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": sorted(list(analysis_reviews['source'].unique())),
        "sample_size": int(len(analysis_reviews)),
        "coverage_notes": [
            f"Total reviews in analysis period: {int(len(analysis_reviews))}",
            f"Sources represented: {', '.join(sorted(analysis_reviews['source'].unique()))}",
            f"Languages: {', '.join(sorted(analysis_reviews['language'].unique()))}"
        ],
        "assumptions": [
            "Rating values are treated as numeric and valid",
            "All reviews in the period are included regardless of text content"
        ],
        "confidence": 1.0
    }
    findings.append(finding1)

# Finding 2: Language Distribution
if len(analysis_reviews) > 0:
    language_counts = analysis_reviews['language'].value_counts()
    
    finding2 = {
        "title": "Review Language Distribution (Jan 26 - Feb 2, 2026)",
        "claim": f"Among {int(len(analysis_reviews))} reviews collected during the analysis period, {int(language_counts.get('English', 0))} were in English and {int(language_counts.get('Arabic', 0))} were in Arabic, providing bilingual coverage of customer feedback.",
        "finding_type": "data_coverage",
        "metrics": {
            "english_reviews": {
                "value": int(language_counts.get('English', 0)),
                "unit": "count",
                "numerator": int(language_counts.get('English', 0)),
                "denominator": int(len(analysis_reviews)),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": int(language_counts.get('Arabic', 0)),
                "unit": "count",
                "numerator": int(language_counts.get('Arabic', 0)),
                "denominator": int(len(analysis_reviews)),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": sorted(list(analysis_reviews['source'].unique())),
        "sample_size": int(len(analysis_reviews)),
        "coverage_notes": [
            f"Total reviews analyzed: {int(len(analysis_reviews))}",
            f"English reviews: {int(language_counts.get('English', 0))}",
            f"Arabic reviews: {int(language_counts.get('Arabic', 0))}"
        ],
        "assumptions": [
            "Language field accurately reflects the language of review text",
            "Reviews are classified as either English or Arabic"
        ],
        "confidence": 1.0
    }
    findings.append(finding2)

# Finding 3: Topic Analysis with Keyword Matching
if len(analysis_reviews) > 0:
    # Define keywords for topic detection in both languages
    service_keywords_en = ['service', 'staff', 'waiter', 'waitress', 'server', 'attendant', 'friendly', 'rude', 'slow', 'fast', 'quick']
    service_keywords_ar = ['خدمة', 'موظف', 'عامل', 'ودود', 'سريع', 'بطيء']
    
    taste_keywords_en = ['taste', 'flavor', 'delicious', 'tasty', 'bland', 'sweet', 'bitter', 'sour', 'fresh']
    taste_keywords_ar = ['طعم', 'لذيذ', 'طازج', 'حلو', 'مر', 'حامض']
    
    quality_keywords_en = ['quality', 'fresh', 'clean', 'dirty', 'hygiene', 'sanitary', 'premium', 'excellent', 'poor']
    quality_keywords_ar = ['جودة', 'نظيف', 'وسخ', 'صحي', 'ممتاز', 'سيء']
    
    # Analyze reviews with text
    reviews_with_text = analysis_reviews[analysis_reviews['text'].notna() & (analysis_reviews['text'].str.len() > 0)].copy()
    
    service_count = 0
    taste_count = 0
    quality_count = 0
    reviews_with_topics = set()
    
    for idx, row in reviews_with_text.iterrows():
        text = str(row['text']).lower()
        language = row['language']
        
        # Check for service mentions
        if language == 'English':
            if any(keyword in text for keyword in service_keywords_en):
                service_count += 1
                reviews_with_topics.add(row['review_id'])
            if any(keyword in text for keyword in taste_keywords_en):
                taste_count += 1
                reviews_with_topics.add(row['review_id'])
            if any(keyword in text for keyword in quality_keywords_en):
                quality_count += 1
                reviews_with_topics.add(row['review_id'])
        elif language == 'Arabic':
            if any(keyword in text for keyword in service_keywords_ar):
                service_count += 1
                reviews_with_topics.add(row['review_id'])
            if any(keyword in text for keyword in taste_keywords_ar):
                taste_count += 1
                reviews_with_topics.add(row['review_id'])
            if any(keyword in text for keyword in quality_keywords_ar):
                quality_count += 1
                reviews_with_topics.add(row['review_id'])
    
    if len(reviews_with_text) > 0:
        reviews_without_topics = len(reviews_with_text) - len(reviews_with_topics)
        finding3 = {
            "title": "Review Topic Analysis (Jan 26 - Feb 2, 2026)",
            "claim": f"Among {int(len(analysis_reviews))} reviews in the analysis period (Jan 26 - Feb 2, 2026), {int(len(reviews_with_text))} contained text content. Keyword analysis identified topics in {int(len(reviews_with_topics))} of these reviews. The most frequently mentioned topics were: service ({int(service_count)} mentions), taste ({int(taste_count)} mentions), and quality ({int(quality_count)} mentions). {int(reviews_without_topics)} reviews contained no extractable topics using the keyword matching approach. Note: {int(len(analysis_reviews) - len(reviews_with_text))} reviews had no text content and were excluded. Analysis covered both English and Arabic text.",
            "finding_type": "topic_analysis",
            "metrics": {
                "reviews_with_text": {
                    "value": int(len(reviews_with_text)),
                    "unit": "count",
                    "numerator": int(len(reviews_with_text)),
                    "denominator": int(len(analysis_reviews)),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "reviews_with_topics": {
                    "value": int(len(reviews_with_topics)),
                    "unit": "count",
                    "numerator": int(len(reviews_with_topics)),
                    "denominator": int(len(reviews_with_text)),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "reviews_without_topics": {
                    "value": int(reviews_without_topics),
                    "unit": "count",
                    "numerator": int(reviews_without_topics),
                    "denominator": int(len(reviews_with_text)),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "service_mentions": {
                    "value": int(service_count),
                    "unit": "count",
                    "numerator": int(service_count),
                    "denominator": int(len(reviews_with_topics)) if len(reviews_with_topics) > 0 else int(len(reviews_with_text)),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "taste_mentions": {
                    "value": int(taste_count),
                    "unit": "count",
                    "numerator": int(taste_count),
                    "denominator": int(len(reviews_with_topics)) if len(reviews_with_topics) > 0 else int(len(reviews_with_text)),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "quality_mentions": {
                    "value": int(quality_count),
                    "unit": "count",
                    "numerator": int(quality_count),
                    "denominator": int(len(reviews_with_topics)) if len(reviews_with_topics) > 0 else int(len(reviews_with_text)),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": sorted(list(analysis_reviews['source'].unique())),
            "sample_size": int(len(reviews_with_text)),
            "coverage_notes": [
                f"Total reviews in period: {int(len(analysis_reviews))}",
                f"Reviews with text content: {int(len(reviews_with_text))}",
                f"Reviews with extractable topics: {int(len(reviews_with_topics))}",
                f"Reviews without extractable topics: {int(reviews_without_topics)}",
                f"Languages analyzed: {', '.join(sorted(reviews_with_text['language'].unique()))}"
            ],
            "assumptions": [
                "Empty or missing review text is excluded from topic analysis",
                "Keyword matching in English and Arabic is used for topic detection",
                "A single review may contain multiple topics",
                "Keyword matching may miss nuanced or implicit topic references",
                "Manual validation of a sample is recommended to assess precision of keyword matching"
            ],
            "confidence": 0.7
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
