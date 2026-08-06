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

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Filter reviews for analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
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
        "claim": f"During the analysis period (Jan 26 - Feb 2, 2026), {len(analysis_reviews)} reviews were collected with an average rating of {avg_rating:.2f} out of 5. Rating distribution: {dict(rating_counts)}",
        "finding_type": "customer_sentiment",
        "metrics": {
            "total_reviews": {
                "value": len(analysis_reviews),
                "unit": "count",
                "numerator": len(analysis_reviews),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(analysis_reviews['rating'].sum(), 2),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_5_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_4_count": {
                "value": int(rating_counts.get(4, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(4, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_3_count": {
                "value": int(rating_counts.get(3, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(3, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_2_count": {
                "value": int(rating_counts.get(2, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(2, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_1_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(analysis_reviews['source'].unique()),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Total reviews in analysis period: {len(analysis_reviews)}",
            f"Sources represented: {', '.join(analysis_reviews['source'].unique())}",
            f"Languages: {', '.join(analysis_reviews['language'].unique())}"
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
        "claim": f"Among {len(analysis_reviews)} reviews collected during the analysis period, {language_counts.get('English', 0)} were in English and {language_counts.get('Arabic', 0)} were in Arabic, providing bilingual coverage of customer feedback.",
        "finding_type": "data_coverage",
        "metrics": {
            "english_reviews": {
                "value": int(language_counts.get('English', 0)),
                "unit": "count",
                "numerator": int(language_counts.get('English', 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": int(language_counts.get('Arabic', 0)),
                "unit": "count",
                "numerator": int(language_counts.get('Arabic', 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(analysis_reviews['source'].unique()),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Total reviews analyzed: {len(analysis_reviews)}",
            f"English reviews: {language_counts.get('English', 0)}",
            f"Arabic reviews: {language_counts.get('Arabic', 0)}"
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
        finding3 = {
            "title": "Review Topic Analysis (Jan 26 - Feb 2, 2026)",
            "claim": f"Among {len(analysis_reviews)} reviews analyzed (Jan 26 - Feb 2, 2026), keyword analysis identified topics in {len(reviews_with_topics)} reviews. The most frequently mentioned topics were: service ({service_count} mentions), taste ({taste_count} mentions), and quality ({quality_count} mentions). Note: {len(analysis_reviews) - len(reviews_with_text)} reviews had no text content, and {len(reviews_with_text) - len(reviews_with_topics)} reviews contained no extractable topics using keyword matching. Analysis covered both English and Arabic text.",
            "finding_type": "topic_analysis",
            "metrics": {
                "reviews_with_text": {
                    "value": len(reviews_with_text),
                    "unit": "count",
                    "numerator": len(reviews_with_text),
                    "denominator": len(analysis_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "reviews_with_topics": {
                    "value": len(reviews_with_topics),
                    "unit": "count",
                    "numerator": len(reviews_with_topics),
                    "denominator": len(reviews_with_text),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "service_mentions": {
                    "value": service_count,
                    "unit": "count",
                    "numerator": service_count,
                    "denominator": len(reviews_with_text),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "taste_mentions": {
                    "value": taste_count,
                    "unit": "count",
                    "numerator": taste_count,
                    "denominator": len(reviews_with_text),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "quality_mentions": {
                    "value": quality_count,
                    "unit": "count",
                    "numerator": quality_count,
                    "denominator": len(reviews_with_text),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": list(analysis_reviews['source'].unique()),
            "sample_size": len(reviews_with_text),
            "coverage_notes": [
                f"Total reviews in period: {len(analysis_reviews)}",
                f"Reviews with text content: {len(reviews_with_text)}",
                f"Reviews with extractable topics: {len(reviews_with_topics)}",
                f"Reviews without extractable topics: {len(reviews_with_text) - len(reviews_with_topics)}",
                f"Languages analyzed: {', '.join(reviews_with_text['language'].unique())}"
            ],
            "assumptions": [
                "Empty or missing review text is excluded from topic analysis",
                "Keyword matching in English and Arabic is used for topic detection",
                "A single review may contain multiple topics",
                "Keyword matching may miss nuanced or implicit topic references"
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
