import os
import json
import pandas as pd
from collections import Counter
from datetime import datetime

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
analysis_start = "2026-01-19T00:00:00+03:00"
analysis_end = "2026-01-26T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Convert reviews date to datetime and handle timezone awareness
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Ensure both are timezone-aware or both are timezone-naive for comparison
if reviews_df['date'].dt.tz is None:
    # If reviews_df is naive, make comparison datetimes naive
    analysis_start_dt = analysis_start_dt.tz_localize(None)
    analysis_end_dt = analysis_end_dt.tz_localize(None)
else:
    # If reviews_df is aware, make comparison datetimes aware
    if analysis_start_dt.tz is None:
        analysis_start_dt = analysis_start_dt.tz_localize('UTC').tz_convert(reviews_df['date'].dt.tz.iloc[0])
        analysis_end_dt = analysis_end_dt.tz_localize('UTC').tz_convert(reviews_df['date'].dt.tz.iloc[0])

# Filter reviews for analysis period
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

findings = []

# Finding 1: Rating Distribution and Average
if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Language breakdown
    language_counts = reviews_analysis['language'].value_counts()
    
    finding1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During the analysis period (Jan 19-26, 2026), the average rating across {len(reviews_analysis)} reviews was {avg_rating:.2f} out of 5, with {int(rating_counts.get(5, 0))} five-star and {int(rating_counts.get(1, 0))} one-star reviews.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "five_star_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "one_star_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Reviews from {len(reviews_analysis['source'].unique())} source(s)",
            f"Language distribution: {dict(language_counts)}",
            "Analysis period: 2026-01-19 to 2026-01-26"
        ],
        "assumptions": [
            "Rating values are valid integers between 1-5",
            "Date field represents review submission date",
            "All reviews in artifact are from the specified sources"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Sentiment/Topic Classification by Language
if len(reviews_analysis) > 0:
    # Separate by language
    english_reviews = reviews_analysis[reviews_analysis['language'] == 'en'].copy()
    arabic_reviews = reviews_analysis[reviews_analysis['language'] == 'ar'].copy()
    
    # Analyze text content for topics (simple keyword-based approach)
    topics_en = []
    topics_ar = []
    
    # English topic keywords
    en_keywords = {
        'quality': ['quality', 'fresh', 'good', 'excellent', 'great', 'best'],
        'service': ['service', 'staff', 'friendly', 'fast', 'slow', 'rude'],
        'price': ['price', 'expensive', 'cheap', 'value', 'cost'],
        'taste': ['taste', 'flavor', 'delicious', 'bland', 'sweet'],
        'cleanliness': ['clean', 'dirty', 'hygiene', 'sanitary']
    }
    
    # Arabic topic keywords
    ar_keywords = {
        'quality': ['جودة', 'طازة', 'ممتاز', 'رائع'],
        'service': ['خدمة', 'موظفين', 'ودود', 'سريع', 'بطيء'],
        'price': ['سعر', 'غالي', 'رخيص', 'قيمة'],
        'taste': ['طعم', 'لذيذ', 'مملح', 'حلو'],
        'cleanliness': ['نظيف', 'وسخ', 'نظافة']
    }
    
    for idx, row in english_reviews.iterrows():
        text = str(row['text']).lower() if pd.notna(row['text']) else ""
        for topic, keywords in en_keywords.items():
            if any(kw in text for kw in keywords):
                topics_en.append(topic)
    
    for idx, row in arabic_reviews.iterrows():
        text = str(row['text']) if pd.notna(row['text']) else ""
        for topic, keywords in ar_keywords.items():
            if any(kw in text for kw in keywords):
                topics_ar.append(topic)
    
    # Count topics
    en_topic_counts = Counter(topics_en) if topics_en else {}
    ar_topic_counts = Counter(topics_ar) if topics_ar else {}
    
    if en_topic_counts or ar_topic_counts:
        finding2 = {
            "title": "Review Topics and Sentiment Themes",
            "claim": f"Among {len(english_reviews)} English reviews, the most frequent topics were {dict(en_topic_counts.most_common(3)) if en_topic_counts else 'none identified'}. Among {len(arabic_reviews)} Arabic reviews, topics included {dict(ar_topic_counts.most_common(3)) if ar_topic_counts else 'none identified'}.",
            "finding_type": "topic_analysis",
            "metrics": {
                "english_reviews_count": {
                    "value": len(english_reviews),
                    "unit": "count",
                    "numerator": len(english_reviews),
                    "denominator": len(reviews_analysis),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "arabic_reviews_count": {
                    "value": len(arabic_reviews),
                    "unit": "count",
                    "numerator": len(arabic_reviews),
                    "denominator": len(reviews_analysis),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "top_english_topic": {
                    "value": en_topic_counts.most_common(1)[0][0] if en_topic_counts else None,
                    "unit": "topic",
                    "numerator": en_topic_counts.most_common(1)[0][1] if en_topic_counts else None,
                    "denominator": len(english_reviews) if len(english_reviews) > 0 else None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "top_arabic_topic": {
                    "value": ar_topic_counts.most_common(1)[0][0] if ar_topic_counts else None,
                    "unit": "topic",
                    "numerator": ar_topic_counts.most_common(1)[0][1] if ar_topic_counts else None,
                    "denominator": len(arabic_reviews) if len(arabic_reviews) > 0 else None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": list(reviews_analysis['source'].unique()),
            "sample_size": len(reviews_analysis),
            "coverage_notes": [
                f"English reviews: {len(english_reviews)}",
                f"Arabic reviews: {len(arabic_reviews)}",
                "Topic identification based on keyword matching in original language",
                "Multiple topics may be present in single review"
            ],
            "assumptions": [
                "Keyword matching is a proxy for topic presence",
                "Language field accurately reflects review language",
                "Text field contains substantive content for analysis"
            ],
            "confidence": 0.75
        }
        findings.append(finding2)

# Finding 3: Low Rating Reviews Analysis
low_rating_reviews = reviews_analysis[reviews_analysis['rating'] <= 2].copy()

if len(low_rating_reviews) > 0:
    low_rating_by_source = low_rating_reviews['source'].value_counts()
    low_rating_by_language = low_rating_reviews['language'].value_counts()
    
    finding3 = {
        "title": "Low Rating Reviews (1-2 Stars)",
        "claim": f"During the analysis period, {len(low_rating_reviews)} reviews ({len(low_rating_reviews)/len(reviews_analysis)*100:.1f}% of total) were rated 1-2 stars. These came from {len(low_rating_by_source)} source(s), with {dict(low_rating_by_language)} language distribution.",
        "finding_type": "low_rating_analysis",
        "metrics": {
            "low_rating_count": {
                "value": len(low_rating_reviews),
                "unit": "count",
                "numerator": len(low_rating_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "low_rating_percentage": {
                "value": round(len(low_rating_reviews)/len(reviews_analysis)*100, 1),
                "unit": "percent",
                "numerator": len(low_rating_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(low_rating_reviews['source'].unique()),
        "sample_size": len(low_rating_reviews),
        "coverage_notes": [
            f"Low rating sources: {dict(low_rating_by_source)}",
            f"Language distribution in low ratings: {dict(low_rating_by_language)}",
            "Low ratings defined as 1-2 stars"
        ],
        "assumptions": [
            "Rating scale is 1-5 stars",
            "Low ratings (1-2) indicate dissatisfaction",
            "Source field accurately identifies review platform"
        ],
        "confidence": 0.90
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
