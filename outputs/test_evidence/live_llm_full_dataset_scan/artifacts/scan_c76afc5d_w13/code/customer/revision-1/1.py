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

# Define analysis period
analysis_start = "2026-04-06T00:00:00+03:00"
analysis_end = "2026-04-13T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Filter reviews to analysis period
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

# Initialize findings list
findings = []

# ============================================================================
# FINDING 1: Rating Distribution and Average
# ============================================================================

if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Get source names from analysis period reviews
    source_names = sorted(reviews_analysis['source'].unique().tolist())
    
    # Get language distribution
    language_dist = reviews_analysis['language'].value_counts().to_dict()
    
    finding_1 = {
        "title": "Rating Distribution and Average (Analysis Period)",
        "claim": f"During the analysis period ({analysis_start} to {analysis_end}), reviews averaged {avg_rating:.2f} stars with {len(reviews_analysis)} total reviews across {len(source_names)} source(s).",
        "finding_type": "customer_satisfaction",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": float(reviews_analysis['rating'].sum()),
                "denominator": len(reviews_analysis),
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
            "rating_5_star_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_4_star_count": {
                "value": int(rating_counts.get(4, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_3_star_count": {
                "value": int(rating_counts.get(3, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_2_star_count": {
                "value": int(rating_counts.get(2, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_1_star_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "language_english_count": {
                "value": int(language_dist.get('en', 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "language_arabic_count": {
                "value": int(language_dist.get('ar', 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Language coverage: English={language_dist.get('en', 0)}, Arabic={language_dist.get('ar', 0)}",
            f"Sources included: {', '.join(source_names)}"
        ],
        "assumptions": [
            "Rating values are numeric and valid",
            "Date filtering uses UTC+3 timezone as specified",
            "All reviews in the artifact are included without exclusion"
        ],
        "confidence": 1.0
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Sentiment/Topic Classification (English Reviews)
# ============================================================================

english_reviews = reviews_analysis[reviews_analysis['language'] == 'en'].copy()

if len(english_reviews) > 0:
    # Simple keyword-based sentiment classification for English
    positive_keywords = ['excellent', 'great', 'good', 'amazing', 'love', 'perfect', 'best', 'wonderful', 'fantastic', 'delicious', 'fresh', 'friendly', 'quick', 'nice']
    negative_keywords = ['bad', 'poor', 'terrible', 'awful', 'hate', 'worst', 'slow', 'rude', 'cold', 'stale', 'dirty', 'expensive', 'disappointed', 'disappointing']
    
    def classify_sentiment_en(text):
        if pd.isna(text) or text == '':
            return 'neutral'
        text_lower = str(text).lower()
        pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
        neg_count = sum(1 for kw in negative_keywords if kw in text_lower)
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'
    
    english_reviews['sentiment'] = english_reviews['text'].apply(classify_sentiment_en)
    sentiment_dist = english_reviews['sentiment'].value_counts().to_dict()
    
    finding_2 = {
        "title": "English Review Sentiment Distribution",
        "claim": f"Among {len(english_reviews)} English-language reviews in the analysis period, sentiment classification shows {sentiment_dist.get('positive', 0)} positive, {sentiment_dist.get('negative', 0)} negative, and {sentiment_dist.get('neutral', 0)} neutral reviews.",
        "finding_type": "sentiment_analysis",
        "metrics": {
            "english_reviews_total": {
                "value": len(english_reviews),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_positive_sentiment": {
                "value": int(sentiment_dist.get('positive', 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_negative_sentiment": {
                "value": int(sentiment_dist.get('negative', 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_neutral_sentiment": {
                "value": int(sentiment_dist.get('neutral', 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_positive_percentage": {
                "value": round(100 * sentiment_dist.get('positive', 0) / len(english_reviews), 1) if len(english_reviews) > 0 else 0,
                "unit": "percent",
                "numerator": sentiment_dist.get('positive', 0),
                "denominator": len(english_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": sorted(english_reviews['source'].unique().tolist()),
        "sample_size": len(english_reviews),
        "coverage_notes": [
            f"English reviews only; {len(english_reviews)} of {len(reviews_analysis)} total reviews",
            "Sentiment classification based on keyword matching",
            f"Sources: {', '.join(sorted(english_reviews['source'].unique().tolist()))}"
        ],
        "assumptions": [
            "Sentiment classification uses keyword matching on review text",
            "Empty or null review texts are classified as neutral",
            "Keyword list is fixed and language-specific"
        ],
        "confidence": 0.6
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Arabic Review Sentiment Distribution
# ============================================================================

arabic_reviews = reviews_analysis[reviews_analysis['language'] == 'ar'].copy()

if len(arabic_reviews) > 0:
    # Simple keyword-based sentiment classification for Arabic
    positive_keywords_ar = ['ممتاز', 'رائع', 'جيد', 'لذيذ', 'طازج', 'سريع', 'لطيف', 'رائع', 'مميز', 'ممتازة']
    negative_keywords_ar = ['سيء', 'سيئة', 'رهيب', 'بطيء', 'بطيئة', 'غالي', 'غالية', 'خيبة', 'مخيب', 'مخيبة']
    
    def classify_sentiment_ar(text):
        if pd.isna(text) or text == '':
            return 'neutral'
        text_lower = str(text).lower()
        pos_count = sum(1 for kw in positive_keywords_ar if kw in text_lower)
        neg_count = sum(1 for kw in negative_keywords_ar if kw in text_lower)
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'
    
    arabic_reviews['sentiment'] = arabic_reviews['text'].apply(classify_sentiment_ar)
    sentiment_dist_ar = arabic_reviews['sentiment'].value_counts().to_dict()
    
    finding_3 = {
        "title": "Arabic Review Sentiment Distribution",
        "claim": f"Among {len(arabic_reviews)} Arabic-language reviews in the analysis period, sentiment classification shows {sentiment_dist_ar.get('positive', 0)} positive, {sentiment_dist_ar.get('negative', 0)} negative, and {sentiment_dist_ar.get('neutral', 0)} neutral reviews.",
        "finding_type": "sentiment_analysis",
        "metrics": {
            "arabic_reviews_total": {
                "value": len(arabic_reviews),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_positive_sentiment": {
                "value": int(sentiment_dist_ar.get('positive', 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_negative_sentiment": {
                "value": int(sentiment_dist_ar.get('negative', 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_neutral_sentiment": {
                "value": int(sentiment_dist_ar.get('neutral', 0)),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_positive_percentage": {
                "value": round(100 * sentiment_dist_ar.get('positive', 0) / len(arabic_reviews), 1) if len(arabic_reviews) > 0 else 0,
                "unit": "percent",
                "numerator": sentiment_dist_ar.get('positive', 0),
                "denominator": len(arabic_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": sorted(arabic_reviews['source'].unique().tolist()),
        "sample_size": len(arabic_reviews),
        "coverage_notes": [
            f"Arabic reviews only; {len(arabic_reviews)} of {len(reviews_analysis)} total reviews",
            "Sentiment classification based on Arabic keyword matching",
            f"Sources: {', '.join(sorted(arabic_reviews['source'].unique().tolist()))}"
        ],
        "assumptions": [
            "Sentiment classification uses Arabic keyword matching on review text",
            "Empty or null review texts are classified as neutral",
            "Keyword list is fixed and language-specific"
        ],
        "confidence": 0.6
    }
    findings.append(finding_3)

# ============================================================================
# Prepare output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
