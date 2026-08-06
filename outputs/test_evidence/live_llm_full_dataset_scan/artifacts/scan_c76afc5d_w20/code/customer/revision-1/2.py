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
analysis_start = "2026-05-25T00:00:00+03:00"
analysis_end = "2026-06-01T00:00:00+03:00"

# Convert to datetime for filtering - handle timezone awareness
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Convert reviews date to datetime and ensure timezone awareness
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# If reviews_df['date'] is tz-naive, localize it to UTC+3
if reviews_df['date'].dt.tz is None:
    reviews_df['date'] = reviews_df['date'].dt.tz_localize('UTC+03:00')
else:
    # If it has timezone info, convert to UTC+3 for consistent comparison
    reviews_df['date'] = reviews_df['date'].dt.tz_convert('UTC+03:00')

# Ensure analysis datetimes are in UTC+3 for comparison
if analysis_start_dt.tz is None:
    analysis_start_dt = analysis_start_dt.tz_localize('UTC+03:00')
if analysis_end_dt.tz is None:
    analysis_end_dt = analysis_end_dt.tz_localize('UTC+03:00')

# Filter reviews to analysis period
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
    
    # Get source names from analysis period
    source_names = sorted(reviews_analysis['source'].unique().tolist())
    
    # Get language distribution
    language_counts = reviews_analysis['language'].value_counts()
    
    finding_1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During {analysis_start} to {analysis_end}, {len(reviews_analysis)} reviews were collected with an average rating of {avg_rating:.2f} out of 5. Rating distribution: {dict(rating_counts)}. Language coverage: {dict(language_counts)}.",
        "finding_type": "rating_distribution",
        "metrics": {
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(reviews_analysis['rating'].sum(), 2),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_5_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_4_count": {
                "value": int(rating_counts.get(4, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(4, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_3_count": {
                "value": int(rating_counts.get(3, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(3, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_2_count": {
                "value": int(rating_counts.get(2, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(2, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_1_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Total reviews in analysis period: {len(reviews_analysis)}",
            f"Language distribution: {dict(language_counts)}",
            f"Sources represented: {', '.join(source_names)}"
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
# FINDING 2: Sentiment/Topic Classification by Language
# ============================================================================

if len(reviews_analysis) > 0:
    # Separate by language
    english_reviews = reviews_analysis[reviews_analysis['language'] == 'en'].copy()
    arabic_reviews = reviews_analysis[reviews_analysis['language'] == 'ar'].copy()
    
    # Simple keyword-based sentiment classification
    positive_keywords_en = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect', 'wonderful', 'fantastic', 'delicious']
    negative_keywords_en = ['bad', 'poor', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting', 'disappointing', 'rude']
    
    positive_keywords_ar = ['جيد', 'رائع', 'ممتاز', 'لذيذ', 'أحب', 'أفضل', 'رائعة', 'جميل']
    negative_keywords_ar = ['سيء', 'سيئة', 'سيئ', 'رهيب', 'مروع', 'سيء', 'خيبة', 'محبط']
    
    def classify_sentiment(text, lang):
        if pd.isna(text) or text == '':
            return 'neutral'
        text_lower = str(text).lower()
        if lang == 'en':
            pos_count = sum(1 for kw in positive_keywords_en if kw in text_lower)
            neg_count = sum(1 for kw in negative_keywords_en if kw in text_lower)
        else:
            pos_count = sum(1 for kw in positive_keywords_ar if kw in text_lower)
            neg_count = sum(1 for kw in negative_keywords_ar if kw in text_lower)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'
    
    english_reviews['sentiment'] = english_reviews.apply(
        lambda row: classify_sentiment(row['text'], 'en'), axis=1
    )
    arabic_reviews['sentiment'] = arabic_reviews.apply(
        lambda row: classify_sentiment(row['text'], 'ar'), axis=1
    )
    
    en_sentiment_counts = english_reviews['sentiment'].value_counts()
    ar_sentiment_counts = arabic_reviews['sentiment'].value_counts()
    
    finding_2 = {
        "title": "Sentiment Distribution by Language",
        "claim": f"English reviews (n={len(english_reviews)}): {dict(en_sentiment_counts)}. Arabic reviews (n={len(arabic_reviews)}): {dict(ar_sentiment_counts)}. Sentiment classification based on keyword matching in original language text.",
        "finding_type": "sentiment_distribution",
        "metrics": {
            "english_reviews_count": {
                "value": len(english_reviews),
                "unit": "count",
                "numerator": len(english_reviews),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_positive_count": {
                "value": int(en_sentiment_counts.get('positive', 0)),
                "unit": "count",
                "numerator": int(en_sentiment_counts.get('positive', 0)),
                "denominator": len(english_reviews) if len(english_reviews) > 0 else None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_negative_count": {
                "value": int(en_sentiment_counts.get('negative', 0)),
                "unit": "count",
                "numerator": int(en_sentiment_counts.get('negative', 0)),
                "denominator": len(english_reviews) if len(english_reviews) > 0 else None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_neutral_count": {
                "value": int(en_sentiment_counts.get('neutral', 0)),
                "unit": "count",
                "numerator": int(en_sentiment_counts.get('neutral', 0)),
                "denominator": len(english_reviews) if len(english_reviews) > 0 else None,
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
            "arabic_positive_count": {
                "value": int(ar_sentiment_counts.get('positive', 0)),
                "unit": "count",
                "numerator": int(ar_sentiment_counts.get('positive', 0)),
                "denominator": len(arabic_reviews) if len(arabic_reviews) > 0 else None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_negative_count": {
                "value": int(ar_sentiment_counts.get('negative', 0)),
                "unit": "count",
                "numerator": int(ar_sentiment_counts.get('negative', 0)),
                "denominator": len(arabic_reviews) if len(arabic_reviews) > 0 else None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_neutral_count": {
                "value": int(ar_sentiment_counts.get('neutral', 0)),
                "unit": "count",
                "numerator": int(ar_sentiment_counts.get('neutral', 0)),
                "denominator": len(arabic_reviews) if len(arabic_reviews) > 0 else None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"English reviews: {len(english_reviews)} ({100*len(english_reviews)/len(reviews_analysis):.1f}% of total)" if len(reviews_analysis) > 0 else "English reviews: 0",
            f"Arabic reviews: {len(arabic_reviews)} ({100*len(arabic_reviews)/len(reviews_analysis):.1f}% of total)" if len(reviews_analysis) > 0 else "Arabic reviews: 0",
            "Sentiment classification uses keyword matching in original language",
            "Empty or missing text classified as neutral"
        ],
        "assumptions": [
            "Keyword lists are representative of positive/negative sentiment in each language",
            "Sentiment is determined by keyword frequency, not context or intensity",
            "Language field is accurate and complete"
        ],
        "confidence": 0.6
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Review Source Distribution
# ============================================================================

if len(reviews_analysis) > 0:
    source_counts = reviews_analysis['source'].value_counts()
    
    finding_3 = {
        "title": "Review Source Distribution",
        "claim": f"Review sources in analysis period: {dict(source_counts)}. Total {len(reviews_analysis)} reviews from {len(source_counts)} distinct sources.",
        "finding_type": "source_distribution",
        "metrics": {
            "total_sources": {
                "value": len(source_counts),
                "unit": "count",
                "numerator": len(source_counts),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Sources identified: {', '.join(source_names)}",
            f"Source distribution: {dict(source_counts)}"
        ],
        "assumptions": [
            "Source field is complete and accurate",
            "All sources in the artifact are included"
        ],
        "confidence": 1.0
    }
    
    # Add individual source metrics
    for source in source_counts.index:
        source_count = source_counts[source]
        safe_source_name = source.replace(' ', '_').replace('-', '_').lower()
        finding_3['metrics'][f'{safe_source_name}_count'] = {
            "value": int(source_count),
            "unit": "count",
            "numerator": int(source_count),
            "denominator": len(reviews_analysis),
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    
    findings.append(finding_3)

# ============================================================================
# Build output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
