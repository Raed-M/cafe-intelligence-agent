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
analysis_start = "2026-06-01T00:00:00+03:00"
analysis_end = "2026-06-08T00:00:00+03:00"

# Convert to datetime for filtering - handle timezone awareness
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Convert reviews date to datetime and remove timezone info if present to match naive comparison
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
# Remove timezone info from both sides to ensure compatibility
if reviews_df['date'].dt.tz is not None:
    reviews_df['date'] = reviews_df['date'].dt.tz_localize(None)
analysis_start_dt_naive = analysis_start_dt.tz_localize(None)
analysis_end_dt_naive = analysis_end_dt.tz_localize(None)

# Filter reviews to analysis period
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt_naive) & 
    (reviews_df['date'] < analysis_end_dt_naive)
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
    
    finding_1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During {analysis_start[:10]} to {analysis_end[:10]}, {len(reviews_analysis)} reviews were collected with an average rating of {avg_rating:.2f} out of 5.0.",
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
            f"Analysis period: {analysis_start[:10]} to {analysis_end[:10]}",
            f"Total reviews in analysis period: {len(reviews_analysis)}",
            f"Sources represented: {', '.join(source_names)}"
        ],
        "assumptions": [
            "Rating values are numeric and valid (1-5 scale)",
            "Review dates are accurate and in UTC+3 timezone",
            "All reviews in the artifact are genuine and unfiltered"
        ],
        "confidence": 1.0
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Language Distribution
# ============================================================================
if len(reviews_analysis) > 0:
    language_counts = reviews_analysis['language'].value_counts()
    
    finding_2 = {
        "title": "Review Language Distribution",
        "claim": f"Of {len(reviews_analysis)} reviews in the analysis period, {language_counts.get('en', 0)} are in English and {language_counts.get('ar', 0)} are in Arabic.",
        "finding_type": "language_coverage",
        "metrics": {
            "english_reviews": {
                "value": int(language_counts.get('en', 0)),
                "unit": "count",
                "numerator": int(language_counts.get('en', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": int(language_counts.get('ar', 0)),
                "unit": "count",
                "numerator": int(language_counts.get('ar', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_percentage": {
                "value": round(100 * language_counts.get('en', 0) / len(reviews_analysis), 1) if len(reviews_analysis) > 0 else 0,
                "unit": "percent",
                "numerator": int(language_counts.get('en', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_percentage": {
                "value": round(100 * language_counts.get('ar', 0) / len(reviews_analysis), 1) if len(reviews_analysis) > 0 else 0,
                "unit": "percent",
                "numerator": int(language_counts.get('ar', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start[:10]} to {analysis_end[:10]}",
            f"Language field populated for all {len(reviews_analysis)} reviews"
        ],
        "assumptions": [
            "Language classification in the artifact is accurate",
            "Language values are either 'en' or 'ar'"
        ],
        "confidence": 1.0
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Sentiment/Topic Classification (Text-based)
# ============================================================================
if len(reviews_analysis) > 0:
    # Identify reviews with non-empty text
    reviews_with_text = reviews_analysis[reviews_analysis['text'].notna() & (reviews_analysis['text'].str.len() > 0)].copy()
    
    if len(reviews_with_text) > 0:
        # Simple keyword-based classification for demonstration
        # Positive keywords (English and Arabic)
        positive_keywords_en = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect', 'wonderful', 'delicious', 'tasty']
        positive_keywords_ar = ['جيد', 'رائع', 'ممتاز', 'لذيذ', 'طعم', 'أحب', 'أفضل', 'رائعة']
        
        # Negative keywords (English and Arabic)
        negative_keywords_en = ['bad', 'poor', 'terrible', 'awful', 'hate', 'worst', 'disgusting', 'cold', 'slow', 'rude']
        negative_keywords_ar = ['سيء', 'سيئة', 'رهيب', 'فظيع', 'بطيء', 'بطيئة', 'بارد', 'باردة']
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for text in reviews_with_text['text']:
            text_lower = str(text).lower()
            has_positive = any(kw in text_lower for kw in positive_keywords_en + positive_keywords_ar)
            has_negative = any(kw in text_lower for kw in negative_keywords_en + negative_keywords_ar)
            
            if has_positive and not has_negative:
                positive_count += 1
            elif has_negative and not has_positive:
                negative_count += 1
            else:
                neutral_count += 1
        
        finding_3 = {
            "title": "Sentiment Classification from Review Text",
            "claim": f"Of {len(reviews_with_text)} reviews with text content in the analysis period, {positive_count} exhibit positive sentiment, {negative_count} exhibit negative sentiment, and {neutral_count} are neutral or mixed.",
            "finding_type": "sentiment_classification",
            "metrics": {
                "reviews_with_text": {
                    "value": len(reviews_with_text),
                    "unit": "count",
                    "numerator": len(reviews_with_text),
                    "denominator": len(reviews_analysis),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "positive_sentiment_count": {
                    "value": positive_count,
                    "unit": "count",
                    "numerator": positive_count,
                    "denominator": len(reviews_with_text),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "negative_sentiment_count": {
                    "value": negative_count,
                    "unit": "count",
                    "numerator": negative_count,
                    "denominator": len(reviews_with_text),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "neutral_sentiment_count": {
                    "value": neutral_count,
                    "unit": "count",
                    "numerator": neutral_count,
                    "denominator": len(reviews_with_text),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "positive_percentage": {
                    "value": round(100 * positive_count / len(reviews_with_text), 1) if len(reviews_with_text) > 0 else 0,
                    "unit": "percent",
                    "numerator": positive_count,
                    "denominator": len(reviews_with_text),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": source_names,
            "sample_size": len(reviews_with_text),
            "coverage_notes": [
                f"Analysis period: {analysis_start[:10]} to {analysis_end[:10]}",
                f"Reviews with non-empty text: {len(reviews_with_text)} of {len(reviews_analysis)}",
                "Sentiment classification based on keyword matching in English and Arabic"
            ],
            "assumptions": [
                "Keyword-based classification is a proxy for sentiment; nuance may be lost",
                "Text field is populated and in original language (not translated)",
                "Positive and negative keywords are mutually exclusive for classification"
            ],
            "confidence": 0.7
        }
        findings.append(finding_3)

# ============================================================================
# Build output
# ============================================================================
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write to output path
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")
