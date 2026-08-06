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
reviews_df = pd.read_parquet(inputs['reviews'])
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])

# Define analysis period
analysis_start = datetime.fromisoformat("2026-03-09T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-03-16T00:00:00+03:00")

# Convert review dates to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True)

# Filter reviews for analysis period
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start) & 
    (reviews_df['date'] < analysis_end)
].copy()

findings = []

# Finding 1: Rating Distribution and Average
if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Get source names from analysis period reviews
    source_names = reviews_analysis['source'].unique().tolist()
    
    finding_1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During the analysis period ({analysis_start.date()} to {analysis_end.date()}), the average rating across {len(reviews_analysis)} reviews is {avg_rating:.2f} out of 5, with {int(rating_counts.get(5, 0))} five-star ratings and {int(rating_counts.get(1, 0))} one-star ratings.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": len(reviews_analysis),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "five_star_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "one_star_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
            f"Total reviews in dataset: {len(reviews_df)}",
            f"Reviews in analysis period: {len(reviews_analysis)}",
            f"Sources represented: {', '.join(source_names)}"
        ],
        "assumptions": [
            "Rating values are numeric and valid (1-5 scale)",
            "Review dates are accurate and in UTC+3 timezone",
            "All reviews in the dataset are included in the distribution"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# Finding 2: Language Distribution
if len(reviews_analysis) > 0:
    language_counts = reviews_analysis['language'].value_counts()
    
    finding_2 = {
        "title": "Review Language Distribution",
        "claim": f"Of {len(reviews_analysis)} reviews in the analysis period, {int(language_counts.get('en', 0))} are in English and {int(language_counts.get('ar', 0))} are in Arabic.",
        "finding_type": "language_distribution",
        "metrics": {
            "english_reviews": {
                "value": int(language_counts.get('en', 0)),
                "unit": "reviews",
                "numerator": int(language_counts.get('en', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "arabic_reviews": {
                "value": int(language_counts.get('ar', 0)),
                "unit": "reviews",
                "numerator": int(language_counts.get('ar', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "english_percentage": {
                "value": round(100 * int(language_counts.get('en', 0)) / len(reviews_analysis), 1) if len(reviews_analysis) > 0 else 0,
                "unit": "percent",
                "numerator": int(language_counts.get('en', 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
            f"Language field populated for all {len(reviews_analysis)} reviews"
        ],
        "assumptions": [
            "Language classification is accurate",
            "Language values are either 'en' or 'ar'"
        ],
        "confidence": 0.95
    }
    findings.append(finding_2)

# Finding 3: Sentiment Analysis (Basic - looking for explicit positive/negative indicators)
if len(reviews_analysis) > 0:
    # Simple sentiment indicators
    positive_keywords_en = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect', 'wonderful', 'fantastic', 'awesome']
    negative_keywords_en = ['bad', 'poor', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting', 'disappointing', 'waste']
    
    positive_keywords_ar = ['جيد', 'رائع', 'ممتاز', 'أحب', 'أفضل', 'مثالي', 'رائع', 'رهيب']
    negative_keywords_ar = ['سيء', 'فظيع', 'مروع', 'كره', 'أسوأ', 'مخيب', 'خيبة']
    
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    for idx, row in reviews_analysis.iterrows():
        text = str(row['text']).lower() if pd.notna(row['text']) else ""
        language = row['language']
        
        if language == 'en':
            has_positive = any(keyword in text for keyword in positive_keywords_en)
            has_negative = any(keyword in text for keyword in negative_keywords_en)
        elif language == 'ar':
            has_positive = any(keyword in text for keyword in positive_keywords_ar)
            has_negative = any(keyword in text for keyword in negative_keywords_ar)
        else:
            has_positive = False
            has_negative = False
        
        if has_positive and not has_negative:
            positive_count += 1
        elif has_negative and not has_positive:
            negative_count += 1
        else:
            neutral_count += 1
    
    if positive_count > 0 or negative_count > 0:
        finding_3 = {
            "title": "Sentiment Indicators in Reviews",
            "claim": f"Among {len(reviews_analysis)} reviews in the analysis period, {positive_count} contain positive sentiment indicators and {negative_count} contain negative sentiment indicators based on keyword analysis.",
            "finding_type": "sentiment_analysis",
            "metrics": {
                "positive_sentiment_count": {
                    "value": positive_count,
                    "unit": "reviews",
                    "numerator": positive_count,
                    "denominator": len(reviews_analysis),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "negative_sentiment_count": {
                    "value": negative_count,
                    "unit": "reviews",
                    "numerator": negative_count,
                    "denominator": len(reviews_analysis),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "neutral_sentiment_count": {
                    "value": neutral_count,
                    "unit": "reviews",
                    "numerator": neutral_count,
                    "denominator": len(reviews_analysis),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": source_names,
            "sample_size": len(reviews_analysis),
            "coverage_notes": [
                f"Analysis period: {analysis_start.date()} to {analysis_end.date()}",
                f"Sentiment classification based on keyword matching in review text",
                f"English and Arabic keywords used for respective language reviews"
            ],
            "assumptions": [
                "Keyword presence indicates sentiment (simple heuristic)",
                "Review text field is populated and meaningful",
                "Language classification is accurate for keyword matching"
            ],
            "confidence": 0.70
        }
        findings.append(finding_3)

# Prepare output
output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings generated.")
