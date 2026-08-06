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
analysis_start = datetime.fromisoformat("2026-04-06T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-04-13T00:00:00+03:00")

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
    
    finding1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During the analysis period (2026-04-06 to 2026-04-13), the average rating across {len(reviews_analysis)} reviews is {avg_rating:.2f} out of 5, with {int(rating_counts.get(5, 0))} five-star and {int(rating_counts.get(1, 0))} one-star reviews.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": len(reviews_analysis),
                "denominator": len(reviews_analysis),
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            },
            "five_star_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": len(reviews_analysis),
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            },
            "one_star_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "reviews",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(reviews_analysis),
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: 2026-04-06 to 2026-04-13",
            f"Total reviews in analysis period: {len(reviews_analysis)}",
            f"Sources represented: {', '.join(source_names)}"
        ],
        "assumptions": [
            "Rating values are numeric and valid",
            "Review dates are accurate and in UTC",
            "All reviews in the period are included"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Language Distribution
if len(reviews_analysis) > 0:
    language_counts = reviews_analysis['language'].value_counts()
    
    finding2 = {
        "title": "Review Language Distribution",
        "claim": f"Of {len(reviews_analysis)} reviews in the analysis period, {int(language_counts.get('English', 0))} are in English and {int(language_counts.get('Arabic', 0))} are in Arabic.",
        "finding_type": "language_coverage",
        "metrics": {
            "english_reviews": {
                "value": int(language_counts.get('English', 0)),
                "unit": "reviews",
                "numerator": int(language_counts.get('English', 0)),
                "denominator": len(reviews_analysis),
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            },
            "arabic_reviews": {
                "value": int(language_counts.get('Arabic', 0)),
                "unit": "reviews",
                "numerator": int(language_counts.get('Arabic', 0)),
                "denominator": len(reviews_analysis),
                "period_start": "2026-04-06T00:00:00+03:00",
                "period_end": "2026-04-13T00:00:00+03:00"
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Language distribution across {len(reviews_analysis)} reviews",
            f"Bilingual coverage: {', '.join(language_counts.index.tolist())}"
        ],
        "assumptions": [
            "Language field is accurately populated",
            "Language values are either 'English' or 'Arabic'"
        ],
        "confidence": 0.95
    }
    findings.append(finding2)

# Finding 3: Sentiment Analysis (Basic - looking for explicit positive/negative indicators)
if len(reviews_analysis) > 0:
    # Simple sentiment indicators
    positive_keywords_en = ['excellent', 'great', 'amazing', 'love', 'perfect', 'best', 'wonderful', 'fantastic', 'good', 'nice', 'delicious']
    negative_keywords_en = ['bad', 'terrible', 'awful', 'hate', 'worst', 'poor', 'disappointing', 'horrible', 'disgusting', 'rude', 'slow']
    
    positive_keywords_ar = ['ممتاز', 'رائع', 'جميل', 'لذيذ', 'ممتازة', 'رائعة', 'جميلة', 'لذيذة', 'أحب', 'أفضل']
    negative_keywords_ar = ['سيء', 'فظيع', 'سيئة', 'فظيعة', 'مخيب', 'مخيبة', 'بطيء', 'بطيئة', 'وقح', 'وقحة']
    
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    for idx, row in reviews_analysis.iterrows():
        if pd.isna(row['text']) or row['text'] == '':
            neutral_count += 1
            continue
            
        text = str(row['text']).lower()
        language = row['language']
        
        has_positive = False
        has_negative = False
        
        if language == 'English':
            has_positive = any(keyword in text for keyword in positive_keywords_en)
            has_negative = any(keyword in text for keyword in negative_keywords_en)
        elif language == 'Arabic':
            has_positive = any(keyword in text for keyword in positive_keywords_ar)
            has_negative = any(keyword in text for keyword in negative_keywords_ar)
        
        if has_positive and not has_negative:
            positive_count += 1
        elif has_negative and not has_positive:
            negative_count += 1
        else:
            neutral_count += 1
    
    if positive_count > 0 or negative_count > 0:
        finding3 = {
            "title": "Sentiment Indicators in Review Text",
            "claim": f"Among {len(reviews_analysis)} reviews in the analysis period, {positive_count} contain positive sentiment indicators and {negative_count} contain negative sentiment indicators based on keyword analysis.",
            "finding_type": "sentiment_analysis",
            "metrics": {
                "positive_sentiment_count": {
                    "value": positive_count,
                    "unit": "reviews",
                    "numerator": positive_count,
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-04-06T00:00:00+03:00",
                    "period_end": "2026-04-13T00:00:00+03:00"
                },
                "negative_sentiment_count": {
                    "value": negative_count,
                    "unit": "reviews",
                    "numerator": negative_count,
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-04-06T00:00:00+03:00",
                    "period_end": "2026-04-13T00:00:00+03:00"
                },
                "neutral_sentiment_count": {
                    "value": neutral_count,
                    "unit": "reviews",
                    "numerator": neutral_count,
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-04-06T00:00:00+03:00",
                    "period_end": "2026-04-13T00:00:00+03:00"
                }
            },
            "source_names": source_names,
            "sample_size": len(reviews_analysis),
            "coverage_notes": [
                f"Sentiment analysis based on keyword matching in {len(reviews_analysis)} reviews",
                f"Bilingual keyword sets used for English and Arabic text",
                f"Reviews with both positive and negative keywords classified as neutral"
            ],
            "assumptions": [
                "Keyword presence indicates sentiment (simple heuristic)",
                "Text field contains review content",
                "Language field correctly identifies review language",
                "Keyword matching is case-insensitive"
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
print(f"Output written to {output_path}")
