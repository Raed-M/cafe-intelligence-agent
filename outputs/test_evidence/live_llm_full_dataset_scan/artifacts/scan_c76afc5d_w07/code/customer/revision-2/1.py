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
analysis_start = "2026-02-23T00:00:00+03:00"
analysis_end = "2026-03-02T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Convert review dates to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# Filter reviews for analysis period
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt) & 
    (reviews_df['date'] < analysis_end_dt)
].copy()

# Get all reviews for baseline comparison
reviews_all = reviews_df.copy()

findings = []

# Finding 1: Rating Distribution and Average
if len(reviews_analysis) > 0:
    rating_dist = reviews_analysis['rating'].value_counts().sort_index().to_dict()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Get source names from analysis period
    source_names_analysis = reviews_analysis['source'].unique().tolist()
    
    finding1 = {
        "title": "Rating Distribution During Analysis Period",
        "claim": f"During the analysis period (Feb 23 - Mar 2, 2026), the average rating was {avg_rating:.2f} across {len(reviews_analysis)} reviews from sources: {', '.join(source_names_analysis)}.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": len(reviews_analysis),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names_analysis,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Total reviews in analysis period: {len(reviews_analysis)}",
            f"Sources represented: {', '.join(source_names_analysis)}"
        ],
        "assumptions": [
            "Rating values are numeric and valid",
            "Review dates are accurate and in UTC+3 timezone",
            "All reviews in the dataset are from the specified sources"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Language Distribution
if len(reviews_analysis) > 0:
    language_dist = reviews_analysis['language'].value_counts().to_dict()
    
    finding2 = {
        "title": "Language Distribution of Reviews",
        "claim": f"During the analysis period, reviews were submitted in {len(language_dist)} language(s): {', '.join([f'{lang} ({count} reviews)' for lang, count in language_dist.items()])}.",
        "finding_type": "language_coverage",
        "metrics": {
            "total_reviews_by_language": {
                "value": json.dumps(language_dist),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names_analysis,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Language distribution: {language_dist}",
            f"Total reviews analyzed: {len(reviews_analysis)}"
        ],
        "assumptions": [
            "Language field accurately reflects the language of the review text",
            "All reviews have a valid language designation"
        ],
        "confidence": 0.95
    }
    findings.append(finding2)

# Finding 3: Sentiment Analysis (Basic - checking for positive/negative keywords)
if len(reviews_analysis) > 0:
    # Simple sentiment indicators
    positive_keywords_en = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect', 'wonderful', 'fantastic', 'awesome']
    negative_keywords_en = ['bad', 'poor', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting', 'disappointing', 'waste']
    
    positive_keywords_ar = ['جيد', 'رائع', 'ممتاز', 'عظيم', 'أحب', 'الأفضل', 'مثالي', 'رائع', 'رهيب', 'مذهل']
    negative_keywords_ar = ['سيء', 'فقير', 'فظيع', 'مرعب', 'أكره', 'الأسوأ', 'مريع', 'مقزز', 'مخيب', 'هدر']
    
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
    
    if positive_count + negative_count + neutral_count > 0:
        finding3 = {
            "title": "Sentiment Indicators in Review Text",
            "claim": f"Among {len(reviews_analysis)} reviews in the analysis period, basic keyword analysis suggests {positive_count} positive-leaning, {negative_count} negative-leaning, and {neutral_count} neutral/mixed reviews.",
            "finding_type": "sentiment_distribution",
            "metrics": {
                "positive_sentiment_count": {
                    "value": positive_count,
                    "unit": "count",
                    "numerator": positive_count,
                    "denominator": len(reviews_analysis),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "negative_sentiment_count": {
                    "value": negative_count,
                    "unit": "count",
                    "numerator": negative_count,
                    "denominator": len(reviews_analysis),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "neutral_sentiment_count": {
                    "value": neutral_count,
                    "unit": "count",
                    "numerator": neutral_count,
                    "denominator": len(reviews_analysis),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": source_names_analysis,
            "sample_size": len(reviews_analysis),
            "coverage_notes": [
                "Sentiment classification based on keyword matching in original review text",
                "Keywords checked in both English and Arabic based on review language field",
                "Classification is indicative only and does not represent full NLP analysis"
            ],
            "assumptions": [
                "Keyword presence indicates sentiment tendency",
                "Review text field contains the actual review content",
                "Language field correctly identifies the review language"
            ],
            "confidence": 0.60
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
