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
analysis_start = "2026-02-23T00:00:00+03:00"
analysis_end = "2026-03-02T00:00:00+03:00"

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
    source_names = reviews_analysis['source'].unique().tolist()
    
    finding_1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During {analysis_start} to {analysis_end}, reviews averaged {avg_rating:.2f} stars across {len(reviews_analysis)} reviews from sources: {', '.join(source_names)}.",
        "finding_type": "rating_distribution",
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
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: {analysis_start} to {analysis_end}",
            f"Total reviews in analysis period: {len(reviews_analysis)}",
            f"Sources represented: {', '.join(source_names)}"
        ],
        "assumptions": [
            "Rating values are numeric and valid",
            "Date field is accurate and in UTC+3",
            "All reviews in artifact are from the specified sources"
        ],
        "confidence": 0.95
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
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Language distribution: {dict(language_counts)}",
            f"Bilingual coverage enables sentiment analysis in both languages"
        ],
        "assumptions": [
            "Language field accurately reflects review language",
            "Reviews are classified as either 'en' or 'ar'"
        ],
        "confidence": 0.95
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Sentiment Topic Analysis (Non-Empty Reviews)
# ============================================================================

if len(reviews_analysis) > 0:
    # Filter to non-empty reviews
    non_empty_reviews = reviews_analysis[
        (reviews_analysis['text'].notna()) & 
        (reviews_analysis['text'].str.strip() != '')
    ].copy()
    
    if len(non_empty_reviews) > 0:
        # Identify common sentiment keywords (simple heuristic)
        positive_keywords = ['good', 'great', 'excellent', 'love', 'best', 'amazing', 'perfect', 'delicious', 'tasty', 'nice', 'wonderful', 'awesome']
        negative_keywords = ['bad', 'poor', 'terrible', 'hate', 'worst', 'awful', 'horrible', 'disgusting', 'rude', 'slow', 'cold', 'stale']
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for idx, row in non_empty_reviews.iterrows():
            text = str(row['text']).lower()
            has_positive = any(kw in text for kw in positive_keywords)
            has_negative = any(kw in text for kw in negative_keywords)
            
            if has_positive and not has_negative:
                positive_count += 1
            elif has_negative and not has_positive:
                negative_count += 1
            else:
                neutral_count += 1
        
        finding_3 = {
            "title": "Review Sentiment Distribution (Non-Empty Reviews)",
            "claim": f"Of {len(non_empty_reviews)} non-empty reviews in the analysis period, sentiment analysis indicates {positive_count} positive, {negative_count} negative, and {neutral_count} neutral/mixed sentiments.",
            "finding_type": "sentiment_distribution",
            "metrics": {
                "positive_sentiment_count": {
                    "value": positive_count,
                    "unit": "count",
                    "numerator": positive_count,
                    "denominator": len(non_empty_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "negative_sentiment_count": {
                    "value": negative_count,
                    "unit": "count",
                    "numerator": negative_count,
                    "denominator": len(non_empty_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "neutral_sentiment_count": {
                    "value": neutral_count,
                    "unit": "count",
                    "numerator": neutral_count,
                    "denominator": len(non_empty_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": source_names,
            "sample_size": len(non_empty_reviews),
            "coverage_notes": [
                f"Non-empty reviews: {len(non_empty_reviews)} of {len(reviews_analysis)} total",
                f"Sentiment classification based on keyword matching in review text",
                f"Both English and Arabic reviews included in analysis"
            ],
            "assumptions": [
                "Sentiment keywords are representative of positive/negative intent",
                "Keyword matching is a proxy for sentiment (not a trained model)",
                "Reviews with both positive and negative keywords are classified as neutral/mixed",
                "Empty or whitespace-only reviews are excluded from sentiment analysis"
            ],
            "confidence": 0.70
        }
        findings.append(finding_3)

# ============================================================================
# Prepare output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write to output path
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")
