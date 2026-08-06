import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter
import re

# Load environment
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Analysis period
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
    
    finding1 = {
        "title": "Customer Rating Distribution (Analysis Period)",
        "claim": f"Average rating is {avg_rating:.2f} out of 5 across {len(reviews_analysis)} reviews in the analysis period (2026-03-09 to 2026-03-16).",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(avg_rating * len(reviews_analysis), 2),
                "denominator": len(reviews_analysis),
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "rating_5_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": len(reviews_analysis),
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "rating_4_count": {
                "value": int(rating_counts.get(4, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(4, 0)),
                "denominator": len(reviews_analysis),
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "rating_3_count": {
                "value": int(rating_counts.get(3, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(3, 0)),
                "denominator": len(reviews_analysis),
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "rating_2_count": {
                "value": int(rating_counts.get(2, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(2, 0)),
                "denominator": len(reviews_analysis),
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            },
            "rating_1_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(reviews_analysis),
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: 2026-03-09 to 2026-03-16",
            f"Total reviews in dataset: {len(reviews_df)}",
            f"Reviews in analysis period: {len(reviews_analysis)}",
            f"Language distribution: {dict(reviews_analysis['language'].value_counts())}"
        ],
        "assumptions": [
            "Rating values are numeric and valid",
            "Review dates are accurate and in UTC+3 timezone",
            "All reviews with non-null ratings are included"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Language Distribution
if len(reviews_analysis) > 0:
    lang_counts = reviews_analysis['language'].value_counts()
    
    finding2 = {
        "title": "Review Language Distribution",
        "claim": f"Reviews are distributed across {len(lang_counts)} languages, with {lang_counts.index[0]} being the most common ({lang_counts.iloc[0]} reviews, {100*lang_counts.iloc[0]/len(reviews_analysis):.1f}%).",
        "finding_type": "language_distribution",
        "metrics": {
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": "2026-03-09T00:00:00+03:00",
                "period_end": "2026-03-16T00:00:00+03:00"
            }
        },
        "source_names": list(reviews_analysis['source'].unique()),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Language breakdown: {dict(lang_counts)}",
            f"Analysis period: 2026-03-09 to 2026-03-16"
        ],
        "assumptions": [
            "Language field is accurately populated",
            "Language codes are standardized"
        ],
        "confidence": 0.95
    }
    
    # Add language-specific metrics
    for lang in lang_counts.index:
        lang_reviews = reviews_analysis[reviews_analysis['language'] == lang]
        lang_avg_rating = lang_reviews['rating'].mean()
        finding2['metrics'][f"{lang}_count"] = {
            "value": int(lang_counts[lang]),
            "unit": "count",
            "numerator": int(lang_counts[lang]),
            "denominator": len(reviews_analysis),
            "period_start": "2026-03-09T00:00:00+03:00",
            "period_end": "2026-03-16T00:00:00+03:00"
        }
        finding2['metrics'][f"{lang}_avg_rating"] = {
            "value": round(lang_avg_rating, 2),
            "unit": "stars",
            "numerator": round(lang_avg_rating * len(lang_reviews), 2),
            "denominator": len(lang_reviews),
            "period_start": "2026-03-09T00:00:00+03:00",
            "period_end": "2026-03-16T00:00:00+03:00"
        }
    
    findings.append(finding2)

# Finding 3: Sentiment Analysis (based on rating levels)
if len(reviews_analysis) > 0:
    # Categorize sentiment based on rating
    reviews_analysis['sentiment'] = reviews_analysis['rating'].apply(
        lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral')
    )
    
    sentiment_counts = reviews_analysis['sentiment'].value_counts()
    
    # Identify reviews with text for sentiment validation
    reviews_with_text = reviews_analysis[reviews_analysis['text'].notna() & (reviews_analysis['text'].str.len() > 0)]
    
    if len(reviews_with_text) > 0:
        finding3 = {
            "title": "Sentiment Distribution Based on Ratings",
            "claim": f"Among {len(reviews_analysis)} reviews, {sentiment_counts.get('positive', 0)} are positive (rating ≥4), {sentiment_counts.get('negative', 0)} are negative (rating ≤2), and {sentiment_counts.get('neutral', 0)} are neutral (rating 3).",
            "finding_type": "sentiment_distribution",
            "metrics": {
                "positive_reviews": {
                    "value": int(sentiment_counts.get('positive', 0)),
                    "unit": "count",
                    "numerator": int(sentiment_counts.get('positive', 0)),
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-03-09T00:00:00+03:00",
                    "period_end": "2026-03-16T00:00:00+03:00"
                },
                "negative_reviews": {
                    "value": int(sentiment_counts.get('negative', 0)),
                    "unit": "count",
                    "numerator": int(sentiment_counts.get('negative', 0)),
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-03-09T00:00:00+03:00",
                    "period_end": "2026-03-16T00:00:00+03:00"
                },
                "neutral_reviews": {
                    "value": int(sentiment_counts.get('neutral', 0)),
                    "unit": "count",
                    "numerator": int(sentiment_counts.get('neutral', 0)),
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-03-09T00:00:00+03:00",
                    "period_end": "2026-03-16T00:00:00+03:00"
                },
                "reviews_with_text": {
                    "value": len(reviews_with_text),
                    "unit": "count",
                    "numerator": len(reviews_with_text),
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-03-09T00:00:00+03:00",
                    "period_end": "2026-03-16T00:00:00+03:00"
                }
            },
            "source_names": list(reviews_analysis['source'].unique()),
            "sample_size": len(reviews_analysis),
            "coverage_notes": [
                f"Sentiment classification based on rating thresholds",
                f"Positive: rating ≥4 ({int(sentiment_counts.get('positive', 0))} reviews)",
                f"Negative: rating ≤2 ({int(sentiment_counts.get('negative', 0))} reviews)",
                f"Neutral: rating = 3 ({int(sentiment_counts.get('neutral', 0))} reviews)",
                f"Reviews with text content: {len(reviews_with_text)} ({100*len(reviews_with_text)/len(reviews_analysis):.1f}%)"
            ],
            "assumptions": [
                "Rating values directly correspond to sentiment (4-5 = positive, 1-2 = negative, 3 = neutral)",
                "Text content is optional and not required for sentiment classification",
                "All reviews have valid rating values"
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
print(f"Output written to {output_path}")
