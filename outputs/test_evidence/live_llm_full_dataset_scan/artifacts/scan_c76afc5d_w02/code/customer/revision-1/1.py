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
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Analysis period
analysis_start = "2026-01-19T00:00:00+03:00"
analysis_end = "2026-01-26T00:00:00+03:00"

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
    
    # Language distribution
    language_counts = analysis_reviews['language'].value_counts()
    
    # Source distribution
    source_counts = analysis_reviews['source'].value_counts()
    
    finding1 = {
        "title": "Review Rating Distribution and Language Coverage",
        "claim": f"During the analysis period (2026-01-19 to 2026-01-26), {len(analysis_reviews)} reviews were collected with an average rating of {avg_rating:.2f}. The reviews span {len(language_counts)} languages and {len(source_counts)} sources.",
        "finding_type": "rating_distribution",
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
            "rating_1_star_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_2_star_count": {
                "value": int(rating_counts.get(2, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(2, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_3_star_count": {
                "value": int(rating_counts.get(3, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(3, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_4_star_count": {
                "value": int(rating_counts.get(4, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(4, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_5_star_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_reviews": {
                "value": int(language_counts.get('en', 0)),
                "unit": "count",
                "numerator": int(language_counts.get('en', 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": int(language_counts.get('ar', 0)),
                "unit": "count",
                "numerator": int(language_counts.get('ar', 0)),
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": sorted(source_counts.index.tolist()),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Analysis period: 2026-01-19 to 2026-01-26 (7 days)",
            f"Total reviews in dataset: {len(reviews_df)}",
            f"Reviews in analysis period: {len(analysis_reviews)}",
            f"Language distribution: {dict(language_counts)}",
            f"Source distribution: {dict(source_counts)}"
        ],
        "assumptions": [
            "Review date field is authoritative for period filtering",
            "Rating values are numeric and valid",
            "Language field accurately reflects review language",
            "Source field identifies the review platform"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Sentiment Classification by Language
if len(analysis_reviews) > 0:
    # Classify sentiment based on rating
    def classify_sentiment(rating):
        if pd.isna(rating):
            return "unknown"
        if rating >= 4:
            return "positive"
        elif rating == 3:
            return "neutral"
        else:
            return "negative"
    
    analysis_reviews['sentiment'] = analysis_reviews['rating'].apply(classify_sentiment)
    
    # Count by language and sentiment
    sentiment_by_lang = analysis_reviews.groupby(['language', 'sentiment']).size().unstack(fill_value=0)
    
    # Get positive reviews with text
    positive_reviews = analysis_reviews[analysis_reviews['sentiment'] == 'positive']
    negative_reviews = analysis_reviews[analysis_reviews['sentiment'] == 'negative']
    
    positive_with_text = positive_reviews[positive_reviews['text'].notna() & (positive_reviews['text'].str.len() > 0)]
    negative_with_text = negative_reviews[negative_reviews['text'].notna() & (negative_reviews['text'].str.len() > 0)]
    
    if len(positive_with_text) > 0 or len(negative_with_text) > 0:
        finding2 = {
            "title": "Sentiment Distribution by Language",
            "claim": f"Of {len(analysis_reviews)} reviews, {len(positive_reviews)} are positive (rating ≥4), {len(negative_reviews)} are negative (rating <3), and {len(analysis_reviews) - len(positive_reviews) - len(negative_reviews)} are neutral (rating=3). Positive reviews comprise {len(positive_with_text)} with substantive text.",
            "finding_type": "sentiment_distribution",
            "metrics": {
                "positive_reviews": {
                    "value": len(positive_reviews),
                    "unit": "count",
                    "numerator": len(positive_reviews),
                    "denominator": len(analysis_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "negative_reviews": {
                    "value": len(negative_reviews),
                    "unit": "count",
                    "numerator": len(negative_reviews),
                    "denominator": len(analysis_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "neutral_reviews": {
                    "value": len(analysis_reviews) - len(positive_reviews) - len(negative_reviews),
                    "unit": "count",
                    "numerator": len(analysis_reviews) - len(positive_reviews) - len(negative_reviews),
                    "denominator": len(analysis_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "positive_reviews_with_text": {
                    "value": len(positive_with_text),
                    "unit": "count",
                    "numerator": len(positive_with_text),
                    "denominator": len(positive_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "negative_reviews_with_text": {
                    "value": len(negative_with_text),
                    "unit": "count",
                    "numerator": len(negative_with_text),
                    "denominator": len(negative_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": sorted(analysis_reviews['source'].unique().tolist()),
            "sample_size": len(analysis_reviews),
            "coverage_notes": [
                f"Sentiment classification based on rating thresholds: positive (≥4), neutral (=3), negative (<3)",
                f"Language coverage: {dict(language_counts)}",
                f"Positive reviews with substantive text: {len(positive_with_text)}",
                f"Negative reviews with substantive text: {len(negative_with_text)}"
            ],
            "assumptions": [
                "Rating is a valid proxy for sentiment",
                "Text field may be empty or null for some reviews",
                "Language field is accurate"
            ],
            "confidence": 0.90
        }
        findings.append(finding2)

# Finding 3: Review Volume by Source
if len(analysis_reviews) > 0:
    source_distribution = analysis_reviews['source'].value_counts()
    
    finding3 = {
        "title": "Review Volume Distribution by Source",
        "claim": f"During the analysis period, reviews were collected from {len(source_distribution)} sources. The distribution shows {source_distribution.to_dict()}.",
        "finding_type": "source_distribution",
        "metrics": {}
    }
    
    # Add metrics for each source
    for source, count in source_distribution.items():
        source_key = f"{source}_review_count"
        finding3["metrics"][source_key] = {
            "value": int(count),
            "unit": "count",
            "numerator": int(count),
            "denominator": len(analysis_reviews),
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    
    finding3["source_names"] = sorted(source_distribution.index.tolist())
    finding3["sample_size"] = len(analysis_reviews)
    finding3["coverage_notes"] = [
        f"Total reviews analyzed: {len(analysis_reviews)}",
        f"Source distribution: {dict(source_distribution)}",
        f"Analysis period: 2026-01-19 to 2026-01-26"
    ]
    finding3["assumptions"] = [
        "Source field accurately identifies the review platform",
        "All reviews in the dataset have a valid source"
    ]
    finding3["confidence"] = 0.95
    
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
