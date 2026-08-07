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

# Analysis period
analysis_start = "2026-07-13T00:00:00+03:00"
analysis_end = "2026-07-20T00:00:00+03:00"

# Convert to datetime for filtering
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Convert review dates to datetime and handle timezone awareness
reviews_df['date'] = pd.to_datetime(reviews_df['date'])

# If review dates are timezone-naive, localize them to UTC+3
if reviews_df['date'].dt.tz is None:
    reviews_df['date'] = reviews_df['date'].dt.tz_localize('UTC+03:00')
else:
    # If they have timezone info, convert to UTC+3 for consistency
    reviews_df['date'] = reviews_df['date'].dt.tz_convert('UTC+03:00')

# Make analysis datetime bounds timezone-aware for comparison
if analysis_start_dt.tz is None:
    analysis_start_dt = analysis_start_dt.tz_localize('UTC+03:00')
if analysis_end_dt.tz is None:
    analysis_end_dt = analysis_end_dt.tz_localize('UTC+03:00')

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
    
    # Sentiment classification based on rating
    positive_reviews = len(reviews_analysis[reviews_analysis['rating'] >= 4])
    negative_reviews = len(reviews_analysis[reviews_analysis['rating'] <= 2])
    neutral_reviews = len(reviews_analysis[(reviews_analysis['rating'] > 2) & (reviews_analysis['rating'] < 4)])
    
    finding_1 = {
        "title": "Customer Rating Distribution and Sentiment",
        "claim": f"During the analysis period (2026-07-13 to 2026-07-20), the cafe received {len(reviews_analysis)} reviews with an average rating of {avg_rating:.2f} out of 5. Positive reviews (rating ≥4) comprise {positive_reviews} reviews ({100*positive_reviews/len(reviews_analysis):.1f}%), while negative reviews (rating ≤2) comprise {negative_reviews} reviews ({100*negative_reviews/len(reviews_analysis):.1f}%).",
        "finding_type": "customer_sentiment",
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
                "numerator": round(avg_rating * len(reviews_analysis), 2),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "positive_ratio": {
                "value": round(positive_reviews / len(reviews_analysis), 3),
                "unit": "proportion",
                "numerator": positive_reviews,
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_ratio": {
                "value": round(negative_reviews / len(reviews_analysis), 3),
                "unit": "proportion",
                "numerator": negative_reviews,
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["reviews"],
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: 2026-07-13 to 2026-07-20 (7 days)",
            f"Total reviews in artifact: {len(reviews_df)}",
            f"Reviews in analysis period: {len(reviews_analysis)}",
            f"Language distribution: {dict(reviews_analysis['language'].value_counts())}"
        ],
        "assumptions": [
            "Rating scale is 1-5 stars",
            "Positive sentiment defined as rating ≥4",
            "Negative sentiment defined as rating ≤2",
            "Review dates are in UTC+3 timezone as specified"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# Finding 2: Platform Distribution
if len(reviews_analysis) > 0:
    platform_counts = reviews_analysis['source'].value_counts()
    
    finding_2 = {
        "title": "Review Platform Distribution",
        "claim": f"Reviews in the analysis period come from {len(platform_counts)} platforms. {platform_counts.index[0]} is the primary source with {platform_counts.iloc[0]} reviews ({100*platform_counts.iloc[0]/len(reviews_analysis):.1f}% of total).",
        "finding_type": "channel_analysis",
        "metrics": {
            "total_platforms": {
                "value": len(platform_counts),
                "unit": "count",
                "numerator": len(platform_counts),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "primary_platform": {
                "value": platform_counts.index[0],
                "unit": "platform_name",
                "numerator": platform_counts.iloc[0],
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["reviews"],
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Platform breakdown: {dict(platform_counts)}",
            f"Analysis period: 2026-07-13 to 2026-07-20"
        ],
        "assumptions": [
            "Platform names in 'source' column are accurate",
            "Each review is attributed to exactly one platform"
        ],
        "confidence": 0.95
    }
    findings.append(finding_2)

# Finding 3: Language Distribution
if len(reviews_analysis) > 0:
    language_counts = reviews_analysis['language'].value_counts()
    
    finding_3 = {
        "title": "Review Language Distribution",
        "claim": f"Customer reviews are submitted in {len(language_counts)} languages. {language_counts.index[0]} reviews comprise {language_counts.iloc[0]} reviews ({100*language_counts.iloc[0]/len(reviews_analysis):.1f}% of total), indicating {language_counts.index[0]}-speaking customers are the primary review source.",
        "finding_type": "customer_demographics",
        "metrics": {
            "total_languages": {
                "value": len(language_counts),
                "unit": "count",
                "numerator": len(language_counts),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "primary_language": {
                "value": language_counts.index[0],
                "unit": "language_code",
                "numerator": language_counts.iloc[0],
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": ["reviews"],
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Language breakdown: {dict(language_counts)}",
            f"Analysis period: 2026-07-13 to 2026-07-20"
        ],
        "assumptions": [
            "Language classification in 'language' column is accurate",
            "Each review is classified as exactly one language"
        ],
        "confidence": 0.95
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
print(f"Output written to {output_path}")
