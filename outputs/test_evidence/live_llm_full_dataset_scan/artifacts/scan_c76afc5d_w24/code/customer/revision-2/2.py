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
analysis_start = "2026-06-22T00:00:00+03:00"
analysis_end = "2026-06-29T00:00:00+03:00"

# Convert to datetime for filtering - handle timezone awareness
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Convert review dates to datetime and ensure timezone-naive for comparison
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True)

# Convert analysis period to UTC for comparison
analysis_start_dt_utc = analysis_start_dt.tz_convert('UTC')
analysis_end_dt_utc = analysis_end_dt.tz_convert('UTC')

# Filter reviews for analysis period
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt_utc) & 
    (reviews_df['date'] < analysis_end_dt_utc)
].copy()

# Get all reviews for baseline comparison
reviews_all = reviews_df.copy()

findings = []

# Finding 1: Rating Distribution and Average
if len(reviews_analysis) > 0:
    rating_dist = reviews_analysis['rating'].value_counts().sort_index().to_dict()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Get source names from analysis period
    source_names_analysis = sorted(reviews_analysis['source'].unique().tolist())
    
    finding1 = {
        "title": "Rating Distribution and Average (Analysis Period)",
        "claim": f"During the analysis period (2026-06-22 to 2026-06-29), the average rating across {len(reviews_analysis)} reviews is {avg_rating:.2f} out of 5, with distribution: {rating_dist}",
        "finding_type": "rating_analysis",
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
            },
            "rating_5_star_count": {
                "value": rating_dist.get(5, 0),
                "unit": "count",
                "numerator": rating_dist.get(5, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_4_star_count": {
                "value": rating_dist.get(4, 0),
                "unit": "count",
                "numerator": rating_dist.get(4, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_3_star_count": {
                "value": rating_dist.get(3, 0),
                "unit": "count",
                "numerator": rating_dist.get(3, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_2_star_count": {
                "value": rating_dist.get(2, 0),
                "unit": "count",
                "numerator": rating_dist.get(2, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "rating_1_star_count": {
                "value": rating_dist.get(1, 0),
                "unit": "count",
                "numerator": rating_dist.get(1, 0),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names_analysis,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: 2026-06-22 to 2026-06-29",
            f"Total reviews in analysis period: {len(reviews_analysis)}",
            f"Sources represented: {', '.join(source_names_analysis)}",
            f"Language distribution: {reviews_analysis['language'].value_counts().to_dict()}"
        ],
        "assumptions": [
            "Rating values are numeric and valid",
            "Review dates are accurate and converted to UTC for comparison",
            "All reviews in the dataset are legitimate customer feedback"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# Finding 2: Language Distribution
if len(reviews_analysis) > 0:
    lang_dist = reviews_analysis['language'].value_counts().to_dict()
    source_names_analysis = sorted(reviews_analysis['source'].unique().tolist())
    
    finding2 = {
        "title": "Language Distribution of Reviews",
        "claim": f"During the analysis period, {len(reviews_analysis)} reviews were submitted in the following languages: {lang_dist}",
        "finding_type": "language_coverage",
        "metrics": {
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
            f"Language distribution: {lang_dist}",
            f"Analysis period: 2026-06-22 to 2026-06-29",
            f"All reviews in analysis period included"
        ],
        "assumptions": [
            "Language field is accurately populated",
            "Language codes are standardized (e.g., 'en', 'ar')"
        ],
        "confidence": 0.95
    }
    
    # Add language-specific metrics
    for lang, count in lang_dist.items():
        finding2["metrics"][f"reviews_{lang}_count"] = {
            "value": count,
            "unit": "count",
            "numerator": count,
            "denominator": len(reviews_analysis),
            "period_start": analysis_start,
            "period_end": analysis_end
        }
    
    findings.append(finding2)

# Finding 3: Review Text Availability and Sentiment Keywords
if len(reviews_analysis) > 0:
    reviews_with_text = reviews_analysis[reviews_analysis['text'].notna() & (reviews_analysis['text'].str.len() > 0)]
    source_names_analysis = sorted(reviews_analysis['source'].unique().tolist())
    
    # Simple keyword-based sentiment indicators
    positive_keywords = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect', 'wonderful', 'fantastic',
                        'جيد', 'رائع', 'ممتاز', 'أحب', 'أفضل', 'مثالي', 'رائع']
    negative_keywords = ['bad', 'poor', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing',
                        'سيء', 'رهيب', 'مروع', 'محبط', 'أسوأ', 'فظيع']
    
    positive_count = 0
    negative_count = 0
    
    for text in reviews_with_text['text']:
        text_lower = str(text).lower()
        if any(keyword in text_lower for keyword in positive_keywords):
            positive_count += 1
        if any(keyword in text_lower for keyword in negative_keywords):
            negative_count += 1
    
    finding3 = {
        "title": "Review Text Availability and Sentiment Indicators",
        "claim": f"Of {len(reviews_analysis)} reviews in the analysis period, {len(reviews_with_text)} contain text. Preliminary keyword analysis suggests {positive_count} reviews contain positive sentiment indicators and {negative_count} contain negative sentiment indicators.",
        "finding_type": "text_availability",
        "metrics": {
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "reviews_with_text": {
                "value": len(reviews_with_text),
                "unit": "count",
                "numerator": len(reviews_with_text),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "reviews_with_positive_keywords": {
                "value": positive_count,
                "unit": "count",
                "numerator": positive_count,
                "denominator": len(reviews_with_text) if len(reviews_with_text) > 0 else None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "reviews_with_negative_keywords": {
                "value": negative_count,
                "unit": "count",
                "numerator": negative_count,
                "denominator": len(reviews_with_text) if len(reviews_with_text) > 0 else None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names_analysis,
        "sample_size": len(reviews_with_text),
        "coverage_notes": [
            f"Analysis period: 2026-06-22 to 2026-06-29",
            f"Total reviews: {len(reviews_analysis)}",
            f"Reviews with text content: {len(reviews_with_text)}",
            f"Text availability rate: {len(reviews_with_text)/len(reviews_analysis)*100:.1f}%",
            "Sentiment analysis based on simple keyword matching, not NLP"
        ],
        "assumptions": [
            "Text field contains original review text",
            "Keyword matching is a proxy for sentiment (not comprehensive)",
            "Keywords are case-insensitive",
            "A review can contain both positive and negative keywords"
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
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings generated.")
