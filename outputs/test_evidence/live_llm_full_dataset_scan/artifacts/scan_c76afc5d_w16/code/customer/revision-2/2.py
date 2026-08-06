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
analysis_start = "2026-04-27T00:00:00+03:00"
analysis_end = "2026-05-04T00:00:00+03:00"

# Convert to datetime for comparison - handle timezone awareness
analysis_start_dt = pd.to_datetime(analysis_start)
analysis_end_dt = pd.to_datetime(analysis_end)

# Convert review dates to datetime and ensure timezone-naive for comparison
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True)

# Convert analysis boundaries to UTC for comparison
analysis_start_dt_utc = analysis_start_dt.tz_convert('UTC')
analysis_end_dt_utc = analysis_end_dt.tz_convert('UTC')

# Filter reviews for analysis period
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start_dt_utc) & 
    (reviews_df['date'] < analysis_end_dt_utc)
].copy()

# Get all reviews for baseline comparison
reviews_all = reviews_df.copy()

# Initialize findings list
findings = []

# FINDING 1: Rating Distribution and Average
if len(reviews_analysis) > 0:
    rating_dist = reviews_analysis['rating'].value_counts().sort_index().to_dict()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Get source names from analysis period
    source_names_analysis = sorted(reviews_analysis['source'].unique().tolist())
    
    finding1 = {
        "title": "Customer Rating Distribution (Analysis Period)",
        "claim": f"During the analysis period (2026-04-27 to 2026-05-04), customer ratings averaged {avg_rating:.2f} out of 5, with {len(reviews_analysis)} reviews collected from {len(source_names_analysis)} source(s): {', '.join(source_names_analysis)}.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(avg_rating * len(reviews_analysis), 2),
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
            },
            **{f"rating_{int(k)}_count": {
                "value": int(v),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            } for k, v in rating_dist.items()}
        },
        "source_names": source_names_analysis,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: 2026-04-27 to 2026-05-04 (8 days)",
            f"Total reviews in analysis period: {len(reviews_analysis)}",
            f"Sources represented: {', '.join(source_names_analysis)}",
            f"Language distribution: {dict(reviews_analysis['language'].value_counts())}"
        ],
        "assumptions": [
            "Review dates are accurate and converted to UTC for comparison",
            "All reviews in the artifact are valid and complete",
            "Rating scale is 1-5 stars"
        ],
        "confidence": 0.95
    }
    findings.append(finding1)

# FINDING 2: Sentiment/Topic Classification by Language
if len(reviews_analysis) > 0:
    # Separate by language
    english_reviews = reviews_analysis[reviews_analysis['language'] == 'en'].copy()
    arabic_reviews = reviews_analysis[reviews_analysis['language'] == 'ar'].copy()
    
    # Classify sentiment based on rating
    def classify_sentiment(rating):
        if rating >= 4:
            return "positive"
        elif rating == 3:
            return "neutral"
        else:
            return "negative"
    
    reviews_analysis['sentiment'] = reviews_analysis['rating'].apply(classify_sentiment)
    
    sentiment_dist = reviews_analysis['sentiment'].value_counts().to_dict()
    
    # Identify common topics from text (simple keyword matching)
    topics_found = {}
    
    # English topics
    if len(english_reviews) > 0:
        english_text = ' '.join(english_reviews['text'].fillna('').str.lower())
        english_keywords = {
            'quality': english_text.count('quality') + english_text.count('good') + english_text.count('excellent'),
            'service': english_text.count('service') + english_text.count('staff') + english_text.count('friendly'),
            'price': english_text.count('price') + english_text.count('expensive') + english_text.count('cheap'),
            'taste': english_text.count('taste') + english_text.count('flavor') + english_text.count('delicious'),
            'wait_time': english_text.count('wait') + english_text.count('slow') + english_text.count('fast')
        }
        topics_found['english'] = {k: v for k, v in english_keywords.items() if v > 0}
    
    # Arabic topics
    if len(arabic_reviews) > 0:
        arabic_text = ' '.join(arabic_reviews['text'].fillna('').str.lower())
        arabic_keywords = {
            'quality': arabic_text.count('جودة') + arabic_text.count('ممتاز') + arabic_text.count('رائع'),
            'service': arabic_text.count('خدمة') + arabic_text.count('موظفين') + arabic_text.count('لطيف'),
            'price': arabic_text.count('سعر') + arabic_text.count('غالي') + arabic_text.count('رخيص'),
            'taste': arabic_text.count('طعم') + arabic_text.count('لذيذ') + arabic_text.count('طعم'),
            'wait_time': arabic_text.count('انتظار') + arabic_text.count('بطيء') + arabic_text.count('سريع')
        }
        topics_found['arabic'] = {k: v for k, v in arabic_keywords.items() if v > 0}
    
    finding2 = {
        "title": "Sentiment Distribution by Language",
        "claim": f"Among {len(reviews_analysis)} reviews in the analysis period, {sentiment_dist.get('positive', 0)} were positive (rating ≥4), {sentiment_dist.get('neutral', 0)} neutral (rating=3), and {sentiment_dist.get('negative', 0)} negative (rating <3). English reviews: {len(english_reviews)}, Arabic reviews: {len(arabic_reviews)}. Sources: {', '.join(source_names_analysis)}.",
        "finding_type": "sentiment_distribution",
        "metrics": {
            "positive_count": {
                "value": sentiment_dist.get('positive', 0),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "neutral_count": {
                "value": sentiment_dist.get('neutral', 0),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "negative_count": {
                "value": sentiment_dist.get('negative', 0),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "english_reviews": {
                "value": len(english_reviews),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "arabic_reviews": {
                "value": len(arabic_reviews),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": source_names_analysis,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Sentiment classification based on rating thresholds (positive: ≥4, neutral: 3, negative: <3)",
            f"English reviews: {len(english_reviews)} ({100*len(english_reviews)/len(reviews_analysis):.1f}%)" if len(reviews_analysis) > 0 else "English reviews: 0",
            f"Arabic reviews: {len(arabic_reviews)} ({100*len(arabic_reviews)/len(reviews_analysis):.1f}%)" if len(reviews_analysis) > 0 else "Arabic reviews: 0",
            f"Topics identified through keyword matching in original language text"
        ],
        "assumptions": [
            "Rating-based sentiment classification is appropriate for this dataset",
            "Text analysis uses simple keyword matching in original language",
            "Empty or missing review text is treated as no topic indicators"
        ],
        "confidence": 0.85
    }
    findings.append(finding2)

# FINDING 3: High-Rating vs Low-Rating Review Comparison
if len(reviews_analysis) > 0:
    high_rating_reviews = reviews_analysis[reviews_analysis['rating'] >= 4]
    low_rating_reviews = reviews_analysis[reviews_analysis['rating'] <= 2]
    
    if len(high_rating_reviews) > 0 and len(low_rating_reviews) > 0:
        # Get sample review IDs for evidence
        high_sample_ids = high_rating_reviews['review_id'].head(3).tolist()
        low_sample_ids = low_rating_reviews['review_id'].head(3).tolist()
        
        finding3 = {
            "title": "High vs Low Rating Review Volume",
            "claim": f"In the analysis period, {len(high_rating_reviews)} reviews had ratings ≥4 (high satisfaction), while {len(low_rating_reviews)} reviews had ratings ≤2 (low satisfaction), indicating a {100*len(high_rating_reviews)/(len(high_rating_reviews)+len(low_rating_reviews)):.1f}% positive-to-total ratio among extreme ratings. Sources: {', '.join(source_names_analysis)}.",
            "finding_type": "rating_comparison",
            "metrics": {
                "high_rating_count": {
                    "value": len(high_rating_reviews),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "low_rating_count": {
                    "value": len(low_rating_reviews),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "positive_ratio": {
                    "value": round(100*len(high_rating_reviews)/(len(high_rating_reviews)+len(low_rating_reviews)), 1),
                    "unit": "percent",
                    "numerator": len(high_rating_reviews),
                    "denominator": len(high_rating_reviews) + len(low_rating_reviews),
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": source_names_analysis,
            "sample_size": len(high_rating_reviews) + len(low_rating_reviews),
            "coverage_notes": [
                f"High rating (≥4): {len(high_rating_reviews)} reviews",
                f"Low rating (≤2): {len(low_rating_reviews)} reviews",
                f"Sample high-rating review IDs: {high_sample_ids}",
                f"Sample low-rating review IDs: {low_sample_ids}",
                f"Neutral ratings (3): {len(reviews_analysis[reviews_analysis['rating'] == 3])} (excluded from this comparison)"
            ],
            "assumptions": [
                "Rating scale is 1-5 stars",
                "High satisfaction defined as rating ≥4",
                "Low satisfaction defined as rating ≤2"
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
