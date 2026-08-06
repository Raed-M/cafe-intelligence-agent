import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter
import re

# Load input/output paths
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)
inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Define analysis period
analysis_start = datetime.fromisoformat("2026-03-16T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-03-23T00:00:00+03:00")

# Convert review dates to datetime
reviews_df['date'] = pd.to_datetime(reviews_df['date'], utc=True)

# Filter reviews for analysis period
reviews_analysis = reviews_df[
    (reviews_df['date'] >= analysis_start) & 
    (reviews_df['date'] < analysis_end)
].copy()

findings = []

# ============================================================================
# FINDING 1: Rating Distribution and Average
# ============================================================================

if len(reviews_analysis) > 0:
    rating_counts = reviews_analysis['rating'].value_counts().sort_index()
    avg_rating = reviews_analysis['rating'].mean()
    
    # Get source names from analysis period reviews
    source_names = reviews_analysis['source'].unique().tolist()
    
    # Get language distribution
    language_dist = reviews_analysis['language'].value_counts().to_dict()
    
    finding_1 = {
        "title": "Review Rating Distribution (Analysis Period)",
        "claim": f"During the analysis period ({analysis_start.date()} to {analysis_end.date()}), the average rating across {len(reviews_analysis)} reviews was {avg_rating:.2f} out of 5.0, with {rating_counts.get(5, 0)} five-star and {rating_counts.get(1, 0)} one-star reviews.",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(reviews_analysis['rating'].sum(), 2),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "five_star_count": {
                "value": int(rating_counts.get(5, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(5, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "one_star_count": {
                "value": int(rating_counts.get(1, 0)),
                "unit": "count",
                "numerator": int(rating_counts.get(1, 0)),
                "denominator": len(reviews_analysis),
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Language distribution: {language_dist}",
            f"Sources represented: {', '.join(source_names)}",
            f"Analysis period: {analysis_start.date()} to {analysis_end.date()}"
        ],
        "assumptions": [
            "Rating values are numeric and valid",
            "Review dates are accurate and in UTC+3 timezone",
            "All reviews in the artifact are included without filtering by source or language"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# ============================================================================
# FINDING 2: Sentiment Classification by Language
# ============================================================================

if len(reviews_analysis) > 0:
    # Simple sentiment classification based on rating thresholds
    # High: 4-5 stars, Medium: 3 stars, Low: 1-2 stars
    
    sentiment_map = {}
    for idx, row in reviews_analysis.iterrows():
        rating = row['rating']
        if rating >= 4:
            sentiment = "positive"
        elif rating == 3:
            sentiment = "neutral"
        else:
            sentiment = "negative"
        sentiment_map[idx] = sentiment
    
    reviews_analysis['sentiment'] = reviews_analysis.index.map(sentiment_map)
    
    # Count by language and sentiment
    lang_sentiment = reviews_analysis.groupby(['language', 'sentiment']).size().reset_index(name='count')
    
    # Get language-specific counts
    english_reviews = reviews_analysis[reviews_analysis['language'] == 'en']
    arabic_reviews = reviews_analysis[reviews_analysis['language'] == 'ar']
    
    en_positive = len(english_reviews[english_reviews['sentiment'] == 'positive'])
    en_total = len(english_reviews)
    ar_positive = len(arabic_reviews[arabic_reviews['sentiment'] == 'positive'])
    ar_total = len(arabic_reviews)
    
    finding_2 = {
        "title": "Sentiment Distribution by Language",
        "claim": f"Among {en_total} English reviews, {en_positive} ({100*en_positive/en_total if en_total > 0 else 0:.1f}%) were positive (4-5 stars). Among {ar_total} Arabic reviews, {ar_positive} ({100*ar_positive/ar_total if ar_total > 0 else 0:.1f}%) were positive.",
        "finding_type": "sentiment_by_language",
        "metrics": {
            "english_positive_count": {
                "value": en_positive,
                "unit": "count",
                "numerator": en_positive,
                "denominator": en_total,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "english_total_count": {
                "value": en_total,
                "unit": "count",
                "numerator": en_total,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "arabic_positive_count": {
                "value": ar_positive,
                "unit": "count",
                "numerator": ar_positive,
                "denominator": ar_total,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            },
            "arabic_total_count": {
                "value": ar_total,
                "unit": "count",
                "numerator": ar_total,
                "denominator": None,
                "period_start": analysis_start.isoformat(),
                "period_end": analysis_end.isoformat()
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"English reviews: {en_total}",
            f"Arabic reviews: {ar_total}",
            f"Sentiment classification: 4-5 stars = positive, 3 stars = neutral, 1-2 stars = negative"
        ],
        "assumptions": [
            "Sentiment is derived from rating thresholds, not text analysis",
            "Language field is accurate",
            "Rating values are valid numeric"
        ],
        "confidence": 0.90
    }
    findings.append(finding_2)

# ============================================================================
# FINDING 3: Low Rating Frequency and Topics
# ============================================================================

if len(reviews_analysis) > 0:
    low_rating_reviews = reviews_analysis[reviews_analysis['rating'] <= 2].copy()
    
    if len(low_rating_reviews) > 0:
        # Extract common words from low-rating reviews (simple keyword extraction)
        low_rating_texts = low_rating_reviews['text'].dropna().astype(str).tolist()
        
        # Simple word frequency for low ratings
        all_words = []
        for text in low_rating_texts:
            # Remove common words and split
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(5)
        
        low_rating_count = len(low_rating_reviews)
        low_rating_pct = 100 * low_rating_count / len(reviews_analysis)
        
        finding_3 = {
            "title": "Low Rating Frequency",
            "claim": f"During the analysis period, {low_rating_count} reviews ({low_rating_pct:.1f}% of {len(reviews_analysis)} total) had ratings of 1-2 stars, indicating dissatisfaction.",
            "finding_type": "low_rating_frequency",
            "metrics": {
                "low_rating_count": {
                    "value": low_rating_count,
                    "unit": "count",
                    "numerator": low_rating_count,
                    "denominator": len(reviews_analysis),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                },
                "low_rating_percentage": {
                    "value": round(low_rating_pct, 1),
                    "unit": "percent",
                    "numerator": low_rating_count,
                    "denominator": len(reviews_analysis),
                    "period_start": analysis_start.isoformat(),
                    "period_end": analysis_end.isoformat()
                }
            },
            "source_names": source_names,
            "sample_size": len(reviews_analysis),
            "coverage_notes": [
                f"Low-rating reviews (1-2 stars): {low_rating_count}",
                f"Total reviews analyzed: {len(reviews_analysis)}",
                f"Top keywords in low-rating text: {', '.join([w[0] for w in top_words[:3]])}" if top_words else "No text content available"
            ],
            "assumptions": [
                "Low rating defined as 1-2 stars",
                "Text content may be missing for some reviews",
                "Keyword extraction is basic and language-agnostic"
            ],
            "confidence": 0.92
        }
        findings.append(finding_3)

# ============================================================================
# Construct output
# ============================================================================

output = {
    "status": "success" if len(findings) > 0 else "insufficient_data",
    "findings": findings
}

# Write output
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Analysis complete. {len(findings)} findings written to {output_path}")
