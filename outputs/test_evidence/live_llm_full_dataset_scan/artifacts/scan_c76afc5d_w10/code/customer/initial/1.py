import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter
import re

# Load environment metadata
with open(os.environ['ANALYST_INPUTS_JSON']) as f:
    run_meta = json.load(f)

inputs = run_meta['inputs']
output_path = run_meta['output_path']

# Read input artifacts
pos_df = pd.read_parquet(inputs['pos'])
menu_df = pd.read_parquet(inputs['menu'])
reviews_df = pd.read_parquet(inputs['reviews'])

# Define analysis period
analysis_start = "2026-03-16T00:00:00+03:00"
analysis_end = "2026-03-23T00:00:00+03:00"

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
    
    # Calculate sentiment distribution
    high_ratings = len(analysis_reviews[analysis_reviews['rating'] >= 4])
    low_ratings = len(analysis_reviews[analysis_reviews['rating'] <= 2])
    
    finding_1 = {
        "title": "Customer Rating Distribution and Average Sentiment",
        "claim": f"During the analysis period (2026-03-16 to 2026-03-23), customers provided {len(analysis_reviews)} reviews with an average rating of {avg_rating:.2f} out of 5. High ratings (4-5 stars) represent {high_ratings} reviews ({100*high_ratings/len(analysis_reviews):.1f}%), while low ratings (1-2 stars) represent {low_ratings} reviews ({100*low_ratings/len(analysis_reviews):.1f}%).",
        "finding_type": "sentiment_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": len(analysis_reviews),
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "total_reviews": {
                "value": len(analysis_reviews),
                "unit": "count",
                "numerator": None,
                "denominator": None,
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "high_ratings_pct": {
                "value": round(100*high_ratings/len(analysis_reviews), 1),
                "unit": "percent",
                "numerator": high_ratings,
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            },
            "low_ratings_pct": {
                "value": round(100*low_ratings/len(analysis_reviews), 1),
                "unit": "percent",
                "numerator": low_ratings,
                "denominator": len(analysis_reviews),
                "period_start": analysis_start,
                "period_end": analysis_end
            }
        },
        "source_names": list(analysis_reviews['source'].unique()),
        "sample_size": len(analysis_reviews),
        "coverage_notes": [
            f"Reviews from {len(analysis_reviews['source'].unique())} source(s)",
            f"Language distribution: {dict(analysis_reviews['language'].value_counts())}"
        ],
        "assumptions": [
            "Rating scale is 1-5 stars",
            "High ratings defined as 4-5 stars",
            "Low ratings defined as 1-2 stars"
        ],
        "confidence": 0.95
    }
    findings.append(finding_1)

# Finding 2: Language Distribution and Sentiment by Language
if len(analysis_reviews) > 0:
    lang_dist = analysis_reviews['language'].value_counts()
    
    # Calculate average rating by language
    lang_ratings = {}
    for lang in analysis_reviews['language'].unique():
        lang_data = analysis_reviews[analysis_reviews['language'] == lang]
        lang_ratings[lang] = {
            'avg_rating': lang_data['rating'].mean(),
            'count': len(lang_data)
        }
    
    # Only report if we have meaningful data for multiple languages
    if len(lang_ratings) > 1:
        finding_2 = {
            "title": "Sentiment Analysis by Language",
            "claim": f"Review sentiment varies by language. {', '.join([f'{lang}: {lang_ratings[lang][\"count\"]} reviews (avg {lang_ratings[lang][\"avg_rating\"]:.2f} stars)' for lang in lang_ratings.keys()])}.",
            "finding_type": "language_sentiment_comparison",
            "metrics": {
                f"avg_rating_{lang}": {
                    "value": round(lang_ratings[lang]['avg_rating'], 2),
                    "unit": "stars",
                    "numerator": lang_ratings[lang]['count'],
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
                for lang in lang_ratings.keys()
            },
            "source_names": list(analysis_reviews['source'].unique()),
            "sample_size": len(analysis_reviews),
            "coverage_notes": [
                f"Language coverage: {dict(lang_dist)}"
            ],
            "assumptions": [
                "Language field accurately reflects review language",
                "Rating scale is consistent across languages"
            ],
            "confidence": 0.85
        }
        findings.append(finding_2)

# Finding 3: Topic/Sentiment Keywords Analysis
if len(analysis_reviews) > 0:
    # Analyze text content for common themes
    english_reviews = analysis_reviews[analysis_reviews['language'] == 'English']
    arabic_reviews = analysis_reviews[analysis_reviews['language'] == 'Arabic']
    
    # Extract common words from positive and negative reviews
    positive_reviews = analysis_reviews[analysis_reviews['rating'] >= 4]
    negative_reviews = analysis_reviews[analysis_reviews['rating'] <= 2]
    
    # Count non-empty reviews
    non_empty_reviews = analysis_reviews[analysis_reviews['text'].notna() & (analysis_reviews['text'].str.len() > 0)]
    
    if len(non_empty_reviews) > 0:
        # Analyze sentiment indicators
        positive_text = ' '.join(positive_reviews[positive_reviews['text'].notna()]['text'].astype(str).tolist()).lower()
        negative_text = ' '.join(negative_reviews[negative_reviews['text'].notna()]['text'].astype(str).tolist()).lower()
        
        # Common positive indicators
        positive_keywords = ['good', 'great', 'excellent', 'love', 'best', 'amazing', 'perfect', 'delicious', 'tasty', 'nice']
        negative_keywords = ['bad', 'poor', 'terrible', 'hate', 'worst', 'awful', 'slow', 'cold', 'stale', 'rude']
        
        positive_mentions = sum(positive_text.count(kw) for kw in positive_keywords)
        negative_mentions = sum(negative_text.count(kw) for kw in negative_keywords)
        
        finding_3 = {
            "title": "Review Content Analysis - Positive vs Negative Sentiment Indicators",
            "claim": f"Analysis of {len(non_empty_reviews)} non-empty reviews reveals {len(positive_reviews)} positive reviews (rating 4-5) and {len(negative_reviews)} negative reviews (rating 1-2). Positive reviews contain {positive_mentions} mentions of quality indicators, while negative reviews contain {negative_mentions} mentions of quality concerns.",
            "finding_type": "content_sentiment_analysis",
            "metrics": {
                "positive_reviews_count": {
                    "value": len(positive_reviews),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "negative_reviews_count": {
                    "value": len(negative_reviews),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                },
                "non_empty_reviews": {
                    "value": len(non_empty_reviews),
                    "unit": "count",
                    "numerator": None,
                    "denominator": None,
                    "period_start": analysis_start,
                    "period_end": analysis_end
                }
            },
            "source_names": list(analysis_reviews['source'].unique()),
            "sample_size": len(non_empty_reviews),
            "coverage_notes": [
                f"Analysis based on {len(non_empty_reviews)} reviews with text content",
                f"Language distribution in analysis: English {len(english_reviews)}, Arabic {len(arabic_reviews)}"
            ],
            "assumptions": [
                "Keyword matching is case-insensitive",
                "Keywords are indicative of sentiment but not exhaustive",
                "Text field contains original customer language"
            ],
            "confidence": 0.75
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
