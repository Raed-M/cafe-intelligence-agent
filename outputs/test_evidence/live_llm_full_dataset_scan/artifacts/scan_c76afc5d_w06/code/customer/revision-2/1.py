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
analysis_start = datetime.fromisoformat("2026-02-16T00:00:00+03:00")
analysis_end = datetime.fromisoformat("2026-02-23T00:00:00+03:00")

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
    
    # Get source names from the data
    source_names = reviews_analysis['source'].unique().tolist()
    
    finding1 = {
        "title": "Review Rating Distribution and Average (Analysis Period)",
        "claim": f"During the analysis period (2026-02-16 to 2026-02-23), reviews averaged {avg_rating:.2f} stars with {len(reviews_analysis)} total reviews across {len(source_names)} source(s).",
        "finding_type": "rating_distribution",
        "metrics": {
            "average_rating": {
                "value": round(avg_rating, 2),
                "unit": "stars",
                "numerator": round(reviews_analysis['rating'].sum(), 2),
                "denominator": len(reviews_analysis),
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-02-23T00:00:00+03:00"
            },
            "total_reviews": {
                "value": len(reviews_analysis),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-02-23T00:00:00+03:00"
            }
        },
        "source_names": source_names,
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Analysis period: 2026-02-16 to 2026-02-23",
            f"Total reviews in dataset: {len(reviews_df)}",
            f"Reviews in analysis period: {len(reviews_analysis)}",
            f"Sources represented: {', '.join(source_names)}"
        ],
        "assumptions": [
            "Review dates are accurate and in UTC+3 timezone",
            "All reviews in the artifact are valid and complete",
            "Rating scale is consistent across all sources"
        ],
        "confidence": 0.95 if len(reviews_analysis) > 10 else 0.70
    }
    findings.append(finding1)

# Finding 2: Language Distribution
if len(reviews_analysis) > 0:
    language_counts = reviews_analysis['language'].value_counts()
    
    finding2 = {
        "title": "Review Language Distribution",
        "claim": f"Reviews in the analysis period are distributed across {len(language_counts)} language(s), with {language_counts.index[0]} being the most common ({language_counts.iloc[0]} reviews).",
        "finding_type": "language_distribution",
        "metrics": {
            "total_reviews_by_language": {
                "value": language_counts.to_dict(),
                "unit": "count",
                "numerator": len(reviews_analysis),
                "denominator": None,
                "period_start": "2026-02-16T00:00:00+03:00",
                "period_end": "2026-02-23T00:00:00+03:00"
            }
        },
        "source_names": reviews_analysis['source'].unique().tolist(),
        "sample_size": len(reviews_analysis),
        "coverage_notes": [
            f"Languages detected: {', '.join(language_counts.index.tolist())}",
            f"Total reviews analyzed: {len(reviews_analysis)}"
        ],
        "assumptions": [
            "Language field is accurately populated in the reviews artifact",
            "Language classification is consistent across all reviews"
        ],
        "confidence": 0.90
    }
    findings.append(finding2)

# Finding 3: Sentiment/Topic Analysis (Basic)
if len(reviews_analysis) > 0:
    # Count non-empty reviews
    non_empty_reviews = reviews_analysis[reviews_analysis['text'].notna() & (reviews_analysis['text'].str.len() > 0)]
    
    # Simple keyword detection for common cafe topics
    keywords = {
        'quality': ['quality', 'جودة', 'excellent', 'great', 'good', 'bad', 'poor'],
        'service': ['service', 'خدمة', 'staff', 'friendly', 'rude', 'slow', 'fast'],
        'price': ['price', 'سعر', 'expensive', 'cheap', 'value', 'cost'],
        'taste': ['taste', 'طعم', 'flavor', 'delicious', 'bitter', 'sweet']
    }
    
    topic_counts = {topic: 0 for topic in keywords}
    
    for text in non_empty_reviews['text']:
        if pd.isna(text):
            continue
        text_lower = str(text).lower()
        for topic, words in keywords.items():
            if any(word in text_lower for word in words):
                topic_counts[topic] += 1
    
    # Find most mentioned topic
    if any(topic_counts.values()):
        top_topic = max(topic_counts, key=topic_counts.get)
        top_count = topic_counts[top_topic]
        
        finding3 = {
            "title": "Review Topic Frequency (Analysis Period)",
            "claim": f"Among {len(non_empty_reviews)} reviews with text content, '{top_topic}' was the most frequently mentioned topic ({top_count} reviews).",
            "finding_type": "topic_frequency",
            "metrics": {
                "topic_mentions": {
                    "value": topic_counts,
                    "unit": "count",
                    "numerator": sum(topic_counts.values()),
                    "denominator": len(non_empty_reviews),
                    "period_start": "2026-02-16T00:00:00+03:00",
                    "period_end": "2026-02-23T00:00:00+03:00"
                },
                "reviews_with_text": {
                    "value": len(non_empty_reviews),
                    "unit": "count",
                    "numerator": len(non_empty_reviews),
                    "denominator": len(reviews_analysis),
                    "period_start": "2026-02-16T00:00:00+03:00",
                    "period_end": "2026-02-23T00:00:00+03:00"
                }
            },
            "source_names": reviews_analysis['source'].unique().tolist(),
            "sample_size": len(non_empty_reviews),
            "coverage_notes": [
                f"Reviews with text content: {len(non_empty_reviews)} out of {len(reviews_analysis)}",
                f"Topic detection based on keyword matching (English and Arabic)",
                f"Topics analyzed: {', '.join(keywords.keys())}"
            ],
            "assumptions": [
                "Keyword matching is a proxy for topic presence",
                "Single keyword occurrence indicates topic relevance",
                "Reviews may mention multiple topics",
                "Language-specific keywords are representative"
            ],
            "confidence": 0.75
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

print(f"Analysis complete. {len(findings)} finding(s) generated.")
print(f"Output written to {output_path}")
