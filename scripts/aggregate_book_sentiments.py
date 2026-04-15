"""
Aggregate emotion labels per book for chatbot lookup.

Takes the labeled reviews and computes per-book sentiment statistics:
- Dominant emotion (most frequent)
- Emotion distribution (counts and percentages)
- Average confidence score
- Total review count

Output is a lightweight lookup table for the chatbot.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Emotions (8 from xlm-emo-t model, data currently has 4: anger, fear, joy, sadness)
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]


def aggregate_emotions(df: pd.DataFrame, min_reviews: int = 1) -> pd.DataFrame:
    """
    Aggregate emotions per book.

    Args:
        df: DataFrame with book_id, emotion, emotion_score columns
        min_reviews: Minimum reviews required per book

    Returns:
        DataFrame with per-book sentiment statistics
    """
    logger.info(f"Aggregating emotions for {df['book_id'].nunique():,} unique books...")

    # Group by book
    grouped = df.groupby('book_id')

    # Compute aggregations
    agg_data = []
    for book_id, group in grouped:
        emotion_counts = group['emotion'].value_counts().to_dict()
        total_reviews = len(group)

        if total_reviews < min_reviews:
            continue

        # Dominant emotion (most frequent)
        dominant_emotion = group['emotion'].value_counts().index[0]
        dominant_count = emotion_counts[dominant_emotion]
        dominant_pct = dominant_count / total_reviews

        # Emotion percentages
        emotion_pcts = {e: emotion_counts.get(e, 0) / total_reviews for e in EMOTIONS}

        # Average confidence
        avg_confidence = group['emotion_score'].mean()

        agg_data.append({
            'book_id': book_id,
            'dominant_emotion': dominant_emotion,
            'dominant_emotion_pct': round(dominant_pct, 3),
            'emotion_counts': emotion_counts,
            'emotion_percentages': {k: round(v, 3) for k, v in emotion_pcts.items()},
            'total_reviews': total_reviews,
            'avg_confidence': round(avg_confidence, 3),
        })

    result_df = pd.DataFrame(agg_data)
    logger.info(f"Aggregated {len(result_df):,} books (min {min_reviews} reviews)")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate emotion labels per book for chatbot lookup"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/reviews_with_emotions.parquet",
        help="Input parquet file with labeled reviews"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/book_sentiments.parquet",
        help="Output parquet file with per-book sentiments"
    )
    parser.add_argument(
        "--min-reviews",
        type=int,
        default=1,
        help="Minimum reviews required per book"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum emotion score to include review"
    )
    args = parser.parse_args()

    # Paths
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / args.input
    output_path = base_dir / args.output

    # Load data
    logger.info(f"Loading labeled reviews from {input_path}...")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} reviews")

    # Filter by confidence
    if args.min_confidence > 0:
        df = df[df['emotion_score'] >= args.min_confidence]
        logger.info(f"After confidence filter (>= {args.min_confidence}): {len(df):,}")

    # Aggregate
    result_df = aggregate_emotions(df, min_reviews=args.min_reviews)

    # Stats
    logger.info("\n" + "=" * 50)
    logger.info("AGGREGATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total books: {len(result_df):,}")
    logger.info(f"Total reviews: {result_df['total_reviews'].sum():,}")
    logger.info(f"Avg reviews per book: {result_df['total_reviews'].mean():.1f}")
    logger.info(f"\nDominant emotion distribution:")
    logger.info(result_df['dominant_emotion'].value_counts().to_string())

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, index=False)
    logger.info(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
