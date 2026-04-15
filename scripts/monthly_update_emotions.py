#!/usr/bin/env python
"""
Monthly emotion update pipeline for book reviews.

This script labels NEW reviews with emotions and re-aggregates book sentiments.
It uses the fine-tuned emotion classifier (FedeIola/emotion-classifier-reviews)
which was trained via knowledge distillation from MilaNLProc/xlm-emo-t.

FIRST RUN: Creates the manifest and processes all existing labeled reviews.

SUBSEQUENT RUNS: Exports new reviews from the platform DB since the last run,
                 labels them with emotions, merges with existing data, and
                 re-aggregates book sentiments.

Usage:
    python scripts/monthly_update_emotions.py [options]

    # dry-run (no writes):
    python scripts/monthly_update_emotions.py --dry-run

    # force update even if no new reviews:
    python scripts/monthly_update_emotions.py --force

    # skip DB export (use existing data only):
    python scripts/monthly_update_emotions.py --no-db
"""

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import pipeline

# ---------------------------------------------------------------------------
# Bootstrap: add project root to path and load .env
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
load_dotenv(_project_root / ".env")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("monthly_update_emotions")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_LIVE_SINCE = datetime(2026, 2, 1, tzinfo=timezone.utc)

_DEFAULT_MANIFEST_PATH = _project_root / "data" / "emotions_manifest.json"
_DEFAULT_RUN_LOG_PATH = _project_root / "data" / "emotions_runs.jsonl"

# Data paths
_LABELED_REVIEWS_PATH = _project_root / "data" / "reviews_with_emotions.parquet"
_BOOK_SENTIMENTS_PATH = _project_root / "data" / "book_sentiments.parquet"
_NEW_REVIEWS_TEMP_PATH = _project_root / "data" / "new_reviews_emotions_temp.parquet"

# Emotions from the classifier
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
def get_device() -> str:
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------
def truncate_text(text: str, max_chars: int = 512) -> str:
    """Truncate text to max characters."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return text[:max_chars]


def label_batch(classifier, texts: list[str]) -> list[dict]:
    """Label a batch of texts and return emotion + score for each."""
    results = []
    try:
        outputs = classifier(texts, truncation=True, max_length=512)
        for output in outputs:
            top = max(output, key=lambda x: x['score'])
            results.append({
                "emotion": top['label'],
                "emotion_score": top['score']
            })
    except Exception as e:
        logger.warning(f"Error in batch: {e}")
        results = [{"emotion": "neutral", "emotion_score": 0.0} for _ in texts]
    return results


# ---------------------------------------------------------------------------
# Export new reviews from platform DB
# ---------------------------------------------------------------------------
def _export_new_reviews(since: datetime) -> Optional[pd.DataFrame]:
    """
    Export reviews created after *since* from the platform database.

    Returns DataFrame with columns: review_id, book_id, review_text, created_at
    Returns None if DB unavailable or no new reviews.
    """
    from sqlalchemy import select
    from sqlalchemy.orm import Session
    from bookdb.db.models import Book, Review
    from bookdb.db.session import SessionLocal

    try:
        session: Session = SessionLocal()
        try:
            stmt = (
                select(
                    Review.id.label("review_id"),
                    Book.goodreads_id.label("book_id"),
                    Review.text.label("review_text"),
                    Review.created_at,
                )
                .join(Book, Book.id == Review.book_id)
                .where(Review.created_at > since)
                .where(Book.goodreads_id.is_not(None))
                .where(Review.text.is_not(None))
                .where(Review.text != "")
            )
            rows = session.execute(stmt).fetchall()
        finally:
            session.close()
    except Exception as exc:
        logger.warning(f"Could not connect to DB for review export: {exc}")
        return None

    if not rows:
        logger.info("No new reviews found since the cutoff date")
        return None

    df = pd.DataFrame([
        {
            "review_id": str(r.review_id),
            "book_id": int(r.book_id),
            "review_text": r.review_text,
            "created_at": r.created_at,
        }
        for r in rows
    ])
    logger.info(f"Exported {len(df):,} new reviews from the database")
    return df


# ---------------------------------------------------------------------------
# Label reviews with emotions
# ---------------------------------------------------------------------------
def _label_reviews_with_emotions(
    df: pd.DataFrame,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Label reviews with emotions using the multilingual classifier.

    Returns DataFrame with added columns: emotion, emotion_score
    """
    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info("Loading emotion classifier (FedeIola/emotion-classifier-reviews)...")

    classifier = pipeline(
        "text-classification",
        model="FedeIola/emotion-classifier-reviews",
        top_k=None,
        device=device if device != "mps" else -1,
    )
    logger.info("Classifier loaded!")

    reviews = df['review_text'].tolist()
    review_ids = df['review_id'].tolist()

    labeled_data = []
    total_batches = (len(reviews) + batch_size - 1) // batch_size

    logger.info(f"Processing {len(reviews):,} reviews in {total_batches:,} batches...")

    for batch_start in tqdm(range(0, len(reviews), batch_size), desc="Labeling"):
        batch_end = min(batch_start + batch_size, len(reviews))
        batch_texts = [truncate_text(t) for t in reviews[batch_start:batch_end]]
        batch_ids = review_ids[batch_start:batch_end]

        valid_indices = [i for i, t in enumerate(batch_texts) if t.strip()]
        if not valid_indices:
            for rid in batch_ids:
                labeled_data.append({
                    "review_id": rid,
                    "emotion": "neutral",
                    "emotion_score": 0.0
                })
            continue

        valid_texts = [batch_texts[i] for i in valid_indices]
        emotions = label_batch(classifier, valid_texts)

        emotion_idx = 0
        for i, rid in enumerate(batch_ids):
            if i in valid_indices:
                labeled_data.append({
                    "review_id": rid,
                    **emotions[emotion_idx]
                })
                emotion_idx += 1
            else:
                labeled_data.append({
                    "review_id": rid,
                    "emotion": "neutral",
                    "emotion_score": 0.0
                })

    emotions_df = pd.DataFrame(labeled_data)
    result_df = df.merge(emotions_df, on='review_id', how='left')

    return result_df


# ---------------------------------------------------------------------------
# Aggregate book sentiments
# ---------------------------------------------------------------------------
def _aggregate_book_sentiments(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate emotion labels per book to create book-level sentiment data.

    Returns DataFrame with columns:
        book_id, dominant_emotion, dominant_emotion_pct, total_reviews,
        anger_pct, anticipation_pct, disgust_pct, fear_pct, joy_pct,
        sadness_pct, surprise_pct, trust_pct
    """
    logger.info("Aggregating book sentiments...")

    # Group by book and emotion
    emotion_counts = (
        labeled_df
        .groupby(['book_id', 'emotion'])
        .size()
        .reset_index(name='count')
    )

    # Pivot to get emotion counts per book
    pivot = emotion_counts.pivot(index='book_id', columns='emotion', values='count').fillna(0)

    # Ensure all emotions are present
    for emotion in EMOTIONS:
        if emotion not in pivot.columns:
            pivot[emotion] = 0

    # Calculate total reviews per book
    pivot['total_reviews'] = pivot[EMOTIONS].sum(axis=1)

    # Calculate percentages
    for emotion in EMOTIONS:
        pivot[f'{emotion}_pct'] = pivot[emotion] / pivot['total_reviews']

    # Find dominant emotion
    emotion_pcts = pivot[[f'{e}_pct' for e in EMOTIONS]]
    pivot['dominant_emotion'] = emotion_pcts.idxmax(axis=1).str.replace('_pct', '')
    pivot['dominant_emotion_pct'] = emotion_pcts.max(axis=1)

    # Select final columns
    result_cols = ['dominant_emotion', 'dominant_emotion_pct', 'total_reviews'] + \
                  [f'{e}_pct' for e in EMOTIONS]

    result_df = pivot[result_cols].reset_index()
    logger.info(f"Aggregated sentiments for {len(result_df):,} books")

    return result_df


# ---------------------------------------------------------------------------
# Manifest management
# ---------------------------------------------------------------------------
def _write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Manifest written -> {path}")


def _load_manifest(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _append_run_log(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# First run
# ---------------------------------------------------------------------------
def _run_first_update(args: argparse.Namespace) -> int:
    """
    First run: create manifest from existing labeled reviews data.
    """
    logger.info("=" * 60)
    logger.info("FIRST RUN - Creating emotions manifest")
    logger.info("=" * 60)

    # Check if labeled reviews exist
    if not _LABELED_REVIEWS_PATH.exists():
        logger.error(f"Labeled reviews not found at {_LABELED_REVIEWS_PATH}")
        logger.error("Run label_emotions.py first to create initial labeled data")
        return 1

    # Load existing labeled reviews
    logger.info(f"Loading existing labeled reviews from {_LABELED_REVIEWS_PATH}")
    labeled_df = pd.read_parquet(_LABELED_REVIEWS_PATH)
    logger.info(f"Loaded {len(labeled_df):,} labeled reviews")

    # Re-aggregate sentiments
    sentiments_df = _aggregate_book_sentiments(labeled_df)

    if not args.dry_run:
        sentiments_df.to_parquet(_BOOK_SENTIMENTS_PATH, index=False)
        logger.info(f"Saved book sentiments to {_BOOK_SENTIMENTS_PATH}")

    now_iso = datetime.now(tz=timezone.utc).isoformat()
    manifest = {
        "is_first_run": True,
        "run_at": now_iso,
        "last_update_at": now_iso,
        "total_labeled_reviews": len(labeled_df),
        "total_books_with_sentiment": len(sentiments_df),
        "outcome": "accepted",
    }

    if not args.dry_run:
        _write_manifest(manifest, Path(args.manifest_path))
    else:
        logger.info("[dry-run] Skipping manifest write")

    _print_summary(manifest)
    return 0


# ---------------------------------------------------------------------------
# Subsequent runs
# ---------------------------------------------------------------------------
def _run_update(args: argparse.Namespace) -> int:
    """
    Regular monthly update: label new reviews and re-aggregate.
    """
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        logger.info("No manifest found - running first update")
        return _run_first_update(args)

    manifest = _load_manifest(manifest_path)
    last_update_str = manifest.get("last_update_at", APP_LIVE_SINCE.isoformat())
    last_update = datetime.fromisoformat(last_update_str)

    logger.info(f"Last update: {last_update_str}")
    logger.info(f"Total labeled reviews (previous): {manifest.get('total_labeled_reviews', 0):,}")

    # Export new reviews
    new_reviews_df: Optional[pd.DataFrame] = None
    if not args.no_db:
        logger.info(f"Exporting reviews created since {last_update_str}...")
        new_reviews_df = _export_new_reviews(since=last_update)

        if new_reviews_df is None or len(new_reviews_df) == 0:
            if not args.force:
                logger.info("No new reviews found. Use --force to re-aggregate anyway.")
                record = {
                    "run_at": datetime.now(tz=timezone.utc).isoformat(),
                    "outcome": "skipped",
                    "reason": "no_new_reviews",
                }
                if not args.dry_run:
                    _append_run_log(record, Path(args.run_log_path))
                return 0
            else:
                logger.info("--force: proceeding with re-aggregation")
    else:
        logger.info("--no-db: skipping review export")

    # Load existing labeled reviews
    if not _LABELED_REVIEWS_PATH.exists():
        logger.error(f"Labeled reviews not found at {_LABELED_REVIEWS_PATH}")
        return 1

    logger.info("Loading existing labeled reviews...")
    existing_df = pd.read_parquet(_LABELED_REVIEWS_PATH)
    logger.info(f"Loaded {len(existing_df):,} existing labeled reviews")

    # Label new reviews if we have any
    if new_reviews_df is not None and len(new_reviews_df) > 0:
        logger.info(f"Labeling {len(new_reviews_df):,} new reviews...")
        new_labeled_df = _label_reviews_with_emotions(new_reviews_df, args.batch_size)

        # Merge with existing
        merged_df = pd.concat([existing_df, new_labeled_df], ignore_index=True)

        # Remove duplicates (keep latest)
        merged_df = merged_df.drop_duplicates(subset=['review_id'], keep='last')
        logger.info(f"Total after merge: {len(merged_df):,} reviews")

        if not args.dry_run:
            merged_df.to_parquet(_LABELED_REVIEWS_PATH, index=False)
            logger.info(f"Updated labeled reviews at {_LABELED_REVIEWS_PATH}")
    else:
        merged_df = existing_df

    # Re-aggregate sentiments
    sentiments_df = _aggregate_book_sentiments(merged_df)

    if not args.dry_run:
        sentiments_df.to_parquet(_BOOK_SENTIMENTS_PATH, index=False)
        logger.info(f"Updated book sentiments at {_BOOK_SENTIMENTS_PATH}")

    now_iso = datetime.now(tz=timezone.utc).isoformat()
    new_reviews_count = len(new_reviews_df) if new_reviews_df is not None else 0

    run_record = {
        "run_at": now_iso,
        "outcome": "accepted",
        "new_reviews_labeled": new_reviews_count,
        "total_labeled_reviews": len(merged_df),
        "total_books_with_sentiment": len(sentiments_df),
    }

    new_manifest = {
        **run_record,
        "is_first_run": False,
        "last_update_at": now_iso,
    }

    if not args.dry_run:
        _write_manifest(new_manifest, Path(args.manifest_path))
        _append_run_log(run_record, Path(args.run_log_path))
    else:
        logger.info("[dry-run] Skipping manifest/log writes")

    _print_summary(new_manifest)
    return 0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _print_summary(record: dict) -> None:
    outcome = record.get("outcome", "unknown").upper()
    sep = "=" * 60
    logger.info(sep)
    logger.info(f"OUTCOME: {outcome}")
    logger.info(f"Total labeled reviews: {record.get('total_labeled_reviews', 'N/A'):,}")
    logger.info(f"Books with sentiment: {record.get('total_books_with_sentiment', 'N/A'):,}")
    if "new_reviews_labeled" in record:
        logger.info(f"New reviews labeled: {record['new_reviews_labeled']:,}")
    logger.info(f"Book sentiments -> {_BOOK_SENTIMENTS_PATH}")
    logger.info(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monthly emotion update pipeline for book reviews"
    )
    parser.add_argument(
        "--manifest-path",
        default=str(_DEFAULT_MANIFEST_PATH),
        help="Path to the emotions manifest JSON file",
    )
    parser.add_argument(
        "--run-log-path",
        default=str(_DEFAULT_RUN_LOG_PATH),
        help="Path to the JSONL run log",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for emotion labeling (default: 32)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force update even if no new reviews found",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip exporting new reviews from the database",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all stages but skip file writes",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.dry_run:
        logger.info("DRY RUN mode - no files will be written")

    manifest_path = Path(args.manifest_path)
    is_first_run = not manifest_path.exists()

    if is_first_run:
        return _run_first_update(args)
    else:
        return _run_update(args)


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        logger.error("Unhandled exception in monthly_update_emotions.py")
        logger.error(traceback.format_exc())
        exit_code = 1
    sys.exit(exit_code)
