"""
Label book reviews with emotions using a pre-trained multilingual emotion classifier.

Uses MilaNLProc/xlm-emo-t (XLM-RoBERTa based) which supports multiple languages and classifies 8 emotions:
- anger, anticipation, disgust, fear, joy, sadness, surprise, trust

knowledge distillation: generate pseudo-labels that will be used to fine-tune a smaller model for production.
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")
import torch
from tqdm import tqdm
from transformers import pipeline

# Emotions from MilaNLProc/xlm-emo-t model (Plutchik's 8 basic emotions)
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

def get_device():
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def truncate_text(text: str, max_chars: int = 512) -> str:
    """Truncate text to max characters (model has token limit)."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return text[:max_chars]


def label_batch(classifier, texts: list[str]) -> list[dict]:
    """Label a batch of texts and return emotion + score for each."""
    results = []
    try:
        outputs = classifier(texts, truncation=True, max_length=512)
        for output in outputs:
            # output is a list of dicts with 'label' and 'score'
            # Get top emotion
            top = max(output, key=lambda x: x['score'])
            results.append({
                "emotion": top['label'],
                "emotion_score": top['score']
            })
    except Exception as e:
        print(f"Error in batch: {e}")
        # Return neutral for failed texts
        results = [{"emotion": "neutral", "emotion_score": 0.0} for _ in texts]
    return results


def main():
    parser = argparse.ArgumentParser(description="Label reviews with emotions")
    parser.add_argument(
        "--input",
        type=str,
        default="data/5_goodreads_reviews_final_clean.parquet",
        help="Input parquet file with reviews"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/reviews_with_emotions.parquet",
        help="Output parquet file with emotion labels"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N reviews instead of processing all (for testing)"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10000,
        help="Save checkpoint every N reviews"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from checkpoint file"
    )
    args = parser.parse_args()

    # Paths
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / args.input
    output_path = base_dir / args.output
    checkpoint_path = base_dir / "data" / "emotion_labeling_checkpoint.parquet"

    print(f"Loading reviews from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Total reviews: {len(df):,}")

    # Sample if requested
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
        print(f"Sampled {len(df):,} reviews")

    # Resume from checkpoint if provided
    start_idx = 0
    labeled_data = []
    if args.resume_from and Path(args.resume_from).exists():
        print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint_df = pd.read_parquet(args.resume_from)
        labeled_data = checkpoint_df.to_dict('records')
        start_idx = len(labeled_data)
        print(f"Resuming from index {start_idx:,}")

    # Load emotion classifier
    device = get_device()
    print(f"Using device: {device}")
    print("Loading multilingual emotion classifier (MilaNLProc/xlm-emo-t)...")

    classifier = pipeline(
        "text-classification",
        model="MilaNLProc/xlm-emo-t",
        top_k=None,  # Return all scores
        device=device if device != "mps" else -1,  # MPS not fully supported
    )
    print("Classifier loaded!")

    # Process reviews in batches
    reviews = df['review_text'].tolist()
    review_ids = df['review_id'].tolist()

    total_batches = (len(reviews) - start_idx + args.batch_size - 1) // args.batch_size

    print(f"\nProcessing {len(reviews) - start_idx:,} reviews in {total_batches:,} batches...")

    for batch_start in tqdm(range(start_idx, len(reviews), args.batch_size), desc="Labeling"):
        batch_end = min(batch_start + args.batch_size, len(reviews))
        batch_texts = [truncate_text(t) for t in reviews[batch_start:batch_end]]
        batch_ids = review_ids[batch_start:batch_end]

        # Skip empty texts
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

        # Get emotions for valid texts
        emotions = label_batch(classifier, valid_texts)

        # Map back to all texts in batch
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

        # Save checkpoint
        if len(labeled_data) % args.checkpoint_every < args.batch_size:
            checkpoint_df = pd.DataFrame(labeled_data)
            checkpoint_df.to_parquet(checkpoint_path, index=False)
            print(f"\nCheckpoint saved: {len(labeled_data):,} reviews processed")

    # Create final output
    print("\nCreating final output...")
    emotions_df = pd.DataFrame(labeled_data)

    # Merge with original data (only keep necessary columns for fine-tuning)
    result_df = df[['review_id', 'book_id', 'review_text']].merge(emotions_df, on='review_id', how='left')

    # Save
    result_df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    # Print statistics
    print("\nEmotion distribution:")
    print(result_df['emotion'].value_counts())

    # Clean up checkpoint
    if checkpoint_path.exists():
        os.remove(checkpoint_path)
        print("Checkpoint cleaned up")


if __name__ == "__main__":
    main()
