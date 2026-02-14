#!/usr/bin/env python
"""
BPR (Bayesian Personalized Ranking) Training Script

This script trains a BPR model on user-book interactions data using the
implicit library and saves the recommendations to a parquet file.

Usage:
    python scripts/train_bpr.py

The script:
1. Loads and preprocesses the interaction data from parquet
2. Trains the BPR model with configurable hyperparameters
3. Generates top-k recommendations for all users (batched for memory efficiency)
4. Saves recommendations to a parquet file
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import polars as pl

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bookdb.models.bpr import BPR
from bookdb.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_PREDICTION_COL,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    # Data paths
    "data_path": "data/sar_interactions_small.parquet",
    "output_path": "predictions/bpr_recommendations.parquet",

    # Data preprocessing
    "col_user": "user_id",
    "col_item": "book_id",
    "col_rating": "weight",

    # BPR model hyperparameters (implicit library naming)
    "factors": 64,  # Latent factor dimension
    "iterations": 100,
    "learning_rate": 0.01,
    "regularization": 0.01,
    "num_threads": 0,  # 0 = auto

    # Recommendation settings
    "top_k": 100,
    "remove_seen": True,
    "batch_size": 1000,  # Users to process at once

    # Other
    "random_state": 42,
}


def load_data(data_path: str) -> pl.DataFrame:
    """Load interaction data from parquet file."""
    logger.info(f"Loading data from {data_path}")
    df = pl.read_parquet(data_path)
    logger.info(f"Loaded {df.height} interactions with schema: {df.schema}")
    return df


def preprocess_data(
    df: pl.DataFrame,
    col_user: str,
    col_item: str,
) -> pl.DataFrame:
    """
    Preprocess the interaction data:
    - Rename columns to expected format (userID, itemID)
    - Remove duplicates
    """
    logger.info("Preprocessing data...")

    # Rename columns to match expected format
    column_mapping = {
        col_user: DEFAULT_USER_COL,
        col_item: DEFAULT_ITEM_COL,
    }
    df = df.rename({k: v for k, v in column_mapping.items() if k in df.columns})

    # Remove duplicates (keep one interaction per user-item pair)
    df = df.unique(subset=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], keep="first")

    # Log statistics
    n_users = df.select(DEFAULT_USER_COL).n_unique()
    n_items = df.select(DEFAULT_ITEM_COL).n_unique()
    logger.info(f"Preprocessed dataset: {n_users} users, {n_items} items, {df.height} interactions")

    return df


def train_bpr_model(
    df: pl.DataFrame,
    config: dict,
) -> tuple[BPR, float]:
    """Train the BPR model on the DataFrame."""
    logger.info("Training BPR model...")
    logger.info(f"  factors={config['factors']}, iterations={config['iterations']}, "
                f"lr={config['learning_rate']}, reg={config['regularization']}")

    model = BPR(
        factors=config["factors"],
        iterations=config["iterations"],
        learning_rate=config["learning_rate"],
        regularization=config["regularization"],
        random_state=config["random_state"],
        num_threads=config["num_threads"],
    )

    start_time = time.time()
    model.fit(
        df,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating="rating" if "rating" in df.columns else None,
    )
    training_time = time.time() - start_time

    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Model: {model.n_users} users, {model.n_items} items")

    return model, training_time


def generate_recommendations(
    model: BPR,
    data: pl.DataFrame,
    top_k: int,
    remove_seen: bool,
    batch_size: int,
) -> pl.DataFrame:
    """Generate recommendations for all users in the data."""
    logger.info(f"Generating top-{top_k} recommendations (remove_seen={remove_seen})...")

    recommendations = model.recommend_k_items(
        data=data,
        top_k=top_k,
        remove_seen=remove_seen,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
        batch_size=batch_size,
    )

    n_users = recommendations.select(DEFAULT_USER_COL).n_unique()
    logger.info(f"Generated recommendations for {n_users} users")

    return recommendations


def save_recommendations(recommendations: pl.DataFrame, output_path: str):
    """Save recommendations to a parquet file."""
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving recommendations to {output_path_obj}")
    recommendations.write_parquet(output_path_obj)
    logger.info(f"Saved {recommendations.height} recommendations")


def main(config: dict):
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Starting BPR Training Pipeline (implicit library)")
    logger.info("=" * 60)

    # Log configuration
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Load data
    df = load_data(config["data_path"])

    # Preprocess
    df = preprocess_data(
        df,
        col_user=config["col_user"],
        col_item=config["col_item"],
    )

    # Train model (fit takes DataFrame directly now)
    model, training_time = train_bpr_model(df, config)

    # Generate recommendations
    recommendations = generate_recommendations(
        model,
        df,
        top_k=config["top_k"],
        remove_seen=config["remove_seen"],
        batch_size=config["batch_size"],
    )

    # Save recommendations
    save_recommendations(recommendations, config["output_path"])

    logger.info("=" * 60)
    logger.info("Training Pipeline Complete!")
    logger.info("=" * 60)

    return model, recommendations


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train BPR recommendation model")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to interaction data (parquet)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save recommendations (parquet)",
    )
    parser.add_argument(
        "-k", "--factors",
        type=int,
        default=None,
        help="Latent factor dimension",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=None,
        help="Regularization parameter",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of recommendations per user",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of users to process at once (memory vs speed tradeoff)",
    )
    parser.add_argument(
        "--keep-seen",
        action="store_true",
        help="Keep items already seen by user in recommendations",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Number of threads (0 = auto)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Start with default config
    config = DEFAULT_CONFIG.copy()

    # Override with command line arguments
    if args.data_path:
        config["data_path"] = args.data_path
    if args.output_path:
        config["output_path"] = args.output_path
    if args.factors:
        config["factors"] = args.factors
    if args.iterations:
        config["iterations"] = args.iterations
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.regularization:
        config["regularization"] = args.regularization
    if args.top_k:
        config["top_k"] = args.top_k
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.keep_seen:
        config["remove_seen"] = False
    if args.num_threads is not None:
        config["num_threads"] = args.num_threads

    # Run training
    model, recommendations = main(config)
