#!/usr/bin/env python
"""
SAR (Simple Algorithm for Recommendations) Training Script

This script trains a SAR model on user-book interactions data and logs all
training metrics, parameters, and model artifacts to MLFlow.

Usage:
    python scripts/train_sar.py [--config config.yaml]

The script:
1. Loads and preprocesses the interaction data
2. Splits data into train/test sets by timestamp
3. Trains the SAR model with configurable hyperparameters
4. Evaluates the model using standard recommendation metrics
5. Logs everything to MLFlow including model artifacts
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import polars as pl
from mlflow.models.signature import infer_signature

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bookdb.evaluation import evaluate_recommendations
from bookdb.models.sar import SARSingleNode, SIM_JACCARD, SIM_COSINE, SIM_LIFT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    # Data paths
    "data_path": "data/sar_interactions_sample.parquet",
    "mlflow_tracking_uri": "https://mlflow.yousef.gg",
    "experiment_name": "SAR_Recommendations",

    # Data preprocessing
    "col_user": "user_id",
    "col_item": "book_id",
    "col_rating": "weight",
    "col_timestamp": "timestamp",
    "test_size_ratio": 0.2,
    "min_user_interactions": 5,
    "min_item_interactions": 5,

    # SAR model hyperparameters
    "similarity_type": SIM_JACCARD,
    "time_decay_coefficient": 30,
    "timedecay_formula": True,
    "threshold": 1,
    "normalize": False,
    "use_torch": True,  # Enable PyTorch MPS/CUDA acceleration

    # Evaluation
    "top_k_values": [5, 10, 20],
    "random_seed": 42,
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
    col_rating: str,
    col_timestamp: str,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> pl.DataFrame:
    """
    Preprocess the interaction data:
    - Rename columns to SAR expected format
    - Filter users/items with too few interactions
    - Handle duplicates
    """
    logger.info("Preprocessing data...")

    # Rename columns to match SAR expected format
    column_mapping = {
        col_user: "userID",
        col_item: "itemID",
        col_rating: "rating",
        col_timestamp: "timestamp",
    }
    df = df.rename(column_mapping)

    initial_count = df.height

    # Filter users with too few interactions
    user_counts = df.group_by("userID").len()
    valid_users = user_counts.filter(pl.col("len") >= min_user_interactions).select("userID")
    df = df.join(valid_users, on="userID", how="inner")
    logger.info(f"Filtered users: {initial_count} -> {df.height} interactions")

    # Filter items with too few interactions
    item_counts = df.group_by("itemID").len()
    valid_items = item_counts.filter(pl.col("len") >= min_item_interactions).select("itemID")
    df = df.join(valid_items, on="itemID", how="inner")
    logger.info(f"Filtered items: {initial_count} -> {df.height} interactions")

    # Remove duplicates (keep most recent interaction per user-item pair)
    df = df.sort("timestamp", descending=True).unique(subset=["userID", "itemID"], keep="first")
    logger.info(f"After deduplication: {df.height} interactions")

    # Log statistics
    n_users = df.select("userID").n_unique()
    n_items = df.select("itemID").n_unique()
    logger.info(f"Final dataset: {n_users} users, {n_items} items, {df.height} interactions")

    return df


def train_test_split_by_time(
    df: pl.DataFrame,
    test_size_ratio: float = 0.2,
    col_timestamp: str = "timestamp",
    random_seed: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split data into train/test sets based on timestamp.
    Uses a stratified approach: for each user, the most recent interactions go to test.
    """
    logger.info(f"Splitting data with test_ratio={test_size_ratio}")

    np.random.seed(random_seed)

    # For each user, split their interactions by time
    # Most recent interactions go to test set
    df_sorted = df.sort([col_timestamp])

    # Calculate cutoff timestamp (e.g., last 20% of time range)
    min_ts = df_sorted.select(pl.col(col_timestamp).min()).item()
    max_ts = df_sorted.select(pl.col(col_timestamp).max()).item()
    time_range = max_ts - min_ts
    cutoff_ts = max_ts - (time_range * test_size_ratio)

    train_df = df_sorted.filter(pl.col(col_timestamp) < cutoff_ts)
    test_df = df_sorted.filter(pl.col(col_timestamp) >= cutoff_ts)

    # Ensure all users in test set also appear in train set
    train_users = set(train_df.select("userID").unique().to_series().to_list())
    test_df = test_df.filter(pl.col("userID").is_in(train_users))

    logger.info(f"Train set: {train_df.height} interactions")
    logger.info(f"Test set: {test_df.height} interactions")

    return train_df, test_df


def train_sar_model(
    train_df: pl.DataFrame,
    config: Dict[str, Any],
) -> SARSingleNode:
    """Train the SAR model."""
    logger.info("Training SAR model...")

    model = SARSingleNode(
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_timestamp="timestamp",
        col_prediction="prediction",
        similarity_type=config["similarity_type"],
        time_decay_coefficient=config["time_decay_coefficient"],
        timedecay_formula=config["timedecay_formula"],
        threshold=config["threshold"],
        normalize=config["normalize"],
        use_torch=config.get("use_torch", True),
    )

    start_time = time.time()
    model.fit(train_df)
    training_time = time.time() - start_time

    logger.info(f"Training completed in {training_time:.2f} seconds")

    return model, training_time


def evaluate_model(
    model: SARSingleNode,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    k_values: list = [5, 10, 20],
) -> Dict[str, float]:
    """
    Evaluate the trained SAR model.

    For each user in the test set, we:
    1. Use their training interactions to make recommendations
    2. Compare recommendations against their test interactions
    """
    logger.info("Evaluating model...")

    # Get unique users in test set
    test_users = test_df.select("userID").unique()

    # Get recommendations for test users
    # We use the model to recommend items, removing items seen in training
    recommendations = model.recommend_k_items(
        test=test_users,
        top_k=max(k_values),
        remove_seen=True,
    )

    # Get all items for coverage calculation
    all_items = train_df.select("itemID").unique().to_series().to_list()

    # Calculate metrics
    metrics = evaluate_recommendations(
        recommendations=recommendations,
        ground_truth=test_df,
        all_items=all_items,
        col_user="userID",
        col_item="itemID",
        col_prediction="prediction",
        k_values=k_values,
    )

    # Log metrics
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")

    return metrics, recommendations


def log_to_mlflow(
    model: SARSingleNode,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    training_time: float,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    recommendations: pl.DataFrame,
    run_name: Optional[str] = None,
):
    """Log all training artifacts to MLFlow."""
    logger.info("Logging to MLFlow...")

    # Set up MLFlow
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params({
            "similarity_type": config["similarity_type"],
            "time_decay_coefficient": config["time_decay_coefficient"],
            "timedecay_formula": config["timedecay_formula"],
            "threshold": config["threshold"],
            "normalize": config["normalize"],
            "test_size_ratio": config["test_size_ratio"],
            "min_user_interactions": config["min_user_interactions"],
            "min_item_interactions": config["min_item_interactions"],
            "random_seed": config["random_seed"],
        })

        # Log dataset statistics
        n_train_users = train_df.select("userID").n_unique()
        n_train_items = train_df.select("itemID").n_unique()
        n_test_users = test_df.select("userID").n_unique()
        n_test_items = test_df.select("itemID").n_unique()

        mlflow.log_params({
            "n_train_interactions": train_df.height,
            "n_test_interactions": test_df.height,
            "n_train_users": n_train_users,
            "n_train_items": n_train_items,
            "n_test_users": n_test_users,
            "n_test_items": n_test_items,
        })

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.log_metric("training_time_seconds", training_time)

        # Log model summary statistics
        mlflow.log_metric("model_n_users", model.n_users)
        mlflow.log_metric("model_n_items", model.n_items)

        # Save model artifacts
        artifacts_dir = Path("mlflow_artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        def safe_log_artifact(path: Path, description: str = None):
            """Safely log an artifact, handling missing boto3 for S3 backends."""
            try:
                mlflow.log_artifact(str(path))
                if description:
                    logger.info(f"Logged artifact: {description}")
            except Exception as e:
                logger.warning(f"Could not log artifact {path}: {e}")
                logger.warning("Artifact saved locally but not uploaded to MLFlow")

        # Save recommendations as artifact
        recommendations_path = artifacts_dir / "recommendations.parquet"
        recommendations.write_parquet(recommendations_path)
        safe_log_artifact(recommendations_path, "recommendations")

        # Save model configuration
        config_path = artifacts_dir / "model_config.json"
        model_config = {
            "similarity_type": config["similarity_type"],
            "time_decay_coefficient": config["time_decay_coefficient"],
            "timedecay_formula": config["timedecay_formula"],
            "threshold": config["threshold"],
            "normalize": config["normalize"],
            "n_users": model.n_users,
            "n_items": model.n_items,
        }
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)
        safe_log_artifact(config_path, "model config")

        # Save evaluation metrics
        metrics_path = artifacts_dir / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        safe_log_artifact(metrics_path, "evaluation metrics")

        # Save model state as custom artifact
        # Note: SAR model contains sparse matrices which need special handling
        model_state_path = artifacts_dir / "sar_model_state.npz"
        if model.user_affinity is not None:
            from scipy.sparse import save_npz
            save_npz(model_state_path, model.user_affinity)
            safe_log_artifact(model_state_path, "model state")

        # Save item similarity matrix
        item_sim_path = artifacts_dir / "item_similarity.npz"
        if model.item_similarity is not None:
            from scipy.sparse import save_npz
            if hasattr(model.item_similarity, 'toarray'):
                # It's a sparse matrix
                save_npz(item_sim_path, model.item_similarity)
            else:
                # It's a dense array - save as npz
                np.savez_compressed(item_sim_path, similarity=model.item_similarity)
            safe_log_artifact(item_sim_path, "item similarity")

        # Save mappings
        mappings_path = artifacts_dir / "mappings.json"
        mappings = {
            "index2item": {str(k): str(v) for k, v in model.index2item.items()},
            "index2user": {str(k): str(v) for k, v in model.index2user.items()},
        }
        with open(mappings_path, "w") as f:
            json.dump(mappings, f, indent=2)
        safe_log_artifact(mappings_path, "mappings")

        # Log the training script
        safe_log_artifact(Path(__file__), "training script")

        # Log tags
        mlflow.set_tags({
            "model_type": "SAR",
            "framework": "bookdb",
            "training_date": datetime.now().isoformat(),
        })

        logger.info(f"MLFlow run ID: {mlflow.active_run().info.run_id}")

        # Clean up artifacts directory
        for artifact in artifacts_dir.glob("*"):
            artifact.unlink()
        artifacts_dir.rmdir()

        return mlflow.active_run().info.run_id


def main(config: Dict[str, Any]):
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Starting SAR Training Pipeline")
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
        col_rating=config["col_rating"],
        col_timestamp=config["col_timestamp"],
        min_user_interactions=config["min_user_interactions"],
        min_item_interactions=config["min_item_interactions"],
    )

    # Split data
    train_df, test_df = train_test_split_by_time(
        df,
        test_size_ratio=config["test_size_ratio"],
        col_timestamp="timestamp",
        random_seed=config["random_seed"],
    )

    # Train model
    model, training_time = train_sar_model(train_df, config)

    # Evaluate model
    metrics, recommendations = evaluate_model(
        model,
        train_df,
        test_df,
        k_values=config["top_k_values"],
    )

    # Log to MLFlow
    run_name = f"sar_{config['similarity_type'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_id = log_to_mlflow(
        model,
        config,
        metrics,
        training_time,
        train_df,
        test_df,
        recommendations,
        run_name=run_name,
    )

    logger.info("=" * 60)
    logger.info("Training Pipeline Complete!")
    logger.info(f"MLFlow Run ID: {run_id}")
    logger.info("=" * 60)

    return model, metrics, run_id


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SAR recommendation model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (JSON)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to interaction data (parquet)",
    )
    parser.add_argument(
        "--similarity-type",
        type=str,
        choices=["cooccurrence", "cosine", "jaccard", "lift", "mutual information"],
        default=None,
        help="Similarity type for SAR model",
    )
    parser.add_argument(
        "--time-decay",
        type=int,
        default=None,
        help="Time decay coefficient in days",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Test set size ratio",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLFlow experiment name",
    )
    parser.add_argument(
        "--no-torch",
        action="store_true",
        help="Disable PyTorch MPS/CUDA acceleration",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Start with default config
    config = DEFAULT_CONFIG.copy()

    # Load config file if provided
    if args.config:
        with open(args.config) as f:
            file_config = json.load(f)
        config.update(file_config)

    # Override with command line arguments
    if args.data_path:
        config["data_path"] = args.data_path
    if args.similarity_type:
        config["similarity_type"] = args.similarity_type
    if args.time_decay:
        config["time_decay_coefficient"] = args.time_decay
    if args.test_size:
        config["test_size_ratio"] = args.test_size
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    if args.no_torch:
        config["use_torch"] = False

    # Run training
    model, metrics, run_id = main(config)
