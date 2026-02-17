#!/usr/bin/env python
"""
BPR (Bayesian Personalized Ranking) Training Script

This script trains a BPR model on user-book interactions data using the
implicit library and saves the recommendations to a parquet file. All
training metrics, parameters, and model artifacts are logged to MLFlow.

Usage:
    python scripts/train_bpr.py

The script:
1. Loads environment variables from .env file
2. Loads and preprocesses the interaction data from parquet
3. Splits data into train/test sets
4. Trains the BPR model with configurable hyperparameters
5. Evaluates the model using standard recommendation metrics
6. Logs everything to MLFlow including model artifacts
7. Generates top-k recommendations for all users (batched for memory efficiency)
8. Saves recommendations to a parquet file
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import mlflow
import polars as pl
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
env_path = project_root / ".env"
load_dotenv(env_path)

# Verify MLflow environment variables are set (without exposing secrets)
def _check_mlflow_env():
    """Check that required MLflow environment variables are set."""
    required_vars = [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "MLFLOW_S3_ENDPOINT_URL",
    ]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        logger.warning(f"Missing MLflow environment variables: {missing}")
        logger.warning(f"Make sure .env file exists at {env_path}")
    return len(missing) == 0

from bookdb.evaluation import evaluate_recommendations
from bookdb.models.bpr import BPR
from bookdb.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
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
    "data_path": "data/bpr_interactions_merged.parquet",
    "output_path": "predictions/bpr_recommendations.parquet",

    # MLflow settings (loaded from environment if not specified)
    "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.yousef.gg"),
    "experiment_name": "BPR_Recommendations",

    # Data preprocessing
    "col_user": "user_id",
    "col_item": "book_id",
    "col_rating": "weight",

    # Data splitting
    "test_size_ratio": 0.15,
    "min_user_interactions": 5,

    # BPR model hyperparameters (implicit library naming)
    "factors": 64,  # Latent factor dimension
    "iterations": 100,
    "learning_rate": 0.01,
    "regularization": 0.01,
    "num_threads": 0,  # 0 = auto

    # Recommendation settings
    "top_k": 30,
    "remove_seen": True,
    "batch_size": 10000,  # Users to process at once
    "model_checkpoint_path": "predictions/bpr_model_checkpoint",

    # Evaluation
    "top_k_values": [5, 10, 20],

    # Other
    "random_state": 42,
}


def load_data(
    data_path: str,
    required_columns: Optional[list[str]] = None,
    optional_columns: Optional[list[str]] = None,
) -> pl.DataFrame:
    """Load interaction data from parquet file."""
    logger.info(f"Loading data from {data_path}")

    lazy_df = pl.scan_parquet(data_path)

    required_columns = list(dict.fromkeys([col for col in (required_columns or []) if col]))
    optional_columns = list(dict.fromkeys([col for col in (optional_columns or []) if col]))

    if required_columns:
        available_columns = set(lazy_df.collect_schema().names())
        missing_columns = [col for col in required_columns if col not in available_columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {data_path}: {missing_columns}"
            )
        selected_columns = required_columns + [
            col for col in optional_columns if col in available_columns and col not in required_columns
        ]
        if optional_columns:
            missing_optional_columns = [col for col in optional_columns if col not in available_columns]
            if missing_optional_columns:
                logger.info(
                    f"Optional columns not found in {data_path}: {missing_optional_columns}"
                )
        lazy_df = lazy_df.select(selected_columns)

    # Streaming collect reduces peak memory during parquet load.
    df = lazy_df.collect(engine="streaming")
    logger.info(f"Loaded {df.height} interactions with schema: {df.schema}")
    return df


def preprocess_data(
    df: pl.DataFrame,
    col_user: str,
    col_item: str,
    col_rating: Optional[str] = None,
    min_user_interactions: int = 5,
) -> pl.DataFrame:
    """
    Preprocess the interaction data:
    - Rename columns to expected format (userID, itemID)
    - Filter users with too few interactions
    - Remove duplicates
    """
    logger.info("Preprocessing data...")

    # Rename columns to match expected format
    column_mapping = {
        col_user: DEFAULT_USER_COL,
        col_item: DEFAULT_ITEM_COL,
    }
    if col_rating:
        column_mapping[col_rating] = DEFAULT_RATING_COL
    df = df.rename({k: v for k, v in column_mapping.items() if k in df.columns})

    initial_count = df.height

    # Filter users with too few interactions
    if min_user_interactions > 1:
        user_counts = df.group_by(DEFAULT_USER_COL).len()
        valid_users = user_counts.filter(pl.col("len") >= min_user_interactions).select(DEFAULT_USER_COL)
        df = df.join(valid_users, on=DEFAULT_USER_COL, how="inner")
        logger.info(f"Filtered users: {initial_count} -> {df.height} interactions")

    # Remove duplicates (keep one interaction per user-item pair)
    df = df.unique(subset=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], keep="first")
    logger.info(f"After deduplication: {df.height} interactions")

    # Use Float32 for ratings to reduce memory footprint on large datasets.
    if DEFAULT_RATING_COL in df.columns:
        df = df.with_columns(pl.col(DEFAULT_RATING_COL).cast(pl.Float32))

    # Log statistics
    n_users = df.select(DEFAULT_USER_COL).n_unique()
    n_items = df.select(DEFAULT_ITEM_COL).n_unique()
    logger.info(f"Preprocessed dataset: {n_users} users, {n_items} items, {df.height} interactions")

    return df


def train_test_split_by_user(
    df: pl.DataFrame,
    test_size_ratio: float = 0.2,
    col_user: str = DEFAULT_USER_COL,
    col_item: str = DEFAULT_ITEM_COL,
    random_seed: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split data into train/test sets with deterministic hash-based sampling.

    This approach ensures:
    - Deterministic split without allocating a full random NumPy array
    - Approximate test_size_ratio on average
    - Every user in the test set also appears in the training set

    Args:
        df: Input DataFrame with user and item columns
        test_size_ratio: Fraction of items to hold out for testing per user
        col_user: User column name
        col_item: Item column name
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Splitting data with test_ratio={test_size_ratio}")

    if not 0 < test_size_ratio < 1:
        raise ValueError("test_size_ratio must be between 0 and 1")

    split_mod = 10_000
    test_threshold = int(test_size_ratio * split_mod)

    # Hash on (user, item) gives deterministic per-interaction pseudo-randomness
    # while avoiding a large in-memory random array.
    df = df.with_columns(
        (pl.struct([col_user, col_item]).hash(seed=random_seed) % split_mod).alias("__split_bucket")
    )

    train_df = df.filter(pl.col("__split_bucket") >= test_threshold).drop("__split_bucket")
    test_df = df.filter(pl.col("__split_bucket") < test_threshold).drop("__split_bucket")

    # Ensure all users in test set also appear in train set by moving one row
    # per missing user from test to train.
    train_users = train_df.select(col_user).unique()
    orphan_test_users = (
        test_df.join(train_users, on=col_user, how="anti")
        .select(col_user)
        .unique()
    )

    if orphan_test_users.height > 0:
        promoted_to_train = (
            test_df.join(orphan_test_users, on=col_user, how="inner")
            .group_by(col_user)
            .head(1)
        )
        test_df = test_df.join(
            promoted_to_train.select([col_user, col_item]),
            on=[col_user, col_item],
            how="anti",
        )
        train_df = pl.concat([train_df, promoted_to_train], how="vertical")
        logger.info(
            f"Moved {promoted_to_train.height} interactions to train "
            f"to keep {orphan_test_users.height} test users represented in training"
        )

    # Log statistics
    n_train_users = train_df.select(col_user).n_unique()
    n_test_users = test_df.select(col_user).n_unique()
    logger.info(f"Train set: {train_df.height} interactions, {n_train_users} users")
    logger.info(f"Test set: {test_df.height} interactions, {n_test_users} users")

    return train_df, test_df


def train_bpr_model(
    df: pl.DataFrame,
    config: dict,
) -> Tuple[BPR, float]:
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
        col_rating=DEFAULT_RATING_COL if DEFAULT_RATING_COL in df.columns else None,
    )
    training_time = time.time() - start_time

    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Model: {model.n_users} users, {model.n_items} items")

    return model, training_time


def evaluate_model(
    model: BPR,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    k_values: list = [5, 10, 20],
    remove_seen: bool = True,
    batch_size: int = 10000,
) -> Tuple[Dict[str, float], pl.DataFrame]:
    """
    Evaluate the trained BPR model.

    For each user in the test set, we:
    1. Use the model to generate recommendations (excluding training items)
    2. Compare recommendations against their test interactions

    Args:
        model: Trained BPR model
        train_df: Training data (used to determine seen items)
        test_df: Test data (ground truth)
        k_values: List of k values for metrics
        remove_seen: Whether to remove items seen in training from recommendations
        batch_size: Batch size for recommendation generation

    Returns:
        Tuple of (metrics dict, recommendations DataFrame)
    """
    logger.info("Evaluating model...")

    # Get unique users in test set
    test_users = test_df.select(DEFAULT_USER_COL).unique()

    # Generate recommendations for test users
    # We use train_df as the reference for seen items
    recommendations = model.recommend_k_items(
        data=test_users,
        top_k=max(k_values),
        remove_seen=remove_seen,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
        batch_size=batch_size,
    )

    # Get all items for coverage calculation. Keep as Series to avoid
    # creating an additional large Python list.
    all_items = train_df.select(DEFAULT_ITEM_COL).unique().to_series()

    # Calculate metrics using the evaluation module
    logger.info("Computing evaluation metrics...")
    metrics = evaluate_recommendations(
        recommendations=recommendations,
        ground_truth=test_df,
        all_items=all_items,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
        k_values=k_values,
        mlflow_compatible_names=True,
    )

    # Log metrics
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")

    return metrics, recommendations


def log_to_mlflow(
    model: BPR,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    training_time: float,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    recommendations_path: Path,
    n_recommendations: int,
    run_name: Optional[str] = None,
) -> str:
    """
    Log all training artifacts to MLFlow.

    Args:
        model: Trained BPR model
        config: Configuration dictionary
        metrics: Evaluation metrics dictionary
        training_time: Training time in seconds
        train_df: Training data
        test_df: Test data
        recommendations_path: Path to generated recommendations parquet
        n_recommendations: Number of generated recommendation rows
        run_name: Optional run name

    Returns:
        MLflow run ID
    """
    logger.info("Logging to MLFlow...")

    # Set up MLFlow
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run(run_name=run_name):
        # Log model hyperparameters
        mlflow.log_params({
            "factors": config["factors"],
            "iterations": config["iterations"],
            "learning_rate": config["learning_rate"],
            "regularization": config["regularization"],
            "num_threads": config["num_threads"],
            "random_state": config["random_state"],
        })

        # Log data processing parameters
        mlflow.log_params({
            "test_size_ratio": config["test_size_ratio"],
            "min_user_interactions": config["min_user_interactions"],
            "remove_seen": config["remove_seen"],
            "batch_size": config["batch_size"],
        })

        # Log dataset statistics
        n_train_users = train_df.select(DEFAULT_USER_COL).n_unique()
        n_train_items = train_df.select(DEFAULT_ITEM_COL).n_unique()
        n_test_users = test_df.select(DEFAULT_USER_COL).n_unique()
        n_test_items = test_df.select(DEFAULT_ITEM_COL).n_unique()

        mlflow.log_params({
            "n_train_interactions": train_df.height,
            "n_test_interactions": test_df.height,
            "n_train_users": n_train_users,
            "n_train_items": n_train_items,
            "n_test_users": n_test_users,
            "n_test_items": n_test_items,
        })

        # Log evaluation metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log training metrics
        mlflow.log_metric("training_time_seconds", training_time)

        # Log model summary statistics
        mlflow.log_metric("model_n_users", model.n_users)
        mlflow.log_metric("model_n_items", model.n_items)
        mlflow.log_metric("n_output_recommendations", n_recommendations)

        # Create artifacts directory
        artifacts_dir = Path("mlflow_artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        def safe_log_artifact(path: Path, description: Optional[str] = None):
            """Safely log an artifact, handling missing boto3 for S3 backends."""
            try:
                mlflow.log_artifact(str(path))
                if description:
                    logger.info(f"Logged artifact: {description}")
            except Exception as e:
                logger.warning(f"Could not log artifact {path}: {e}")
                logger.warning("Artifact saved locally but not uploaded to MLFlow")

        # Log the final output recommendations parquet as artifact
        if recommendations_path.exists():
            safe_log_artifact(recommendations_path, "recommendations parquet")
        else:
            logger.warning(
                f"Recommendations file not found, cannot log artifact: {recommendations_path}"
            )

        # Save model using BPR's save method
        model_dir = artifacts_dir / "bpr_model"
        model.save(model_dir)
        # Log the entire model directory
        for artifact_file in model_dir.glob("*"):
            safe_log_artifact(artifact_file, f"model/{artifact_file.name}")

        # Save model configuration
        config_path = artifacts_dir / "model_config.json"
        model_config = {
            "factors": config["factors"],
            "iterations": config["iterations"],
            "learning_rate": config["learning_rate"],
            "regularization": config["regularization"],
            "n_users": model.n_users,
            "n_items": model.n_items,
            "random_state": config["random_state"],
        }
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)
        safe_log_artifact(config_path, "model config")

        # Save evaluation metrics
        metrics_path = artifacts_dir / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        safe_log_artifact(metrics_path, "evaluation metrics")

        # Log the training script
        safe_log_artifact(Path(__file__), "training script")

        # Log tags
        mlflow.set_tags({
            "model_type": "BPR",
            "framework": "implicit",
            "training_date": datetime.now().isoformat(),
        })

        # Get run ID safely
        active_run = mlflow.active_run()
        run_id = active_run.info.run_id if active_run else "unknown"
        logger.info(f"MLFlow run ID: {run_id}")

        # Clean up artifacts directory
        for artifact in artifacts_dir.glob("*"):
            if artifact.is_file():
                artifact.unlink()
            elif artifact.is_dir():
                for sub_artifact in artifact.glob("*"):
                    sub_artifact.unlink()
                artifact.rmdir()
        artifacts_dir.rmdir()

        return run_id


def save_recommendations_streaming(
    model: BPR,
    output_path: str,
    top_k: int,
    remove_seen: bool,
    batch_size: int,
) -> Tuple[Path, int]:
    """Generate and save recommendations directly to a single parquet file."""
    import pyarrow.parquet as pq

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path_obj.with_suffix(output_path_obj.suffix + ".tmp")

    if temp_output_path.exists():
        temp_output_path.unlink()

    logger.info(f"Generating and streaming recommendations to {output_path_obj}")

    writer = None
    total_rows = 0
    total_users = model.n_users
    processed_users = 0

    try:
        # Iterate over model user mappings directly to avoid a full unique-user collect.
        for batch_index, batch_users in enumerate(
            _iter_user_batches(model._user_id_map.keys(), batch_size),
            start=1,
        ):
            batch_users_df = pl.DataFrame({DEFAULT_USER_COL: batch_users})
            batch_recommendations = model.recommend_k_items(
                data=batch_users_df,
                top_k=top_k,
                remove_seen=remove_seen,
                col_user=DEFAULT_USER_COL,
                col_item=DEFAULT_ITEM_COL,
                col_prediction=DEFAULT_PREDICTION_COL,
                batch_size=batch_size,
            )

            if batch_recommendations.height > 0:
                batch_table = batch_recommendations.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(str(temp_output_path), batch_table.schema)
                writer.write_table(batch_table)
                total_rows += batch_recommendations.height

            processed_users += len(batch_users)
            if batch_index % 10 == 0 or processed_users >= total_users:
                logger.info(f"  Processed {processed_users}/{total_users} users")

            del batch_users_df, batch_recommendations
            if batch_index % 25 == 0:
                gc.collect()
    except Exception:
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise
    finally:
        if writer is not None:
            writer.close()

    if total_rows > 0:
        temp_output_path.replace(output_path_obj)
    else:
        # Preserve schema in the output even when there are no recommendations.
        empty_df = pl.DataFrame({
            DEFAULT_USER_COL: pl.Series([], dtype=model._user_dtype or pl.String),
            DEFAULT_ITEM_COL: pl.Series([], dtype=model._item_dtype or pl.String),
            DEFAULT_PREDICTION_COL: pl.Series([], dtype=pl.Float32),
        })
        empty_df.write_parquet(output_path_obj)
        if temp_output_path.exists():
            temp_output_path.unlink()

    logger.info(f"Saved {total_rows} recommendations to {output_path_obj}")
    return output_path_obj, total_rows


def save_model_checkpoint(model: BPR, checkpoint_path: str) -> Path:
    """Save a local model checkpoint that can be reused if later stages fail."""
    checkpoint_path_obj = Path(checkpoint_path)
    checkpoint_path_obj.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving local model checkpoint to {checkpoint_path_obj}")
    model.save(checkpoint_path_obj)
    logger.info(f"Saved local model checkpoint to {checkpoint_path_obj}")

    return checkpoint_path_obj


def _iter_user_batches(users: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    """Yield user IDs in fixed-size batches without materializing all users at once."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    user_iterator = iter(users)
    while True:
        batch = list(islice(user_iterator, batch_size))
        if not batch:
            break
        yield batch


def main(config: dict) -> Tuple[BPR, str, Dict[str, float], str]:
    """
    Main training pipeline.

    Returns:
        Tuple of (model, recommendations_output_path, metrics, mlflow_run_id)
    """
    # Check MLflow environment before starting
    _check_mlflow_env()

    logger.info("=" * 60)
    logger.info("Starting BPR Training Pipeline (implicit library)")
    logger.info("=" * 60)

    # Log configuration
    logger.info("Configuration:")
    for key, value in config.items():
        # Mask sensitive values
        if "key" in key.lower() or "secret" in key.lower() or "token" in key.lower():
            value = "***"
        logger.info(f"  {key}: {value}")

    # Load data
    df = load_data(
        config["data_path"],
        required_columns=[
            config["col_user"],
            config["col_item"],
        ],
        optional_columns=[config.get("col_rating")],
    )

    # Preprocess
    df = preprocess_data(
        df,
        col_user=config["col_user"],
        col_item=config["col_item"],
        col_rating=config.get("col_rating"),
        min_user_interactions=config["min_user_interactions"],
    )

    # Split data into train/test
    train_df, test_df = train_test_split_by_user(
        df,
        test_size_ratio=config["test_size_ratio"],
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        random_seed=config["random_state"],
    )
    del df
    gc.collect()

    # Train model on training data
    model, training_time = train_bpr_model(train_df, config)

    checkpoint_start_time = time.time()
    checkpoint_path = save_model_checkpoint(
        model=model,
        checkpoint_path=config["model_checkpoint_path"],
    )
    checkpoint_time = time.time() - checkpoint_start_time
    logger.info(
        f"Model checkpoint save completed in {checkpoint_time:.2f} seconds "
        f"({checkpoint_path})"
    )

    recommendations_start_time = time.time()
    recommendations_output_path, n_recommendations = save_recommendations_streaming(
        model=model,
        output_path=config["output_path"],
        top_k=config["top_k"],
        remove_seen=config["remove_seen"],
        batch_size=config["batch_size"],
    )
    recommendations_time = time.time() - recommendations_start_time
    logger.info(
        f"Recommendation export completed in {recommendations_time:.2f} seconds "
        f"({n_recommendations} rows)"
    )

    # Evaluate model on test data
    evaluation_start_time = time.time()
    metrics, evaluation_recommendations = evaluate_model(
        model,
        train_df,
        test_df,
        k_values=config["top_k_values"],
        remove_seen=config["remove_seen"],
        batch_size=config["batch_size"],
    )
    evaluation_time = time.time() - evaluation_start_time
    logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
    del evaluation_recommendations
    gc.collect()

    # Log to MLFlow
    mlflow_start_time = time.time()
    run_name = f"bpr_f{config['factors']}_i{config['iterations']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_id = log_to_mlflow(
        model,
        config,
        metrics,
        training_time,
        train_df,
        test_df,
        recommendations_output_path,
        n_recommendations,
        run_name=run_name,
    )
    mlflow_time = time.time() - mlflow_start_time
    logger.info(f"MLflow logging completed in {mlflow_time:.2f} seconds")

    del train_df, test_df
    gc.collect()

    logger.info("=" * 60)
    logger.info("Training Pipeline Complete!")
    logger.info(f"MLFlow Run ID: {run_id}")
    logger.info("=" * 60)

    return model, str(recommendations_output_path), metrics, run_id


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
        "--model-checkpoint-path",
        type=str,
        default=None,
        help="Path to save local model checkpoint before evaluation",
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
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Test set size ratio (default: 0.2)",
    )
    parser.add_argument(
        "--min-user-interactions",
        type=int,
        default=None,
        help="Minimum interactions per user (default: 5)",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name",
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
    if args.model_checkpoint_path:
        config["model_checkpoint_path"] = args.model_checkpoint_path
    if args.keep_seen:
        config["remove_seen"] = False
    if args.num_threads is not None:
        config["num_threads"] = args.num_threads
    if args.test_size is not None:
        config["test_size_ratio"] = args.test_size
    if args.min_user_interactions is not None:
        config["min_user_interactions"] = args.min_user_interactions
    if args.mlflow_tracking_uri:
        config["mlflow_tracking_uri"] = args.mlflow_tracking_uri
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name

    # Run training
    model, recommendations, metrics, run_id = main(config)
