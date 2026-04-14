#!/usr/bin/env python
"""
Neural Collaborative Filtering (NCF) Training Script

This script trains an NCF model on user-book interactions data using PyTorch
and saves the recommendations to a parquet file. All training metrics,
parameters, and model artifacts are logged to MLFlow.

Usage:
    python scripts/train_ncf.py

The script:
1. Loads environment variables from .env file
2. Loads and preprocesses the interaction data from parquet
3. Splits data into train/test sets
4. Trains the NCF model with configurable hyperparameters
5. Evaluates the model using standard recommendation metrics
6. Logs everything to MLFlow including model artifacts
7. Generates top-k recommendations for all users
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import polars as pl
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
env_path = project_root / ".env"
load_dotenv(env_path)


# Verify MLflow environment variables are set
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
from bookdb.models.ncf import NCF
from bookdb.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_RATING_COL,
    DEFAULT_USER_COL,
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
    "data_path": "data/ncf_interactions_merged.parquet",
    "output_path": "predictions/ncf_recommendations.parquet",
    # MLflow settings
    "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.yousef.gg"),
    "experiment_name": "NCF_Recommendations",
    # Data preprocessing
    "col_user": "user_id",
    "col_item": "book_id",
    "col_rating": "weight",
    # Data splitting
    "test_size_ratio": 0.15,
    "min_user_interactions": 5,
    # NCF model hyperparameters
    "factors": 64,
    "layers": [64, 32, 16],
    "epochs": 20,
    "batch_size": 256,
    "learning_rate": 0.001,
    "dropout": 0.2,
    "n_negatives": 4,
    "device": None,  # Auto-detect
    # Recommendation settings
    "top_k": 30,
    "remove_seen": True,
    "batch_size": 1000,
    "model_checkpoint_path": "predictions/ncf_model_checkpoint",
    # Evaluation
    "top_k_values": [5, 10, 20],
    # Other
    "random_state": 42,
}


def load_data(
    data_path: str,
    required_columns: Optional[list[str]] = None,
) -> pl.DataFrame:
    """Load interaction data from parquet file."""
    logger.info(f"Loading data from {data_path}")

    lazy_df = pl.scan_parquet(data_path)

    required_columns = list(
        dict.fromkeys([col for col in (required_columns or []) if col])
    )

    if required_columns:
        available_columns = set(lazy_df.collect_schema().names())
        missing_columns = [
            col for col in required_columns if col not in available_columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {data_path}: {missing_columns}"
            )
        lazy_df = lazy_df.select(required_columns)

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
    """Preprocess the interaction data."""
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
        valid_users = user_counts.filter(pl.col("len") >= min_user_interactions).select(
            DEFAULT_USER_COL
        )
        df = df.join(valid_users, on=DEFAULT_USER_COL, how="inner")
        logger.info(f"Filtered users: {initial_count} -> {df.height} interactions")

    # Remove duplicates
    df = df.unique(subset=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], keep="first")
    logger.info(f"After deduplication: {df.height} interactions")

    # Use Float32 for ratings
    if DEFAULT_RATING_COL in df.columns:
        df = df.with_columns(pl.col(DEFAULT_RATING_COL).cast(pl.Float32))

    n_users = df.select(DEFAULT_USER_COL).n_unique()
    n_items = df.select(DEFAULT_ITEM_COL).n_unique()
    logger.info(
        f"Preprocessed dataset: {n_users} users, {n_items} items, {df.height} interactions"
    )

    return df


def train_test_split_by_user(
    df: pl.DataFrame,
    test_size_ratio: float = 0.2,
    col_user: str = DEFAULT_USER_COL,
    col_item: str = DEFAULT_ITEM_COL,
    random_seed: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Split data into train/test sets with deterministic hash-based sampling."""
    logger.info(f"Splitting data with test_ratio={test_size_ratio}")

    if not 0 < test_size_ratio < 1:
        raise ValueError("test_size_ratio must be between 0 and 1")

    split_mod = 10_000
    test_threshold = int(test_size_ratio * split_mod)

    df = df.with_columns(
        (pl.struct([col_user, col_item]).hash(seed=random_seed) % split_mod).alias(
            "__split_bucket"
        )
    )

    train_df = df.filter(pl.col("__split_bucket") >= test_threshold).drop(
        "__split_bucket"
    )
    test_df = df.filter(pl.col("__split_bucket") < test_threshold).drop(
        "__split_bucket"
    )

    # Ensure all users in test set also appear in train set
    train_users = train_df.select(col_user).unique()
    orphan_test_users = (
        test_df.join(train_users, on=col_user, how="anti").select(col_user).unique()
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
            f"to keep {orphan_test_users.height} test users"
        )

    n_train_users = train_df.select(col_user).n_unique()
    n_test_users = test_df.select(col_user).n_unique()
    logger.info(f"Train set: {train_df.height} interactions, {n_train_users} users")
    logger.info(f"Test set: {test_df.height} interactions, {n_test_users} users")

    return train_df, test_df


def train_ncf_model(
    df: pl.DataFrame,
    config: dict,
) -> Tuple[NCF, float]:
    """Train the NCF model on the DataFrame."""
    logger.info("Training NCF model...")
    logger.info(f"  factors={config['factors']}, layers={config['layers']}")
    logger.info(f"  epochs={config['epochs']}, batch_size={config['batch_size']}")

    model = NCF(
        factors=config["factors"],
        layers=config["layers"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        dropout=config["dropout"],
        n_negatives=config["n_negatives"],
        device=config.get("device"),
        random_state=config["random_state"],
    )

    start_time = time.time()
    model.fit(
        df,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        verbose=True,
    )
    training_time = time.time() - start_time

    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Model: {model.n_users} users, {model.n_items} items")

    return model, training_time


def evaluate_model(
    model: NCF,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    k_values: list = [5, 10, 20],
    remove_seen: bool = True,
    batch_size: int = 10000,
) -> Tuple[Dict[str, float], pl.DataFrame]:
    """Evaluate the trained NCF model."""
    logger.info("Evaluating model...")

    test_users = test_df.select(DEFAULT_USER_COL).unique()

    recommendations = model.recommend_k_items(
        data=test_users,
        top_k=max(k_values),
        remove_seen=remove_seen,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
        batch_size=batch_size,
    )

    all_items = train_df.select(DEFAULT_ITEM_COL).unique().to_series()

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

    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")

    return metrics, recommendations


def log_to_mlflow(
    model: NCF,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    training_time: float,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    recommendations_path: Path,
    n_recommendations: int,
    run_name: Optional[str] = None,
) -> str:
    """Log all training artifacts to MLFlow."""
    logger.info("Logging to MLFlow...")

    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run(run_name=run_name):
        # Log model hyperparameters
        mlflow.log_params(
            {
                "factors": config["factors"],
                "layers": config["layers"],
                "epochs": config["epochs"],
                "batch_size": config["batch_size"],
                "learning_rate": config["learning_rate"],
                "dropout": config["dropout"],
                "n_negatives": config["n_negatives"],
            }
        )

        # Log data processing parameters
        mlflow.log_params(
            {
                "test_size_ratio": config["test_size_ratio"],
                "min_user_interactions": config["min_user_interactions"],
                "remove_seen": config["remove_seen"],
            }
        )

        # Log dataset statistics
        n_train_users = train_df.select(DEFAULT_USER_COL).n_unique()
        n_train_items = train_df.select(DEFAULT_ITEM_COL).n_unique()
        n_test_users = test_df.select(DEFAULT_USER_COL).n_unique()
        n_test_items = test_df.select(DEFAULT_ITEM_COL).n_unique()

        mlflow.log_params(
            {
                "n_train_interactions": train_df.height,
                "n_test_interactions": test_df.height,
                "n_train_users": n_train_users,
                "n_train_items": n_train_items,
                "n_test_users": n_test_users,
                "n_test_items": n_test_items,
            }
        )

        # Log evaluation metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log training metrics
        mlflow.log_metric("training_time_seconds", training_time)

        # Log model summary
        mlflow.log_metric("model_n_users", model.n_users)
        mlflow.log_metric("model_n_items", model.n_items)
        mlflow.log_metric("n_output_recommendations", n_recommendations)

        # Save model checkpoint for MLFlow
        artifacts_dir = Path("mlflow_artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        model_dir = artifacts_dir / "ncf_model"
        model.save(model_dir)

        def safe_log_artifact(path: Path, description: Optional[str] = None):
            """Safely log an artifact."""
            try:
                mlflow.log_artifact(str(path))
                if description:
                    logger.info(f"Logged artifact: {description}")
            except Exception as e:
                logger.warning(f"Could not log artifact {path}: {e}")

        # Log recommendations
        if recommendations_path.exists():
            safe_log_artifact(recommendations_path, "recommendations parquet")

        # Log model directory as a single addressable artifact path for registry
        mlflow.log_artifacts(str(model_dir), artifact_path="ncf_model")

        # Save config
        config_path = artifacts_dir / "model_config.json"
        model_config = {
            "factors": config["factors"],
            "layers": config["layers"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "dropout": config["dropout"],
            "n_users": model.n_users,
            "n_items": model.n_items,
        }
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)
        safe_log_artifact(config_path, "model config")

        # Save metrics
        metrics_path = artifacts_dir / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        safe_log_artifact(metrics_path, "evaluation metrics")

        # Log training script
        safe_log_artifact(Path(__file__), "training script")

        # Log tags
        mlflow.set_tags(
            {
                "model_type": "NCF",
                "framework": "PyTorch",
                "training_date": datetime.now().isoformat(),
            }
        )

        active_run = mlflow.active_run()
        run_id = active_run.info.run_id if active_run else "unknown"
        logger.info(f"MLFlow run ID: {run_id}")

        # Cleanup
        for artifact in artifacts_dir.glob("*"):
            if artifact.is_file():
                artifact.unlink()
            elif artifact.is_dir():
                for sub in artifact.glob("*"):
                    sub.unlink()
                artifact.rmdir()
        artifacts_dir.rmdir()

        return run_id


def save_recommendations(
    model: NCF,
    output_path: str,
    top_k: int,
    remove_seen: bool,
    batch_size: int,
) -> Tuple[Path, int]:
    """Generate and save recommendations to parquet."""
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating recommendations to {output_path_obj}")

    # Get all users from model
    all_users = list(model._user_id_map.keys())
    users_df = pl.DataFrame({DEFAULT_USER_COL: all_users})

    recommendations = model.recommend_k_items(
        data=users_df,
        top_k=top_k,
        remove_seen=remove_seen,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
        batch_size=batch_size,
    )

    recommendations.write_parquet(output_path_obj)
    logger.info(f"Saved {recommendations.height} recommendations to {output_path_obj}")

    return output_path_obj, recommendations.height


def main(config: dict) -> Tuple[NCF, str, Dict[str, float], str]:
    """Main training pipeline."""
    _check_mlflow_env()

    logger.info("=" * 60)
    logger.info("Starting NCF Training Pipeline (PyTorch)")
    logger.info("=" * 60)

    logger.info("Configuration:")
    for key, value in config.items():
        if "key" in key.lower() or "secret" in key.lower() or "token" in key.lower():
            value = "***"
        logger.info(f"  {key}: {value}")

    # Load data
    df = load_data(
        config["data_path"],
        required_columns=[config["col_user"], config["col_item"]],
    )

    # Preprocess
    df = preprocess_data(
        df,
        col_user=config["col_user"],
        col_item=config["col_item"],
        col_rating=config.get("col_rating"),
        min_user_interactions=config["min_user_interactions"],
    )

    # Split
    train_df, test_df = train_test_split_by_user(
        df,
        test_size_ratio=config["test_size_ratio"],
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        random_seed=config["random_state"],
    )
    del df
    gc.collect()

    # Train
    model, training_time = train_ncf_model(train_df, config)

    # Save checkpoint
    checkpoint_path = Path(config["model_checkpoint_path"])
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_path)
    logger.info(f"Model checkpoint saved to {checkpoint_path}")

    # Generate recommendations
    recommendations_path, n_recommendations = save_recommendations(
        model=model,
        output_path=config["output_path"],
        top_k=config["top_k"],
        remove_seen=config["remove_seen"],
        batch_size=config["batch_size"],
    )

    # Evaluate
    metrics, _ = evaluate_model(
        model,
        train_df,
        test_df,
        k_values=config["top_k_values"],
        remove_seen=config["remove_seen"],
        batch_size=config["batch_size"],
    )

    # Log to MLFlow
    run_name = f"ncf_f{config['factors']}_e{config['epochs']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_id = log_to_mlflow(
        model,
        config,
        metrics,
        training_time,
        train_df,
        test_df,
        recommendations_path,
        n_recommendations,
        run_name=run_name,
    )

    del train_df, test_df
    gc.collect()

    logger.info("=" * 60)
    logger.info("Training Pipeline Complete!")
    logger.info(f"MLFlow Run ID: {run_id}")
    logger.info("=" * 60)

    return model, str(recommendations_path), metrics, run_id


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train NCF recommendation model")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--factors", type=int, default=None)
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated list, e.g., '64,32,16'",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda", "mps"]
    )
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--min-user-interactions", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = DEFAULT_CONFIG.copy()

    if args.data_path:
        config["data_path"] = args.data_path
    if args.output_path:
        config["output_path"] = args.output_path
    if args.factors:
        config["factors"] = args.factors
    if args.layers:
        config["layers"] = [int(x) for x in args.layers.split(",")]
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.dropout:
        config["dropout"] = args.dropout
    if args.top_k:
        config["top_k"] = args.top_k
    if args.device:
        config["device"] = args.device
    if args.test_size:
        config["test_size_ratio"] = args.test_size
    if args.min_user_interactions:
        config["min_user_interactions"] = args.min_user_interactions

    model, recommendations, metrics, run_id = main(config)
