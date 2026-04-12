#!/usr/bin/env python
"""
Weekly offline retraining pipeline for the BPR recommendation model.

FIRST RUN: detected automatically when BPR_Recommender does not exist in the
           MLflow Model Registry. Loads baseline metrics from sample-artifacts/,
           counts the preprocessed historical dataset size, exports any app
           interactions recorded since APP_LIVE_SINCE, merges, trains, registers
           the model, and writes the manifest.

SUBSEQUENT RUNS: loads the manifest written by the previous accepted run, exports
                 app interactions since the last accepted training cutoff, checks
                 whether enough new data has arrived (threshold), trains, compares
                 metrics against the manifest baseline, and promotes or rejects.

Usage:
    python scripts/weekly_retrain.py [options]

    # dry-run against tiny test data (no DB, no writes):
    python scripts/weekly_retrain.py \\
        --dry-run --no-db \\
        --bpr-data-path data/bpr_interactions_merged_tiny.parquet

    # force a retrain even if data threshold not met:
    python scripts/weekly_retrain.py --force
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

import mlflow
import polars as pl
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# ---------------------------------------------------------------------------
# Bootstrap: add project root to path and load .env
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
load_dotenv(_project_root / ".env")

# Imported here (not from bookdb.db) to avoid triggering SQLAlchemy engine
# creation at module level, which would block on DB connect.
import scripts.train_bpr as _bpr

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("weekly_retrain")

# ---------------------------------------------------------------------------
# Constants
#   APP_LIVE_SINCE distinguishes between our data and goodreads, only for first run
# ---------------------------------------------------------------------------
APP_LIVE_SINCE = datetime(2026, 2, 1, tzinfo=timezone.utc)

_DEFAULT_MANIFEST_PATH = _project_root / "data" / "retrain_manifest.json"
_DEFAULT_RUN_LOG_PATH = _project_root / "data" / "retrain_runs.jsonl"
_SAMPLE_ARTIFACTS_DIR = _project_root / "sample-artifacts"

_BPR_REGISTRY_NAME = "BPR_Recommender"

# Merged parquet path written during a retrain run (overwritten each time).
_MERGED_BPR_PATH = _project_root / "data" / "bpr_interactions_retrain.parquet"

# ---------------------------------------------------------------------------
# MLflow Model Registry helpers
# ---------------------------------------------------------------------------

def _setup_mlflow() -> MlflowClient:
    """Configure MLflow tracking URI and return a client."""
    uri = os.getenv("MLFLOW_TRACKING_URI")
    logger.info(f"Connecting to MLflow at {uri} …")
    mlflow.set_tracking_uri(uri)
    return MlflowClient()


def _model_registered(client: MlflowClient, name: str) -> bool:
    """Return True if *name* exists in the MLflow Model Registry."""
    try:
        client.get_registered_model(name)
        return True
    except MlflowException:
        return False


def _register_model(
    client: MlflowClient,
    run_id: str,
    artifact_filename: str,
    registry_name: str,
    stage: str,
    dry_run: bool,
) -> Optional[str]:
    """
    Register a run artifact as a new model version and transition its stage.

    Returns the version string, or None on dry_run.
    """
    model_uri = f"runs:/{run_id}/{artifact_filename}"
    if dry_run:
        logger.info(f"[dry-run] Would register {model_uri} → {registry_name} ({stage})")
        return None

    logger.info(f"Registering {model_uri} → {registry_name}")
    version_info = mlflow.register_model(model_uri=model_uri, name=registry_name)
    version = version_info.version

    client.transition_model_version_stage(
        name=registry_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )
    logger.info(f"  {registry_name} v{version} → {stage}")
    return version

# ---------------------------------------------------------------------------
# Baseline metrics helpers
# ---------------------------------------------------------------------------

def _load_bpr_baseline_metrics() -> dict:
    """
    Load the BPR baseline evaluation metrics from sample-artifacts.
    These were produced by the original Goodreads training run.
    """
    path = _SAMPLE_ARTIFACTS_DIR / "bpr" / "evaluation_metrics.json"
    if not path.exists():
        logger.warning(f"BPR sample metrics not found at {path}; using empty baseline")
        return {}
    with open(path) as f:
        metrics = json.load(f)
    logger.info(f"Loaded BPR baseline metrics from {path}")
    return metrics

# ---------------------------------------------------------------------------
# Dataset size counter — reuses train_bpr's preprocessing exactly
# ---------------------------------------------------------------------------

def _count_bpr_preprocessed_rows(data_path: str) -> int:
    """
    Load and preprocess the BPR parquet using the exact same pipeline as
    train_bpr.main() to get the post-preprocessing row count.
    """
    cfg = _bpr.DEFAULT_CONFIG
    df = _bpr.load_data(
        data_path,
        required_columns=[cfg["col_user"], cfg["col_item"]],
        optional_columns=[cfg["col_rating"]],
    )
    df = _bpr.preprocess_data(
        df,
        col_user=cfg["col_user"],
        col_item=cfg["col_item"],
        col_rating=cfg.get("col_rating"),
        min_user_interactions=cfg["min_user_interactions"],
    )
    return df.height

# ---------------------------------------------------------------------------
# App interaction export
# ---------------------------------------------------------------------------

def _export_app_interactions(since: datetime) -> Optional[pl.DataFrame]:
    """
    Query the production book_ratings table for all ratings created after
    *since*, join with books to get the Goodreads book ID, and return a
    Polars DataFrame whose columns exactly match the historical training parquet:

        user_id   (Utf8)   — "app_<pg_user_id>"
        book_id   (Int64)  — books.goodreads_id
        weight    (Float32)— rating / 5.0
        timestamp (Int64)  — Unix epoch of created_at

    DB imports are deferred to this function so that module-level import does
    not trigger SQLAlchemy engine creation (which would block on DB connect).

    Returns None when the DB is unavailable or no rows exist.
    """
    # Lazy imports — only hit the DB when we actually need it.
    from sqlalchemy import select
    from sqlalchemy.orm import Session
    from bookdb.db.models import Book, BookRating
    from bookdb.db.session import SessionLocal

    try:
        session: Session = SessionLocal()
        try:
            stmt = (
                select(
                    BookRating.user_id,
                    Book.goodreads_id.label("book_id"),
                    BookRating.rating,
                    BookRating.created_at,
                )
                .join(Book, Book.id == BookRating.book_id)
                .where(BookRating.created_at > since)
                .where(Book.goodreads_id.is_not(None))
            )
            rows = session.execute(stmt).fetchall()
        finally:
            session.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not connect to DB for interaction export: {exc}")
        return None

    if not rows:
        logger.info("No new app interactions found since the cutoff date")
        return None

    df = pl.DataFrame(
        {
            "user_id": pl.Series([f"app_{r.user_id}" for r in rows], dtype=pl.Utf8),
            "book_id": pl.Series([int(r.book_id) for r in rows], dtype=pl.Int64),
            "weight": pl.Series([float(r.rating) / 5.0 for r in rows], dtype=pl.Float32),
            "timestamp": pl.Series([int(r.created_at.timestamp()) for r in rows], dtype=pl.Int64),
        }
    )
    logger.info(f"Exported {df.height} new app interactions from the database")
    return df

# ---------------------------------------------------------------------------
# Dataset merge
# ---------------------------------------------------------------------------

def _merge_bpr(historical_path: str, new_interactions: Optional[pl.DataFrame]) -> Path:
    """
    Merge the historical BPR parquet with new app interactions.

    Produces data/bpr_interactions_retrain.parquet with columns:
        user_id (Utf8), book_id (Int64), weight (Float32)

    The historical user_id column (Int64 Goodreads IDs) is cast to Utf8 so
    that it can be concatenated with the app_-prefixed string IDs while
    remaining compatible with BPR's internal string-to-int mapping.
    The file is then passed to train_bpr.main() which applies the full
    preprocessing pipeline (user filter ≥5, dedup, Float32 cast) uniformly
    across all rows.
    """
    cfg = _bpr.DEFAULT_CONFIG
    historical = _bpr.load_data(
        historical_path,
        required_columns=[cfg["col_user"], cfg["col_item"]],
        optional_columns=[cfg["col_rating"]],
    )
    # Normalise user_id to string so historical and app rows share the same type.
    historical = historical.with_columns(
        pl.col(cfg["col_user"]).cast(pl.Utf8).alias(cfg["col_user"])
    )

    if new_interactions is not None and new_interactions.height > 0:
        app_bpr = new_interactions.select(["user_id", "book_id", "weight"])
        merged = pl.concat([historical, app_bpr], how="diagonal_relaxed")
    else:
        merged = historical

    # Resolve duplicate user-book pairs: keep maximum weight (most positive signal).
    merged = (
        merged
        .group_by([cfg["col_user"], cfg["col_item"]])
        .agg(pl.col(cfg["col_rating"]).max())
    )

    merged.write_parquet(_MERGED_BPR_PATH)
    logger.info(f"Wrote merged BPR dataset ({merged.height} rows) → {_MERGED_BPR_PATH}")
    return _MERGED_BPR_PATH

# ---------------------------------------------------------------------------
# Promotion decision
# ---------------------------------------------------------------------------

def _metrics_acceptable(
    new_metrics: dict,
    baseline_metrics: dict,
    tolerance: float,
) -> bool:
    """
    Accept the new model if NDCG@10 does not regress by more than *tolerance*
    relative to the baseline. If either dict is missing the key, allow
    promotion (no comparison is possible).
    """
    key = "ndcg_at_10"
    if key not in baseline_metrics or key not in new_metrics:
        logger.info(
            f"Cannot compare {key}: missing from baseline or new metrics. Allowing promotion."
        )
        return True
    floor = baseline_metrics[key] * (1.0 - tolerance)
    ok = new_metrics[key] >= floor
    logger.info(
        f"  {key}: new={new_metrics[key]:.4f}  baseline={baseline_metrics[key]:.4f}"
        f"  floor={floor:.4f}  → {'PASS' if ok else 'FAIL'}"
    )
    return ok

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Manifest written → {path}")


def _load_manifest(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _append_run_log(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

# ---------------------------------------------------------------------------
# First-run flow
# ---------------------------------------------------------------------------

def _run_first_train(args: argparse.Namespace, client: MlflowClient) -> int:
    """
    Execute the first-ever retraining run.

    - Loads baseline metrics from sample-artifacts/.
    - Counts preprocessed rows in the historical parquet (becomes the baseline
      for subsequent threshold checks).
    - Exports any app interactions recorded since APP_LIVE_SINCE.
    - Merges data, trains the model, registers it in Production.
    - Writes the manifest.

    Returns 0 on success.
    """
    logger.info("=" * 60)
    logger.info("FIRST RUN detected — BPR_Recommender not found in MLflow")
    logger.info(f"  BPR data : {args.bpr_data_path}")
    logger.info("=" * 60)

    bpr_baseline = _load_bpr_baseline_metrics()

    # Count post-preprocessing rows from the historical data.
    # This becomes the denominator for threshold checks in future runs.
    logger.info("Counting preprocessed rows in historical dataset …")
    bpr_historical_rows = _count_bpr_preprocessed_rows(args.bpr_data_path)
    logger.info(f"  BPR historical rows (post-preprocessing): {bpr_historical_rows:,}")

    # Export any app interactions accumulated since the application went live.
    new_interactions: Optional[pl.DataFrame] = None
    if not args.no_db:
        logger.info(f"Exporting app interactions since {APP_LIVE_SINCE.isoformat()} …")
        new_interactions = _export_app_interactions(since=APP_LIVE_SINCE)
        if new_interactions is not None:
            logger.info(f"  Exported {new_interactions.height:,} raw interaction rows")
        elif not args.force:
            logger.info(
                "No new app interactions found — nothing has changed since the historical "
                "training data was collected. Skipping retrain. Use --force to override."
            )
            return 0
    else:
        logger.info("--no-db: skipping app interaction export")

    # Build the merged dataset (historical + new app rows).
    logger.info("Merging BPR dataset …")
    merged_bpr_path = _merge_bpr(args.bpr_data_path, new_interactions)

    # Train — delegate entirely to the existing training script.
    bpr_config = _bpr.DEFAULT_CONFIG.copy()
    bpr_config["data_path"] = str(merged_bpr_path)
    logger.info("Training BPR …")
    _bpr_model, bpr_recs_path, bpr_metrics, bpr_run_id = _bpr.main(bpr_config)
    logger.info(f"BPR training complete (run_id={bpr_run_id})")
    logger.info(f"  BPR metrics: { {k: round(v, 4) for k, v in bpr_metrics.items()} }")

    # Register — on first run we always promote to Production.
    bpr_version = _register_model(
        client, bpr_run_id, "model.pkl", _BPR_REGISTRY_NAME, "Production", args.dry_run
    )

    now_iso = datetime.now(tz=timezone.utc).isoformat()
    manifest = {
        "is_first_run": True,
        "run_at": now_iso,
        "last_accepted_train_at": now_iso,
        "bpr_training_row_count": bpr_historical_rows,
        "bpr_mlflow_run_id": bpr_run_id,
        "bpr_model_version": bpr_version,
        "bpr_baseline_metrics": bpr_metrics,
        "outcome": "accepted",
        "bpr_recs_path": str(bpr_recs_path),
    }

    if not args.dry_run:
        _write_manifest(manifest, Path(args.manifest_path))
    else:
        logger.info("[dry-run] Skipping manifest write")

    _print_summary(manifest)
    return 0

# ---------------------------------------------------------------------------
# Subsequent-run flow
# ---------------------------------------------------------------------------

def _run_retrain(args: argparse.Namespace, client: MlflowClient) -> int:
    """
    Execute a regular weekly retraining run.

    Loads the manifest, checks the data threshold, merges, trains, evaluates,
    and promotes or rejects.

    Returns 0 on success / skip, 1 on rejection.
    """
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        logger.error(
            f"Manifest not found at {manifest_path}. "
            "Ensure a first run has completed, or delete the registry entry to re-trigger it."
        )
        return 1

    manifest = _load_manifest(manifest_path)
    last_train_at_str = manifest["last_accepted_train_at"]
    last_train_at = datetime.fromisoformat(last_train_at_str)
    logger.info(f"Last accepted training cutoff : {last_train_at_str}")
    logger.info(f"BPR data path                : {args.bpr_data_path}")

    # Export new interactions since the last accepted run.
    new_interactions: Optional[pl.DataFrame] = None
    if not args.no_db:
        logger.info(f"Exporting app interactions since {last_train_at_str} …")
        new_interactions = _export_app_interactions(since=last_train_at)
        if new_interactions is not None:
            logger.info(f"  Exported {new_interactions.height:,} raw interaction rows")
        else:
            logger.info("  No new interactions found")
    else:
        logger.info("--no-db: skipping app interaction export")

    # Count new deduplicated BPR rows using the same preprocessing pipeline.
    # We build a temporary in-memory frame just for counting to avoid wasted
    # I/O before the threshold check.
    new_bpr_rows = 0
    if new_interactions is not None and new_interactions.height > 0:
        cfg = _bpr.DEFAULT_CONFIG
        tmp = new_interactions.select(["user_id", "book_id", "weight"])
        tmp = _bpr.preprocess_data(
            tmp,
            col_user="user_id",
            col_item="book_id",
            col_rating="weight",
            min_user_interactions=cfg["min_user_interactions"],
        )
        new_bpr_rows = tmp.height
    logger.info(f"New deduplicated BPR rows: {new_bpr_rows:,}")

    previous_row_count = manifest.get("bpr_training_row_count", 0)
    threshold_ratio = new_bpr_rows / max(previous_row_count, 1)
    logger.info(
        f"Threshold check: {new_bpr_rows:,} new / {previous_row_count:,} previous = "
        f"{threshold_ratio:.3f}  (required ≥ {args.threshold})"
    )

    if not args.force and threshold_ratio < args.threshold:
        logger.info(
            f"Threshold not met ({threshold_ratio:.3f} < {args.threshold}). "
            "Recording skipped run."
        )
        record = {
            "run_at": datetime.now(tz=timezone.utc).isoformat(),
            "outcome": "skipped",
            "threshold_ratio": threshold_ratio,
            "new_bpr_rows": new_bpr_rows,
            "previous_row_count": previous_row_count,
        }
        if not args.dry_run:
            _append_run_log(record, Path(args.run_log_path))
        return 0

    logger.info("Threshold met — proceeding with retrain")

    logger.info("Merging BPR dataset …")
    merged_bpr_path = _merge_bpr(args.bpr_data_path, new_interactions)

    bpr_config = _bpr.DEFAULT_CONFIG.copy()
    bpr_config["data_path"] = str(merged_bpr_path)
    logger.info("Training BPR …")
    _bpr_model, bpr_recs_path, bpr_metrics, bpr_run_id = _bpr.main(bpr_config)
    logger.info(f"BPR training complete (run_id={bpr_run_id})")
    logger.info(f"  BPR metrics: { {k: round(v, 4) for k, v in bpr_metrics.items()} }")

    # Evaluate against manifest baseline metrics.
    accepted = _metrics_acceptable(
        bpr_metrics, manifest.get("bpr_baseline_metrics", {}), args.tolerance
    )
    logger.info(f"Metrics gate: BPR={'PASS' if accepted else 'FAIL'} → {'ACCEPTED' if accepted else 'REJECTED'}")

    bpr_stage = "Production" if accepted else "Archived"
    bpr_version = _register_model(
        client, bpr_run_id, "model.pkl", _BPR_REGISTRY_NAME, bpr_stage, args.dry_run
    )

    now_iso = datetime.now(tz=timezone.utc).isoformat()
    run_record = {
        "is_first_run": False,
        "run_at": now_iso,
        "outcome": "accepted" if accepted else "rejected",
        "bpr_mlflow_run_id": bpr_run_id,
        "bpr_model_version": bpr_version,
        "bpr_metrics": bpr_metrics,
        "threshold_ratio": threshold_ratio,
    }

    if accepted:
        new_manifest = {
            **run_record,
            "last_accepted_train_at": now_iso,
            "bpr_training_row_count": new_bpr_rows + previous_row_count,
            "bpr_baseline_metrics": bpr_metrics,
            "bpr_recs_path": str(bpr_recs_path),
        }
        if not args.dry_run:
            _write_manifest(new_manifest, Path(args.manifest_path))
            _append_run_log(run_record, Path(args.run_log_path))
        _print_summary(new_manifest)
        return 0
    else:
        logger.warning(
            "New model did NOT pass the metrics gate. "
            "Previous Production version remains unchanged."
        )
        if not args.dry_run:
            _append_run_log(run_record, Path(args.run_log_path))
        _print_summary(run_record)
        return 1

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(record: dict) -> None:
    outcome = record.get("outcome", "unknown").upper()
    sep = "=" * 60
    logger.info(sep)
    logger.info(f"OUTCOME: {outcome}")
    if "bpr_recs_path" in record:
        logger.info(f"BPR recommendations → {record['bpr_recs_path']}")
        logger.info("  Update BPR_PARQUET_URL in your backend .env to this path.")
    if "bpr_model_version" in record and record["bpr_model_version"]:
        logger.info(f"BPR model version  : {record['bpr_model_version']}")
    logger.info(sep)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline weekly retraining pipeline for the BPR recommendation model"
    )
    parser.add_argument(
        "--bpr-data-path",
        default=str(_project_root / "data" / "bpr_interactions_merged.parquet"),
        help="Path to the historical BPR interactions parquet",
    )
    parser.add_argument(
        "--manifest-path",
        default=str(_DEFAULT_MANIFEST_PATH),
        help="Path to the retrain manifest JSON file",
    )
    parser.add_argument(
        "--run-log-path",
        default=str(_DEFAULT_RUN_LOG_PATH),
        help="Path to the JSONL run log (one record per line)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Minimum ratio of new/previous training rows to trigger retrain (default: 0.10)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Max allowed NDCG@10 regression fraction before rejection (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the data threshold check and always retrain",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip exporting app interactions from the database",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run all stages but skip MLflow Model Registry writes, "
            "manifest updates, and run log appends"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.dry_run:
        logger.info("DRY RUN mode — no registry writes or manifest updates will occur")

    client = _setup_mlflow()

    is_first_run = not _model_registered(client, _BPR_REGISTRY_NAME)
    logger.info(f"BPR_Recommender registered: {not is_first_run} → {'first run' if is_first_run else 'retrain'}")

    if is_first_run:
        return _run_first_train(args, client)
    else:
        return _run_retrain(args, client)


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        logger.error("Unhandled exception in weekly_retrain.py")
        logger.error(traceback.format_exc())
        exit_code = 1
    sys.exit(exit_code)
