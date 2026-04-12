# Weekly Retrain Guide

`scripts/weekly_retrain.py` — offline retraining pipeline for the BPR recommendation model.

---

## What it does

The script orchestrates the full retraining cycle in one command. It handles two distinct cases:

**First run** — detected when `BPR_Recommender` does not exist in the MLflow Model Registry. Exports app interactions since `APP_LIVE_SINCE`; if none are found, exits early (nothing has changed from the historical baseline). Otherwise merges, trains, registers, and writes the manifest.

**Subsequent runs** — loads the manifest from the previous accepted run, exports new user interactions since the last training cutoff, checks whether enough new data has arrived, trains, compares metrics, and promotes or rejects.

---

## Prerequisites

### 1. Environment variables (`.env` or shell)

| Variable | Required | Purpose |
|---|---|---|
| `MLFLOW_TRACKING_URI` | Yes | MLflow server (e.g. `https://mlflow.yousef.gg`) |
| `MLFLOW_TRACKING_USERNAME` | Yes | MLflow basic-auth username |
| `MLFLOW_TRACKING_PASSWORD` | Yes | MLflow basic-auth password |
| `AWS_ACCESS_KEY_ID` | Yes | S3 key for MLflow artifact storage |
| `AWS_SECRET_ACCESS_KEY` | Yes | S3 secret |
| `MLFLOW_S3_ENDPOINT_URL` | Yes | Custom S3 endpoint (if not AWS) |
| `DATABASE_URL` | Only if exporting app interactions | Production Postgres connection string |

The script loads `.env` automatically from the project root.

### 2. Local data files

| File | Required | Description |
|---|---|---|
| `data/bpr_interactions_merged.parquet` | Yes (first run) | Historical BPR training data. Columns: `user_id` (Int64), `book_id` (Int64), `weight` (Float32) |
| `sample-artifacts/bpr/evaluation_metrics.json` | Yes (first run) | BPR baseline metrics from the original Goodreads training |
| `data/retrain_manifest.json` | Auto-created | Written after each accepted run. Required for subsequent runs. |

Override the data path with `--bpr-data-path` if your file lives elsewhere.

### 3. Python environment

Same as the existing project environment — no new dependencies are introduced. Run from the project root so imports resolve correctly.

---

## The `APP_LIVE_SINCE` constant

Near the top of `weekly_retrain.py`:

```python
APP_LIVE_SINCE = datetime(2026, 2, 1, tzinfo=timezone.utc)
```

**Update this to your actual application launch date.** On a first run, this is the cutoff used when querying the database for app interactions to merge into the initial training set. On subsequent runs the manifest's `last_accepted_train_at` is used instead.

---

## Usage

```bash
# Typical run (all defaults):
python scripts/weekly_retrain.py

# Dry run — runs every stage but makes no writes (good for testing):
python scripts/weekly_retrain.py --dry-run --no-db \
    --bpr-data-path data/bpr_interactions_merged_tiny.parquet

# Force retrain even if data threshold not met:
python scripts/weekly_retrain.py --force

# Skip DB export (train on historical data only):
python scripts/weekly_retrain.py --no-db

# Custom data path and lower threshold:
python scripts/weekly_retrain.py \
    --bpr-data-path /path/to/bpr_interactions_merged.parquet \
    --threshold 0.05
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--bpr-data-path` | `data/bpr_interactions_merged.parquet` | Historical BPR parquet |
| `--manifest-path` | `data/retrain_manifest.json` | Manifest location |
| `--run-log-path` | `data/retrain_runs.jsonl` | JSONL log of all runs |
| `--threshold` | `0.10` | Min ratio of new/previous rows to trigger retrain |
| `--tolerance` | `0.02` | Max allowed NDCG@10 regression (2%) |
| `--force` | off | Skip threshold check |
| `--no-db` | off | Skip database interaction export |
| `--dry-run` | off | No registry writes, no manifest/log writes |

---

## Execution flow

```
Start
│
├── Connect to MLflow, check Model Registry
│
├── First run? (BPR_Recommender not registered)
│   │
│   ├── YES ─────────────────────────────────────────────────────────────┐
│   │   Load BPR baseline from sample-artifacts/bpr/evaluation_metrics.json
│   │   Count preprocessed rows in historical parquet
│   │   Export app interactions since APP_LIVE_SINCE (unless --no-db)
│   │       No new interactions? → exit 0 (skip, use --force to override)
│   │   Merge → data/bpr_interactions_retrain.parquet
│   │   train_bpr.main()  →  predictions/bpr_recommendations.parquet
│   │   Register BPR_Recommender v1 → Production
│   │   Write manifest
│   │   Exit 0 ◄────────────────────────────────────────────────────────┘
│   │
│   └── NO ──────────────────────────────────────────────────────────────┐
│       Load manifest
│       Export app interactions since manifest.last_accepted_train_at
│       Count new deduplicated BPR rows (same preprocessing as training)
│       Threshold check: new_rows / previous_rows ≥ threshold?
│           NO → write "skipped" to run log, exit 0
│           YES (or --force) ↓
│       Merge → data/bpr_interactions_retrain.parquet
│       train_bpr.main()  →  predictions/bpr_recommendations.parquet
│       Compare NDCG@10 vs manifest baseline (±2% tolerance)
│           PASS → register new version → Production
│                  write updated manifest, append run log
│                  exit 0
│           FAIL → register new version → Archived
│                  append run log, exit 1 ◄──────────────────────────────┘
```

---

## Preprocessing consistency

New app interactions are exported with **exactly the same column names and types** as the historical training parquet:

| Column | Type | Source |
|---|---|---|
| `user_id` | `Utf8` | `"app_{BookRating.user_id}"` — historical Goodreads IDs are cast to string for uniform type |
| `book_id` | `Int64` | `books.goodreads_id` — same canonical ID space as historical data |
| `weight` | `Float32` | `rating / 5.0` (ratings are 1–5, so weight is 0.2–1.0) |
| `timestamp` | `Int64` | Unix epoch of `BookRating.created_at` |

After the merge, the resulting parquet is passed directly to `train_bpr.main()`, which runs the **identical** `preprocess_data()` function on the entire combined dataset:

- Filter users with fewer than 5 interactions
- Deduplicate user-book pairs (keep max weight)
- Cast weight to Float32

### App user ID namespace

New app users have no Goodreads ID. Their `user_id` values in the training data and the output recommendation parquet are prefixed with `app_`: e.g. `"app_42"`.

The existing `_bpr_recommendations()` function in `apps/api/routers/discovery.py` queries the parquet with `WHERE user_id = <goodreads_user_id>` using an integer. To serve personalized recommendations to app users after a retrain, update that function to query `WHERE user_id = 'app_<user_id>'` for authenticated users who have no Goodreads ID on file.

---

## What it produces

### After an accepted run

| Output | Location | Description |
|---|---|---|
| BPR recommendations | `predictions/bpr_recommendations.parquet` | New top-30 recommendations per user. Columns: `user_id` (Utf8), `item_id` (Utf8), `prediction` (Float32) |
| BPR model checkpoint | `predictions/bpr_model_checkpoint/` | `model.pkl`, `mappings.json`, `user_item_matrix.npz` |
| Retrain manifest | `data/retrain_manifest.json` | JSON record of this run — used as baseline for next run |
| Run log entry | `data/retrain_runs.jsonl` | Append-only log of all runs (accepted, rejected, skipped) |
| MLflow run | MLflow experiment `BPR_Recommendations` | Params, metrics, artifacts |
| MLflow model version | Registry: `BPR_Recommender` | New version promoted to `Production` |
| Merged retrain parquet | `data/bpr_interactions_retrain.parquet` | Intermediate file — safe to delete after training |

### After a rejected run

Same MLflow run and registry entry are created, but the version is transitioned to `Archived` instead of `Production`. The manifest is **not** updated — the previous accepted baseline remains in effect. The script exits with code 1.

### After a skipped run (threshold not met)

No training occurs. One line is appended to `data/retrain_runs.jsonl` recording the ratio and timestamp. The script exits with code 0.

---

## Updating the backend after an accepted run

The discovery API reads the BPR recommendations parquet from `BPR_PARQUET_URL`. After an accepted run, `predictions/bpr_recommendations.parquet` has been updated in place — no env var change is needed if your backend reads from a local path.

If you are serving the parquet from S3, upload the new file and update `BPR_PARQUET_URL` in your deployment environment:

```bash
aws s3 cp predictions/bpr_recommendations.parquet s3://your-bucket/artifacts/bpr_recommendations.parquet
# then update BPR_PARQUET_URL in your backend env
```

---

## Rolling back

The run log (`data/retrain_runs.jsonl`) records the MLflow run ID and model version for every run. To roll back:

1. Look up the previous accepted run's `bpr_model_version` in the run log or manifest.
2. In MLflow UI (or via the client), transition that version back to `Production`.
3. Download the corresponding `bpr_recommendations.parquet` artifact from that MLflow run and replace `predictions/bpr_recommendations.parquet`.

---

## Manifest reference

`data/retrain_manifest.json` is read and written by the script. Never edit it manually between runs unless you are deliberately resetting the baseline.

```jsonc
{
  "is_first_run": false,
  "run_at": "2026-04-14T03:00:00Z",
  "last_accepted_train_at": "2026-04-14T03:00:00Z",   // cutoff for next run's export
  "bpr_training_row_count": 2961840,                   // denominator for threshold check
  "bpr_mlflow_run_id": "abc123...",
  "bpr_model_version": "2",
  "bpr_baseline_metrics": { "ndcg_at_10": 0.104, ... }, // gate for next run's promotion
  "outcome": "accepted",
  "bpr_recs_path": "predictions/bpr_recommendations.parquet"
}
```
