# Weekly Retrain Guide

Two offline retraining pipelines share the same design:

| Script | Model | Registry name |
|---|---|---|
| `scripts/weekly_retrain_bpr.py` | BPR | `BPR_Recommender` |
| `scripts/weekly_retrain_ncf.py` | NCF | `NCF_Recommender` |

Both scripts follow an identical flow and accept the same flags (only the data-path flag is named differently). The sections below call out per-model differences where they exist.

---

## What they do

**First run** — detected when the model does not exist in the MLflow Model Registry. Exports app interactions since `APP_LIVE_SINCE`; if none are found, exits early (nothing has changed from the historical baseline). Otherwise merges, trains, registers, and writes the manifest.

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

Both scripts load `.env` automatically from the project root.

### 2. Local data files

**BPR**

| File | Required | Description |
|---|---|---|
| `data/bpr_interactions_merged.parquet` | Yes (first run) | Historical BPR training data. Columns: `user_id` (Int64), `book_id` (Int64), `weight` (Float32) |
| `sample-artifacts/bpr/evaluation_metrics.json` | No | BPR baseline metrics from the original training run. If absent, the first accepted run's metrics become the baseline. |
| `data/retrain_manifest.json` | Auto-created | Written after each accepted BPR run. Required for subsequent runs. |

**NCF**

| File | Required | Description |
|---|---|---|
| `data/ncf_interactions_merged.parquet` | Yes (first run) | Historical NCF training data. Same column schema as BPR. |
| `sample-artifacts/ncf/evaluation_metrics.json` | No | NCF baseline metrics from the original training run. If absent, the first accepted run's metrics become the baseline. |
| `data/retrain_manifest_ncf.json` | Auto-created | Written after each accepted NCF run. Required for subsequent runs. |

Override data paths with `--bpr-data-path` / `--ncf-data-path` if your files live elsewhere.

### 3. Python environment

Same as the existing project environment — no new dependencies. Run from the project root so imports resolve correctly.

---

## The `APP_LIVE_SINCE` constant

Near the top of each script:

```python
APP_LIVE_SINCE = datetime(2026, 2, 1, tzinfo=timezone.utc)
```

**Update this to your actual application launch date.** On a first run, this is the cutoff used when querying the database for app interactions to merge into the initial training set. On subsequent runs the manifest's `last_accepted_train_at` is used instead.

---

## Usage

```bash
# --- BPR ---

# Typical run (all defaults):
python scripts/weekly_retrain_bpr.py

# Dry run — runs every stage but makes no writes (good for testing):
python scripts/weekly_retrain_bpr.py --dry-run --no-db \
    --bpr-data-path data/bpr_interactions_merged_tiny.parquet

# Force retrain even if data threshold not met:
python scripts/weekly_retrain_bpr.py --force

# --- NCF ---

# Typical run (all defaults):
python scripts/weekly_retrain_ncf.py

# Dry run:
python scripts/weekly_retrain_ncf.py --dry-run --no-db \
    --ncf-data-path data/ncf_interactions_merged.parquet

# Force retrain:
python scripts/weekly_retrain_ncf.py --force
```

### All flags

Flags are identical for both scripts except where noted.

| Flag | Default (BPR) | Default (NCF) | Description |
|---|---|---|---|
| `--bpr-data-path` | `data/bpr_interactions_merged.parquet` | _(n/a)_ | Historical BPR parquet |
| `--ncf-data-path` | _(n/a)_ | `data/ncf_interactions_merged.parquet` | Historical NCF parquet |
| `--manifest-path` | `data/retrain_manifest.json` | `data/retrain_manifest_ncf.json` | Manifest location |
| `--run-log-path` | `data/retrain_runs.jsonl` | `data/retrain_runs_ncf.jsonl` | JSONL log of all runs |
| `--threshold` | `0.10` | `0.10` | Min ratio of new/previous rows to trigger retrain |
| `--tolerance` | `0.02` | `0.02` | Max allowed NDCG@10 regression (2%) |
| `--force` | off | off | Skip threshold check |
| `--no-db` | off | off | Skip database interaction export |
| `--dry-run` | off | off | No registry writes, no manifest/log writes |

---

## Execution flow

The flow is identical for both scripts (substitute NCF names where BPR appears):

```
Start
│
├── Connect to MLflow, check Model Registry
│
├── First run? (model not registered)
│   │
│   ├── YES ─────────────────────────────────────────────────────────────┐
│   │   Load baseline from sample-artifacts/{bpr,ncf}/evaluation_metrics.json
│   │   Count preprocessed rows in historical parquet
│   │   Export app interactions since APP_LIVE_SINCE (unless --no-db)
│   │       No new interactions? → exit 0 (skip, use --force to override)
│   │   Merge → data/{bpr,ncf}_interactions_retrain.parquet
│   │   train_{bpr,ncf}.main()  →  predictions/{bpr,ncf}_recommendations.parquet
│   │   Register {BPR,NCF}_Recommender v1 → Production
│   │   Write manifest
│   │   Exit 0 ◄────────────────────────────────────────────────────────┘
│   │
│   └── NO ──────────────────────────────────────────────────────────────┐
│       Load manifest
│       Export app interactions since manifest.last_accepted_train_at
│       Count new deduplicated rows (same preprocessing as training)
│       Threshold check: new_rows / previous_rows ≥ threshold?
│           NO → write "skipped" to run log, exit 0
│           YES (or --force) ↓
│       Merge → data/{bpr,ncf}_interactions_retrain.parquet
│       train_{bpr,ncf}.main()  →  predictions/{bpr,ncf}_recommendations.parquet
│       Compare NDCG@10 vs manifest baseline (±2% tolerance)
│           PASS → register new version → Production
│                  write updated manifest, append run log
│                  exit 0
│           FAIL → register new version → Archived
│                  append run log, exit 1 ◄──────────────────────────────┘
```

---

## Preprocessing consistency

New app interactions are exported with **exactly the same column names and types** as the historical training parquet (for both BPR and NCF):

| Column | Type | Source |
|---|---|---|
| `user_id` | `Utf8` | `"app_{BookRating.user_id}"` — historical Goodreads IDs are cast to string for uniform type |
| `book_id` | `Int64` | `books.goodreads_id` — same canonical ID space as historical data |
| `weight` | `Float32` | `rating / 5.0` (ratings are 1–5, so weight is 0.2–1.0) |
| `timestamp` | `Int64` | Unix epoch of `BookRating.created_at` |

After the merge, the resulting parquet is passed directly to `train_{bpr,ncf}.main()`, which runs the **identical** `preprocess_data()` function on the entire combined dataset:

- Filter users with fewer than 5 interactions
- Deduplicate user-book pairs (keep max weight)
- Cast weight to Float32

### App user ID namespace

New app users have no Goodreads ID. Their `user_id` values in the training data and the output recommendation parquet are prefixed with `app_`: e.g. `"app_42"`.

---

## What they produce

### After an accepted run

**BPR**

| Output | Location | Description |
|---|---|---|
| BPR recommendations | `predictions/bpr_recommendations.parquet` | Top-30 recs per user. Columns: `user_id` (Utf8), `item_id` (Utf8), `prediction` (Float32) |
| BPR model checkpoint | `predictions/bpr_model_checkpoint/` | `model.pkl`, `mappings.json`, `user_item_matrix.npz` |
| Retrain manifest | `data/retrain_manifest.json` | JSON record of this run |
| Run log entry | `data/retrain_runs.jsonl` | Append-only log of all runs |
| MLflow run | Experiment `BPR_Recommendations` | Params, metrics, artifacts |
| MLflow model version | Registry: `BPR_Recommender` | New version promoted to `Production` |
| Merged retrain parquet | `data/bpr_interactions_retrain.parquet` | Intermediate — safe to delete after training |

**NCF**

| Output | Location | Description |
|---|---|---|
| NCF recommendations | `predictions/ncf_recommendations.parquet` | Top-30 recs per user. Same schema as BPR. |
| NCF model checkpoint | `predictions/ncf_model_checkpoint/` | `model.pt`, `config.json`, `mappings.json` |
| Retrain manifest | `data/retrain_manifest_ncf.json` | JSON record of this run |
| Run log entry | `data/retrain_runs_ncf.jsonl` | Append-only log of all runs |
| MLflow run | Experiment `NCF_Recommendations` | Params, metrics, artifacts |
| MLflow model version | Registry: `NCF_Recommender` | New version promoted to `Production` |
| Merged retrain parquet | `data/ncf_interactions_retrain.parquet` | Intermediate — safe to delete after training |

### After a rejected run

Same MLflow run and registry entry are created, but the version is transitioned to `Archived` instead of `Production`. The manifest is **not** updated — the previous accepted baseline remains in effect. The script exits with code 1.

### After a skipped run (threshold not met)

No training occurs. One line is appended to the run log recording the ratio and timestamp. The script exits with code 0.

---

## Updating the backend after an accepted run

The discovery API reads the recommendations parquet from `BPR_PARQUET_URL` / `NCF_PARQUET_URL`. After an accepted run, the parquet is updated in place — no env var change is needed if your backend reads from a local path.

If you are serving the parquet from S3, upload the new file and update the env var in your deployment environment:

```bash
# BPR
aws s3 cp predictions/bpr_recommendations.parquet s3://your-bucket/artifacts/bpr_recommendations.parquet
# then update BPR_PARQUET_URL in your backend env

# NCF
aws s3 cp predictions/ncf_recommendations.parquet s3://your-bucket/artifacts/ncf_recommendations.parquet
# then update NCF_PARQUET_URL in your backend env
```

---

## Rolling back

The run log records the MLflow run ID and model version for every run. To roll back:

1. Look up the previous accepted run's model version in the run log or manifest.
2. In MLflow UI (or via the client), transition that version back to `Production`.
3. Download the corresponding recommendations parquet artifact from that MLflow run and replace the local file.

---

## Manifest reference

### BPR (`data/retrain_manifest.json`)

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

### NCF (`data/retrain_manifest_ncf.json`)

```jsonc
{
  "is_first_run": false,
  "run_at": "2026-04-14T03:00:00Z",
  "last_accepted_train_at": "2026-04-14T03:00:00Z",
  "ncf_training_row_count": 2961840,
  "ncf_mlflow_run_id": "def456...",
  "ncf_model_version": "2",
  "ncf_baseline_metrics": { "ndcg_at_10": 0.112, ... },
  "outcome": "accepted",
  "ncf_recs_path": "predictions/ncf_recommendations.parquet"
}
```

Never edit either manifest manually between runs unless you are deliberately resetting the baseline.
