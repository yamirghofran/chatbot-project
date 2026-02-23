---
icon: lucide/brain
---

# Recommendation Models

BookDB implements multiple recommendation algorithms for personalized book discovery. The models are trained offline and served through the API for real-time recommendations.

## Overview

| Model | Type | Use Case |
|-------|------|----------|
| **SAR** | Collaborative Filtering | Item-item similarity, cold-start from seed items |
| **BPR** | Matrix Factorization | Personalized ranking for known users |
| **Vector Search** | Content-based | Semantic similarity via embeddings |

## SAR (Simple Algorithm for Recommendations)

Location: `bookdb/models/sar.py`

SAR is a fast, scalable algorithm based on user transaction history and item similarity.

### Features

- Multiple similarity metrics: Jaccard, Cosine, Lift, Mutual Information, Inclusion Index
- Time decay support for recency weighting
- GPU acceleration via PyTorch (MPS/CUDA)
- Cold-start recommendations from seed items

### Training

```bash
uv run python scripts/train_sar.py
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_type` | `jaccard` | Similarity metric |
| `time_decay_coefficient` | 30 | Half-life in days |
| `timedecay_formula` | False | Apply time decay |
| `threshold` | 1 | Min co-occurrence threshold |
| `normalize` | False | Normalize predictions |
| `use_torch` | True | Use GPU acceleration |

### Usage

```python
from bookdb.models.sar import SARSingleNode

model = SARSingleNode(
    col_user="user_id",
    col_item="book_id",
    col_rating="rating",
    similarity_type="jaccard",
)

model.fit(train_df)
recommendations = model.recommend_k_items(test_df, top_k=10)
```

### Similarity Functions

- `cooccurrence` - Raw co-occurrence counts
- `jaccard` - Jaccard similarity (default)
- `cosine` - Cosine similarity
- `lift` - Lift measure
- `mutual_information` - Mutual information
- `lexicographers_mutual_information` - Normalized MI
- `inclusion_index` - Inclusion index

## BPR (Bayesian Personalized Ranking)

Location: `bookdb/models/bpr.py`

BPR is a matrix factorization model optimized for implicit feedback using the implicit library.

### Features

- Memory-efficient training on large datasets
- String-to-integer ID mapping internally
- Batched recommendation generation
- Save/load model state

### Training

```bash
uv run python scripts/train_bpr.py
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `factors` | 64 | Number of latent factors |
| `iterations` | 100 | Training iterations |
| `learning_rate` | 0.01 | SGD learning rate |
| `regularization` | 0.01 | L2 regularization |

### Usage

```python
from bookdb.models.bpr import BPR

model = BPR(factors=64, iterations=100)
model.fit(train_df, col_user="user_id", col_item="book_id")

# Recommendations for a user
recs = model.recommend(user_id, top_k=10)

# Batch recommendations
all_recs = model.recommend_k_items(test_df, top_k=100, remove_seen=True)

# Find similar items
similar = model.similar_items(item_id, top_k=10)

# Save/load
model.save("models/bpr/")
model = BPR.load("models/bpr/")
```

## Vector-Based Recommendations

The API uses Qdrant vector search for content-based recommendations:

1. **Semantic Search**: Query embedding matched against book embeddings
2. **Related Books**: Find similar books using vector similarity
3. **Interaction-Based**: Aggregate user's interacted books to find similar items

See [Qdrant Integration](qdrant.md) for details.

## Recommendation Pipeline

The `/discovery/recommendations` endpoint combines multiple strategies:

```
User Request
     │
     ▼
┌─────────────────┐
│ BPR Model       │ ── Pre-computed recommendations for Goodreads users
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Interaction     │ ── Real-time vector search from ratings/shell/lists
│ Vectors         │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Merge & Dedupe  │ ── Combine with BPR dominant
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Cold Start      │ ── Fill remaining slots with popular books
└─────────────────┘
     │
     ▼
   Results
```

## Evaluation Metrics

Location: `bookdb/evaluation/__init__.py`

Standard recommender evaluation metrics:

| Metric | Description |
|--------|-------------|
| `precision_at_k` | Relevant items in top-k / k |
| `recall_at_k` | Relevant items in top-k / total relevant |
| `map_at_k` | Mean Average Precision |
| `ndcg_at_k` | Normalized Discounted Cumulative Gain |
| `hit_rate_at_k` | Users with at least one hit |
| `coverage` | Catalog coverage |
| `diversity` | Intra-list diversity |

### Usage

```python
from bookdb.evaluation import evaluate_recommendations

metrics = evaluate_recommendations(
    recommendations=recommendations_df,
    ground_truth=test_df,
    col_user="user_id",
    col_item="book_id",
    k_values=[5, 10, 20],
)

# Returns: {"precision_at_10": 0.15, "recall_at_10": 0.08, ...}
```

## Training Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train_sar.py` | Train SAR model |
| `scripts/train_bpr.py` | Train BPR model, generate recommendations parquet |

## Model Artifacts

- `models/sar/` - SAR model checkpoints
- `data/bpr_model_predictions/bpr_recommendations.parquet` - BPR recommendations for API
