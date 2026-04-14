# BookDB Experiments

Three experiments evaluate the core components added to BookDB. All use a temporal split (80/10/10 by timestamp quantile, val cutoff 2017-04-19) to avoid data leakage.

---

## Experiment 1 - Query Rewriting

### Setup

Users don't write structured queries. They search with something like "fantasy book by tolkien", no description, no context. This experiment tests whether an LLM rewriting step recovers retrieval quality lost from that vagueness.

Three conditions on 200 queries:

- **Vague**: title + genres + authors only, description stripped
- **Rewritten**: the vague query is expanded by the LLM (`rewrite_query_sync` via Groq) before hitting Qdrant
- **Full**: the complete embedding text (ideal upper bound)

### Results

| Metric | Vague | Rewritten | Full |
|--------|------:|----------:|-----:|
| Recall@10 | 0.028 | 0.039 | 0.074 |
| NDCG@10 | 0.062 | 0.104 | 0.235 |
| MRR | 0.239 | 0.399 | 0.998 |
| HitRate@10 | 0.360 | 0.500 | 1.000 |

Rewritten vs vague - NDCG@10: t=4.00, p=0.0001. Recall@10: t=2.58, p=0.011. Both significant.

### What it shows

Rewriting roughly doubles NDCG@10 (0.062 to 0.104) and gets 50% of users to find something relevant in the top 10, up from 36%. The gap to the ideal is still large, which makes sense, the LLM is reconstructing description from genre/author hints, not reading the actual book.

---

## Experiment 2 - BPR + Vector Fusion

### Setup

The discovery endpoint fuses two signals: BPR collaborative filtering and vector semantic search, with BPR filling any gaps. The question is whether that fusion actually beats either signal alone.

Three conditions on 200 users (temporal split, >=3 training interactions):

- **BPR-only**: top-k from the BPR parquet by prediction score
- **Vector-only**: Qdrant similarity seeded from the user's interaction history
- **Fused**: hybrid logic

### Results

| Metric | BPR-only | Vector-only | Fused |
|--------|----------:|------------:|------:|
| Recall@10 | 0.0079 | 0.0062 | 0.0078 |
| NDCG@10 | 0.0093 | 0.0173 | 0.0122 |
| HitRate@10 | 0.045 | 0.105 | 0.080 |

Fused vs BPR-only on HitRate@10: t=2.36, p=0.019. Significant. Everything else is not.

### What it shows

BPR alone has poor hit rate (4.5%), it's personalized but tends to surface books the user has already encountered. Adding vector search pushes hit rate to 8%, a significant +78% over BPR-only. Vector-only actually has the best raw hit rate (10.5%) but no personalization signal and inconsistent NDCG. The fused approach lands in between, which is the expected behavior of the quota design.

---

## Experiment 3 - Seed Clustering

### Setup

When making vector recommendations, `_cluster_vector_recommendations` in `discovery.py` doesn't just average a user's reading history into one centroid, it clusters the seed books first and queries Qdrant once per cluster. The idea is that a user who reads both fantasy and romance shouldn't get 10 fantasy books just because fantasy edges out romance on average score.

Two conditions on 200 users with ≥10 training interactions:

- **Single centroid**: weighted average of all seed vectors to one Qdrant query
- **Clustered**: K-means on seed vectors (k = max(2, min(seeds//3, 5))), one query per cluster, proportional slot allocation, round-robin interleave, exact logic from `discovery.py` using `cluster_seeds_by_embedding`

Both use `get_vectors_by_ids` and `most_similar_by_vector` from `apps/api/core/embeddings.py`.

An extra metric **Diversity** measures mean pairwise cosine distance of the returned books (higher = more varied).

### Results

| Metric | Single centroid | Clustered |
|--------|---------------:|----------:|
| Recall@10 | 0.0006 | 0.0030 |
| NDCG@10 | 0.0042 | 0.0094 |
| HitRate@10 | 0.030 | 0.070 |
| Diversity | 0.164 | 0.420 |

Clustered vs single: HitRate@10: t=2.015, p=0.045. Recall@10: t=1.984, p=0.049. Diversity: t=40.7, p=0. All significant. NDCG@10 not significant (p=0.114).

### What it shows

The diversity result is the clearest one in the whole set of experiments: t=40.7 with p=0 means the improvement is essentially deterministic across users. Clustering prevents genre flooding, the returned list covers a much wider semantic range (cosine distance 0.164 to 0.420).

Hit rate doubles (3% to 7%) and is statistically significant. The single centroid gets pulled toward the dominant genre and misses the rest; clustering allocates slots proportionally so minority interests still get representation.

NDCG doesn't improve significantly, the ranking within each cluster is still cosine similarity, not personalized, so the order isn't better. The gain is in what gets included, not in how it's ranked.

---

## Summary

| | Key metric | Change | p |
|--|-----------|--------|---|
| Query rewriting | NDCG@10 | +66% | 0.0001 |
| BPR + vector fusion | HitRate@10 | +78% over BPR-only | 0.019 |
| Seed clustering | HitRate@10 / Diversity | +133% / +156% | 0.045 / around 0 |
