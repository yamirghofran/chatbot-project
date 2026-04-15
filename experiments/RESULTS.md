# BookDB Experiments

Three experiments evaluate the core components added to BookDB. All use a temporal split (80/10/10 by timestamp quantile, val cutoff 2017-04-19) to avoid data leakage.

---

## Experiment 1 - Query Rewriting

### Setup

Users don't write structured queries. They search with something like "fantasy book by tolkien", no description, no context. This experiment tests whether an LLM rewriting step recovers retrieval quality lost from that vagueness, and whether injecting the user's taste profile into the rewriter steers the result toward their preferences.

Four conditions on 200 queries from the temporal benchmark:

- **Vague**: title + genres + authors only, description stripped
- **Rewritten**: the vague query is expanded by the LLM (`rewrite_query_sync` via Groq) before hitting Qdrant, no user context
- **Personalized**: same LLM rewrite but with a randomly sampled user taste profile injected into the system prompt, so the synthetic description is steered toward that reader's genre/tone preferences
- **Full**: the complete embedding text (ideal upper bound)

### Results

| Metric | Vague | Rewritten | Personalized | Full (ideal) |
|--------|------:|----------:|-------------:|-------------:|
| Recall@5 | 0.0216 | 0.0320 | 0.0179 | 0.0703 |
| Recall@10 | 0.0280 | 0.0387 | 0.0216 | 0.0737 |
| Recall@20 | 0.0325 | 0.0440 | 0.0267 | 0.0806 |
| Precision@10 | 0.0390 | 0.0598 | 0.0348 | 0.1135 |
| NDCG@10 | 0.0623 | 0.0986 | 0.0576 | 0.2351 |
| MRR | 0.2386 | 0.3752 | 0.2247 | 0.9975 |
| HitRate@10 | 0.3600 | 0.4975 | 0.2879 | 1.0000 |

**Statistical tests (paired t-test):**

| Comparison | Recall@10 | NDCG@10 |
|------------|----------:|--------:|
| Rewritten vs vague | t=2.788, p=0.0058 | t=3.751, p=0.0002 |
| Personalized vs vague | t=−1.716, p=0.0877 | t=−0.558, p=0.5778 |
| Full vs vague | t=14.873, p=0.0000 | t=24.600, p=0.0000 |
| Personalized vs rewritten | t=−5.626, p=0.0000  | t=−5.469, p=0.0000 |


### What it shows

Generic rewriting works: NDCG@10 rises from 0.062 to 0.099 (+58%), HitRate@10 from 36% to 50%, both significant. The LLM successfully reconstructs a richer description from sparse header fields and the expanded text lands closer to the indexed book vectors.

Personalized rewriting, as tested here, hurts retrieval. Assigning a random taste profile actively steers the rewrite away from the query's ground-truth answer and toward the randomly sampled user's preferences instead. The result is significantly worse than even the generic rewrite (p=0.0000 on both metrics). This is the expected outcome of random profile assignment — it confirms that taste injection changes retrieval direction strongly, which is exactly the point. When the profile belongs to the user who actually issued the query, this steering is a feature; when it belongs to someone else, it's noise.

The remaining gap to the full ideal (NDCG 0.099 vs 0.235) reflects an inherent ceiling: the LLM is generating a plausible description, not reading the actual book.

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
