"""Filter review embeddings to match reduced reviews dataset."""

import os
import time
import polars as pl

project_root = __import__("pathlib").Path(__file__).resolve().parents[4]

reduced_reviews_path = os.path.join(project_root, "data", "5_goodreads_reviews_final_clean.parquet")
embeddings_path = os.path.join(project_root, "data", "reviews_base_embeddings.parquet")
output_path = os.path.join(project_root, "data", "reviews_reduced_embeddings.parquet")

print("Loading reduced review IDs...")
t0 = time.time()

# Get the set of review IDs that passed the reduction filter
reduced_ids = (
    pl.scan_parquet(reduced_reviews_path)
    .select("review_id")
    .collect()
    .to_series()
    .to_list()
)
print(f"Loaded {len(reduced_ids):,} reduced review IDs ({time.time()-t0:.1f}s)")

print("Filtering embeddings...")
t1 = time.time()

# Filter embeddings to keep only those in the reduced set
df_embeddings = (
    pl.scan_parquet(embeddings_path)
    .filter(pl.col("review_id").is_in(reduced_ids))
    .collect()
)
print(f"Filtered to {len(df_embeddings):,} embeddings ({time.time()-t1:.1f}s)")

print("Saving...")
df_embeddings.write_parquet(output_path)
print(f"Saved to {output_path}")
print(f"Total time: {time.time()-t0:.1f}s")

# Validation
print("\n Validation ")
print(f"Reviews in 5_final_clean: {len(reduced_ids):,}")
print(f"Embeddings in output:     {len(df_embeddings):,}")
if len(reduced_ids) == len(df_embeddings):
    print("Match: OK")
else:
    diff = len(reduced_ids) - len(df_embeddings)
    print(f"MISMATCH: {diff:,} reviews have no embedding")
