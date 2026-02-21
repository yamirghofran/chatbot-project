"""Embedding index for book similarity lookups."""

from __future__ import annotations

import os

import numpy as np


class EmbeddingIndex:
    def __init__(self, embeddings: dict[int, np.ndarray]):
        self._embeddings = embeddings
        if embeddings:
            ids = list(embeddings.keys())
            self._ids = np.array(ids, dtype=np.int64)
            matrix = np.stack(list(embeddings.values()), axis=0).astype(np.float32)
            # Pre-normalise rows for cosine similarity via dot product.
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._matrix = matrix / norms
        else:
            self._ids = np.array([], dtype=np.int64)
            self._matrix = np.empty((0, 0), dtype=np.float32)

    def get(self, book_id: int) -> np.ndarray | None:
        return self._embeddings.get(book_id)

    def most_similar(self, query: np.ndarray, top_k: int = 20, exclude_ids: set[int] | None = None) -> list[int]:
        """Return internal book IDs (goodreads_ids) of the top_k most similar books."""
        if self._matrix.size == 0:
            return []
        q = query.astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        scores = self._matrix @ q
        if exclude_ids:
            for i, bid in enumerate(self._ids):
                if int(bid) in exclude_ids:
                    scores[i] = -1.0
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [int(self._ids[i]) for i in top_indices]


def load_embedding_index(parquet_path: str) -> EmbeddingIndex:
    if not os.path.exists(parquet_path):
        print(f"WARNING: Embedding file not found: {parquet_path}. Using empty index.")
        return EmbeddingIndex({})
    try:
        import polars as pl
        df = pl.read_parquet(parquet_path)
        # Expected columns: book_id (goodreads_id), embedding (list of floats)
        embedding_col = "embedding"
        id_col = "book_id"
        embeddings: dict[int, np.ndarray] = {}
        for row in df.iter_rows(named=True):
            bid = int(row[id_col])
            emb = row[embedding_col]
            if emb is not None:
                embeddings[bid] = np.array(emb, dtype=np.float32)
        print(f"Loaded {len(embeddings):,} book embeddings.")
        return EmbeddingIndex(embeddings)
    except Exception as e:
        print(f"WARNING: Failed to load embeddings: {e}. Using empty index.")
        return EmbeddingIndex({})
