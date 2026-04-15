
from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl


def map_similar_books(book_ids_list: list, book_id_lookup: dict) -> list[int]:
    """Map a list of book ID strings to CSV integer IDs, dropping any unmapped IDs.

    Args:
        book_ids_list: List of book ID strings from the raw Goodreads JSON.
        book_id_lookup: Mapping from string book ID to integer CSV book ID.

    Returns:
        List of integer book IDs with unmapped entries removed.
    """
    return [book_id_lookup[bid] for bid in book_ids_list if bid in book_id_lookup]


def build_book_edition_lookup(data_dir: str | Path) -> pl.DataFrame:
    """Build an edition-to-canonical-book lookup table in CSV ID space.

    Reads ``best_book_id_map.json`` (Goodreads IDs) and translates both sides
    to CSV integer IDs via ``raw_book_id_map.parquet``. Self-maps are removed.

    Args:
        data_dir: Directory containing ``best_book_id_map.json`` and
            ``raw_book_id_map.parquet``.

    Returns:
        DataFrame with columns ``edition_csv_id`` (Int64) and
        ``best_csv_id`` (Int64), one row per non-trivial edition mapping.
    """
    data_dir = Path(data_dir)

    with open(data_dir / "best_book_id_map.json") as f:
        best_book_id_map = json.load(f)

    edition_to_best_gr = {
        int(eid): int(best_id)
        for best_id, edition_ids in best_book_id_map.items()
        for eid in edition_ids
    }

    gr_lookup = pl.DataFrame({
        "edition_book_id": list(edition_to_best_gr.keys()),
        "best_book_id": list(edition_to_best_gr.values()),
    })

    book_id_map = pl.read_parquet(data_dir / "raw_book_id_map.parquet")

    return (
        gr_lookup
        .join(
            book_id_map.select(
                pl.col("book_id").alias("edition_book_id"),
                pl.col("book_id_csv").alias("edition_csv_id"),
            ),
            on="edition_book_id",
            how="inner",
        )
        .join(
            book_id_map.select(
                pl.col("book_id").alias("best_book_id"),
                pl.col("book_id_csv").alias("best_csv_id"),
            ),
            on="best_book_id",
            how="inner",
        )
        .select("edition_csv_id", "best_csv_id")
        .filter(pl.col("edition_csv_id") != pl.col("best_csv_id"))
    )


def drop_unmatched_book_ids(
    lf: pl.LazyFrame,
    books_ref: pl.DataFrame,
    book_id_col: str = "book_id",
) -> pl.LazyFrame:
    """Remove rows whose book_id has no match in the reference books table.

    Args:
        lf: Source lazy frame to filter.
        books_ref: DataFrame containing valid book IDs (must include ``book_id_col``).
        book_id_col: Name of the book ID column (default ``"book_id"``).

    Returns:
        Filtered lazy frame keeping only rows with a matching book ID.
    """
    return lf.join(books_ref.lazy().select(book_id_col), on=book_id_col, how="semi")
