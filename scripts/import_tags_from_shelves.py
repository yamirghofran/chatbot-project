"""Import genre tags from Goodreads popular_shelves into PostgreSQL.

Usage:
    uv run python scripts/import_tags_from_shelves.py

Reads data/3_goodreads_books_with_metrics.parquet, extracts canonical genre tags
using the same logic as notebooks/data/processing/construct_embedding_text.py,
then upserts them into the tags and book_tags tables.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import polars as pl
from sqlalchemy import text as sa_text

# Add project root to path so bookdb is importable when run directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from bookdb.db.crud import TagCRUD
from bookdb.db.session import SessionLocal

DEFAULT_BOOKS_PATH = os.path.join(PROJECT_ROOT, "data", "3_goodreads_books_with_metrics.parquet")
DEFAULT_BATCH_SIZE = 10_000

# ---------------------------------------------------------------------------
# Genre extraction (mirrored from notebooks/data/processing/construct_embedding_text.py)
# ---------------------------------------------------------------------------

GENRE_PATTERNS = [
    ("science fiction", r"\b(?:science[\s_-]*fiction|sci[\s_-]?fi|scifi|sf)\b"),
    ("historical fiction", r"\bhistorical[\s_-]*fiction\b"),
    ("literary fiction", r"\bliterary[\s_-]*fiction\b"),
    ("young adult", r"\b(?:young[\s_-]*adult|ya)\b"),
    ("middle grade", r"\bmiddle[\s_-]*grade\b"),
    ("graphic novel", r"\bgraphic[\s_-]*novel\b"),
    ("self-help", r"\bself[\s_-]*help\b"),
    ("true crime", r"\btrue[\s_-]*crime\b"),
    ("urban fantasy", r"\burban[\s_-]*fantasy\b"),
    ("chick lit", r"\bchick[\s_-]*lit\b"),
    ("fantasy", r"\bfantasy\b"),
    ("romance", r"\bromance\b"),
    ("mystery", r"\bmystery\b"),
    ("thriller", r"\bthriller\b"),
    ("horror", r"\bhorror\b"),
    ("dystopian", r"\bdystopian\b"),
    ("paranormal", r"\bparanormal\b"),
    ("adventure", r"\badventure\b"),
    ("crime", r"\bcrime\b"),
    ("fiction", r"\bfiction\b"),
    ("nonfiction", r"\bnon[\s_-]?fiction\b"),
    ("biography", r"\bbiograph(?:y|ies)\b"),
    ("memoir", r"\bmemoir\b"),
    ("poetry", r"\bpoetry\b"),
    ("classics", r"\bclassics?\b"),
    ("contemporary", r"\bcontemporary\b"),
    ("children", r"\bchildren(?:s)?\b"),
    ("comics", r"\bcomics?\b"),
    ("manga", r"\bmanga\b"),
    ("history", r"\bhistory\b"),
    ("philosophy", r"\bphilosophy\b"),
    ("business", r"\bbusiness\b"),
    ("science", r"\bscience\b"),
]

IGNORE_SHELF_REGEX = (
    r"(?:^|\b)(?:to[\s_-]*read|currently[\s_-]*reading|owned|my[\s_-]*books?|"
    r"favorites?|favourites?|wishlist|wish[\s_-]*list|library|kindle|e[\s_-]?book|"
    r"audiobooks?|book[\s_-]*club|did[\s_-]*not[\s_-]*finish|dnf|series|default)"
    r"(?:\b|$)"
)


def extract_genres_expr() -> pl.Expr:
    normalized_shelves = (
        pl.col("popular_shelves")
        .list.eval(
            pl.element()
            .struct.field("name")
            .fill_null("")
            .str.to_lowercase()
            .str.replace_all(r"[_/]", " ")
            .str.replace_all(r"[^a-z0-9+\-\s]", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )
        .list.eval(
            pl.when(
                (pl.element().str.len_chars() > 0)
                & ~pl.element().str.contains(IGNORE_SHELF_REGEX)
            )
            .then(pl.element())
            .otherwise(None)
        )
        .list.drop_nulls()
    )

    shelf_text = normalized_shelves.list.join(" | ")

    return (
        pl.concat_list(
            [
                pl.when(shelf_text.str.contains(pattern))
                .then(pl.lit(label))
                .otherwise(None)
                for label, pattern in GENRE_PATTERNS
            ]
        )
        .list.drop_nulls()
        .list.unique()
        .list.sort()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import genre tags from Goodreads shelves into PostgreSQL.")
    parser.add_argument(
        "--books-path",
        default=DEFAULT_BOOKS_PATH,
        help="Path to parquet file with book_id and popular_shelves columns.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for book_tags inserts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of books to process from parquet.",
    )
    return parser.parse_args()


def load_books_with_genres(books_path: str, limit: int | None) -> tuple[pl.DataFrame, pl.DataFrame]:
    query = (
        pl.scan_parquet(books_path)
        .select(["book_id", "popular_shelves"])
        .with_columns(genre_list=extract_genres_expr())
        .select(["book_id", "genre_list"])
    )
    if limit is not None:
        query = query.limit(limit)

    books_df = query.collect()
    books_with_genres = books_df.filter(pl.col("genre_list").list.len() > 0)
    return books_df, books_with_genres


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        print("ERROR: --batch-size must be greater than 0.")
        sys.exit(1)

    if not os.path.exists(args.books_path):
        print(f"ERROR: Data file not found: {args.books_path}")
        sys.exit(1)

    started_at = time.perf_counter()
    print(f"Reading {args.books_path} ...")
    books_df, books_with_genres = load_books_with_genres(args.books_path, args.limit)
    print(f"Books with genre tags: {len(books_with_genres):,} / {len(books_df):,}")
    if books_with_genres.is_empty():
        print("No tags extracted. Exiting.")
        return

    with SessionLocal() as session:
        result = session.execute(sa_text("SELECT goodreads_id, id FROM books WHERE goodreads_id IS NOT NULL"))
        goodreads_to_internal: dict[int, int] = {row[0]: row[1] for row in result}
        print(f"Books in DB: {len(goodreads_to_internal):,}")

        all_genres = (
            books_with_genres
            .select(pl.col("genre_list").explode().unique().sort().alias("genre"))
            .get_column("genre")
            .to_list()
        )
        print(f"Unique genres ({len(all_genres)}): {all_genres}")

        tag_map = TagCRUD.bulk_get_or_create(session, all_genres)
        session.commit()
        print(f"Tags upserted: {len(tag_map)}")

        rows_processed = 0
        rows_skipped_missing_book = 0
        rows_link_attempted = 0
        rows_link_reported_inserted = 0
        batch: list[dict] = []
        insert_stmt = sa_text(
            "INSERT INTO book_tags (book_id, tag_id, created_at, updated_at) "
            "VALUES (:book_id, :tag_id, now(), now()) "
            "ON CONFLICT DO NOTHING"
        )

        for goodreads_id, genre_list in books_with_genres.iter_rows():
            internal_id = goodreads_to_internal.get(int(goodreads_id))
            if internal_id is None:
                rows_skipped_missing_book += 1
                continue

            for genre in genre_list:
                tag = tag_map.get(genre)
                if tag is None:
                    continue
                batch.append({"book_id": internal_id, "tag_id": tag.id})
            rows_processed += 1

            if len(batch) >= args.batch_size:
                result = session.execute(insert_stmt, batch)
                rows_link_attempted += len(batch)
                rows_link_reported_inserted += int(getattr(result, "rowcount", 0) or 0)
                batch = []

        if batch:
            result = session.execute(insert_stmt, batch)
            rows_link_attempted += len(batch)
            rows_link_reported_inserted += int(getattr(result, "rowcount", 0) or 0)

        session.commit()
        print(f"Books processed: {rows_processed:,}")
        print(f"Books skipped (missing in DB): {rows_skipped_missing_book:,}")
        print(f"book_tags rows attempted: {rows_link_attempted:,}")
        print(f"book_tags rows reported inserted: {rows_link_reported_inserted:,}")
        print(f"Total runtime: {time.perf_counter() - started_at:.2f}s")
        print("Done.")


if __name__ == "__main__":
    main()
