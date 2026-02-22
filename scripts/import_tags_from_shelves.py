"""Import genre tags from Goodreads popular_shelves into the database.

Usage:
    python scripts/import_tags_from_shelves.py

Reads data/goodreads_books_standardized.parquet, extracts canonical genre tags
using the same logic as notebooks/data/processing/construct_embedding_text.py,
then upserts them into the tags and book_tags tables.
"""

from __future__ import annotations

import os
import sys

import polars as pl

# Add project root to path so bookdb is importable when run directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from bookdb.db.crud import TagCRUD
from bookdb.db.session import SessionLocal

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


def main() -> None:
    data_path = os.path.join(PROJECT_ROOT, "data", "goodreads_books_standardized.parquet")
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    print(f"Reading {data_path} ...")
    books_df = (
        pl.scan_parquet(data_path)
        .select(["book_id", "popular_shelves"])
        .with_columns(genre_list=extract_genres_expr())
        .select(["book_id", "genre_list"])
        .collect()
    )

    # Filter to books that have at least one genre tag.
    books_with_genres = books_df.filter(pl.col("genre_list").list.len() > 0)
    print(f"Books with genre tags: {len(books_with_genres):,} / {len(books_df):,}")

    with SessionLocal() as session:
        # Resolve goodreads_id -> internal book id.
        from sqlalchemy import text as sa_text
        result = session.execute(sa_text("SELECT goodreads_id, id FROM books"))
        goodreads_to_internal: dict[int, int] = {row[0]: row[1] for row in result}
        print(f"Books in DB: {len(goodreads_to_internal):,}")

        # Collect all unique genre names and pre-create tags.
        all_genres: set[str] = set()
        for genre_list in books_with_genres["genre_list"].to_list():
            all_genres.update(genre_list)
        print(f"Unique genres: {sorted(all_genres)}")

        tag_map = TagCRUD.bulk_get_or_create(session, list(all_genres))
        session.commit()
        print(f"Tags upserted: {len(tag_map)}")

        # Bulk insert book_tags using ON CONFLICT DO NOTHING.
        from sqlalchemy import text as sa_text

        rows_processed = 0
        rows_linked = 0
        batch: list[dict] = []

        for row in books_with_genres.iter_rows(named=True):
            goodreads_id = int(row["book_id"])
            internal_id = goodreads_to_internal.get(goodreads_id)
            if internal_id is None:
                continue
            for genre in row["genre_list"]:
                tag = tag_map.get(genre)
                if tag is None:
                    continue
                batch.append({"book_id": internal_id, "tag_id": tag.id})
            rows_processed += 1

            if len(batch) >= 1000:
                session.execute(
                    sa_text(
                        "INSERT INTO book_tags (book_id, tag_id, created_at, updated_at) "
                        "VALUES (:book_id, :tag_id, now(), now()) "
                        "ON CONFLICT DO NOTHING"
                    ),
                    batch,
                )
                rows_linked += len(batch)
                batch = []

        if batch:
            session.execute(
                sa_text(
                    "INSERT INTO book_tags (book_id, tag_id, created_at, updated_at) "
                    "VALUES (:book_id, :tag_id, now(), now()) "
                    "ON CONFLICT DO NOTHING"
                ),
                batch,
            )
            rows_linked += len(batch)

        session.commit()
        print(f"Books processed: {rows_processed:,}")
        print(f"book_tags rows inserted: {rows_linked:,}")
        print("Done.")


if __name__ == "__main__":
    main()
