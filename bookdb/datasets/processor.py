"""
Simple dataset importer for books and authors.

Reads CSV or Parquet files using polars and inserts data into the DB
in batches. Keeps logic obvious and debuggable.
"""

from pathlib import Path
from typing import Any

import polars as pl
from sqlalchemy.orm import Session

from bookdb.db.session import SessionLocal
from .crud import BookCRUD, AuthorCRUD


# Column mappings

DEFAULT_COLUMNS = {
    "title": "title",
    "authors": "authors",
    "pages_number": "pages_number",
    "publisher_name": "publisher_name",
    "publish_year": "publish_year",
}


# Helpers

def read_file(path: str | Path) -> pl.DataFrame:
    # Support both csv and parquet
    
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".csv":
        return pl.read_csv(path, ignore_errors=True)
    if ext == ".parquet":
        return pl.read_parquet(path)

    raise ValueError(f"Unsupported file format: {ext}")


def split_authors(raw: Any) -> list[str]:
    # Split author column is multiple are included
    if not raw:
        return []

    if isinstance(raw, str):
        return [a.strip() for a in raw.split(",") if a.strip()]

    if isinstance(raw, list):
        return [str(a).strip() for a in raw if a]

    return []


def safe_int(val: Any) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def extract_book(row: dict, columns: dict) -> tuple[dict, list[str]]:
    title = row.get(columns["title"])
    if not title or not str(title).strip():
        raise ValueError("Missing title")

    authors = split_authors(row.get(columns["authors"]))

    book_data = {
        "title": str(title).strip(),
        "pages_number": safe_int(row.get(columns["pages_number"])),
        "publisher_name": row.get(columns["publisher_name"]),
        "publish_year": safe_int(row.get(columns["publish_year"])),
    }

    return book_data, authors


# Main processor
def import_dataset(
    file_path: str | Path,
    batch_size: int = 500,
    limit: int | None = None,
    session: Session | None = None,
) -> dict[str, int]:
    """
    Import books + authors from a dataset file.
    """

    stats = {
        "rows": 0,
        "books_created": 0,
        "books_skipped": 0,
        "errors": 0,
    }

    df = read_file(file_path)

    if limit:
        df = df.head(limit)

    rows = df.to_dicts()
    stats["rows"] = len(rows)

    own_session = session is None
    if own_session:
        session = SessionLocal()

    seen_titles: set[str] = set()

    try:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]

            for row in batch:
                try:
                    book_data, author_names = extract_book(row, DEFAULT_COLUMNS)

                    title_key = book_data["title"].lower()
                    if title_key in seen_titles:
                        stats["books_skipped"] += 1
                        continue

                    seen_titles.add(title_key)

                    book = BookCRUD.create_with_authors(
                        session,
                        author_names=author_names,
                        **book_data,
                    )

                    stats["books_created"] += 1

                except Exception:
                    stats["errors"] += 1
                    continue

            if own_session:
                session.commit()

    finally:
        if own_session:
            session.close()

    return stats


# Preview helper to inspect datasets before upload
def preview_dataset(file_path: str | Path, n: int = 5) -> list[dict]:
    df = read_file(file_path)
    rows = df.head(n).to_dicts()

    preview = []

    for row in rows:
        try:
            book, authors = extract_book(row, DEFAULT_COLUMNS)
            preview.append({
                "book": book,
                "authors": authors,
            })
        except Exception as e:
            preview.append({
                "error": str(e),
                "row": row,
            })

    return preview



