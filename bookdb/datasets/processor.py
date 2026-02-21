"""
Dataset importer for books and authors.

Supports:
- Generic CSV/Parquet import via import_dataset()
- Goodreads-specific imports via import_authors() and import_books()
"""

import logging
from pathlib import Path
from typing import Any

import polars as pl
from sqlalchemy.orm import Session

from bookdb.db.session import SessionLocal
from .crud import BookCRUD, AuthorCRUD

logger = logging.getLogger(__name__)


# ── Column mappings: Goodreads dataset column → DB column ─────────────

GOODREADS_BOOK_COLUMNS = {
    "title": "title",
    "original_title": "title",  # book_works dataset uses original_title
    "description": "description",
    "image_url": "image_url",
    "publication_year": "publication_year",
    "book_id": "book_id",
    "similar_books": "similar_books",
    # Columns in dataset but NOT stored in our DB:
    # isbn, isbn13, asin, kindle_asin, text_reviews_count, series,
    # country_code, language_code, popular_shelves, is_ebook,
    # average_rating, format, link, publisher, num_pages,
    # publication_day, publication_month, edition_information, url,
    # ratings_count, work_id, title_without_series
}

GOODREADS_AUTHOR_COLUMNS = {
    "name": "name",
    "author_id": "external_id",
    # Columns in dataset but NOT stored in our DB:
    # average_rating, ratings_count, text_reviews_count
}

# Generic CSV/Parquet column mapping (used by import_dataset)
DEFAULT_COLUMNS = {
    "title": "title",
    "authors": "authors",
    "publication_year": "publish_year",
}


# Helpers

def read_file(path: str | Path) -> pl.DataFrame:
    """Read CSV or Parquet file."""
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
        "publication_year": safe_int(row.get(columns["publication_year"])),
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
        "authors_created": 0,
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

                    BookCRUD.create_with_authors(
                        session,
                        author_names=author_names,
                        **book_data,
                    )
                    stats["books_created"] += 1

                except Exception:
                    session.rollback()
                    logger.exception(f"Failed to process row: {row}")
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
    """Return the first n rows of a dataset file as-is."""
    df = read_file(file_path)
    return df.head(n).to_dicts()


def import_authors(
    file_path: str | Path = None,
    batch_size: int = 1000,
    limit: int | None = None,
    session: Session | None = None,
) -> dict[str, int]:
    """
    Import authors from Goodreads authors dataset.

    Expected columns: name, author_id, average_rating, ratings_count, text_reviews_count
    """
    stats = {
        "rows": 0,
        "authors_created": 0,
        "authors_skipped": 0,
        "errors": 0,
    }

    df = pl.read_parquet(file_path)

    if limit:
        df = df.head(limit)

    rows = df.to_dicts()
    stats["rows"] = len(rows)

    own_session = session is None
    if own_session:
        session = SessionLocal()

    seen_names: set[str] = set()

    try:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]

            for row in batch:
                try:
                    name = row.get("name")
                    if not name or not str(name).strip():
                        stats["errors"] += 1
                        continue

                    name = str(name).strip()
                    name_key = name.lower()

                    if name_key in seen_names:
                        stats["authors_skipped"] += 1
                        continue

                    seen_names.add(name_key)

                    ext_id = row.get("author_id")

                    # Skip if author already exists (by external_id or name)
                    if ext_id is not None and AuthorCRUD.get_by_external_id(session, str(ext_id)):
                        stats["authors_skipped"] += 1
                        continue
                    if AuthorCRUD.get_by_name(session, name):
                        stats["authors_skipped"] += 1
                        continue

                    author_data = {}
                    if ext_id is not None:
                        author_data["external_id"] = str(ext_id)

                    AuthorCRUD.create(session, name, **author_data)
                    stats["authors_created"] += 1

                except Exception:
                    session.rollback()
                    logger.exception(f"Failed to process row: {row}")
                    stats["errors"] += 1
                    continue

            if own_session:
                session.commit()

    finally:
        if own_session:
            session.close()

    return stats


def parse_authors_field(raw) -> list[str]:
    """Extract author external IDs from the 'authors' column.

    The column is a list of dicts like [{"author_id": "9212", "role": ""}].
    Returns a list of author_id strings.
    """
    if not raw:
        return []
    if isinstance(raw, list):
        ids = []
        for entry in raw:
            if isinstance(entry, dict):
                aid = entry.get("author_id")
                if aid:
                    ids.append(str(aid))
        return ids
    return []


def import_books(
    file_path: str | Path = None,
    batch_size: int = 10000,
    limit: int | None = None,
    session: Session | None = None,
) -> dict[str, int]:
    """
    Import books from Goodreads books dataset.

    Expected columns: title, publication_year, description, image_url,
                      book_id, authors (list of {author_id, role} dicts)
    """
    stats = {
        "rows": 0,
        "books_created": 0,
        "books_skipped": 0,
        "authors_linked": 0,
        "errors": 0,
    }

    df = pl.read_parquet(file_path)

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

            # Collect all author external IDs in this batch for bulk lookup
            batch_author_ids: set[str] = set()
            for row in batch:
                batch_author_ids.update(parse_authors_field(row.get("authors")))

            author_map = AuthorCRUD.bulk_get_by_external_ids(
                session, list(batch_author_ids)
            )

            for row in batch:
                try:
                    title = row.get("original_title") or row.get("title")
                    if not title or not str(title).strip():
                        stats["errors"] += 1
                        continue

                    title = str(title).strip()
                    title_key = title.lower()

                    if title_key in seen_titles:
                        stats["books_skipped"] += 1
                        continue

                    seen_titles.add(title_key)

                    ext_book_id = str(row["book_id"]) if row.get("book_id") else None

                    # Skip if book already exists (by book_id or title)
                    if ext_book_id and BookCRUD.get_by_book_id(session, ext_book_id):
                        stats["books_skipped"] += 1
                        continue

                    book_data = {
                        "title": title,
                        "publication_year": safe_int(row.get("publication_year")),
                        "description": row.get("description") or None,
                        "image_url": row.get("image_url") or None,
                        "book_id": ext_book_id,
                    }

                    book = BookCRUD.create(session, **book_data)

                    # Link authors via junction table
                    ext_ids = parse_authors_field(row.get("authors"))
                    linked = [author_map[eid] for eid in ext_ids if eid in author_map]
                    if linked:
                        book.authors = linked
                        stats["authors_linked"] += len(linked)

                    stats["books_created"] += 1

                except Exception:
                    session.rollback()
                    logger.exception(f"Failed to process row: {row}")
                    stats["errors"] += 1
                    continue

            if own_session:
                session.commit()

    finally:
        if own_session:
            session.close()

    return stats



